package core

import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import java.io.{File, PrintWriter}
import scala.io.Source

object Word2VecEmbedder {

  // Plain average-pooled Word2Vec embeddings.
  def buildAndSave(spark: SparkSession, passagesPath: String, outputDir: String, modelDir: String): Unit = {
    import spark.implicits._

    val tokenized = TextPreprocessor.transform(spark.read.parquet(passagesPath), "passage_text", "filtered_tokens")

    val w2v = new Word2Vec()
      .setInputCol("filtered_tokens").setOutputCol("features")
      .setVectorSize(100).setMinCount(5).setMaxIter(1).setWindowSize(5)

    println("Training Word2Vec...")
    val model = w2v.fit(tokenized)

    new File(modelDir).mkdirs()
    model.write.overwrite().save(modelDir + "/word2vec")

    model.transform(tokenized)
      .select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    println(s"Word2Vec embeddings saved to $outputDir")
  }

  def embedQuery(spark: SparkSession, queryText: String, modelDir: String,
                 preloadedModel: Option[Word2VecModel] = None): Vector = {
    import spark.implicits._
    val model     = preloadedModel.getOrElse(Word2VecModel.load(modelDir + "/word2vec"))
    val tokens    = TextPreprocessor.tokenize(queryText)
    model.setInputCol("filtered_tokens").setOutputCol("features")
      .transform(Seq(Tuple1(tokens.toSeq)).toDF("filtered_tokens"))
      .select("features").head().getAs[Vector](0)
  }

  // SIF (Arora et al. 2017): weight each word vector by a/(a+p(w)) then remove first PC.
  // Better than plain average pooling — rare words contribute more, common words less.
  def buildAndSaveSIF(spark: SparkSession, passagesPath: String, outputDir: String,
                      modelDir: String, a: Double = 1e-3, removePc: Boolean = true): Unit = {
    import spark.implicits._

    val tokenized = TextPreprocessor.transform(spark.read.parquet(passagesPath), "passage_text", "filtered_tokens")
    tokenized.cache()

    val modelPath = modelDir + "/word2vec"
    val model = if (new File(modelPath).exists()) {
      println("Reusing existing Word2Vec model for SIF...")
      Word2VecModel.load(modelPath)
    } else {
      println("Training Word2Vec for SIF...")
      val m = new Word2Vec()
        .setInputCol("filtered_tokens").setOutputCol("_raw")
        .setVectorSize(100).setMinCount(5).setMaxIter(1).setWindowSize(5)
        .fit(tokenized)
      new File(modelDir).mkdirs()
      m.write.overwrite().save(modelPath)
      m
    }

    val wordVecs: Map[String, Array[Double]] = model.getVectors.collect()
      .map(r => r.getString(0) -> r.getAs[DenseVector](1).values).toMap
    val dim = wordVecs.head._2.length

    println("Computing word frequencies for SIF weights...")
    val wordCounts = tokenized.select(explode(col("filtered_tokens")).as("word"))
      .groupBy("word").count().collect()
    val total = wordCounts.map(_.getLong(1)).sum.toDouble
    val sifWeights = wordCounts.map(r => r.getString(0) -> (a / (a + r.getLong(1) / total))).toMap

    val pw = new PrintWriter(new File(modelDir + "/word2vec-sif-weights.txt"))
    sifWeights.foreach { case (w, wt) => pw.println(s"$w\t$wt") }
    pw.close()

    val bcVecs    = spark.sparkContext.broadcast(wordVecs)    // fits in memory, broadcast once
    val bcWeights = spark.sparkContext.broadcast(sifWeights)

    val sifUDF = udf { (tokens: Seq[String]) =>
      val result = Array.fill(dim)(0.0)
      var cnt    = 0
      tokens.foreach { w =>
        bcVecs.value.get(w).foreach { vec =>
          val wt = bcWeights.value.getOrElse(w, a / (a + 1e-9))
          var j  = 0; while (j < dim) { result(j) += wt * vec(j); j += 1 }
          cnt += 1
        }
      }
      if (cnt > 0) { var j = 0; while (j < dim) { result(j) /= cnt; j += 1 } }
      new DenseVector(result)
    }

    val embedded = tokenized.withColumn("features", sifUDF(col("filtered_tokens")))

    val finalDF = if (removePc) {
      println("Computing first principal component for removal...")
      val allVecs = embedded.select("features").collect().map(_.getAs[DenseVector](0).values)
      val pc      = firstPC(allVecs, dim)

      val pcPw = new PrintWriter(new File(modelDir + "/word2vec-sif-pc.txt"))
      pcPw.println(pc.mkString(" ")); pcPw.close()

      val bcPC = spark.sparkContext.broadcast(pc)
      val removePcUDF = udf { (v: Vector) =>
        val arr = v.toArray
        val dot = arr.indices.map(j => arr(j) * bcPC.value(j)).sum
        new DenseVector(arr.indices.map(j => arr(j) - dot * bcPC.value(j)).toArray)
      }
      embedded.withColumn("features", removePcUDF(col("features")))
    } else embedded

    finalDF.select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    tokenized.unpersist()
    println(s"Word2Vec-SIF embeddings saved to $outputDir")
  }

  // Power-iteration to find the dominant direction in the embedding space.
  private def firstPC(vecs: Array[Array[Double]], dim: Int): Array[Double] = {
    val mean = Array.fill(dim)(0.0)
    vecs.foreach(v => { var j = 0; while (j < dim) { mean(j) += v(j); j += 1 } })
    var j0 = 0; while (j0 < dim) { mean(j0) /= vecs.length; j0 += 1 }

    val rng = new scala.util.Random(42)
    var pc  = Array.fill(dim)(rng.nextDouble() - 0.5)
    val n0  = math.sqrt(pc.map(x => x * x).sum)
    pc = pc.map(_ / n0)

    for (_ <- 0 until 100) {
      val next = Array.fill(dim)(0.0)
      vecs.foreach { v =>
        val c   = v.indices.map(j => v(j) - mean(j)).toArray
        val dot = c.indices.map(j => c(j) * pc(j)).sum
        var j   = 0; while (j < dim) { next(j) += dot * c(j); j += 1 }
      }
      val norm = math.sqrt(next.map(x => x * x).sum)
      pc = if (norm > 1e-12) next.map(_ / norm) else next
    }
    pc
  }

  // Load helpers — called once at retriever startup.
  def loadSIFWeights(modelDir: String): Map[String, Double] = {
    val src = Source.fromFile(modelDir + "/word2vec-sif-weights.txt")
    try src.getLines().filter(_.nonEmpty).map { l => val t = l.split("\t"); t(0) -> t(1).toDouble }.toMap
    finally src.close()
  }

  def loadSIFPC(modelDir: String): Option[Array[Double]] = {
    val f = new File(modelDir + "/word2vec-sif-pc.txt")
    if (!f.exists()) None
    else { val src = Source.fromFile(f); try Some(src.mkString.trim.split(" ").map(_.toDouble)) finally src.close() }
  }

  def loadWordVecs(modelDir: String): Map[String, Array[Double]] =
    Word2VecModel.load(modelDir + "/word2vec").getVectors.collect()
      .map(r => r.getString(0) -> r.getAs[DenseVector](1).values).toMap

  // Fully local SIF encoding — no Spark job needed at query time.
  def embedQuerySIF(queryText: String, wordVecs: Map[String, Array[Double]],
                    sifWeights: Map[String, Double], pc: Option[Array[Double]], a: Double = 1e-3): Vector = {
    val tokens = TextPreprocessor.tokenize(queryText)
    val dim    = wordVecs.head._2.length
    val result = Array.fill(dim)(0.0)
    var cnt    = 0
    tokens.foreach { w =>
      wordVecs.get(w).foreach { vec =>
        val wt = sifWeights.getOrElse(w, a / (a + 1e-9))
        var j  = 0; while (j < dim) { result(j) += wt * vec(j); j += 1 }
        cnt += 1
      }
    }
    if (cnt > 0) { var j = 0; while (j < dim) { result(j) /= cnt; j += 1 } }
    pc.foreach { pcArr =>
      val dot = result.indices.map(j => result(j) * pcArr(j)).sum
      var j   = 0; while (j < dim) { result(j) -= dot * pcArr(j); j += 1 }
    }
    new DenseVector(result)
  }

  def embedAllQueriesSIF(queries: Seq[(String, String)], wordVecs: Map[String, Array[Double]],
                         sifWeights: Map[String, Double], pc: Option[Array[Double]]): Array[(String, Vector)] =
    queries.map { case (qid, text) => qid -> embedQuerySIF(text, wordVecs, sifWeights, pc) }.toArray
}
