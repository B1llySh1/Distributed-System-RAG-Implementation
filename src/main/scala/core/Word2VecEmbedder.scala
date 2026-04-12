package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, explode}

import java.io.{File, PrintWriter}
import scala.io.Source

object Word2VecEmbedder {

  // ── Plain Word2Vec ──────────────────────────────────────────────────────────

  def buildAndSave(
      spark: SparkSession,
      passagesPath: String,
      outputDir: String,
      modelDir: String
  ): Unit = {
    import spark.implicits._

    val passages    = spark.read.parquet(passagesPath)
    val tokenizedDf = TextPreprocessor.transform(passages, "passage_text", "filtered_tokens")

    val word2vec = new Word2Vec()
      .setInputCol("filtered_tokens")
      .setOutputCol("features")
      .setVectorSize(100)
      .setMinCount(5)
      .setMaxIter(1)
      .setWindowSize(5)

    println("Training Word2Vec...")
    val w2vModel = word2vec.fit(tokenizedDf)
    val result   = w2vModel.transform(tokenizedDf)

    new File(modelDir).mkdirs()
    w2vModel.write.overwrite().save(modelDir + "/word2vec")

    println(s"Saving Word2Vec embeddings to $outputDir ...")
    result.select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    println(s"Saved Word2Vec model to $modelDir/word2vec")
  }

  def embedQuery(
      spark: SparkSession,
      queryText: String,
      modelDir: String,
      preloadedModel: Option[Word2VecModel] = None
  ): Vector = {
    import spark.implicits._

    val w2vModel = preloadedModel.getOrElse(Word2VecModel.load(modelDir + "/word2vec"))
    val tokens   = TextPreprocessor.tokenize(queryText)
    val queryDF  = Seq(Tuple1(tokens.toSeq)).toDF("filtered_tokens")
    w2vModel.setInputCol("filtered_tokens").setOutputCol("features")
      .transform(queryDF).select("features").head().getAs[Vector](0)
  }

  // ── Word2Vec-SIF ────────────────────────────────────────────────────────────
  //
  // Arora et al. 2017 "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"
  //
  // v(sentence) = (1/|s|) * Σ  a/(a+p(w)) * wordVec(w)
  // Then optionally subtract the first principal component (common component removal).

  /** Build SIF-weighted Word2Vec embeddings and save to outputDir. */
  def buildAndSaveSIF(
      spark: SparkSession,
      passagesPath: String,
      outputDir: String,
      modelDir: String,
      a: Double       = 1e-3,
      removePc: Boolean = true
  ): Unit = {
    import spark.implicits._

    val passages  = spark.read.parquet(passagesPath)
    val tokenized = TextPreprocessor.transform(passages, "passage_text", "filtered_tokens")
    tokenized.cache()

    // Reuse existing Word2Vec model if present, otherwise train a new one
    val modelPath = modelDir + "/word2vec"
    val w2vModel  =
      if (new File(modelPath).exists()) {
        println("Loading existing Word2Vec model...")
        Word2VecModel.load(modelPath)
      } else {
        println("Training Word2Vec model...")
        val m = new Word2Vec()
          .setInputCol("filtered_tokens").setOutputCol("_raw")
          .setVectorSize(100).setMinCount(5).setMaxIter(1).setWindowSize(5)
          .fit(tokenized)
        new File(modelDir).mkdirs()
        m.write.overwrite().save(modelPath)
        m
      }

    // Collect word vectors to driver: word -> Array[Double]
    val wordVecsMap: Map[String, Array[Double]] = w2vModel.getVectors.collect()
      .map(r => r.getString(0) -> r.getAs[DenseVector](1).values)
      .toMap
    val dim = wordVecsMap.head._2.length

    // Compute word frequencies from the corpus
    println("Computing corpus word frequencies...")
    val wordCountRows = tokenized
      .select(explode(col("filtered_tokens")).as("word"))
      .groupBy("word").count().collect()
    val totalCount = wordCountRows.map(_.getLong(1)).sum.toDouble

    // Precompute SIF weights: a / (a + p(w))
    val sifWeights: Map[String, Double] = wordCountRows.map { r =>
      val freq = r.getLong(1).toDouble / totalCount
      r.getString(0) -> (a / (a + freq))
    }.toMap

    // Save SIF weights to file for later use at query time
    val weightsFile = modelDir + "/word2vec-sif-weights.txt"
    val pw = new PrintWriter(new File(weightsFile))
    sifWeights.foreach { case (w, wt) => pw.println(s"$w\t$wt") }
    pw.close()
    println(s"Saved ${sifWeights.size} SIF weights to $weightsFile")

    // Broadcast for UDF
    val bcVecs    = spark.sparkContext.broadcast(wordVecsMap)
    val bcWeights = spark.sparkContext.broadcast(sifWeights)
    val bcDim     = dim
    val bcA       = a

    import org.apache.spark.sql.functions.udf
    val sifUDF = udf { (tokens: Seq[String]) =>
      val vecs    = bcVecs.value
      val weights = bcWeights.value
      val result  = Array.fill(bcDim)(0.0)
      var cnt     = 0
      tokens.foreach { w =>
        vecs.get(w).foreach { vec =>
          val wt = weights.getOrElse(w, bcA / (bcA + 1e-9))
          var j = 0; while (j < bcDim) { result(j) += wt * vec(j); j += 1 }
          cnt += 1
        }
      }
      if (cnt > 0) { var j = 0; while (j < bcDim) { result(j) /= cnt; j += 1 } }
      new DenseVector(result)
    }

    println("Computing SIF embeddings...")
    val embedded = tokenized.withColumn("features", sifUDF(col("filtered_tokens")))

    val finalDF = if (removePc) {
      println("Computing first principal component for removal...")
      val allVecs = embedded.select("features").collect()
        .map(_.getAs[DenseVector](0).values)
      val pc = computeFirstPC(allVecs, dim)

      val pcFile = new PrintWriter(new File(modelDir + "/word2vec-sif-pc.txt"))
      pcFile.println(pc.mkString(" "))
      pcFile.close()
      println(s"Saved first PC to $modelDir/word2vec-sif-pc.txt")

      val bcPC = spark.sparkContext.broadcast(pc)
      val removePcUDF = udf { (v: Vector) =>
        val arr = v.toArray; val pcArr = bcPC.value
        val dot = arr.indices.map(j => arr(j) * pcArr(j)).sum
        new DenseVector(arr.indices.map(j => arr(j) - dot * pcArr(j)).toArray)
      }
      embedded.withColumn("features", removePcUDF(col("features")))
    } else embedded

    println(s"Saving SIF embeddings to $outputDir ...")
    finalDF.select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    tokenized.unpersist()
    println("Word2Vec-SIF embeddings built and saved.")
  }

  /** Power-iteration SVD to compute the first principal component. */
  private def computeFirstPC(vecs: Array[Array[Double]], dim: Int): Array[Double] = {
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
        val centered = v.indices.map(j => v(j) - mean(j)).toArray
        val dot      = centered.indices.map(j => centered(j) * pc(j)).sum
        var j = 0; while (j < dim) { next(j) += dot * centered(j); j += 1 }
      }
      val norm = math.sqrt(next.map(x => x * x).sum)
      pc = if (norm > 1e-12) next.map(_ / norm) else next
    }
    pc
  }

  // ── SIF helpers (used by BruteForceRetriever / LSHRetriever) ────────────────

  def loadSIFWeights(modelDir: String): Map[String, Double] = {
    val src = Source.fromFile(modelDir + "/word2vec-sif-weights.txt")
    try src.getLines().filter(_.nonEmpty).map { line =>
      val t = line.split("\t"); t(0) -> t(1).toDouble
    }.toMap
    finally src.close()
  }

  def loadSIFPC(modelDir: String): Option[Array[Double]] = {
    val f = new File(modelDir + "/word2vec-sif-pc.txt")
    if (!f.exists()) None
    else {
      val src = Source.fromFile(f)
      try Some(src.mkString.trim.split(" ").map(_.toDouble))
      finally src.close()
    }
  }

  def loadWordVecs(modelDir: String): Map[String, Array[Double]] =
    Word2VecModel.load(modelDir + "/word2vec").getVectors.collect()
      .map(r => r.getString(0) -> r.getAs[DenseVector](1).values)
      .toMap

  /** Encode a single query with SIF weighting + optional PC removal. Fully local — no Spark. */
  def embedQuerySIF(
      queryText: String,
      wordVecsMap: Map[String, Array[Double]],
      sifWeights: Map[String, Double],
      pc: Option[Array[Double]],
      a: Double = 1e-3
  ): Vector = {
    val tokens = TextPreprocessor.tokenize(queryText)
    val dim    = wordVecsMap.head._2.length
    val result = Array.fill(dim)(0.0)
    var cnt    = 0
    tokens.foreach { w =>
      wordVecsMap.get(w).foreach { vec =>
        val wt = sifWeights.getOrElse(w, a / (a + 1e-9))
        var j = 0; while (j < dim) { result(j) += wt * vec(j); j += 1 }
        cnt += 1
      }
    }
    if (cnt > 0) { var j = 0; while (j < dim) { result(j) /= cnt; j += 1 } }
    pc.foreach { pcArr =>
      val dot = result.indices.map(j => result(j) * pcArr(j)).sum
      var j = 0; while (j < dim) { result(j) -= dot * pcArr(j); j += 1 }
    }
    new DenseVector(result)
  }

  /** Encode all queries with SIF — fully local, no Spark job needed. */
  def embedAllQueriesSIF(
      queries: Seq[(String, String)],
      wordVecsMap: Map[String, Array[Double]],
      sifWeights: Map[String, Double],
      pc: Option[Array[Double]],
      a: Double = 1e-3
  ): Array[(String, Vector)] =
    queries.map { case (qid, text) =>
      qid -> embedQuerySIF(text, wordVecsMap, sifWeights, pc, a)
    }.toArray

  // ── Entry point ─────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var inputPath = ""
    var outputDir = ""
    var modelDir  = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--input"  => inputPath = args(i + 1); i += 2
        case "--output" => outputDir = args(i + 1); i += 2
        case "--model"  => modelDir  = args(i + 1); i += 2
        case _          => i += 1
      }
    }

    require(inputPath.nonEmpty, "--input is required")
    require(outputDir.nonEmpty, "--output is required")
    require(modelDir.nonEmpty,  "--model is required")

    val spark = SparkSession.builder()
      .appName("Word2VecEmbedder")
      .getOrCreate()

    buildAndSave(spark, inputPath, outputDir, modelDir)

    spark.stop()
  }
}
