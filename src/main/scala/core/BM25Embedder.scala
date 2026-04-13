package core

import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import java.io.{File, PrintWriter}
import scala.io.Source

object BM25Embedder {

  private val NumFeatures = 65536
  private val k1 = 1.2
  private val b  = 0.75

  case class BM25Stats(N: Long, avgdl: Double, numFeatures: Int, dfMap: Map[Int, Long])

  def buildAndSave(spark: SparkSession, passagesPath: String, outputDir: String, modelDir: String): Unit = {
    import spark.implicits._

    val passages  = spark.read.parquet(passagesPath)
    val tokenized = TextPreprocessor.transform(passages, "passage_text", "filtered_tokens")
    val withLen   = tokenized.withColumn("docLen", size(col("filtered_tokens")))

    val statsRow = withLen.agg(count("*").as("N"), avg("docLen").as("avgdl")).head()
    val N     = statsRow.getLong(0)
    val avgdl = statsRow.getDouble(1)
    println(s"BM25 corpus: N=$N  avgdl=$avgdl")

    val htf  = new HashingTF().setNumFeatures(NumFeatures).setInputCol("filtered_tokens").setOutputCol("rawTf")
    val tfDf = htf.transform(withLen)

    // Extract non-zero feature indices per doc to count document frequency
    val indicesUDF = udf((v: Vector) => v match {
      case sv: SparseVector => sv.indices.toSeq
      case _                => v.toSparse.indices.toSeq
    })

    val dfDF  = tfDf
      .withColumn("feat_index", explode(indicesUDF(col("rawTf"))))
      .groupBy("feat_index").agg(countDistinct("pid").as("docFreq"))
      .select(col("feat_index").as("feature"), col("docFreq"))

    val dfMap    = dfDF.collect().map(r => r.getInt(0) -> r.getLong(1)).toMap
    val bcDfMap  = spark.sparkContext.broadcast(dfMap)  // small enough to broadcast (~65K entries)

    val bm25UDF = udf((rawTf: Vector, docLen: Int) => {
      val sv = rawTf match { case s: SparseVector => s; case _ => rawTf.toSparse }
      val bm25Values = sv.indices.indices.map { j =>
        val tf    = sv.values(j)
        val df    = bcDfMap.value.getOrElse(sv.indices(j), 1L).toDouble
        val idf   = math.log((N - df + 0.5) / (df + 0.5) + 1)
        val score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * docLen / avgdl))
        math.max(0.0, score)  // clip negatives (can happen with very high df)
      }.toArray
      Vectors.sparse(rawTf.size, sv.indices, bm25Values)
    })

    tfDf.withColumn("features", bm25UDF(col("rawTf"), col("docLen")))
      .select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    new File(modelDir).mkdirs()

    val pw = new PrintWriter(new File(modelDir + "/bm25-stats.txt"))
    pw.println(s"N=$N"); pw.println(s"avgdl=$avgdl"); pw.println(s"numFeatures=$NumFeatures")
    pw.close()

    dfDF.write.mode("overwrite").parquet(modelDir + "/bm25-df")
    bcDfMap.unpersist()

    println(s"BM25 embeddings saved to $outputDir")
  }

  def loadStats(spark: SparkSession, modelDir: String): BM25Stats = {
    val m = Source.fromFile(modelDir + "/bm25-stats.txt").getLines()
      .map { l => val p = l.split("=", 2); p(0).trim -> p(1).trim }.toMap
    val dfMap = spark.read.parquet(modelDir + "/bm25-df")
      .collect().map(r => r.getInt(0) -> r.getLong(1)).toMap
    BM25Stats(m("N").toLong, m("avgdl").toDouble, m("numFeatures").toInt, dfMap)
  }

  // Pure local BM25 scoring on a raw-TF vector from Spark's HashingTF.
  // Must use the same HashingTF (Murmur3 seed=42) that built the passage index — don't re-hash locally.
  private def applyBm25Formula(rawTf: Vector, stats: BM25Stats): Vector = {
    val sv     = rawTf match { case s: SparseVector => s; case _ => rawTf.toSparse }
    val docLen = sv.values.sum
    val scores = sv.indices.indices.map { j =>
      val tf  = sv.values(j)
      val df  = stats.dfMap.getOrElse(sv.indices(j), 1L).toDouble
      val idf = math.log((stats.N - df + 0.5) / (df + 0.5) + 1)
      math.max(0.0, idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * docLen / stats.avgdl)))
    }.toArray
    Vectors.sparse(stats.numFeatures, sv.indices, scores)
  }

  def embedQuery(spark: SparkSession, query: String, modelDir: String,
                 preloadedStats: Option[BM25Stats] = None): Vector = {
    import spark.implicits._
    val stats = preloadedStats.getOrElse(loadStats(spark, modelDir))
    val tokens = TextPreprocessor.tokenize(query)
    if (tokens.isEmpty) return Vectors.sparse(stats.numFeatures, Array.empty, Array.empty)

    // One small Spark job ensures we use the same Murmur3 hash as the passage index
    val htf   = new HashingTF().setNumFeatures(stats.numFeatures).setInputCol("filtered_tokens").setOutputCol("rawTf")
    val rawTf = htf.transform(Seq(Tuple1(tokens.toSeq)).toDF("filtered_tokens")).select("rawTf").head().getAs[Vector](0)
    applyBm25Formula(rawTf, stats)
  }

  // Hash all queries in one Spark job, then apply BM25 formula locally — no per-query overhead.
  def embedAllQueries(spark: SparkSession, queries: Seq[(String, String)], stats: BM25Stats): Array[(String, Vector)] = {
    import spark.implicits._
    val tokenized = TextPreprocessor.transform(queries.toDF("qid", "query_text"), "query_text", "filtered_tokens")
    val htf       = new HashingTF().setNumFeatures(stats.numFeatures).setInputCol("filtered_tokens").setOutputCol("rawTf")
    htf.transform(tokenized).select("qid", "rawTf").collect()
      .map(r => r.getString(0) -> applyBm25Formula(r.getAs[Vector](1), stats))
  }
}
