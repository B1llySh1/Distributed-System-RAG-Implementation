package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import java.io.{File, PrintWriter}
import scala.io.Source

object BM25Embedder {

  private val NumFeatures = 65536
  private val k1          = 1.2
  private val b           = 0.75

  def buildAndSave(
      spark: SparkSession,
      passagesPath: String,
      outputDir: String,
      modelDir: String
  ): Unit = {
    import spark.implicits._

    val passages = spark.read.parquet(passagesPath)

    // Step 1: tokenize
    val tokenized = TextPreprocessor.transform(passages, "passage_text", "filtered_tokens")

    // Step 2: compute docLen
    val withLen = tokenized.withColumn("docLen", size(col("filtered_tokens")))

    // Step 3: compute avgdl and N
    val statsRow = withLen.agg(count("*").as("N"), avg("docLen").as("avgdl")).head()
    val N        = statsRow.getLong(0)
    val avgdl    = statsRow.getDouble(1)
    println(s"BM25 corpus stats: N=$N, avgdl=$avgdl")

    // Step 4: apply HashingTF
    val hashingTF = new HashingTF()
      .setNumFeatures(NumFeatures)
      .setInputCol("filtered_tokens")
      .setOutputCol("rawTf")

    val tfDf = hashingTF.transform(withLen)

    // Step 5: compute document frequency per feature
    val indicesUDF = udf((v: Vector) => v match {
      case sv: SparseVector => sv.indices.toSeq
      case _                => v.toSparse.indices.toSeq
    })

    val dfDF = tfDf
      .withColumn("feat_index", explode(indicesUDF(col("rawTf"))))
      .groupBy("feat_index")
      .agg(countDistinct("pid").as("docFreq"))
      .select(col("feat_index").as("feature"), col("docFreq"))

    val dfMap: Map[Int, Long] = dfDF.collect()
      .map(r => r.getInt(0) -> r.getLong(1))
      .toMap

    val bcDfMap = spark.sparkContext.broadcast(dfMap)
    val bcN     = N
    val bcAvgdl = avgdl

    // Step 6: BM25 UDF
    val bm25VecUDF = udf((rawTf: Vector, docLen: Int) => {
      val dfMapVal = bcDfMap.value
      val sv       = rawTf match {
        case s: SparseVector => s
        case _               => rawTf.toSparse
      }
      val indices  = sv.indices
      val tfValues = sv.values
      val bm25Values = new Array[Double](indices.length)
      var j = 0
      while (j < indices.length) {
        val feat    = indices(j)
        val tf      = tfValues(j)
        val dfVal   = dfMapVal.getOrElse(feat, 1L).toDouble
        val idf     = math.log((bcN - dfVal + 0.5) / (dfVal + 0.5) + 1)
        val numer   = tf * (k1 + 1)
        val denom   = tf + k1 * (1 - b + b * docLen / bcAvgdl)
        val score   = idf * numer / denom
        bm25Values(j) = math.max(0.0, score)  // clip negatives for MinHashLSH
        j += 1
      }
      Vectors.sparse(rawTf.size, indices, bm25Values)
    })

    // Step 7: apply BM25 UDF
    val result = tfDf.withColumn("features", bm25VecUDF(col("rawTf"), col("docLen")))

    // Step 8: save passage embeddings
    println(s"Saving BM25 embeddings to $outputDir ...")
    result.select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    // Step 9: save model artifacts
    new File(modelDir).mkdirs()

    // Save stats
    val statsPw = new PrintWriter(new File(modelDir + "/bm25-stats.txt"))
    statsPw.println(s"N=$N")
    statsPw.println(s"avgdl=$avgdl")
    statsPw.println(s"numFeatures=$NumFeatures")
    statsPw.close()

    // Save dfDF as Parquet
    dfDF.write.mode("overwrite").parquet(modelDir + "/bm25-df")

    println(s"Saved BM25 stats to $modelDir/bm25-stats.txt")
    println(s"Saved BM25 df to $modelDir/bm25-df")

    bcDfMap.unpersist()
  }

  // ── BM25Stats: pre-loaded corpus statistics ──────────────────────────────
  case class BM25Stats(N: Long, avgdl: Double, numFeatures: Int, dfMap: Map[Int, Long])

  /** Load corpus stats from disk once. Cache the result in BruteForceRetriever / LSHRetriever. */
  def loadStats(spark: SparkSession, modelDir: String): BM25Stats = {
    val statsLines = Source.fromFile(modelDir + "/bm25-stats.txt").getLines().toArray
    val statsMap   = statsLines.map { l => val p = l.split("=", 2); p(0).trim -> p(1).trim }.toMap
    val N           = statsMap("N").toLong
    val avgdl       = statsMap("avgdl").toDouble
    val numFeatures = statsMap("numFeatures").toInt
    val dfMap       = spark.read.parquet(modelDir + "/bm25-df")
      .collect().map(r => r.getInt(0) -> r.getLong(1)).toMap
    BM25Stats(N, avgdl, numFeatures, dfMap)
  }

  /**
   * Apply the BM25 formula to a raw TF SparseVector produced by Spark's HashingTF.
   * Pure local computation — no Spark jobs.
   */
  private def applyBm25Formula(rawTf: Vector, stats: BM25Stats): Vector = {
    val sv     = rawTf match { case s: SparseVector => s; case _ => rawTf.toSparse }
    val docLen = sv.values.sum   // total term count = sum of TF values
    val indices    = sv.indices
    val tfValues   = sv.values
    val bm25Values = new Array[Double](indices.length)
    var j = 0
    while (j < indices.length) {
      val feat  = indices(j)
      val tf    = tfValues(j)
      val dfVal = stats.dfMap.getOrElse(feat, 1L).toDouble
      val idf   = math.log((stats.N - dfVal + 0.5) / (dfVal + 0.5) + 1)
      val numer = tf * (k1 + 1)
      val denom = tf + k1 * (1 - b + b * docLen / stats.avgdl)
      bm25Values(j) = math.max(0.0, idf * numer / denom)
      j += 1
    }
    Vectors.sparse(stats.numFeatures, indices, bm25Values)
  }

  /**
   * Encode a single query as a BM25 sparse vector.
   * Uses Spark's HashingTF (one small Spark job) to guarantee identical feature
   * indices to those used when building the passage index.
   */
  def embedQuery(
      spark: SparkSession,
      query: String,
      modelDir: String,
      preloadedStats: Option[BM25Stats] = None
  ): Vector = {
    import spark.implicits._
    val stats = preloadedStats.getOrElse(loadStats(spark, modelDir))

    val tokens = TextPreprocessor.tokenize(query)
    if (tokens.isEmpty) return Vectors.sparse(stats.numFeatures, Array.empty, Array.empty)

    val queryDF = Seq(Tuple1(tokens.toSeq)).toDF("filtered_tokens")
    val htf     = new HashingTF().setNumFeatures(stats.numFeatures)
      .setInputCol("filtered_tokens").setOutputCol("rawTf")
    val rawTf   = htf.transform(queryDF).select("rawTf").head().getAs[Vector](0)
    applyBm25Formula(rawTf, stats)
  }

  /**
   * Batch-encode all queries in ONE Spark job (HashingTF over a multi-row DataFrame),
   * then apply BM25 formula locally to each row.
   * Returns Array[(qid, BM25Vector)] with zero per-query Spark overhead.
   */
  def embedAllQueries(
      spark: SparkSession,
      queries: Seq[(String, String)],   // (qid, query_text)
      stats: BM25Stats
  ): Array[(String, Vector)] = {
    import spark.implicits._

    val queryDF   = queries.toDF("qid", "query_text")
    val tokenized = TextPreprocessor.transform(queryDF, "query_text", "filtered_tokens")
    val htf       = new HashingTF().setNumFeatures(stats.numFeatures)
      .setInputCol("filtered_tokens").setOutputCol("rawTf")
    val tfDF      = htf.transform(tokenized)

    tfDF.select("qid", "rawTf").collect().map { row =>
      val qid   = row.getString(0)
      val rawTf = row.getAs[Vector](1)
      qid -> applyBm25Formula(rawTf, stats)
    }
  }

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
      .appName("BM25Embedder")
      .getOrCreate()

    buildAndSave(spark, inputPath, outputDir, modelDir)

    spark.stop()
  }
}
