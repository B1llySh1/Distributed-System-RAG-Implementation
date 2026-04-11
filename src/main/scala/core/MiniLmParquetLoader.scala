package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object MiniLmParquetLoader {

  /**
   * Load passage embeddings from TSV, join with passages parquet to get passage_text,
   * convert embeddings to DenseVector stored in "features" column, save as Parquet.
   *
   * TSV format: pid\tf1,f2,...,f384
   */
  def loadPassageEmbeddings(
      spark: SparkSession,
      tsvPath: String,
      passagesParquetPath: String,
      outputDir: String
  ): Unit = {
    import spark.implicits._

    val toVecUDF = udf((arr: Seq[Double]) => Vectors.dense(arr.toArray): Vector)

    // Read TSV lines
    val rawLines = spark.sparkContext.textFile(tsvPath)
    val embDF = rawLines.map { line =>
      val tab  = line.indexOf('\t')
      val pid  = line.substring(0, tab)
      val vals = line.substring(tab + 1).split(",").map(_.toDouble)
      (pid, vals.toSeq)
    }.toDF("pid", "embedding_arr")

    // Join with passages to get passage_text
    val passages = spark.read.parquet(passagesParquetPath).select("pid", "passage_text")
    val joined   = embDF.join(passages, "pid")

    // Convert embedding Array[Double] to DenseVector
    val result = joined.withColumn("features", toVecUDF(col("embedding_arr")))

    println(s"Saving MiniLM passage embeddings to $outputDir ...")
    result.select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)
  }

  /**
   * Load query embeddings from TSV, save as Parquet with columns (qid, features: DenseVector).
   *
   * TSV format: qid\tf1,f2,...,f384
   */
  def loadQueryEmbeddings(
      spark: SparkSession,
      tsvPath: String,
      outputDir: String
  ): Unit = {
    import spark.implicits._

    val toVecUDF = udf((arr: Seq[Double]) => Vectors.dense(arr.toArray): Vector)

    val rawLines = spark.sparkContext.textFile(tsvPath)
    val embDF = rawLines.map { line =>
      val tab  = line.indexOf('\t')
      val qid  = line.substring(0, tab)
      val vals = line.substring(tab + 1).split(",").map(_.toDouble)
      (qid, vals.toSeq)
    }.toDF("qid", "embedding_arr")

    val result = embDF.withColumn("features", toVecUDF(col("embedding_arr")))

    println(s"Saving MiniLM query embeddings to $outputDir ...")
    result.select("qid", "features")
      .write.mode("overwrite").parquet(outputDir)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var inputPath    = ""
    var passagesPath = ""
    var outputDir    = ""
    var embType      = "passage"

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--input"    => inputPath    = args(i + 1); i += 2
        case "--passages" => passagesPath = args(i + 1); i += 2
        case "--output"   => outputDir    = args(i + 1); i += 2
        case "--type"     => embType      = args(i + 1); i += 2
        case _            => i += 1
      }
    }

    require(inputPath.nonEmpty,  "--input is required")
    require(outputDir.nonEmpty,  "--output is required")

    val spark = SparkSession.builder()
      .appName("MiniLmParquetLoader")
      .getOrCreate()

    embType match {
      case "passage" =>
        require(passagesPath.nonEmpty, "--passages is required for type=passage")
        loadPassageEmbeddings(spark, inputPath, passagesPath, outputDir)
      case "query" =>
        loadQueryEmbeddings(spark, inputPath, outputDir)
      case other =>
        throw new IllegalArgumentException(s"Unknown --type: $other. Expected 'passage' or 'query'.")
    }

    spark.stop()
  }
}
