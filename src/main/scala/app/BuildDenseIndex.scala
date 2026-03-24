package app

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object BuildDenseIndex {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var inputTsv   = ""
    var outputDir  = ""
    var passagesParquet = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--input"    => inputTsv        = args(i + 1); i += 2
        case "--output"   => outputDir       = args(i + 1); i += 2
        case "--passages" => passagesParquet = args(i + 1); i += 2
        case _            => i += 1
      }
    }

    require(inputTsv.nonEmpty,  "--input (dense-raw TSV) is required")
    require(outputDir.nonEmpty, "--output (parquet dir) is required")

    val spark = SparkSession.builder()
      .appName("BuildDenseIndex")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    println(s"Reading dense embeddings from $inputTsv ...")

    // Each line: "pid\tf1,f2,...,f384"
    val rawRDD = spark.sparkContext.textFile(inputTsv)

    // Parse each line into (pid, Array[Double]) — avoids VectorUDT access issues
    val parsed = rawRDD
      .filter(_.nonEmpty)
      .map { line =>
        val parts  = line.split("\t", 2)
        val pid    = parts(0).trim
        val floats = parts(1).trim.split(",").map(_.trim.toDouble)
        (pid, floats)
      }

    var embeddingsDF = parsed.toDF("pid", "embedding")

    // Optionally join with passages to attach passage_text
    if (passagesParquet.nonEmpty) {
      val passages = spark.read.parquet(passagesParquet).select("pid", "passage_text")
      embeddingsDF = embeddingsDF.join(passages, "pid")
    }

    println(s"Saving dense index to $outputDir ...")
    embeddingsDF.write.mode("overwrite").parquet(outputDir)
    println(s"Done. ${embeddingsDF.count()} passages indexed.")

    spark.stop()
  }
}
