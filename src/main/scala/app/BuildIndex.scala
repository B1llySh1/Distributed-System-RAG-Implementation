package app

import core.TfIdfEmbedder
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object BuildIndex {

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

    require(inputPath.nonEmpty, "--input (passages parquet dir) is required")
    require(outputDir.nonEmpty, "--output (embeddings dir) is required")
    require(modelDir.nonEmpty,  "--model (model save dir) is required")

    val spark = SparkSession.builder()
      .appName("BuildIndex")
      .master("local[*]")
      .getOrCreate()

    println(s"Building TF-IDF index from $inputPath ...")
    TfIdfEmbedder.buildAndSave(spark, inputPath, outputDir, modelDir)
    println("Index build complete.")

    spark.stop()
  }
}
