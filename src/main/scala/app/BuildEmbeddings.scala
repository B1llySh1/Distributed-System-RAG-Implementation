package app

import core.{BM25Embedder, TfIdfEmbedder, Word2VecEmbedder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

// Unified embedding builder.
//
// Usage:
//   --method  {tfidf|bm25|word2vec|word2vec-sif}
//   --input   <passages parquet>
//   --output  <output parquet dir>
//   --model   <model save dir>
object BuildEmbeddings {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var method    = ""
    var inputPath = ""
    var outputDir = ""
    var modelDir  = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--method" => method    = args(i + 1); i += 2
        case "--input"  => inputPath = args(i + 1); i += 2
        case "--output" => outputDir = args(i + 1); i += 2
        case "--model"  => modelDir  = args(i + 1); i += 2
        case _          => i += 1
      }
    }

    require(method.nonEmpty,    "--method is required")
    require(inputPath.nonEmpty, "--input is required")
    require(outputDir.nonEmpty, "--output is required")
    require(modelDir.nonEmpty,  "--model is required")

    val spark = SparkSession.builder().appName(s"BuildEmbeddings-$method").getOrCreate()

    method match {
      case "tfidf"        => TfIdfEmbedder.buildAndSave(spark, inputPath, outputDir, modelDir)
      case "bm25"         => BM25Embedder.buildAndSave(spark, inputPath, outputDir, modelDir)
      case "word2vec"     => Word2VecEmbedder.buildAndSave(spark, inputPath, outputDir, modelDir)
      case "word2vec-sif" => Word2VecEmbedder.buildAndSaveSIF(spark, inputPath, outputDir, modelDir)
      case other => throw new IllegalArgumentException(
        s"Unknown --method: $other. Expected: tfidf, bm25, word2vec, word2vec-sif")
    }

    spark.stop()
  }
}
