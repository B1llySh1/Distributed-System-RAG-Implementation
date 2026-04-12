package app

import core.{BM25Embedder, MiniLmParquetLoader, TfIdfEmbedder, Word2VecEmbedder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

/**
 * Unified embedding builder. Replaces BuildIndex + BuildDenseIndex.
 *
 * Usage:
 *   --method  {tfidf|bm25|word2vec|minilm-passage|minilm-query}
 *   --input   <passages parquet OR raw TSV for minilm>
 *   --output  <output parquet dir>
 *   --model   <model save dir>          (not needed for minilm-*)
 *   --passages <passages parquet>       (only for minilm-passage, to join passage_text)
 */
object BuildEmbeddings {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var method       = ""
    var inputPath    = ""
    var outputDir    = ""
    var modelDir     = ""
    var passagesPath = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--method"   => method       = args(i + 1); i += 2
        case "--input"    => inputPath    = args(i + 1); i += 2
        case "--output"   => outputDir    = args(i + 1); i += 2
        case "--model"    => modelDir     = args(i + 1); i += 2
        case "--passages" => passagesPath = args(i + 1); i += 2
        case _            => i += 1
      }
    }

    require(method.nonEmpty,    "--method is required")
    require(inputPath.nonEmpty, "--input is required")
    require(outputDir.nonEmpty, "--output is required")

    val spark = SparkSession.builder()
      .appName(s"BuildEmbeddings-$method")
      .getOrCreate()

    method match {
      case "tfidf" =>
        require(modelDir.nonEmpty, "--model is required for method=tfidf")
        println(s"Building TF-IDF embeddings from $inputPath ...")
        TfIdfEmbedder.buildAndSave(spark, inputPath, outputDir, modelDir)

      case "bm25" =>
        require(modelDir.nonEmpty, "--model is required for method=bm25")
        println(s"Building BM25 embeddings from $inputPath ...")
        BM25Embedder.buildAndSave(spark, inputPath, outputDir, modelDir)

      case "word2vec" =>
        require(modelDir.nonEmpty, "--model is required for method=word2vec")
        println(s"Building Word2Vec embeddings from $inputPath ...")
        Word2VecEmbedder.buildAndSave(spark, inputPath, outputDir, modelDir)

      case "word2vec-sif" =>
        require(modelDir.nonEmpty, "--model is required for method=word2vec-sif")
        println(s"Building Word2Vec-SIF embeddings from $inputPath ...")
        Word2VecEmbedder.buildAndSaveSIF(spark, inputPath, outputDir, modelDir)

      case "minilm-passage" =>
        require(passagesPath.nonEmpty, "--passages is required for method=minilm-passage")
        println(s"Loading MiniLM passage embeddings from TSV $inputPath ...")
        MiniLmParquetLoader.loadPassageEmbeddings(spark, inputPath, passagesPath, outputDir)

      case "minilm-query" =>
        println(s"Loading MiniLM query embeddings from TSV $inputPath ...")
        MiniLmParquetLoader.loadQueryEmbeddings(spark, inputPath, outputDir)

      case other =>
        throw new IllegalArgumentException(
          s"Unknown --method: $other. Expected: tfidf, bm25, word2vec, word2vec-sif, minilm-passage, minilm-query")
    }

    println(s"Embeddings saved to $outputDir")
    spark.stop()
  }
}
