package app

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, MinHashLSH}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.SparkSession

/**
 * Build and save a LSH index for a given embedding.
 *
 * Usage:
 *   --embedding       {tfidf|bm25|word2vec|minilm}
 *   --input           <embeddings parquet dir>
 *   --output          <LSH model save path>
 *   --num-hash-tables 5          (default 5)
 *   --bucket-length   2.0        (for BRP/dense only, default 2.0)
 */
object BuildLSHIndex {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var embedding      = ""
    var inputPath      = ""
    var outputPath     = ""
    var numHashTables  = 5
    var bucketLength   = 2.0

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--embedding"       => embedding     = args(i + 1); i += 2
        case "--input"           => inputPath     = args(i + 1); i += 2
        case "--output"          => outputPath    = args(i + 1); i += 2
        case "--num-hash-tables" => numHashTables = args(i + 1).toInt; i += 2
        case "--bucket-length"   => bucketLength  = args(i + 1).toDouble; i += 2
        case _                   => i += 1
      }
    }

    require(embedding.nonEmpty,  "--embedding is required")
    require(inputPath.nonEmpty,  "--input is required")
    require(outputPath.nonEmpty, "--output is required")

    val spark = SparkSession.builder()
      .appName(s"BuildLSHIndex-$embedding")
      .getOrCreate()

    val embeddingsDF = spark.read.parquet(inputPath)

    // Detect vector type from the first row's "features" column
    val sampleVec = embeddingsDF.select("features").head().getAs[org.apache.spark.ml.linalg.Vector](0)
    val isSparse  = sampleVec.isInstanceOf[SparseVector]

    if (isSparse) {
      // Use MinHashLSH for sparse vectors (tfidf, bm25)
      println(s"Detected sparse vectors — using MinHashLSH (numHashTables=$numHashTables)")
      val mh = new MinHashLSH()
        .setInputCol("features")
        .setOutputCol("hashes")
        .setNumHashTables(numHashTables)
      val model = mh.fit(embeddingsDF)
      model.write.overwrite().save(outputPath)
    } else {
      // Use BucketedRandomProjectionLSH for dense vectors (word2vec, minilm)
      println(s"Detected dense vectors — using BucketedRandomProjectionLSH " +
              s"(numHashTables=$numHashTables, bucketLength=$bucketLength)")
      val brp = new BucketedRandomProjectionLSH()
        .setInputCol("features")
        .setOutputCol("hashes")
        .setNumHashTables(numHashTables)
        .setBucketLength(bucketLength)
      val model = brp.fit(embeddingsDF)
      model.write.overwrite().save(outputPath)
    }

    println(s"LSH index saved to $outputPath")
    spark.stop()
  }
}
