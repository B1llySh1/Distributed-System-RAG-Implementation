package app

import core.{BruteForceRetriever, HybridRetriever, Retriever}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import scala.io.StdIn

// Interactive search REPL.
//
// Usage:
//   --embedding      {tfidf|bm25|word2vec|word2vec-sif}
//   --index          <embeddings parquet>
//   --model          <model dir>
//   --k              10
//   --hybrid         enable hybrid mode
//   --dense-index    <dense embeddings parquet>    (hybrid only)
//   --dense-model    <dense model dir>             (hybrid only)
//   --sparse-embedding {tfidf|bm25}                (hybrid, default bm25)
//   --dense-embedding  {word2vec|word2vec-sif}     (hybrid, default word2vec)
//   --fusion         {linear|rrf}                  (hybrid, default linear)
//   --alpha          0.5                           (linear fusion only)
object SearchApp {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var embedding       = "tfidf"
    var indexPath       = ""
    var modelDir        = ""
    var k               = 10
    var hybridMode      = false
    var denseIndex      = ""
    var denseModelDir   = ""
    var sparseEmbedding = "bm25"
    var denseEmbedding  = "word2vec"
    var fusion          = "linear"
    var alpha           = 0.5

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--embedding"        => embedding       = args(i + 1); i += 2
        case "--index"            => indexPath       = args(i + 1); i += 2
        case "--model"            => modelDir        = args(i + 1); i += 2
        case "--k"                => k               = args(i + 1).toInt; i += 2
        case "--hybrid"           => hybridMode      = true; i += 1
        case "--dense-index"      => denseIndex      = args(i + 1); i += 2
        case "--dense-model"      => denseModelDir   = args(i + 1); i += 2
        case "--sparse-embedding" => sparseEmbedding = args(i + 1); i += 2
        case "--dense-embedding"  => denseEmbedding  = args(i + 1); i += 2
        case "--fusion"           => fusion          = args(i + 1); i += 2
        case "--alpha"            => alpha           = args(i + 1).toDouble; i += 2
        case _                    => i += 1
      }
    }

    require(indexPath.nonEmpty, "--index is required")
    require(modelDir.nonEmpty,  "--model is required")
    if (hybridMode) {
      require(denseIndex.nonEmpty,    "--dense-index is required for --hybrid")
      require(denseModelDir.nonEmpty, "--dense-model is required for --hybrid")
    }

    val spark = SparkSession.builder().appName("SearchApp").getOrCreate()

    val retriever: Retriever = if (hybridMode) {
      val sparseR = new BruteForceRetriever(spark, sparseEmbedding, indexPath,   modelDir)
      val denseR  = new BruteForceRetriever(spark, denseEmbedding,  denseIndex,  denseModelDir)
      new HybridRetriever(sparseR, denseR, alpha, fusion)
    } else {
      new BruteForceRetriever(spark, embedding, indexPath, modelDir)
    }

    // Read passage texts from disk once; reuse across all queries
    println("Loading passage texts...")
    val passageTexts: Map[String, String] = {
      val base = spark.read.parquet(indexPath).select("pid", "passage_text").collect()
        .map(r => r.getString(0) -> r.getString(1)).toMap
      // In hybrid mode the dense index may cover different passages — merge both
      if (hybridMode && denseIndex != indexPath)
        base ++ spark.read.parquet(denseIndex).select("pid", "passage_text").collect()
          .map(r => r.getString(0) -> r.getString(1)).toMap
      else base
    }
    println(s"${passageTexts.size} passages ready.")

    // A warmup query forces all lazy initialisation (model loading, passage vector
    // collection) before the first real query, so the first result comes back fast.
    print("Warming up...")
    val warmupT0 = System.currentTimeMillis()
    retriever.retrieve("warmup", 1)
    println(f" done (${System.currentTimeMillis() - warmupT0} ms)")

    println(s"\nSearchApp ready  [${retriever.name}  k=$k]")
    println("Enter a query (or 'quit' to exit):\n")

    var line = StdIn.readLine()
    while (line != null && line.trim.toLowerCase != "quit") {
      val query = line.trim
      if (query.nonEmpty) {
        println(s"\nQuery: $query")
        println("-" * 70)
        try {
          val t0      = System.currentTimeMillis()
          val results = retriever.retrieve(query, k)
          val ms      = System.currentTimeMillis() - t0

          if (results.isEmpty) {
            println(s"No results. (${ms}ms)")
          } else {
            println(f"${results.length} results in ${ms}ms\n")
            results.zipWithIndex.foreach { case ((pid, score), rank) =>
              val text    = passageTexts.getOrElse(pid, "")
              val preview = if (text.length > 150) text.substring(0, 150) + "..." else text
              println(f"${rank + 1}%2d. score=${score}%.4f  pid=$pid")
              println(s"    $preview\n")
            }
          }
        } catch {
          case e: Exception => println(s"[ERROR] ${e.getMessage}")
        }
      }
      println("Enter next query (or 'quit' to exit):")
      line = StdIn.readLine()
    }

    println("Exiting.")
    spark.stop()
  }
}
