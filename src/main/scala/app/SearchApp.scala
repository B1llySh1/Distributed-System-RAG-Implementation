package app

import core.{BruteForceRetriever, LSHRetriever, Retriever}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import scala.io.StdIn

/**
 * Interactive search REPL.
 *
 * CLI:
 *   --embedding     {tfidf|bm25|word2vec}   (minilm not supported)
 *   --retrieval     {bruteforce|lsh}         default: bruteforce
 *   --index         <embeddings parquet>     sparse index (or only index for non-hybrid)
 *   --model         <model dir>
 *   --lsh-model     <LSH model dir>          required when --retrieval lsh
 *   --k             10
 *   --hybrid        enable hybrid mode (sparse + dense)
 *   --dense-index   <dense embeddings parquet>
 *   --dense-model   <dense model dir>
 *   --sparse-embedding {tfidf|bm25}          default: bm25
 *   --dense-embedding  {word2vec}            default: word2vec
 *   --fusion        {linear|rrf}             default: linear
 *   --alpha         0.5                      for fusion=linear
 */
object SearchApp {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var embedding       = "tfidf"
    var retrieval       = "bruteforce"
    var indexPath       = ""
    var modelDir        = ""
    var lshModel        = ""
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
        case "--embedding"       => embedding       = args(i + 1); i += 2
        case "--retrieval"       => retrieval       = args(i + 1); i += 2
        case "--index"           => indexPath       = args(i + 1); i += 2
        case "--model"           => modelDir        = args(i + 1); i += 2
        case "--lsh-model"       => lshModel        = args(i + 1); i += 2
        case "--k"               => k               = args(i + 1).toInt; i += 2
        case "--hybrid"          => hybridMode      = true; i += 1
        case "--dense-index"     => denseIndex      = args(i + 1); i += 2
        case "--dense-model"     => denseModelDir   = args(i + 1); i += 2
        case "--sparse-embedding"=> sparseEmbedding = args(i + 1); i += 2
        case "--dense-embedding" => denseEmbedding  = args(i + 1); i += 2
        case "--fusion"          => fusion          = args(i + 1); i += 2
        case "--alpha"           => alpha           = args(i + 1).toDouble; i += 2
        case _                   => i += 1
      }
    }

    require(indexPath.nonEmpty, "--index is required")
    require(modelDir.nonEmpty,  "--model is required")
    if (retrieval == "lsh") require(lshModel.nonEmpty, "--lsh-model is required for --retrieval lsh")
    if (hybridMode) {
      require(denseIndex.nonEmpty,    "--dense-index is required for --hybrid")
      require(denseModelDir.nonEmpty, "--dense-model is required for --hybrid")
    }
    if (!hybridMode && embedding == "minilm") {
      println("[ERROR] MiniLM interactive search is not supported (requires precomputed query embeddings).")
      println("        Use --embedding tfidf, bm25, or word2vec.")
      System.exit(1)
    }

    val spark = SparkSession.builder()
      .appName("SearchApp")
      .getOrCreate()

    // Build the retriever
    val retriever: Retriever = if (hybridMode) {
      val sparseR = new BruteForceRetriever(spark, sparseEmbedding, indexPath,    modelDir)
      val denseR  = new BruteForceRetriever(spark, denseEmbedding,  denseIndex, denseModelDir)
      new core.HybridRetriever(sparseR, denseR, alpha, fusion)
    } else {
      retrieval match {
        case "lsh" =>
          new LSHRetriever(spark, embedding, indexPath, lshModel, modelDir)
        case _ =>
          new BruteForceRetriever(spark, embedding, indexPath, modelDir)
      }
    }

    // Load passage texts once — avoids re-reading parquet on every query.
    // For hybrid mode read from both indexes (pids may differ); union them.
    println("Loading passage index...")
    val passageTexts: Map[String, String] = {
      val base = spark.read.parquet(indexPath).select("pid", "passage_text").collect()
        .map(r => r.getString(0) -> r.getString(1)).toMap
      if (hybridMode && denseIndex != indexPath)
        base ++ spark.read.parquet(denseIndex).select("pid", "passage_text").collect()
          .map(r => r.getString(0) -> r.getString(1)).toMap
      else base
    }
    println(s"${passageTexts.size} passages ready.")

    // Warm up the retriever — forces all lazy initialization (model loading,
    // passage vector collection, LSH bucket index) before the first real query.
    print("Warming up retriever...")
    val warmupT0 = System.currentTimeMillis()
    retriever.retrieve("warmup", 1)
    println(f" done (${System.currentTimeMillis() - warmupT0} ms)")

    println(s"\nSearchApp ready  [embedding=$embedding  retrieval=$retrieval  k=$k]")
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
          val queryMs = System.currentTimeMillis() - t0

          if (results.isEmpty) {
            println(s"No results found.  (${queryMs} ms)")
          } else {
            println(f"Retrieved ${results.length} results in ${queryMs} ms\n")
            results.zipWithIndex.foreach { case ((pid, score), rank) =>
              val text    = passageTexts.getOrElse(pid, "")
              val preview = if (text.length > 150) text.substring(0, 150) + "..." else text
              println(f"${rank + 1}%2d. score=${score}%.4f  pid=${pid}  (${queryMs} ms total)")
              println(s"    $preview")
              println()
            }
          }
        } catch {
          case e: Exception =>
            println(s"[ERROR] ${e.getMessage}")
        }
      }

      println("Enter next query (or 'quit' to exit):")
      line = StdIn.readLine()
    }

    println("Exiting SearchApp.")
    spark.stop()
  }
}
