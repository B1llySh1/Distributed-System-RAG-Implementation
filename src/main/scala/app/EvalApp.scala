package app

import core._
import core.Evaluator.QueryResult
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import java.io.{File, PrintWriter}

/**
 * Unified evaluation app.
 *
 * CLI:
 *   --embedding      {tfidf|bm25|word2vec|minilm|hybrid}
 *   --retrieval      {bruteforce|lsh}
 *   --k              10
 *   --queries        <TSV path>
 *   --qrels          <TSV path>
 *   --embedding-path <parquet dir for single embedding, or sparse parquet for hybrid>
 *   --model-path     <model dir>
 *   --lsh-model-path <LSH model dir>              (for retrieval=lsh)
 *   --dense-path     <dense embeddings parquet>   (for hybrid)
 *   --dense-model    <dense model dir>            (for hybrid)
 *   --query-embeddings <query embeddings parquet> (for minilm or hybrid with dense=minilm)
 *   --sparse-embedding {tfidf|bm25}               (for hybrid, default bm25)
 *   --dense-embedding  {word2vec|minilm}          (for hybrid, default minilm)
 *   --alpha          0.5                          (for hybrid)
 *   --max-queries    N                            (optional, for quick testing)
 *   --output         <CSV path>                  (optional)
 */
object EvalApp {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var embedding        = "tfidf"
    var retrieval        = "bruteforce"
    var k                = 10
    var queriesPath      = ""
    var qrelsPath        = ""
    var embeddingPath    = ""
    var modelPath        = ""
    var lshModelPath     = ""
    var densePath        = ""
    var denseModel       = ""
    var queryEmbeddings  = ""
    var sparseEmbedding  = "bm25"
    var denseEmbedding   = "minilm"
    var alpha            = 0.5
    var fusion           = "linear"
    var maxQueries       = -1
    var outputPath       = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--embedding"        => embedding       = args(i + 1); i += 2
        case "--retrieval"        => retrieval       = args(i + 1); i += 2
        case "--k"                => k               = args(i + 1).toInt; i += 2
        case "--queries"          => queriesPath     = args(i + 1); i += 2
        case "--qrels"            => qrelsPath       = args(i + 1); i += 2
        case "--embedding-path"   => embeddingPath   = args(i + 1); i += 2
        case "--model-path"       => modelPath       = args(i + 1); i += 2
        case "--lsh-model-path"   => lshModelPath    = args(i + 1); i += 2
        case "--dense-path"       => densePath       = args(i + 1); i += 2
        case "--dense-model"      => denseModel      = args(i + 1); i += 2
        case "--query-embeddings" => queryEmbeddings = args(i + 1); i += 2
        case "--sparse-embedding" => sparseEmbedding = args(i + 1); i += 2
        case "--dense-embedding"  => denseEmbedding  = args(i + 1); i += 2
        case "--alpha"            => alpha           = args(i + 1).toDouble; i += 2
        case "--fusion"           => fusion          = args(i + 1); i += 2
        case "--max-queries"      => maxQueries      = args(i + 1).toInt; i += 2
        case "--output"           => outputPath      = args(i + 1); i += 2
        case _                    => i += 1
      }
    }

    require(queriesPath.nonEmpty,   "--queries is required")
    require(qrelsPath.nonEmpty,     "--qrels is required")
    require(embeddingPath.nonEmpty, "--embedding-path is required")

    val spark = SparkSession.builder()
      .appName(s"EvalApp-$embedding-$retrieval")
      .getOrCreate()

    // Load qrels
    println(s"Loading qrels from $qrelsPath ...")
    val qrelsDF = DataLoader.loadQrels(spark, qrelsPath)
    val qrelsMap: Map[String, Set[String]] = qrelsDF
      .filter(qrelsDF("relevance") >= 1)
      .collect()
      .groupBy(_.getString(0))
      .map { case (qid, rows) => qid -> rows.map(_.getString(2)).toSet }

    val qidsWithLabels = qrelsMap.keySet
    println(s"  ${qrelsMap.size} queries have relevance labels.")

    // Load queries
    println(s"Loading queries from $queriesPath ...")
    val allQueries = DataLoader.loadQueries(spark, queriesPath)
      .collect()
      .map(r => (r.getString(0), r.getString(1)))

    var queries = allQueries.filter { case (qid, _) => qidsWithLabels.contains(qid) }
    println(s"  ${allQueries.length} total queries -> ${queries.length} have qrels entries.")

    if (maxQueries > 0 && queries.length > maxQueries) {
      queries = queries.take(maxQueries)
      println(s"  Capped to $maxQueries queries (--max-queries).")
    }

    if (queries.isEmpty) {
      println("No queries to evaluate. Check that query IDs match qrels.")
      spark.stop()
      return
    }

    // Build retriever
    val retrieverObj: Retriever = embedding match {
      case "hybrid" =>
        val sparseRetriever = new BruteForceRetriever(
          spark, sparseEmbedding, embeddingPath, modelPath,
          if (sparseEmbedding == "minilm") queryEmbeddings else "")
        val denseRetriever = new BruteForceRetriever(
          spark, denseEmbedding, densePath, denseModel,
          if (denseEmbedding == "minilm") queryEmbeddings else "")
        new HybridRetriever(sparseRetriever, denseRetriever, alpha, fusion)

      case emb =>
        retrieval match {
          case "bruteforce" =>
            new BruteForceRetriever(spark, emb, embeddingPath, modelPath,
              if (emb == "minilm") queryEmbeddings else "")
          case "lsh" =>
            require(lshModelPath.nonEmpty, "--lsh-model-path is required for retrieval=lsh")
            new LSHRetriever(spark, emb, embeddingPath, lshModelPath, modelPath,
              if (emb == "minilm") queryEmbeddings else "")
          case other =>
            throw new IllegalArgumentException(s"Unknown --retrieval: $other")
        }
    }

    println(s"Retriever: ${retrieverObj.name}")

    // Evaluate
    val t0 = System.currentTimeMillis()

    val queryResults: Seq[QueryResult] = retrieval match {
      case "bruteforce" =>
        // Batch retrieve for BruteForceRetriever or HybridRetriever
        val resultMap: Map[String, List[(String, Double)]] = retrieverObj match {
          case bf: BruteForceRetriever =>
            bf.batchRetrieve(queries.toSeq, k)
          case hy: HybridRetriever =>
            hy.batchRetrieve(queries.toSeq, k)
          case _ =>
            queries.map { case (qid, text) =>
              val t1  = System.currentTimeMillis()
              val res = retrieverObj.retrieve(text, k)
              qid -> res
            }.toMap
        }
        queries.map { case (qid, _) =>
          val pids = resultMap.getOrElse(qid, List.empty).map(_._1)
          QueryResult(qid, pids)
        }.toSeq

      case "lsh" =>
        val lshRetriever = retrieverObj.asInstanceOf[LSHRetriever]
        val reportEvery  = math.max(1, queries.length / 200)  // print ~20 progress lines
        lshRetriever.batchRetrieve(queries.toSeq, k, reportEvery).map {
          case (qid, res, ms) => QueryResult(qid, res.map(_._1), ms)
        }

      case other =>
        throw new IllegalArgumentException(s"Unknown --retrieval: $other")
    }

    val totalMs = System.currentTimeMillis() - t0

    // Compute metrics
    val (meanPrecision, meanRecall, mrr) = Evaluator.evaluate(qrelsMap, queryResults, k)

    println()
    println("=" * 60)
    println(s"Evaluation Results (embedding=$embedding, retrieval=$retrieval, k=$k)")
    println("=" * 60)
    println(f"Queries evaluated  : ${queries.length}")
    println(f"Total time         : ${totalMs / 1000.0}%.1f s")
    println(f"Avg time / query   : ${totalMs.toDouble / queries.length}%.1f ms")
    println(f"Mean Precision@$k  : $meanPrecision%.4f")
    println(f"Mean Recall@$k     : $meanRecall%.4f")
    println(f"MRR                : $mrr%.4f")
    println("=" * 60)

    if (outputPath.nonEmpty) {
      new File(outputPath).getParentFile match {
        case d if d != null => d.mkdirs()
        case _              =>
      }
      val pw = new PrintWriter(new File(outputPath))
      pw.println("embedding,retrieval,k,num_queries,precision_at_k,recall_at_k,mrr")
      pw.println(f"$embedding,$retrieval,$k,${queries.length},$meanPrecision%.6f,$meanRecall%.6f,$mrr%.6f")
      pw.println()
      pw.println("qid,precision_at_k,recall_at_k,reciprocal_rank,query_time_ms")
      queryResults.foreach { qr =>
        val relevantPids = qrelsMap.getOrElse(qr.qid, Set.empty[String])
        val m = Evaluator.computeMetrics(qr.retrievedPids, relevantPids, k)
        pw.println(f"${qr.qid},${m.precisionAtK}%.6f,${m.recallAtK}%.6f,${m.reciprocalRank}%.6f,${qr.queryTimeMs}")
      }
      pw.close()
      println(s"Results saved to $outputPath")
    }

    spark.stop()
  }
}
