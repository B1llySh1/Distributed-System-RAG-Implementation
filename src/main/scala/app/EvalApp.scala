package app

import core.{DataLoader, Evaluator, Retriever}
import core.Evaluator.QueryResult
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import java.io.{File, PrintWriter}

object EvalApp {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var method      = "tfidf"
    var indexPath   = ""
    var modelDir    = ""
    var queriesPath = ""
    var qrelsPath   = ""
    var k           = 10
    var outputPath  = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--method"  => method      = args(i + 1); i += 2
        case "--index"   => indexPath   = args(i + 1); i += 2
        case "--model"   => modelDir    = args(i + 1); i += 2
        case "--queries" => queriesPath = args(i + 1); i += 2
        case "--qrels"   => qrelsPath   = args(i + 1); i += 2
        case "--k"       => k           = args(i + 1).toInt; i += 2
        case "--output"  => outputPath  = args(i + 1); i += 2
        case _           => i += 1
      }
    }

    require(indexPath.nonEmpty,   "--index is required")
    require(queriesPath.nonEmpty, "--queries is required")
    require(qrelsPath.nonEmpty,   "--qrels is required")
    if (method == "tfidf") require(modelDir.nonEmpty, "--model is required for tfidf method")

    val spark = SparkSession.builder()
      .appName("EvalApp")
      .master("local[*]")
      .getOrCreate()

    // ── Load qrels ────────────────────────────────────────────────────────────
    println(s"Loading qrels from $qrelsPath ...")
    val qrelsDF = DataLoader.loadQrels(spark, qrelsPath)
    val qrelsMap: Map[String, Set[String]] = qrelsDF
      .filter(qrelsDF("relevance") >= 1)
      .collect()
      .groupBy(_.getString(0))
      .map { case (qid, rows) => qid -> rows.map(_.getString(2)).toSet }

    val qidsWithLabels = qrelsMap.keySet
    println(s"  ${qrelsMap.size} queries have relevance labels.")

    // ── Load queries, filter to only those present in qrels ──────────────────
    println(s"Loading queries from $queriesPath ...")
    val allQueries = DataLoader.loadQueries(spark, queriesPath)
      .collect()
      .map(r => (r.getString(0), r.getString(1)))

    val queries = allQueries.filter { case (qid, _) => qidsWithLabels.contains(qid) }
    println(s"  ${allQueries.length} total queries -> ${queries.length} have qrels entries (filtered).")

    if (queries.isEmpty) {
      println("No queries to evaluate. Check that query IDs match qrels.")
      spark.stop()
      return
    }

    // ── Batch retrieval ───────────────────────────────────────────────────────
    val t0 = System.currentTimeMillis()

    val retrievalMap: Map[String, Array[(String, String, Double)]] = method match {
      case "tfidf" =>
        Retriever.batchRetrieveTfIdf(spark, queries.toSeq, indexPath, modelDir, k)
      case "dense" =>
        println("[WARN] Dense batch evaluation not yet implemented.")
        Map.empty
      case other =>
        println(s"[WARN] Unknown method: $other")
        Map.empty
    }

    val totalMs = System.currentTimeMillis() - t0

    // ── Compute metrics ───────────────────────────────────────────────────────
    val queryResults: Seq[QueryResult] = queries.map { case (qid, _) =>
      val pids = retrievalMap.getOrElse(qid, Array.empty).map(_._1)
      QueryResult(qid, pids)
    }.toSeq

    val (meanPrecision, meanRecall, mrr) = Evaluator.evaluate(qrelsMap, queryResults, k)

    println()
    println("=" * 50)
    println(s"Evaluation Results (k=$k, method=$method)")
    println("=" * 50)
    println(f"Queries evaluated  : ${queries.length}")
    println(f"Total time         : ${totalMs / 1000.0}%.1f s")
    println(f"Avg time / query   : ${totalMs.toDouble / queries.length}%.1f ms")
    println(f"Mean Precision@$k  : $meanPrecision%.4f")
    println(f"Mean Recall@$k     : $meanRecall%.4f")
    println(f"MRR                : $mrr%.4f")
    println("=" * 50)

    if (outputPath.nonEmpty) {
      new File(outputPath).getParentFile match {
        case d if d != null => d.mkdirs()
        case _ =>
      }
      val pw = new PrintWriter(new File(outputPath))
      pw.println("method,k,num_queries,precision_at_k,recall_at_k,mrr")
      pw.println(f"$method,$k,${queries.length},$meanPrecision%.6f,$meanRecall%.6f,$mrr%.6f")
      pw.println()
      pw.println("qid,precision_at_k,recall_at_k,reciprocal_rank")
      queryResults.foreach { qr =>
        val relevantPids = qrelsMap.getOrElse(qr.qid, Set.empty[String])
        val m = Evaluator.computeMetrics(qr.retrievedPids, relevantPids, k)
        pw.println(f"${qr.qid},${m.precisionAtK}%.6f,${m.recallAtK}%.6f,${m.reciprocalRank}%.6f")
      }
      pw.close()
      println(s"Results saved to $outputPath")
    }

    spark.stop()
  }
}
