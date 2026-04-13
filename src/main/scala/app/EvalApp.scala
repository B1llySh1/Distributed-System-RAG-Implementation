package app

import core._
import core.Evaluator.QueryResult
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import java.io.{File, PrintWriter}

// Evaluate a retrieval configuration against MS MARCO qrels.
//
// Usage:
//   --embedding        {tfidf|bm25|word2vec|word2vec-sif|hybrid}
//   --k                10
//   --queries          <TSV path>
//   --qrels            <TSV path>
//   --embedding-path   <passage embeddings parquet>
//   --model-path       <model dir>
//   --dense-path       <dense embeddings parquet>      (hybrid only)
//   --dense-model      <dense model dir>               (hybrid only)
//   --sparse-embedding {tfidf|bm25}                    (hybrid, default bm25)
//   --dense-embedding  {word2vec|word2vec-sif}         (hybrid, default word2vec)
//   --fusion           {linear|rrf}                    (hybrid, default linear)
//   --alpha            0.5                             (linear fusion only)
//   --max-queries      N                               (optional cap for quick testing)
//   --output           <CSV path>                      (optional)
object EvalApp {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var embedding       = "tfidf"
    var k               = 10
    var queriesPath     = ""
    var qrelsPath       = ""
    var embeddingPath   = ""
    var modelPath       = ""
    var densePath       = ""
    var denseModel      = ""
    var sparseEmbedding = "bm25"
    var denseEmbedding  = "word2vec"
    var fusion          = "linear"
    var alpha           = 0.5
    var maxQueries      = -1
    var outputPath      = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--embedding"        => embedding       = args(i + 1); i += 2
        case "--k"                => k               = args(i + 1).toInt; i += 2
        case "--queries"          => queriesPath     = args(i + 1); i += 2
        case "--qrels"            => qrelsPath       = args(i + 1); i += 2
        case "--embedding-path"   => embeddingPath   = args(i + 1); i += 2
        case "--model-path"       => modelPath       = args(i + 1); i += 2
        case "--dense-path"       => densePath       = args(i + 1); i += 2
        case "--dense-model"      => denseModel      = args(i + 1); i += 2
        case "--sparse-embedding" => sparseEmbedding = args(i + 1); i += 2
        case "--dense-embedding"  => denseEmbedding  = args(i + 1); i += 2
        case "--fusion"           => fusion          = args(i + 1); i += 2
        case "--alpha"            => alpha           = args(i + 1).toDouble; i += 2
        case "--max-queries"      => maxQueries      = args(i + 1).toInt; i += 2
        case "--output"           => outputPath      = args(i + 1); i += 2
        case _                    => i += 1
      }
    }

    require(queriesPath.nonEmpty,   "--queries is required")
    require(qrelsPath.nonEmpty,     "--qrels is required")
    require(embeddingPath.nonEmpty, "--embedding-path is required")

    val spark = SparkSession.builder().appName(s"EvalApp-$embedding").getOrCreate()

    println(s"Loading qrels from $qrelsPath ...")
    val qrelsMap: Map[String, Set[String]] = DataLoader.loadQrels(spark, qrelsPath)
      .filter(_.getInt(3) >= 1)
      .collect()
      .groupBy(_.getString(0))
      .map { case (qid, rows) => qid -> rows.map(_.getString(2)).toSet }

    println(s"  ${qrelsMap.size} queries have relevance labels.")

    println(s"Loading queries from $queriesPath ...")
    var queries = DataLoader.loadQueries(spark, queriesPath)
      .collect().map(r => r.getString(0) -> r.getString(1))
      .filter { case (qid, _) => qrelsMap.contains(qid) }  // skip queries with no labels

    println(s"  ${queries.length} queries with qrels entries.")

    if (maxQueries > 0 && queries.length > maxQueries) {
      queries = queries.take(maxQueries)
      println(s"  Capped to $maxQueries queries.")
    }

    if (queries.isEmpty) { println("No queries to evaluate."); spark.stop(); return }

    val retriever: Retriever = embedding match {
      case "hybrid" =>
        require(densePath.nonEmpty,  "--dense-path is required for hybrid")
        require(denseModel.nonEmpty, "--dense-model is required for hybrid")
        val sparseR = new BruteForceRetriever(spark, sparseEmbedding, embeddingPath, modelPath)
        val denseR  = new BruteForceRetriever(spark, denseEmbedding,  densePath,     denseModel)
        new HybridRetriever(sparseR, denseR, alpha, fusion)

      case emb =>
        new BruteForceRetriever(spark, emb, embeddingPath, modelPath)
    }

    println(s"Retriever: ${retriever.name}")

    val t0 = System.currentTimeMillis()

    val resultMap: Map[String, List[(String, Double)]] = retriever match {
      case hy: HybridRetriever     => hy.batchRetrieve(queries.toSeq, k)
      case bf: BruteForceRetriever => bf.batchRetrieve(queries.toSeq, k)
      case _ => queries.map { case (qid, text) => qid -> retriever.retrieve(text, k) }.toMap
    }

    val queryResults = queries.map { case (qid, _) =>
      QueryResult(qid, resultMap.getOrElse(qid, List.empty).map(_._1))
    }.toSeq

    val totalMs = System.currentTimeMillis() - t0
    val (meanP, meanR, mrr) = Evaluator.evaluate(qrelsMap, queryResults, k)

    println()
    println("=" * 60)
    println(s"Evaluation Results (embedding=$embedding, k=$k)")
    println("=" * 60)
    println(f"Queries evaluated  : ${queries.length}")
    println(f"Total time         : ${totalMs / 1000.0}%.1f s")
    println(f"Avg time / query   : ${totalMs.toDouble / queries.length}%.1f ms")
    println(f"Mean Precision@$k  : $meanP%.4f")
    println(f"Mean Recall@$k     : $meanR%.4f")
    println(f"MRR                : $mrr%.4f")
    println("=" * 60)

    if (outputPath.nonEmpty) {
      new File(outputPath).getParentFile match { case d if d != null => d.mkdirs(); case _ => }
      val pw = new PrintWriter(new File(outputPath))
      pw.println("embedding,k,num_queries,precision_at_k,recall_at_k,mrr")
      pw.println(f"$embedding,$k,${queries.length},$meanP%.6f,$meanR%.6f,$mrr%.6f")
      pw.println()
      pw.println("qid,precision_at_k,recall_at_k,reciprocal_rank")
      queryResults.foreach { qr =>
        val m = Evaluator.computeMetrics(qr.retrievedPids, qrelsMap.getOrElse(qr.qid, Set.empty), k)
        pw.println(f"${qr.qid},${m.precisionAtK}%.6f,${m.recallAtK}%.6f,${m.reciprocalRank}%.6f")
      }
      pw.close()
      println(s"Results saved to $outputPath")
    }

    spark.stop()
  }
}
