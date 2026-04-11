package core

object Evaluator {

  case class QueryResult(qid: String, retrievedPids: Seq[String], queryTimeMs: Long = 0L)

  case class Metrics(precisionAtK: Double, recallAtK: Double, reciprocalRank: Double)

  def computeMetrics(
      retrievedPids: Seq[String],
      relevantPids: Set[String],
      k: Int
  ): Metrics = {
    if (relevantPids.isEmpty) {
      return Metrics(0.0, 0.0, 0.0)
    }

    val topK = retrievedPids.take(k)

    val truePositives = topK.count(pid => relevantPids.contains(pid))

    val precisionAtK = if (k == 0) 0.0 else truePositives.toDouble / k
    val recallAtK    = truePositives.toDouble / relevantPids.size

    // Reciprocal rank: position of first relevant doc (1-indexed)
    val rankOfFirst    = retrievedPids.indexWhere(pid => relevantPids.contains(pid))
    val reciprocalRank = if (rankOfFirst < 0) 0.0 else 1.0 / (rankOfFirst + 1)

    Metrics(precisionAtK, recallAtK, reciprocalRank)
  }

  def evaluate(
      qrels: Map[String, Set[String]],
      results: Seq[QueryResult],
      k: Int
  ): (Double, Double, Double) = {
    if (results.isEmpty) return (0.0, 0.0, 0.0)

    val metricsPerQuery = results.map { qr =>
      val relevantPids = qrels.getOrElse(qr.qid, Set.empty[String])
      computeMetrics(qr.retrievedPids, relevantPids, k)
    }

    val n             = metricsPerQuery.size.toDouble
    val meanPrecision = metricsPerQuery.map(_.precisionAtK).sum / n
    val meanRecall    = metricsPerQuery.map(_.recallAtK).sum / n
    val mrr           = metricsPerQuery.map(_.reciprocalRank).sum / n

    // Report avg query time if available
    val timings = results.map(_.queryTimeMs)
    if (timings.exists(_ > 0)) {
      val avgMs = timings.sum.toDouble / timings.length
      println(f"  Avg query time: $avgMs%.1f ms")
    }

    (meanPrecision, meanRecall, mrr)
  }
}
