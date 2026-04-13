package core

object Evaluator {

  case class QueryResult(qid: String, retrievedPids: Seq[String], queryTimeMs: Long = 0L)
  case class Metrics(precisionAtK: Double, recallAtK: Double, reciprocalRank: Double)

  def computeMetrics(retrievedPids: Seq[String], relevantPids: Set[String], k: Int): Metrics = {
    if (relevantPids.isEmpty) return Metrics(0.0, 0.0, 0.0)
    val topK = retrievedPids.take(k)
    val hits = topK.count(relevantPids.contains)
    val rr   = retrievedPids.indexWhere(relevantPids.contains) match {
      case -1  => 0.0
      case pos => 1.0 / (pos + 1)
    }
    Metrics(if (k == 0) 0.0 else hits.toDouble / k, hits.toDouble / relevantPids.size, rr)
  }

  def evaluate(qrels: Map[String, Set[String]], results: Seq[QueryResult], k: Int): (Double, Double, Double) = {
    if (results.isEmpty) return (0.0, 0.0, 0.0)
    val metrics = results.map(qr => computeMetrics(qr.retrievedPids, qrels.getOrElse(qr.qid, Set.empty), k))
    val n       = metrics.size.toDouble
    (metrics.map(_.precisionAtK).sum / n, metrics.map(_.recallAtK).sum / n, metrics.map(_.reciprocalRank).sum / n)
  }
}
