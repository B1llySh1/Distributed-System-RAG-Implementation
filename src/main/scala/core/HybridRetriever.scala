package core

/**
 * Hybrid retriever combining a sparse and dense BruteForceRetriever.
 *
 * fusion = "linear" : min-max normalise both score lists then blend
 *                     final = alpha * sparse + (1-alpha) * dense
 *
 * fusion = "rrf"    : Reciprocal Rank Fusion (rank-based, no normalisation needed)
 *                     final = 1/(rrfK + rank_sparse) + 1/(rrfK + rank_dense)
 *                     Documents absent from one list get rank = candidatePool + 1.
 *                     rrfK=60 is the standard constant from the original RRF paper.
 */
class HybridRetriever(
    sparseRetriever: BruteForceRetriever,
    denseRetriever: BruteForceRetriever,
    alpha: Double  = 0.5,
    fusion: String = "linear",
    rrfK: Int      = 60
) extends Retriever {

  val name =
    s"Hybrid(${sparseRetriever.embeddingMethod}+${denseRetriever.embeddingMethod}," +
    s"fusion=$fusion" +
    (if (fusion == "linear") s",alpha=$alpha" else s",rrfK=$rrfK") + ")"

  def retrieve(queryText: String, k: Int): List[(String, Double)] =
    fuse(
      sparseRetriever.retrieve(queryText, k * 5),
      denseRetriever.retrieve(queryText, k * 5),
      k
    )

  def batchRetrieve(
      queries: Seq[(String, String)],
      k: Int
  ): Map[String, List[(String, Double)]] = {
    val sparseBatch = sparseRetriever.batchRetrieve(queries, k * 5)
    val denseBatch  = denseRetriever.batchRetrieve(queries, k * 5)
    queries.map { case (qid, _) =>
      val sp = sparseBatch.getOrElse(qid, List.empty)
      val de = denseBatch.getOrElse(qid, List.empty)
      qid -> fuse(sp, de, k)
    }.toMap
  }

  // ── fusion helpers ─────────────────────────────────────────────────────────

  private def fuse(
      sparseRanked: List[(String, Double)],
      denseRanked:  List[(String, Double)],
      k: Int
  ): List[(String, Double)] =
    if (fusion == "rrf") fuseRRF(sparseRanked, denseRanked, k)
    else                 fuseLinear(sparseRanked, denseRanked, k)

  private def fuseLinear(
      sparseRanked: List[(String, Double)],
      denseRanked:  List[(String, Double)],
      k: Int
  ): List[(String, Double)] = {
    val sp      = sparseRanked.toMap
    val de      = denseRanked.toMap
    val allPids = (sp.keySet ++ de.keySet).toSeq

    def normalize(scores: Map[String, Double]): Map[String, Double] = {
      val vals = allPids.map(p => scores.getOrElse(p, 0.0))
      val mn   = vals.min
      val mx   = vals.max
      if (mx == mn) allPids.map(_ -> 0.0).toMap
      else allPids.map(p => p -> (scores.getOrElse(p, 0.0) - mn) / (mx - mn)).toMap
    }

    val ns = normalize(sp)
    val nd = normalize(de)

    allPids
      .map(pid => pid -> (alpha * ns(pid) + (1 - alpha) * nd(pid)))
      .sortBy(-_._2)
      .take(k)
      .toList
  }

  private def fuseRRF(
      sparseRanked: List[(String, Double)],
      denseRanked:  List[(String, Double)],
      k: Int
  ): List[(String, Double)] = {
    // Build rank maps (1-based); missing docs get rank = pool size + 1
    val sparsePool = sparseRanked.length
    val densePool  = denseRanked.length
    val sparseRank = sparseRanked.zipWithIndex.map { case ((pid, _), i) => pid -> (i + 1) }.toMap
    val denseRank  = denseRanked.zipWithIndex.map  { case ((pid, _), i) => pid -> (i + 1) }.toMap

    val allPids = (sparseRank.keySet ++ denseRank.keySet).toSeq

    allPids
      .map { pid =>
        val rs = sparseRank.getOrElse(pid, sparsePool + 1)
        val rd = denseRank.getOrElse(pid,  densePool  + 1)
        pid -> (1.0 / (rrfK + rs) + 1.0 / (rrfK + rd))
      }
      .sortBy(-_._2)
      .take(k)
      .toList
  }
}
