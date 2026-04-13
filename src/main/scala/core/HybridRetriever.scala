package core

// Combines a sparse and a dense BruteForceRetriever via score fusion.
//
// fusion="linear" : min-max normalise both score lists, then blend with alpha
//   final = alpha * sparse + (1-alpha) * dense
//
// fusion="rrf"    : Reciprocal Rank Fusion — rank-based, no score normalisation needed
//   final = 1/(rrfK + rank_sparse) + 1/(rrfK + rank_dense)
//   Works well when the two retrievers have comparable quality;
//   use linear with high alpha when one retriever is much stronger.
class HybridRetriever(
    sparseRetriever: BruteForceRetriever,
    denseRetriever: BruteForceRetriever,
    alpha: Double  = 0.5,
    fusion: String = "linear",
    rrfK: Int      = 60
) extends Retriever {

  val name = s"Hybrid(${sparseRetriever.embeddingMethod}+${denseRetriever.embeddingMethod}," +
    s"fusion=$fusion" + (if (fusion == "linear") s",alpha=$alpha" else s",rrfK=$rrfK") + ")"

  def retrieve(queryText: String, k: Int): List[(String, Double)] =
    fuse(sparseRetriever.retrieve(queryText, k * 5), denseRetriever.retrieve(queryText, k * 5), k)

  def batchRetrieve(queries: Seq[(String, String)], k: Int): Map[String, List[(String, Double)]] = {
    val sparseBatch = sparseRetriever.batchRetrieve(queries, k * 5)
    val denseBatch  = denseRetriever.batchRetrieve(queries, k * 5)
    queries.map { case (qid, _) =>
      qid -> fuse(sparseBatch.getOrElse(qid, List.empty), denseBatch.getOrElse(qid, List.empty), k)
    }.toMap
  }

  private def fuse(sparse: List[(String, Double)], dense: List[(String, Double)], k: Int) =
    if (fusion == "rrf") fuseRRF(sparse, dense, k) else fuseLinear(sparse, dense, k)

  private def fuseLinear(sparse: List[(String, Double)], dense: List[(String, Double)], k: Int) = {
    val sp      = sparse.toMap; val de = dense.toMap
    val allPids = (sp.keySet ++ de.keySet).toSeq

    def norm(scores: Map[String, Double]): Map[String, Double] = {
      val vals = allPids.map(p => scores.getOrElse(p, 0.0))
      val mn = vals.min; val mx = vals.max
      if (mx == mn) allPids.map(_ -> 0.0).toMap
      else allPids.map(p => p -> (scores.getOrElse(p, 0.0) - mn) / (mx - mn)).toMap
    }

    val ns = norm(sp); val nd = norm(de)
    allPids.map(p => p -> (alpha * ns(p) + (1 - alpha) * nd(p))).sortBy(-_._2).take(k).toList
  }

  private def fuseRRF(sparse: List[(String, Double)], dense: List[(String, Double)], k: Int) = {
    val sparseRank = sparse.zipWithIndex.map { case ((pid, _), i) => pid -> (i + 1) }.toMap
    val denseRank  = dense.zipWithIndex.map  { case ((pid, _), i) => pid -> (i + 1) }.toMap
    val allPids    = (sparseRank.keySet ++ denseRank.keySet).toSeq

    allPids.map { pid =>
      val rs = sparseRank.getOrElse(pid, sparse.length + 1)
      val rd = denseRank.getOrElse(pid,  dense.length  + 1)
      pid -> (1.0 / (rrfK + rs) + 1.0 / (rrfK + rd))
    }.sortBy(-_._2).take(k).toList
  }
}
