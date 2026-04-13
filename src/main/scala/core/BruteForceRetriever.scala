package core

import org.apache.spark.ml.feature.{HashingTF, IDFModel, Word2VecModel}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.SparkSession

class BruteForceRetriever(
    spark: SparkSession,
    val embeddingMethod: String,
    embeddingsPath: String,
    modelDir: String
) extends Retriever {

  val name = s"BruteForce-$embeddingMethod"

  // Pull all passage vectors to the driver once — 200K × 100-dim fits in memory easily
  private lazy val (pids, vecs, norms) = {
    val rows = spark.read.parquet(embeddingsPath).select("pid", "features").collect()
    val ps   = rows.map(_.getString(0))
    val vs   = rows.map(_.getAs[Vector](1))
    val ns   = vs.map(v => math.sqrt(v match {
      case sv: SparseVector => sv.values.map(x => x * x).sum
      case dv: DenseVector  => dv.values.map(x => x * x).sum
    }))
    (ps, vs, ns)
  }

  // Models loaded once on first use and reused for all interactive queries
  private lazy val cachedNumFeatures = TfIdfEmbedder.loadNumFeatures(modelDir)
  private lazy val cachedIdfModel    = IDFModel.load(modelDir + "/tfidf-idf")
  private lazy val cachedW2vModel    = Word2VecModel.load(modelDir + "/word2vec")
  private lazy val cachedBm25Stats   = BM25Embedder.loadStats(spark, modelDir)
  private lazy val cachedWordVecs    = Word2VecEmbedder.loadWordVecs(modelDir)
  private lazy val cachedSIFWeights  = Word2VecEmbedder.loadSIFWeights(modelDir)
  private lazy val cachedSIFPC       = Word2VecEmbedder.loadSIFPC(modelDir)

  def retrieve(queryText: String, k: Int): List[(String, Double)] =
    scoreAndRank(encodeQuery(queryText), k)

  private def encodeQuery(queryText: String): Vector = embeddingMethod match {
    case "tfidf"       => TfIdfEmbedder.embedQuery(spark, queryText, cachedNumFeatures, cachedIdfModel)
    case "bm25"        => BM25Embedder.embedQuery(spark, queryText, modelDir, Some(cachedBm25Stats))
    case "word2vec"    => Word2VecEmbedder.embedQuery(spark, queryText, modelDir, Some(cachedW2vModel))
    case "word2vec-sif"=> Word2VecEmbedder.embedQuerySIF(queryText, cachedWordVecs, cachedSIFWeights, cachedSIFPC)
    case other         => throw new IllegalArgumentException(s"Unknown embedding: $other")
  }

  // Encode all queries in one Spark job, then score all passages locally.
  def batchRetrieve(queries: Seq[(String, String)], k: Int): Map[String, List[(String, Double)]] = {
    import spark.implicits._

    println(s"  [$name] ${pids.length} passages loaded. Encoding ${queries.length} queries...")

    val queryVectors: Seq[(String, Vector)] = embeddingMethod match {
      case "tfidf" =>
        val htf      = new HashingTF().setNumFeatures(cachedNumFeatures).setInputCol("filtered_tokens").setOutputCol("tf")
        val idfStage = cachedIdfModel.setInputCol("tf").setOutputCol("features")
        val tokenized = TextPreprocessor.transform(queries.toDF("qid", "query_text"), "query_text", "filtered_tokens")
        idfStage.transform(htf.transform(tokenized))
          .select("qid", "features").collect()
          .map(r => r.getString(0) -> r.getAs[Vector](1))

      case "bm25" =>
        BM25Embedder.embedAllQueries(spark, queries, cachedBm25Stats)

      case "word2vec" =>
        val tokenized = TextPreprocessor.transform(queries.toDF("qid", "query_text"), "query_text", "filtered_tokens")
        cachedW2vModel.setInputCol("filtered_tokens").setOutputCol("features")
          .transform(tokenized).select("qid", "features").collect()
          .map(r => r.getString(0) -> r.getAs[Vector](1))

      case "word2vec-sif" =>
        // Fully local — word vecs and SIF weights already on the driver
        Word2VecEmbedder.embedAllQueriesSIF(queries, cachedWordVecs, cachedSIFWeights, cachedSIFPC)

      case other => throw new IllegalArgumentException(s"Unknown embedding: $other")
    }

    val t0 = System.currentTimeMillis()
    val result = queryVectors.map { case (qid, qvec) => qid -> scoreAndRank(qvec, k) }.toMap
    val elapsed = System.currentTimeMillis() - t0
    println(f"  [$name] Ranked in ${elapsed / 1000.0}%.1f s (${elapsed.toDouble / queryVectors.length}%.1f ms/query avg)")
    result
  }

  private def scoreAndRank(queryVec: Vector, k: Int): List[(String, Double)] = {
    val heap = new java.util.PriorityQueue[(Double, Int)](k + 1, Ordering.by[(Double, Int), Double](_._1))
    var i = 0
    while (i < pids.length) {
      val score = RetrieverUtils.cosineSimilarity(queryVec, vecs(i))
      if (heap.size() < k) heap.offer((score, i))
      else if (score > heap.peek()._1) { heap.poll(); heap.offer((score, i)) }
      i += 1
    }
    val result = new Array[(String, Double)](heap.size())
    var hi = heap.size() - 1
    while (!heap.isEmpty) { val (s, idx) = heap.poll(); result(hi) = (pids(idx), s); hi -= 1 }
    result.toList
  }
}
