package core

import org.apache.spark.ml.feature.{IDFModel, Word2VecModel}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.SparkSession

class BruteForceRetriever(
    spark: SparkSession,
    val embeddingMethod: String,
    embeddingsPath: String,
    modelDir: String,
    queryEmbeddingsPath: String = ""
) extends Retriever {

  val name = s"BruteForce-$embeddingMethod"

  // Lazy-load passage data to driver
  private lazy val (pids, texts, vecs, norms) = {
    val rows = spark.read.parquet(embeddingsPath)
      .select("pid", "passage_text", "features")
      .collect()
    val ps = rows.map(_.getString(0))
    val ts = rows.map(_.getString(1))
    val vs = rows.map(_.getAs[Vector](2))
    val ns = vs.map(v => math.sqrt(v match {
      case sv: SparseVector => sv.values.map(x => x * x).sum
      case dv: DenseVector  => dv.values.map(x => x * x).sum
    }))
    (ps, ts, vs, ns)
  }

  def retrieve(queryText: String, k: Int): List[(String, Double)] = {
    val queryVec = encodeQuery(queryText)
    scoreAndRank(queryVec, k)
  }

  // Models/stats loaded once lazily, reused across all queries
  private lazy val cachedNumFeatures: Int              = TfIdfEmbedder.loadNumFeatures(modelDir)
  private lazy val cachedIdfModel:    IDFModel          = IDFModel.load(modelDir + "/tfidf-idf")
  private lazy val cachedW2vModel:    Word2VecModel     = Word2VecModel.load(modelDir + "/word2vec")
  private lazy val cachedBm25Stats:   BM25Embedder.BM25Stats = BM25Embedder.loadStats(spark, modelDir)

  // Encode a single query (used for interactive search) — no model reloads
  private def encodeQuery(queryText: String): Vector = embeddingMethod match {
    case "tfidf" =>
      TfIdfEmbedder.embedQuery(spark, queryText, cachedNumFeatures, cachedIdfModel)
    case "bm25" =>
      BM25Embedder.embedQuery(spark, queryText, modelDir, Some(cachedBm25Stats))
    case "word2vec" =>
      Word2VecEmbedder.embedQuery(spark, queryText, modelDir, Some(cachedW2vModel))
    case "minilm" =>
      throw new UnsupportedOperationException(
        "MiniLM interactive search requires precomputed query embeddings")
  }

  /**
   * Batch retrieve: encode all queries in ONE Spark job then score locally.
   * Returns Map[qid -> List[(pid, score)]]
   */
  def batchRetrieve(queries: Seq[(String, String)], k: Int): Map[String, List[(String, Double)]] = {
    import spark.implicits._

    // Force passage loading
    val n = pids.length
    println(s"  [$name] $n passages loaded. Encoding ${queries.length} queries...")

    // Encode all queries
    val queryVectors: Seq[(String, Vector)] = embeddingMethod match {

      case "tfidf" =>
        val numFeatures = TfIdfEmbedder.loadNumFeatures(modelDir)
        val idfModel    = IDFModel.load(modelDir + "/tfidf-idf")
        val queryDF     = queries.toDF("qid", "query_text")
        val tokenized   = TextPreprocessor.transform(queryDF, "query_text", "filtered_tokens")
        import org.apache.spark.ml.feature.HashingTF
        val htf = new HashingTF()
          .setNumFeatures(numFeatures)
          .setInputCol("filtered_tokens")
          .setOutputCol("tf")
        val idfStage = idfModel.setInputCol("tf").setOutputCol("features")
        idfStage.transform(htf.transform(tokenized))
          .select("qid", "features").collect()
          .map(r => r.getString(0) -> r.getAs[Vector](1))

      case "bm25" =>
        // HashingTF applied to all queries in ONE Spark job; BM25 formula applied locally
        BM25Embedder.embedAllQueries(spark, queries, cachedBm25Stats)

      case "word2vec" =>
        val w2vModel  = Word2VecModel.load(modelDir + "/word2vec")
        val queryDF   = queries.toDF("qid", "query_text")
        val tokenized = TextPreprocessor.transform(queryDF, "query_text", "filtered_tokens")
        val w2vStage  = w2vModel.setInputCol("filtered_tokens").setOutputCol("features")
        w2vStage.transform(tokenized)
          .select("qid", "features").collect()
          .map(r => r.getString(0) -> r.getAs[Vector](1))

      case "minilm" =>
        require(queryEmbeddingsPath.nonEmpty, "queryEmbeddingsPath required for minilm")
        val qidSet = queries.map(_._1).toSet
        spark.read.parquet(queryEmbeddingsPath)
          .select("qid", "features").collect()
          .filter(r => qidSet.contains(r.getString(0)))
          .map(r => r.getString(0) -> r.getAs[Vector](1))

      case other =>
        throw new IllegalArgumentException(s"Unknown embedding method: $other")
    }

    // Score all passages locally
    val t0 = System.currentTimeMillis()
    val result = queryVectors.map { case (qid, queryVec) =>
      qid -> scoreAndRank(queryVec, k)
    }.toMap

    val elapsed = System.currentTimeMillis() - t0
    println(f"  [$name] Ranked in ${elapsed / 1000.0}%.1f s " +
            f"(${elapsed.toDouble / queryVectors.length}%.1f ms/query avg)")
    result
  }

  private def scoreAndRank(queryVec: Vector, k: Int): List[(String, Double)] = {
    val n = pids.length
    val heap = new java.util.PriorityQueue[(Double, Int)](k + 1,
      Ordering.by[(Double, Int), Double](_._1))
    var i = 0
    while (i < n) {
      val score = RetrieverUtils.cosineSimilarity(queryVec, vecs(i))
      if (heap.size() < k) heap.offer((score, i))
      else if (score > heap.peek()._1) { heap.poll(); heap.offer((score, i)) }
      i += 1
    }
    val result = new Array[(String, Double)](heap.size())
    var hi = heap.size() - 1
    while (!heap.isEmpty) {
      val (score, idx) = heap.poll()
      result(hi) = (pids(idx), score)
      hi -= 1
    }
    result.toList
  }
}
