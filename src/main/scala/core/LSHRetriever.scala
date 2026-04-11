package core

import org.apache.spark.ml.feature.{BucketedRandomProjectionLSHModel, IDFModel, MinHashLSHModel, Word2VecModel}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.SparkSession

/**
 * LSHRetriever — approximate nearest neighbour retrieval using Spark MLlib LSH.
 *
 * Batch evaluation path mirrors BruteForceRetriever:
 *  1. Load passage vectors + their LSH hash buckets to driver (one Spark job).
 *  2. Build an in-memory inverted index: (tableIdx, bucketVal) -> List[passageIdx].
 *  3. Encode ALL queries in one Spark job (same query encoders as BruteForce).
 *  4. Hash ALL query vectors in one Spark job via lshModel.transform.
 *  5. For each query: local bucket lookup -> candidate set -> cosine similarity scoring.
 *
 * Total Spark jobs for batch eval: 3-4 (vs N per query with approxNearestNeighbors).
 *
 * Single-query interactive path still uses approxNearestNeighbors for simplicity.
 */
class LSHRetriever(
    spark: SparkSession,
    val embeddingMethod: String,
    embeddingsPath: String,
    lshModelPath: String,
    modelDir: String,
    queryEmbeddingsPath: String = ""
) extends Retriever {

  val name = s"LSH-$embeddingMethod"

  // ── Lazy-loaded state ──────────────────────────────────────────────────────

  private lazy val lshModel: Either[BucketedRandomProjectionLSHModel, MinHashLSHModel] = {
    try { Left(BucketedRandomProjectionLSHModel.load(lshModelPath)) }
    catch { case _: Exception => Right(MinHashLSHModel.load(lshModelPath)) }
  }

  // Passage embeddings DataFrame — needed only for single-query approxNearestNeighbors
  private lazy val embeddingsDF = spark.read.parquet(embeddingsPath).cache()

  // Passage data collected to driver: pid, text, feature vector
  private lazy val (pids, texts, vecs) = {
    val rows = spark.read.parquet(embeddingsPath)
      .select("pid", "passage_text", "features")
      .collect()
    (rows.map(_.getString(0)),
     rows.map(_.getString(1)),
     rows.map(_.getAs[Vector](2)))
  }

  // LSH bucket values per passage: passageHashes(i) = Seq of bucket values,
  // one per hash table.  Computed by running lshModel.transform once on driver data.
  private lazy val passageHashes: Array[Seq[Double]] = {
    import spark.implicits._
    println(s"  [$name] Computing LSH hash values for ${pids.length} passages...")
    val vecDF = vecs.toSeq.zipWithIndex.map { case (v, i) => (i, v) }.toDF("_idx", "features")
    val hashed = lshModel match {
      case Left(m)  => m.transform(vecDF)
      case Right(m) => m.transform(vecDF)
    }
    // "hashes" column: Seq[Vector], each Vector is a 1-element DenseVector (the bucket id)
    val collected = hashed.orderBy("_idx")
      .select("hashes").collect()
      .map(r => r.getSeq[Vector](0).map(_(0)))  // extract the Double bucket id from each 1-d Vector
    println(s"  [$name] Passage hashing done.")
    collected
  }

  // Inverted index: (tableIdx, bucketVal) -> List[passageIdx]
  private lazy val bucketIndex: Map[(Int, Double), List[Int]] = {
    val buf = scala.collection.mutable.HashMap[(Int, Double), scala.collection.mutable.ListBuffer[Int]]()
    pids.indices.foreach { pi =>
      passageHashes(pi).zipWithIndex.foreach { case (bucketVal, tableIdx) =>
        buf.getOrElseUpdate((tableIdx, bucketVal), scala.collection.mutable.ListBuffer()).append(pi)
      }
    }
    buf.map { case (k, v) => k -> v.toList }.toMap
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Interactive single-query path — uses approxNearestNeighbors. */
  def retrieve(queryText: String, k: Int): List[(String, Double)] = {
    val queryVec = encodeQuerySingle(queryText)
    approxSearch(queryVec, k)
  }

  def retrieveWithVector(queryVec: Vector, k: Int): List[(String, Double)] =
    approxSearch(queryVec, k)

  /**
   * Batch evaluation — mirrors BruteForceRetriever.batchRetrieve.
   * 3-4 Spark jobs total regardless of query count.
   * Returns Seq[(qid, List[(pid, score)], queryTimeMs)]
   */
  def batchRetrieve(
      queries: Seq[(String, String)],
      k: Int,
      reportEvery: Int = 100
  ): Seq[(String, List[(String, Double)], Long)] = {
    import spark.implicits._

    val n = pids.length

    // Step 1: force passage + bucket init (counted above in lazy vals)
    println(s"  [$name] $n passages loaded. Building bucket index...")
    val numBuckets = bucketIndex.size
    println(s"  [$name] Bucket index built ($numBuckets buckets). Encoding ${queries.length} queries...")

    // Step 2: encode all queries in one Spark job
    val queryVectors: Array[(String, Vector)] = encodeAllQueries(queries)

    // Step 3: hash all query vectors in one Spark job
    println(s"  [$name] Hashing query vectors...")
    val (nonZeroQVecs, zeroQids) = queryVectors.partition { case (_, v) =>
      v match {
        case sv: SparseVector => sv.numNonzeros > 0
        case dv: DenseVector  => dv.values.exists(_ != 0.0)
      }
    }

    val queryHashes: Map[String, Seq[Double]] = if (nonZeroQVecs.nonEmpty) {
      val qVecDF = nonZeroQVecs.toSeq.map { case (qid, v) => (qid, v) }.toDF("qid", "features")
      val hashedDF = lshModel match {
        case Left(m)  => m.transform(qVecDF)
        case Right(m) => m.transform(qVecDF)
      }
      hashedDF.select("qid", "hashes").collect()
        .map(r => r.getString(0) -> r.getSeq[Vector](1).map(_(0)))
        .toMap
    } else Map.empty

    // Step 4: local candidate lookup + cosine similarity scoring
    println(s"  [$name] Running local candidate lookup and scoring...")
    val t0 = System.currentTimeMillis()
    val total = queryVectors.length

    val results = queryVectors.zipWithIndex.map { case ((qid, qvec), idx) =>
      val t1 = System.currentTimeMillis()

      val topK: List[(String, Double)] = queryHashes.get(qid) match {
        case None =>
          // Zero vector — skip (MinHashLSH requires non-zero)
          List.empty
        case Some(hashes) =>
          val candidateSet = scala.collection.mutable.HashSet[Int]()
          hashes.zipWithIndex.foreach { case (bucketVal, tableIdx) =>
            bucketIndex.getOrElse((tableIdx, bucketVal), Nil).foreach(candidateSet.add)
          }
          if (candidateSet.isEmpty) List.empty
          else {
            val heap = new java.util.PriorityQueue[(Double, Int)](k + 1,
              Ordering.by[(Double, Int), Double](_._1))
            candidateSet.foreach { pi =>
              val score = RetrieverUtils.cosineSimilarity(qvec, vecs(pi))
              if (heap.size() < k) heap.offer((score, pi))
              else if (score > heap.peek()._1) { heap.poll(); heap.offer((score, pi)) }
            }
            val arr = new Array[(String, Double)](heap.size())
            var hi  = heap.size() - 1
            while (!heap.isEmpty) { val (s, i) = heap.poll(); arr(hi) = (pids(i), s); hi -= 1 }
            arr.toList
          }
      }

      val ms = System.currentTimeMillis() - t1
      if ((idx + 1) % reportEvery == 0 || idx + 1 == total) {
        val candidates = queryHashes.get(qid).map { hashes =>
          hashes.zipWithIndex.flatMap { case (bv, ti) =>
            bucketIndex.getOrElse((ti, bv), Nil)
          }.toSet.size
        }.getOrElse(0)
        println(f"  [$name] ${idx + 1}%6d / $total  candidates=$candidates  ${ms}ms")
      }
      (qid, topK, ms)
    }

    val elapsed = System.currentTimeMillis() - t0
    println(f"  [$name] Done. ${total} queries in ${elapsed / 1000.0}%.1f s " +
            f"(${elapsed.toDouble / total}%.1f ms/query avg)  " +
            f"zero-vector queries skipped: ${zeroQids.length}")
    results
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /** Single-query path: uses approxNearestNeighbors (keeps DF in memory). */
  private def approxSearch(queryVec: Vector, k: Int): List[(String, Double)] = {
    val hasNonZero = queryVec match {
      case sv: SparseVector => sv.numNonzeros > 0
      case dv: DenseVector  => dv.values.exists(_ != 0.0)
    }
    if (!hasNonZero) return List.empty

    val resultDF = lshModel match {
      case Left(m)  => m.approxNearestNeighbors(embeddingsDF, queryVec, k, "lsh_dist")
      case Right(m) => m.approxNearestNeighbors(embeddingsDF, queryVec, k, "lsh_dist")
    }
    resultDF.select("pid", "lsh_dist").collect().map { r =>
      val dist  = r.getDouble(1)
      val score = lshModel match {
        case Left(_)  => 1.0 / (1.0 + dist)
        case Right(_) => 1.0 - dist
      }
      (r.getString(0), score)
    }.sortBy(-_._2).toList
  }

  /** Encode a single query — loads models each call (interactive search only). */
  private def encodeQuerySingle(queryText: String): Vector = embeddingMethod match {
    case "tfidf" =>
      val numFeatures = TfIdfEmbedder.loadNumFeatures(modelDir)
      TfIdfEmbedder.embedQuery(spark, queryText, numFeatures, IDFModel.load(modelDir + "/tfidf-idf"))
    case "bm25"    => BM25Embedder.embedQuery(spark, queryText, modelDir)
    case "word2vec" => Word2VecEmbedder.embedQuery(spark, queryText, modelDir)
    case "minilm"  =>
      throw new UnsupportedOperationException("MiniLM LSH: use batchRetrieve with precomputed embeddings")
  }

  /** Encode all queries in one Spark job — same logic as BruteForceRetriever. */
  private def encodeAllQueries(queries: Seq[(String, String)]): Array[(String, Vector)] = {
    import spark.implicits._
    embeddingMethod match {

      case "tfidf" =>
        val numFeatures = TfIdfEmbedder.loadNumFeatures(modelDir)
        val idfModel    = IDFModel.load(modelDir + "/tfidf-idf")
        import org.apache.spark.ml.feature.HashingTF
        val qDF      = queries.toDF("qid", "query_text")
        val tokenized = TextPreprocessor.transform(qDF, "query_text", "filtered_tokens")
        val htf      = new HashingTF().setNumFeatures(numFeatures).setInputCol("filtered_tokens").setOutputCol("tf")
        val idfStage = idfModel.setInputCol("tf").setOutputCol("features")
        idfStage.transform(htf.transform(tokenized))
          .select("qid", "features").collect()
          .map(r => r.getString(0) -> r.getAs[Vector](1))

      case "bm25" =>
        BM25Embedder.embedAllQueries(spark, queries, BM25Embedder.loadStats(spark, modelDir))

      case "word2vec" =>
        val w2vModel  = Word2VecModel.load(modelDir + "/word2vec")
        val qDF       = queries.toDF("qid", "query_text")
        val tokenized = TextPreprocessor.transform(qDF, "query_text", "filtered_tokens")
        val w2vStage  = w2vModel.setInputCol("filtered_tokens").setOutputCol("features")
        w2vStage.transform(tokenized)
          .select("qid", "features").collect()
          .map(r => r.getString(0) -> r.getAs[Vector](1))

      case "minilm" =>
        require(queryEmbeddingsPath.nonEmpty, "--query-embeddings required for minilm LSH")
        val qidSet = queries.map(_._1).toSet
        spark.read.parquet(queryEmbeddingsPath)
          .select("qid", "features").collect()
          .filter(r => qidSet.contains(r.getString(0)))
          .map(r => r.getString(0) -> r.getAs[Vector](1))

      case other => throw new IllegalArgumentException(s"Unknown embedding: $other")
    }
  }
}
