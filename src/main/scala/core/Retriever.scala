package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{HashingTF, IDFModel, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object Retriever {

  def cosineSimilarity(v1: Vector, v2: Vector): Double = {
    var dot   = 0.0
    var norm1 = 0.0
    var norm2 = 0.0

    (v1, v2) match {
      case (sv1: SparseVector, sv2: SparseVector) =>
        // Efficient sparse x sparse
        val indices1 = sv1.indices
        val values1  = sv1.values
        val indices2 = sv2.indices
        val values2  = sv2.values

        var p = 0
        var q = 0
        while (p < indices1.length && q < indices2.length) {
          if (indices1(p) == indices2(q)) {
            dot += values1(p) * values2(q)
            p += 1
            q += 1
          } else if (indices1(p) < indices2(q)) {
            p += 1
          } else {
            q += 1
          }
        }
        values1.foreach(v => norm1 += v * v)
        values2.foreach(v => norm2 += v * v)

      case _ =>
        // Fallback: dense iteration
        val arr1 = v1.toArray
        val arr2 = v2.toArray
        var idx  = 0
        while (idx < arr1.length) {
          dot   += arr1(idx) * arr2(idx)
          norm1 += arr1(idx) * arr1(idx)
          norm2 += arr2(idx) * arr2(idx)
          idx += 1
        }
    }

    val denom = math.sqrt(norm1) * math.sqrt(norm2)
    if (denom == 0.0) 0.0 else dot / denom
  }

  def retrieveTfIdf(
      spark: SparkSession,
      query: String,
      embeddingsPath: String,
      modelDir: String,
      k: Int
  ): Array[(String, String, Double)] = {
    import spark.implicits._

    // Load embeddings
    val embeddings = spark.read.parquet(embeddingsPath)

    // Load saved models
    val idfModel  = IDFModel.load(modelDir + "/idf")
    val hashingTF = HashingTF.load(modelDir + "/hashingTF")

    // Embed query using a single-row DataFrame pipeline
    val queryDF = Seq(query).toDF("query_text")

    val tokenizer = new RegexTokenizer()
      .setInputCol("query_text")
      .setOutputCol("tokens")
      .setPattern("\\W+")

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    val htf = new HashingTF()
      .setNumFeatures(hashingTF.getNumFeatures)
      .setInputCol("filtered")
      .setOutputCol("tf")

    val idfStage = idfModel
      .setInputCol("tf")
      .setOutputCol("tfidf")

    val tokenized = tokenizer.transform(queryDF)
    val filtered  = remover.transform(tokenized)
    val tfed      = htf.transform(filtered)
    val tfidfed   = idfStage.transform(tfed)

    val queryVec = tfidfed.select("tfidf").head().getAs[Vector](0)
    val bcQuery  = spark.sparkContext.broadcast(queryVec)

    val simUDF = udf((passVec: Vector) => cosineSimilarity(bcQuery.value, passVec))

    val results = embeddings
      .withColumn("score", simUDF(col("tfidf")))
      .orderBy(desc("score"))
      .limit(k)
      .select("pid", "passage_text", "score")
      .collect()
      .map(row => (row.getString(0), row.getString(1), row.getDouble(2)))

    bcQuery.unpersist()
    results
  }

  /**
   * Batch TF-IDF retrieval optimised for evaluation:
   *  1. Load passage embeddings once and collect to driver.
   *  2. Pre-compute per-passage L2 norms once.
   *  3. Embed ALL queries in a single Spark job.
   *  4. Do all cosine-similarity ranking locally — zero per-query Spark overhead.
   *
   * @param queries  Sequence of (qid, query_text) pairs (already filtered to qids in qrels).
   * @return Map qid -> top-k Array[(pid, passage_text, score)]
   */
  def batchRetrieveTfIdf(
      spark: SparkSession,
      queries: Seq[(String, String)],
      embeddingsPath: String,
      modelDir: String,
      k: Int
  ): Map[String, Array[(String, String, Double)]] = {
    import spark.implicits._

    // ── Step 1: load passage embeddings once, collect to driver ──────────────
    println(s"  [batch] Loading passage embeddings from $embeddingsPath ...")
    val passageRows = spark.read.parquet(embeddingsPath)
      .select("pid", "passage_text", "tfidf")
      .collect()

    val pids    = passageRows.map(_.getString(0))
    val texts   = passageRows.map(_.getString(1))
    val vecs    = passageRows.map(_.getAs[Vector](2))
    val norms   = vecs.map(v => math.sqrt(v match {
      case sv: SparseVector => sv.values.map(x => x * x).sum
      case dv: DenseVector  => dv.values.map(x => x * x).sum
    }))
    val n = pids.length
    println(s"  [batch] ${n} passages loaded to driver.")

    // ── Step 2: load models once ─────────────────────────────────────────────
    val idfModel  = IDFModel.load(modelDir + "/idf")
    val hashingTF = HashingTF.load(modelDir + "/hashingTF")

    // ── Step 3: embed ALL queries in a single Spark job ──────────────────────
    println(s"  [batch] Embedding ${queries.length} queries ...")
    val queryDF = queries.toDF("qid", "query_text")

    val tokenizer = new RegexTokenizer()
      .setInputCol("query_text").setOutputCol("tokens").setPattern("\\W+")
    val remover = new StopWordsRemover()
      .setInputCol("tokens").setOutputCol("filtered")
    val htf = new HashingTF()
      .setNumFeatures(hashingTF.getNumFeatures)
      .setInputCol("filtered").setOutputCol("tf")
    val idfStage = idfModel.setInputCol("tf").setOutputCol("tfidf")

    val embedded = idfStage.transform(
      htf.transform(remover.transform(tokenizer.transform(queryDF)))
    ).select("qid", "tfidf").collect()

    // ── Step 4: cosine similarity and top-k — entirely on driver ─────────────
    // cosineSimilarity already handles sparse–sparse efficiently.
    // Passage norms are pre-computed so each call only needs the dot product
    // and the query norm — no repeated sqrt over passage values.
    println(s"  [batch] Ranking passages for all queries ...")
    val t0 = System.currentTimeMillis()

    val result = embedded.map { row =>
      val qid      = row.getString(0)
      val queryVec = row.getAs[Vector](1)

      // score every passage and take top-k with a min-heap of size k
      val heap = new java.util.PriorityQueue[(Double, Int)](k + 1,
        Ordering.by[(Double, Int), Double](_._1))

      var pi = 0
      while (pi < n) {
        val score = cosineSimilarity(queryVec, vecs(pi))
        if (heap.size() < k) {
          heap.offer((score, pi))
        } else if (score > heap.peek()._1) {
          heap.poll()
          heap.offer((score, pi))
        }
        pi += 1
      }

      // drain heap in descending order
      val topK = new Array[(String, String, Double)](heap.size())
      var hi = heap.size() - 1
      while (!heap.isEmpty) {
        val (score, idx) = heap.poll()
        topK(hi) = (pids(idx), texts(idx), score)
        hi -= 1
      }

      qid -> topK
    }.toMap

    val elapsed = (System.currentTimeMillis() - t0) / 1000.0
    println(f"  [batch] Done. ${embedded.length} queries ranked in ${elapsed}%.1f s " +
            f"(${elapsed / embedded.length * 1000}%.1f ms/query avg)")
    result
  }

  def retrieveDense(
      spark: SparkSession,
      queryEmbedding: Array[Double],
      embeddingsPath: String,
      k: Int
  ): Array[(String, String, Double)] = {
    import spark.implicits._

    val embeddings = spark.read.parquet(embeddingsPath)

    val queryVec = Vectors.dense(queryEmbedding)
    val bcQuery  = spark.sparkContext.broadcast(queryVec)

    val simUDF = udf((embArr: Seq[Double]) => {
      val passVec = Vectors.dense(embArr.toArray)
      cosineSimilarity(bcQuery.value, passVec)
    })

    val results = embeddings
      .withColumn("score", simUDF(col("embedding")))
      .orderBy(desc("score"))
      .limit(k)
      .select("pid", "passage_text", "score")
      .collect()
      .map(row => (row.getString(0), row.getString(1), row.getDouble(2)))

    bcQuery.unpersist()
    results
  }
}
