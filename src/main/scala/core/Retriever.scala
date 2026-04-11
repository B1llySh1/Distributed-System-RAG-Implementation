package core

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

trait Retriever {
  def retrieve(queryText: String, k: Int): List[(String, Double)]
  def name: String
}

object RetrieverUtils {

  def cosineSimilarity(v1: Vector, v2: Vector): Double = {
    var dot   = 0.0
    var norm1 = 0.0
    var norm2 = 0.0

    (v1, v2) match {
      case (sv1: SparseVector, sv2: SparseVector) =>
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
}
