package core

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

trait Retriever {
  def retrieve(queryText: String, k: Int): List[(String, Double)]
  def name: String
}

object RetrieverUtils {

  def cosineSimilarity(v1: Vector, v2: Vector): Double = {
    var dot = 0.0; var norm1 = 0.0; var norm2 = 0.0

    (v1, v2) match {
      case (sv1: SparseVector, sv2: SparseVector) =>
        // Two-pointer merge on sorted index arrays — avoids converting to dense
        var p = 0; var q = 0
        while (p < sv1.indices.length && q < sv2.indices.length) {
          if      (sv1.indices(p) == sv2.indices(q)) { dot += sv1.values(p) * sv2.values(q); p += 1; q += 1 }
          else if (sv1.indices(p) <  sv2.indices(q)) { p += 1 }
          else                                        { q += 1 }
        }
        sv1.values.foreach(v => norm1 += v * v)
        sv2.values.foreach(v => norm2 += v * v)

      case _ =>
        val a1  = v1.toArray; val a2 = v2.toArray
        var idx = 0
        while (idx < a1.length) {
          dot += a1(idx) * a2(idx); norm1 += a1(idx) * a1(idx); norm2 += a2(idx) * a2(idx)
          idx += 1
        }
    }

    val denom = math.sqrt(norm1) * math.sqrt(norm2)
    if (denom == 0.0) 0.0 else dot / denom
  }
}
