package core

import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession

import java.io.{File, PrintWriter}
import scala.io.Source

object TfIdfEmbedder {

  private val NumFeatures = 65536

  def buildAndSave(spark: SparkSession, passagesPath: String, outputDir: String, modelDir: String): Unit = {
    import spark.implicits._

    val passages  = spark.read.parquet(passagesPath)
    val tokenized = TextPreprocessor.transform(passages, "passage_text", "filtered_tokens")

    val htf = new HashingTF().setNumFeatures(NumFeatures).setInputCol("filtered_tokens").setOutputCol("tf")
    val idf = new IDF().setInputCol("tf").setOutputCol("features")

    val tfDf     = htf.transform(tokenized)
    val idfModel = idf.fit(tfDf)  // scan corpus once to compute IDF weights

    idfModel.transform(tfDf)
      .select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    new File(modelDir).mkdirs()
    idfModel.write.overwrite().save(modelDir + "/tfidf-idf")

    // HashingTF isn't MLReadable, so we just persist numFeatures as a text file
    val pw = new PrintWriter(new File(modelDir + "/tfidf-numfeatures.txt"))
    pw.println(NumFeatures); pw.close()

    println(s"TF-IDF embeddings saved to $outputDir")
  }

  def embedQuery(spark: SparkSession, query: String, numFeatures: Int, idfModel: IDFModel): Vector = {
    import spark.implicits._

    val queryDF   = Seq(Tuple1(query)).toDF("query_text")
    val tokenized = TextPreprocessor.transform(queryDF, "query_text", "filtered_tokens")
    val htf       = new HashingTF().setNumFeatures(numFeatures).setInputCol("filtered_tokens").setOutputCol("tf")
    val idfStage  = idfModel.setInputCol("tf").setOutputCol("features")

    idfStage.transform(htf.transform(tokenized)).select("features").head().getAs[Vector](0)
  }

  def loadNumFeatures(modelDir: String): Int = {
    val src = Source.fromFile(modelDir + "/tfidf-numfeatures.txt")
    try src.getLines().next().trim.toInt finally src.close()
  }
}
