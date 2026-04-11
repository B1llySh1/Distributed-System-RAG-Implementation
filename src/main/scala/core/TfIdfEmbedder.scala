package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.io.{File, PrintWriter}
import scala.io.Source

object TfIdfEmbedder {

  private val NumFeatures = 65536

  def buildAndSave(
      spark: SparkSession,
      passagesPath: String,
      outputDir: String,
      modelDir: String
  ): Unit = {
    import spark.implicits._

    val passages = spark.read.parquet(passagesPath)

    // Use TextPreprocessor for tokenization + stop word removal
    val tokenized = TextPreprocessor.transform(passages, "passage_text", "filtered_tokens")

    val hashingTF = new HashingTF()
      .setNumFeatures(NumFeatures)
      .setInputCol("filtered_tokens")
      .setOutputCol("tf")

    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("features")

    println("Computing TF...")
    val tfDf = hashingTF.transform(tokenized)

    println("Fitting IDF...")
    val idfModel = idf.fit(tfDf)

    val transformed = idfModel.transform(tfDf)

    println(s"Saving TF-IDF embeddings to $outputDir ...")
    transformed.select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    // Save IDF model
    new File(modelDir).mkdirs()
    idfModel.write.overwrite().save(modelDir + "/tfidf-idf")

    // Save numFeatures instead of HashingTF (HashingTF is not MLReadable)
    val pw = new PrintWriter(new File(modelDir + "/tfidf-numfeatures.txt"))
    pw.println(NumFeatures.toString)
    pw.close()

    println(s"Saved IDF model to $modelDir/tfidf-idf")
    println(s"Saved numFeatures to $modelDir/tfidf-numfeatures.txt")
  }

  def embedQuery(
      spark: SparkSession,
      query: String,
      numFeatures: Int,
      idfModel: IDFModel
  ): Vector = {
    import spark.implicits._

    val queryDF = Seq(Tuple1(query)).toDF("query_text")
    val tokenized = TextPreprocessor.transform(queryDF, "query_text", "filtered_tokens")

    val htf = new HashingTF()
      .setNumFeatures(numFeatures)
      .setInputCol("filtered_tokens")
      .setOutputCol("tf")

    val idfStage = idfModel
      .setInputCol("tf")
      .setOutputCol("features")

    val tfed    = htf.transform(tokenized)
    val tfidfed = idfStage.transform(tfed)

    tfidfed.select("features").head().getAs[Vector](0)
  }

  /** Load numFeatures from the stats file saved during buildAndSave. */
  def loadNumFeatures(modelDir: String): Int = {
    val src = Source.fromFile(modelDir + "/tfidf-numfeatures.txt")
    try src.getLines().next().trim.toInt
    finally src.close()
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var inputPath = ""
    var outputDir = ""
    var modelDir  = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--input"  => inputPath = args(i + 1); i += 2
        case "--output" => outputDir = args(i + 1); i += 2
        case "--model"  => modelDir  = args(i + 1); i += 2
        case _          => i += 1
      }
    }

    require(inputPath.nonEmpty, "--input is required")
    require(outputDir.nonEmpty, "--output is required")
    require(modelDir.nonEmpty,  "--model is required")

    val spark = SparkSession.builder()
      .appName("TfIdfEmbedder")
      .getOrCreate()

    buildAndSave(spark, inputPath, outputDir, modelDir)

    spark.stop()
  }
}
