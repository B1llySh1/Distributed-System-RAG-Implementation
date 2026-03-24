package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

object TfIdfEmbedder {

  def buildAndSave(
      spark: SparkSession,
      passagesPath: String,
      outputDir: String,
      modelDir: String
  ): Unit = {
    import spark.implicits._

    val passages = spark.read.parquet(passagesPath)

    val tokenizer = new RegexTokenizer()
      .setInputCol("passage_text")
      .setOutputCol("tokens")
      .setPattern("\\W+")

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setNumFeatures(65536)
      .setInputCol("filtered")
      .setOutputCol("tf")

    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf))

    println("Fitting TF-IDF pipeline...")
    val model = pipeline.fit(passages)

    val transformed = model.transform(passages)

    println(s"Saving TF-IDF embeddings to $outputDir ...")
    transformed.select("pid", "passage_text", "tfidf")
      .write.mode("overwrite").parquet(outputDir)

    // Extract sub-models for saving
    val idfModel = model.stages(3).asInstanceOf[IDFModel]
    idfModel.save(modelDir + "/idf")

    // HashingTF is a Transformer (not a Model), save it via the pipeline stage
    hashingTF.save(modelDir + "/hashingTF")

    println(s"Saved IDF model to $modelDir/idf")
    println(s"Saved HashingTF to $modelDir/hashingTF")
  }

  def embedQuery(
      spark: SparkSession,
      query: String,
      hashingTF: HashingTF,
      idfModel: IDFModel
  ): Vector = {
    import spark.implicits._

    val queryDF = Seq((query)).toDF("query_text")

    val tokenizer = new RegexTokenizer()
      .setInputCol("query_text")
      .setOutputCol("tokens")
      .setPattern("\\W+")

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    val htf = hashingTF
      .setInputCol("filtered")
      .setOutputCol("tf")

    val idfStage = idfModel
      .setInputCol("tf")
      .setOutputCol("tfidf")

    val tokenized  = tokenizer.transform(queryDF)
    val filtered   = remover.transform(tokenized)
    val tfed       = htf.transform(filtered)
    val tfidfed    = idfStage.transform(tfed)

    tfidfed.select("tfidf").head().getAs[Vector](0)
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
      .master("local[*]")
      .getOrCreate()

    buildAndSave(spark, inputPath, outputDir, modelDir)

    spark.stop()
  }
}
