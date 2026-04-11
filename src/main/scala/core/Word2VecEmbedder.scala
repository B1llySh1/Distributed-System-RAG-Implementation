package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession

import java.io.File

object Word2VecEmbedder {

  def buildAndSave(
      spark: SparkSession,
      passagesPath: String,
      outputDir: String,
      modelDir: String
  ): Unit = {
    import spark.implicits._

    val passages = spark.read.parquet(passagesPath)

    // Step 2: tokenize
    val tokenizedDf = TextPreprocessor.transform(passages, "passage_text", "filtered_tokens")

    // Step 3: train Word2Vec
    val word2vec = new Word2Vec()
      .setInputCol("filtered_tokens")
      .setOutputCol("features")
      .setVectorSize(100)
      .setMinCount(5)
      .setMaxIter(1)
      .setWindowSize(5)

    println("Training Word2Vec...")
    val w2vModel = word2vec.fit(tokenizedDf)

    // Step 4: transform passages
    val result = w2vModel.transform(tokenizedDf)

    // Step 5: save model
    new File(modelDir).mkdirs()
    w2vModel.write.overwrite().save(modelDir + "/word2vec")

    // Step 6: save embeddings
    println(s"Saving Word2Vec embeddings to $outputDir ...")
    result.select("pid", "passage_text", "features")
      .write.mode("overwrite").parquet(outputDir)

    println(s"Saved Word2Vec model to $modelDir/word2vec")
  }

  def embedQuery(
      spark: SparkSession,
      queryText: String,
      modelDir: String,
      preloadedModel: Option[Word2VecModel] = None
  ): Vector = {
    import spark.implicits._

    val w2vModel = preloadedModel.getOrElse(Word2VecModel.load(modelDir + "/word2vec"))
    val tokens   = TextPreprocessor.tokenize(queryText)

    val queryDF  = Seq(Tuple1(tokens.toSeq)).toDF("filtered_tokens")
    val w2vStage = w2vModel.setInputCol("filtered_tokens").setOutputCol("features")
    val result   = w2vStage.transform(queryDF)

    result.select("features").head().getAs[Vector](0)
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
      .appName("Word2VecEmbedder")
      .getOrCreate()

    buildAndSave(spark, inputPath, outputDir, modelDir)

    spark.stop()
  }
}
