package core

import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.DataFrame

object TextPreprocessor {

  private val stopWords = StopWordsRemover.loadDefaultStopWords("english").toSet

  // Tokenize a DataFrame column: split on non-word chars, lowercase, remove stop words.
  // "_tp_raw" is a temp column dropped before returning.
  def transform(df: DataFrame, inputCol: String, outputCol: String): DataFrame = {
    val tokenizer = new RegexTokenizer()
      .setInputCol(inputCol)
      .setOutputCol("_tp_raw")
      .setPattern("\\W+")
      .setToLowercase(true)

    val remover = new StopWordsRemover()
      .setInputCol("_tp_raw")
      .setOutputCol(outputCol)

    remover.transform(tokenizer.transform(df)).drop("_tp_raw")
  }

  // Local tokenization for query encoding on the driver — same logic as the Spark path.
  def tokenize(text: String): Array[String] =
    text.toLowerCase.split("\\W+").filter(t => t.nonEmpty && !stopWords.contains(t))
}
