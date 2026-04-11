package core

import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object TextPreprocessor {

  private val stopWords = StopWordsRemover.loadDefaultStopWords("english").toSet

  /**
   * Transform a DataFrame: inputCol (String) -> outputCol (Array[String] of filtered tokens).
   * Uses RegexTokenizer (\\W+, lowercase=true) then StopWordsRemover.
   * Intermediate column is named "_tp_raw_tokens" (dropped from output).
   */
  def transform(df: DataFrame, inputCol: String, outputCol: String): DataFrame = {
    val tokenizer = new RegexTokenizer()
      .setInputCol(inputCol)
      .setOutputCol("_tp_raw_tokens")
      .setPattern("\\W+")
      .setToLowercase(true)

    val remover = new StopWordsRemover()
      .setInputCol("_tp_raw_tokens")
      .setOutputCol(outputCol)

    val tokenized = tokenizer.transform(df)
    val filtered  = remover.transform(tokenized)
    filtered.drop("_tp_raw_tokens")
  }

  /**
   * Tokenize a single string on the driver (for local query encoding, BM25 stats).
   */
  def tokenize(text: String): Array[String] = {
    text.toLowerCase
      .split("\\W+")
      .filter(t => t.nonEmpty && !stopWords.contains(t))
  }
}
