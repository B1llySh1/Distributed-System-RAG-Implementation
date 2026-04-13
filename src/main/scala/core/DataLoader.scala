package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

object DataLoader {

  def loadPassages(spark: SparkSession, path: String): DataFrame =
    spark.read.option("delimiter", "\t").option("header", "false")
      .schema(StructType(Seq(
        StructField("pid", StringType),
        StructField("passage_text", StringType)
      ))).csv(path)

  def loadQueries(spark: SparkSession, path: String): DataFrame =
    spark.read.option("delimiter", "\t").option("header", "false")
      .schema(StructType(Seq(
        StructField("qid", StringType),
        StructField("query_text", StringType)
      ))).csv(path)

  def loadQrels(spark: SparkSession, path: String): DataFrame =
    spark.read.option("delimiter", "\t").option("header", "false")
      .schema(StructType(Seq(
        StructField("qid", StringType),
        StructField("dummy", IntegerType),
        StructField("pid", StringType),
        StructField("relevance", IntegerType)
      ))).csv(path)

  // Preprocess TSV files into Parquet once, so all downstream jobs read efficiently.
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var passagesPath = ""
    var queriesPath  = ""
    var qrelsPath    = ""
    var outputDir    = ""

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--passages" => passagesPath = args(i + 1); i += 2
        case "--queries"  => queriesPath  = args(i + 1); i += 2
        case "--qrels"    => qrelsPath    = args(i + 1); i += 2
        case "--output"   => outputDir    = args(i + 1); i += 2
        case _            => i += 1
      }
    }

    require(passagesPath.nonEmpty, "--passages is required")
    require(queriesPath.nonEmpty,  "--queries is required")
    require(qrelsPath.nonEmpty,    "--qrels is required")
    require(outputDir.nonEmpty,    "--output is required")

    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("DataLoader").getOrCreate()

    val passages = loadPassages(spark, passagesPath)
    val queries  = loadQueries(spark, queriesPath)
    val qrels    = loadQrels(spark, qrelsPath)

    passages.write.mode("overwrite").parquet(outputDir + "/passages")
    queries.write.mode("overwrite").parquet(outputDir + "/queries")
    qrels.write.mode("overwrite").parquet(outputDir + "/qrels")

    println(s"Passages: ${passages.count()}")
    println(s"Queries : ${queries.count()}")
    println(s"Qrels   : ${qrels.count()}")

    spark.stop()
  }
}
