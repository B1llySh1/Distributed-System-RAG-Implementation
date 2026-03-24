package core

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

object DataLoader {

  def createSpark(appName: String): SparkSession = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    SparkSession.builder()
      .appName(appName)
      .master("local[*]")
      .getOrCreate()
  }

  def loadPassages(spark: SparkSession, path: String): DataFrame = {
    val schema = StructType(Seq(
      StructField("pid", StringType, nullable = true),
      StructField("passage_text", StringType, nullable = true)
    ))
    spark.read
      .option("delimiter", "\t")
      .option("header", "false")
      .schema(schema)
      .csv(path)
  }

  def loadQueries(spark: SparkSession, path: String): DataFrame = {
    val schema = StructType(Seq(
      StructField("qid", StringType, nullable = true),
      StructField("query_text", StringType, nullable = true)
    ))
    spark.read
      .option("delimiter", "\t")
      .option("header", "false")
      .schema(schema)
      .csv(path)
  }

  def loadQrels(spark: SparkSession, path: String): DataFrame = {
    val schema = StructType(Seq(
      StructField("qid", StringType, nullable = true),
      StructField("dummy", IntegerType, nullable = true),
      StructField("pid", StringType, nullable = true),
      StructField("relevance", IntegerType, nullable = true)
    ))
    spark.read
      .option("delimiter", "\t")
      .option("header", "false")
      .schema(schema)
      .csv(path)
  }

  def main(args: Array[String]): Unit = {
    var passagesPath = ""
    var queriesPath = ""
    var qrelsPath = ""
    var outputDir = ""

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

    val spark = createSpark("DataLoader")

    val passages = loadPassages(spark, passagesPath)
    val queries  = loadQueries(spark, queriesPath)
    val qrels    = loadQrels(spark, qrelsPath)

    passages.write.mode("overwrite").parquet(outputDir + "/passages")
    queries.write.mode("overwrite").parquet(outputDir + "/queries")
    qrels.write.mode("overwrite").parquet(outputDir + "/qrels")

    println(s"Passages count : ${passages.count()}")
    println(s"Queries count  : ${queries.count()}")
    println(s"Qrels count    : ${qrels.count()}")
    println(s"Saved parquet files to $outputDir")

    spark.stop()
  }
}
