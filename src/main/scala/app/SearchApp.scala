package app

import core.Retriever
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import scala.io.StdIn

object SearchApp {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    var method    = "tfidf"
    var indexPath = ""
    var modelDir  = ""
    var k         = 10

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--method" => method    = args(i + 1); i += 2
        case "--index"  => indexPath = args(i + 1); i += 2
        case "--model"  => modelDir  = args(i + 1); i += 2
        case "--k"      => k         = args(i + 1).toInt; i += 2
        case _          => i += 1
      }
    }

    require(indexPath.nonEmpty, "--index is required")
    if (method == "tfidf") require(modelDir.nonEmpty, "--model is required for tfidf method")

    val spark = SparkSession.builder()
      .appName("SearchApp")
      .master("local[*]")
      .getOrCreate()

    println(s"SearchApp ready (method=$method, k=$k).")
    println("Enter a query (or 'quit' to exit):")

    var line = StdIn.readLine()
    while (line != null && line.trim.toLowerCase != "quit") {
      val query = line.trim
      if (query.nonEmpty) {
        println(s"\nQuery: $query")
        println("-" * 60)

        val results: Array[(String, String, Double)] = method match {
          case "tfidf" =>
            Retriever.retrieveTfIdf(spark, query, indexPath, modelDir, k)

          case "dense" =>
            // For interactive dense search, we need a query embedding.
            // Without DJL at runtime from Scala side, print a message.
            println("Dense interactive search requires a pre-embedded query.")
            println("Please use EvalApp for batch dense evaluation.")
            Array.empty

          case other =>
            println(s"Unknown method: $other. Use 'tfidf' or 'dense'.")
            Array.empty
        }

        results.zipWithIndex.foreach { case ((pid, text, score), rank) =>
          val preview = if (text != null && text.length > 120) text.substring(0, 120) + "..." else text
          println(f"${rank + 1}%2d. [pid=$pid, score=$score%.4f]")
          println(s"    $preview")
          println()
        }

        if (results.isEmpty && method == "tfidf") println("No results found.")
      }
      println("Enter next query (or 'quit' to exit):")
      line = StdIn.readLine()
    }

    println("Exiting SearchApp.")
    spark.stop()
  }
}
