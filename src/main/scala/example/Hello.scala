package example

import org.apache.spark
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

object Hello {


  def main(args: Array[String]): Unit = {
    val path = "/Users/juanluisrto/Downloads/2008.csv.bz2"
    val conf = new SparkConf().setAppName("predictor")
    val sc = new SparkContext(conf)


    val spark = SparkSession
      .builder
      .appName("predictor")
      .config("spark.master", "local")
      .master("local")
      .getOrCreate()

    val rawData = spark.read.format("csv").option("header","true").load("path")

    println("\nsuccess\n")

  }
}

