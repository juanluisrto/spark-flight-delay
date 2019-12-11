package example

import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.expr


object Hello {


  def main(args: Array[String]): Unit = {
    val path = "D:/Dataset/2008.csv.bz2"
    //
    val conf = new SparkConf().setAppName("predictor")

    val spark = SparkSession
      .builder
      .appName("predictor")
      .config("spark.master", "local")
      .master("local")
      .getOrCreate()

    // Take a look to diverted and cancelled variables
    val dataset = spark.read.format("csv").option("header", "true").load(path)
    // var dataset = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path)

    // Dropping variables
    val dropped_dataset = dataset
      //.limit(250)
      // Removing the forbidden variables
      .drop("ArrTime")
      .drop("ActualElapsedTime")
      .drop("AirTime")
      .drop("TaxiIn")
      .drop("Diverted")
      .drop("CarrierDelay")
      .drop("WeatherDelay")
      .drop("NASDelay")
      .drop("SecurityDelay")
      .drop("LateAircraftDelay")
      // Tail Number: is an identifier, so it is not going to be useful.
      .drop("Year")
      .drop("FlightNum")
      .drop("TailNum")
      .drop("Cancelled")
      .drop("CancellationCode")

    // We transform DepTime to integer
    val expressionDepTime = "(60*((DepTime - (DepTime%100))/100))+(DepTime%100)"
    val expressionCRSDepTime = "(60*((CRSDepTime - (CRSDepTime%100))/100))+(CRSDepTime%100)"
    val expressionCRSArrTime = "(60*((CRSArrTime - (CRSArrTime%100))/100))+(CRSArrTime%100)"

    val typed_dataset = dropped_dataset
      //Transformation of HHMM variables to integer minutes
      .withColumn("DepTime", expr(expressionDepTime))
      .withColumn("CRSDepTime", expr(expressionCRSDepTime))
      .withColumn("CRSArrTime", expr(expressionCRSArrTime))
      //Variables cast
      .withColumn("Month", col("Month").cast(IntegerType))
      .withColumn("DayofMonth", col("DayofMonth").cast(IntegerType))
      .withColumn("DayOfWeek", col("DayOfWeek").cast(IntegerType))
      .withColumn("DepTime", col("DepTime").cast(IntegerType))
      .withColumn("CRSDepTime", col("CRSDepTime").cast(IntegerType))
      .withColumn("CRSArrTime", col("CRSArrTime").cast(IntegerType))
      .withColumn("CRSElapsedTime", col("CRSElapsedTime").cast(IntegerType))
      .withColumn("ArrDelay", col("ArrDelay").cast(IntegerType))
      .withColumn("DepDelay", col("DepDelay").cast(IntegerType))
      .withColumn("Distance", col("Distance").cast(IntegerType))
      .withColumn("TaxiOut", col("TaxiOut").cast(IntegerType))

    println("Rows number before duplication removal" + typed_dataset.count())

    val cleaned_dataset = typed_dataset
      .dropDuplicates()

    println("Rows number before duplication removal" + cleaned_dataset.count())

    spark.stop()
  }
}

