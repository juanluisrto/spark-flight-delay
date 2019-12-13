package example

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.expr

import scala.io.Source

object Hello extends Serializable {
    def main(args: Array[String]): Unit = {

        val conf = new SparkConf().setAppName("predictor")

        val spark = SparkSession
          .builder
          .appName("predictor")
          .config("spark.master", "local")
          .master("local")
          .getOrCreate()

        val textIterator : Iterator[String] = Source.fromResource("path").getLines()
        val path = textIterator.next()
        // Take a look to diverted and cancelled variables
        val dataset = spark.read.format("csv").option("header", "true").load(path)
        // var dataset = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path)

        // Dropping variables
        val dropped_dataset = dataset
          .limit(250)
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

        // We transform DepTime, CRSDepTime and CRSArrTime to integer
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

        val cols = typed_dataset.columns.
          filter( _ != "ArrDelay").
          filter( _ != "UniqueCarrier").
          filter( _ != "Origin").
          filter( _ != "Dest")

        val assembler = new VectorAssembler()
          .setInputCols(cols)
          .setOutputCol("features")
          .setHandleInvalid("skip")

        val output = assembler.transform(typed_dataset)
        output.show(truncate=false)

        val lr = new LinearRegression()
          .setLabelCol("ArrDelay")
          .setFeaturesCol("features")
          .setMaxIter(10)
          .setElasticNetParam(0.8)

        val lrmodel = lr.fit(output)

        val trainingSummary = lrmodel.summary
        println(s"numIterations: ${trainingSummary.totalIterations}")
        println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
        trainingSummary.residuals.show()
        println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
        println(s"r2: ${trainingSummary.r2}")

        spark.stop()
    }
}