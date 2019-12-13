package example

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.expr

import scala.io.Source

object Hello {


  def main(args: Array[String]): Unit = {

    // Each one can have its own path in the path.txt file, which should not be synced with git.
    val textIterator : Iterator[String] = Source.fromResource("path").getLines()
    val path = textIterator.next()
    //val path = "D:/Dataset/2008.csv.bz2"


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

    println("Rows number before duplication removal" + typed_dataset.count())


   // val cleaned_dataset = typed_dataset.dropDuplicates()

    //println("Rows number before duplication removal" + cleaned_dataset.count())

    val cols = typed_dataset.columns.filter( _ != "ArrDelay").filter( _ != "UniqueCarrier").filter( _ != "Origin").filter( _ != "Dest")

    val categorical = Array("UniqueCarrier", "Origin", "Dest")

    println(cols.toList)


    val assembler = new VectorAssembler()
        .setInputCols(cols)
        .setOutputCol("features")
        //.setHandleInvalid("skip")

    val lr = new LinearRegression()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")

/*
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
      .build()


    val steps = Array(assembler, lr)

    val pipeline = new Pipeline().setStages(steps)


    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator().setLabelCol("DelayOutputVar"))
      //.setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)*/

    val Array(training, test) = typed_dataset.randomSplit(Array(0.70, 0.30), seed = 12345)

    //     [a,b] = function() //[a,b]


  //  val model = tvs.fit(training)

    val output = assembler.transform(training)
    val lrmodel = lr.fit(output)

    println(lrmodel.getClass())


    //val holdout = model.transform(test).select("prediction", "DelayOutputVar")


    val trainingSummary = lrmodel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")


/*
    //Linear Regression
    //https://scalac.io/scala-spark-ml-machine-learning-introduction/#pipelines
    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setStandardization(true)
    //.labelCol( "MV")

    //.featuresCol//("features")




    // Fit the model
    val lrModel = lr.fit(typed_dataset)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
*/
    spark.stop()
  }
}

