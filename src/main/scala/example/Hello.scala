package example

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
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

    val textIterator: Iterator[String] = Source.fromResource("path").getLines()
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

    val cols = typed_dataset.columns
      .filter(_ != "ArrDelay")
      // Removing the categorical ones
      .filter( _ != "UniqueCarrier")
      .filter( _ != "Origin")
      .filter( _ != "Dest")

    // But adding the one hot encoded vector that the pipeline is going to create below.
    val cols_pipeline = Vector("UniqueCarrier_Vec","Origin_Vec","Dest_Vec")
    val final_cols = cols ++ cols_pipeline

    // Creating the assembler with the final columns
    val assembler = new VectorAssembler()
      .setInputCols(final_cols)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    // Now we are going to create the pipeline to transform the categorical variables to a OneHotEncoding vector
    // https://towardsdatascience.com/feature-encoding-with-spark-2-3-0-part-1-9ede45562740
    // https://towardsdatascience.com/feature-encoding-made-simple-with-spark-2-3-0-part-2-5bfc869a809a

    // First we define the categorical features of the original dataset that need to be encoded
    val categorical_features = Array("UniqueCarrier", "Origin", "Dest")

    // Basically, we create a feature map. The map transformation takes in a function and applies
    // it to each element in the RDD and the result of the function is a new value of each element in the resulting RDD
    val stringIndexer = categorical_features.map { categorical_variable_name =>
      // VariableName would be changed to VariableName_Index
      new StringIndexer()
        .setInputCol(categorical_variable_name)
        .setOutputCol(categorical_variable_name + "_Index")
        .setHandleInvalid("skip")
    }
    val oneHotEncoder = categorical_features.map { categorical_variable_name =>
      new OneHotEncoderEstimator()
        .setInputCols(Array(categorical_variable_name + "_Index"))
        .setOutputCols(Array(categorical_variable_name + "_Vec"))
        .setDropLast(false)
    }

    val lr = new LinearRegression()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    // Pipeline
    val encoder_pipeline = new Pipeline().setStages(stringIndexer ++ oneHotEncoder ++ Array(assembler,lr))

    val lr_model = encoder_pipeline.fit(typed_dataset)

    // The last stage of the output will be the training model (we need to cast it)
    val training_model = lr_model.stages(lr_model.stages.length-1).asInstanceOf[LinearRegressionModel]

    val training_summary = training_model.summary
    println(s"numIterations: ${training_summary.totalIterations}")
    println(s"objectiveHistory: ${training_summary.objectiveHistory.toList}")
    training_summary.residuals.show()
    println(s"RMSE: ${training_summary.rootMeanSquaredError}")
    println(s"r2: ${training_summary.r2}")

    spark.stop()
  }
}