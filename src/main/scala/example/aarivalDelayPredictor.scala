package example

import java.nio.file.{Files, Paths}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.expr

object aarivalDelayPredictor extends Serializable {
  def main(args: Array[String]): Unit = {

    println("BIG DATA: Arrival Delay Predictor")
    println("")
    print("Provide the file source: ")

    // Read file path from console.
    var dataPath = scala.io.StdIn.readLine()
    var existe_archivo = Files.exists(Paths.get(dataPath))

    while (!existe_archivo) {
      print("The file path is not correct! Try again: ")
      // We need StdIn version of readLine (not deprecated)
      dataPath = scala.io.StdIn.readLine()
      existe_archivo = Files.exists(Paths.get(dataPath))
    }

    // Read machine learning option from console
    println("Correct path! Now you can select between two different algorithms to apply")
    println("    - If you want to use Linear Regression: Write 1 and press enter")
    println("    - If you want to use Random Forest:     Write 2 and press enter")
    println("    - If you want to close the application  Write 0 and press enter")

    var caracter = scala.io.StdIn.readChar()

    while (!caracter.equals('1') & !caracter.equals('2') & !caracter.equals('0')) {
      println("The option selected is not valid! Try again: ")
      println("    - If you want to use Linear Regression: Write 1")
      println("    - If you want to use Random Forest:     Write 2")
      println("    - If you want to close the application: Write 0")
      // We need StdIn version of readLine (not deprecated)
      caracter = scala.io.StdIn.readChar()
    }

    // Function toInt for caracter returns the value of the caracter as a byte, so we need to rest the value as a character of zero to get the correct result.
    val machineLearningOption = caracter.toInt - '0'.toInt

    if (machineLearningOption == 0) {
      System.exit(1)
    } else if (machineLearningOption == 1) {
      println("Linear Regression algorithm has been selected. This learning process can take several minutes.")
    } else if (machineLearningOption == 2) {
      println("Classification Trees algorithm has been selected. This learning process can take several minutes.")
    }

    val conf = new SparkConf().setAppName("predictor")

    val spark = SparkSession
      .builder
      .appName("predictor")
      .config("spark.master", "local")
      .master("local")
      .getOrCreate()

    // Take a look to diverted and cancelled variables
    val dataset = spark.read.format("csv").option("header", "true").load(dataPath)

    // Flights diverted or cancelled should not be in the dataset
    val dataset_filtered = dataset.filter("(Cancelled=0) AND (Diverted=0)")

    // Dropping variables
    val dropped_dataset = dataset_filtered
      //.limit(250)
      // Removing the forbidden variables
      .drop("ArrTime", "ActualElapsedTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
      // Year: is always the same for each dataset.
      .drop("Year")
      // Identifiers must be removed.
      .drop("FlightNum", "TailNum")
      // Cancelled flights must be removed
      .drop("Cancelled", "CancellationCode")

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
      .filter(_ != "UniqueCarrier")
      .filter(_ != "Origin")
      .filter(_ != "Dest")

    // But adding the one hot encoded vector that the pipeline is going to create below.
    val cols_pipeline = Vector("UniqueCarrier_Vec", "Origin_Vec", "Dest_Vec")
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

    val Array(training, test) = typed_dataset.randomSplit(Array(0.8, 0.2), seed = 1)

    machineLearningOption match {
      //Linear Regression
      case 1 =>
        val lr = new LinearRegression()
          .setLabelCol("ArrDelay")
          .setFeaturesCol("features")
          .setMaxIter(10)
          .setElasticNetParam(0.8)

        // Pipeline
        val encoder_pipeline = new Pipeline().setStages(stringIndexer ++ oneHotEncoder ++ Array(assembler, lr))

        val lr_model = encoder_pipeline.fit(training)

        // The last stage of the output will be the training model (we need to cast it)
        //val training_model = lr_model.stages(lr_model.stages.length - 1).asInstanceOf[LinearRegressionModel]

        val predictions = lr_model.transform(test).select("prediction", "ArrDelay")

        val evaluator = new RegressionEvaluator()
          .setLabelCol("ArrDelay")
          .setPredictionCol("prediction")
          .setMetricName("rmse")

        val rmse = evaluator.evaluate(predictions)
        println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

      // Classification Forest
      case 2 =>
        val rf = new RandomForestRegressor()
          .setLabelCol("ArrDelay")
          .setFeaturesCol("features")
          .setNumTrees(3)
          .setFeatureSubsetStrategy("auto")
          .setMaxDepth(4)
          .setMaxBins(32)

        // Pipeline
        val encoder_pipeline = new Pipeline().setStages(stringIndexer ++ oneHotEncoder ++ Array(assembler, rf))

        val rf_model = encoder_pipeline.fit(training)

        // The last stage of the output will be the training model (we need to cast it)
        //val training_model = rf_model.stages(rf_model.stages.length - 1).asInstanceOf[RandomForestRegressionModel]

        val predictions = rf_model.transform(test).select("prediction", "ArrDelay")

        val evaluator = new RegressionEvaluator()
          .setLabelCol("ArrDelay")
          .setPredictionCol("prediction")
          .setMetricName("rmse")

        val rmse = evaluator.evaluate(predictions)
        println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    }

    spark.stop()
  }
}