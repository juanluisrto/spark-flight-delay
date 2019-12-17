package example

import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object Example {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    val df = Seq((-3.0965012, 5.2371198, "casa", -0.7370271),
      (-0.2100299, -0.7810844,"edificio", -1.3284768),
      (8.3525083, 5.3337562,"colegio", 21.8897181),
      (-3.0380369, 6.5357180,"casa", 0.3469820),
      (5.9354651, 6.0223208,"edificio", 17.9566144),
      (-6.8357707, 5.6629804,"colegio", -8.1598308),
      (8.8919844, -2.5149762,"casa", 15.3622538),
      (6.3404984, 4.1778706,"colegio", 16.7931822))
      .toDF("x1", "x2","xc", "y")

    val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "xc"))
      .setOutputCol("features")

    val categorical_features = Array("xc")

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
        .setOutputCols(Array(categorical_variable_name + "_vec"))
        .setDropLast(false)
    }

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    // Pipeline
    val encoder_pipeline = new Pipeline().setStages(stringIndexer ++ oneHotEncoder ++ Array(assembler,lr))

    val indexer_model = encoder_pipeline.fit(df)
    val df_transformed = indexer_model.transform(df)

    println("Aquí estaría el transformado")
    df_transformed.show()

    val lrModel = lr.fit(df)
    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
}
