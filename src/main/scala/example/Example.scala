package example

import org.apache.spark
import org.apache.spark.ml.feature.VectorAssembler
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

    val df = Seq((-3.0965012, 5.2371198, -0.7370271),
      (-0.2100299, -0.7810844, -1.3284768),
      (8.3525083, 5.3337562, 21.8897181),
      (-3.0380369, 6.5357180, 0.3469820),
      (5.9354651, 6.0223208, 17.9566144),
      (-6.8357707, 5.6629804, -8.1598308),
      (8.8919844, -2.5149762, 15.3622538),
      (6.3404984, 4.1778706, 16.7931822))
      .toDF("x1", "x2", "y")

    val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2"))
      .setOutputCol("features")
    val output = assembler.transform(df)
    output.show(truncate = false)

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setMaxIter(10)
      .setElasticNetParam(0.8)
    val lrModel = lr.fit(output)
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