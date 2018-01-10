package se.kth.spark.lab1.task_bonus

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.linalg.Vector
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {

    // conf = new SparkConf().setAppName("lab1").setMaster("local")
    val conf = new SparkConf().setAppName("lab2")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "hdfs://10.0.104.163:8020/Projects/datasets/million_song/csv/all.txt"
    //val filePath = "src/main/resources/all.txt"
    val obsDF: DataFrame = sqlContext.read.textFile(filePath).toDF()
    val Array(training, test) = obsDF.randomSplit(Array(0.7, 0.3),1234)

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("tokens")
      .setPattern(",")

    val arr2Vect = new Array2Vector()
      .setInputCol("tokens")
      .setOutputCol("vectors")

    val lSlicer = new VectorSlicer()
      .setInputCol("vectors")
      .setOutputCol("vectorLabel")

    lSlicer.setIndices(Array(0))

    val v2d = new Vector2DoubleUDF((v:Vector) => v(0))
      .setInputCol("vectorLabel")
      .setOutputCol("doubleLabel")

    val lShifter = new DoubleUDF((d:Double) => (d - 1922.0 ))
      .setInputCol("doubleLabel")
      .setOutputCol("label")

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("vectors")
      .setOutputCol("features")

    fSlicer.setIndices(1 to 13 toArray)

    val myLR = new LinearRegression()
    myLR.setElasticNetParam(0.1).setMaxIter(10).setRegParam(.05)

    val lrStage = 6

    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLR))

    val pipelineModel: PipelineModel = pipeline.fit(training)

    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]
    //print rmse of our model
    //println(s"Best linear regression Model by Cross Validation params. Max Iteratiions: ${lrModel.getMaxIter}  Regularization Parameters: ${lrModel.getRegParam}")
    println(s"Training RMSE : ${lrModel.summary.rootMeanSquaredError}")

    //do prediction - print first k
    val predictions = pipelineModel.transform(test)
    predictions.select("features","label","prediction").take(5).foreach(println)

    // RMSE on Test Set
    println()
    println()
    val testEvaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = testEvaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)
    sc.stop()
  }
}