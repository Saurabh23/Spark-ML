package se.kth.spark.lab1.task4

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
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
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

    fSlicer.setIndices(Array(1,2,3))

    val myLR = new LinearRegression()
    myLR.setElasticNetParam(0.1)

    val lrStage = 6

    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLR))

    //build the parameter grid by setting the values for maxIter and regParam
    val paramGrid = new ParamGridBuilder().addGrid(myLR.maxIter,Array(5,10,30,50,100,200,300)).addGrid(myLR.regParam,Array(.001,.01,.05,.1,.5,.9)).build()
    val evaluator = new RegressionEvaluator

    val cv = new CrossValidator().setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel: CrossValidatorModel = cv.fit(training)

    val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    println(s"Best linear regression Model by Cross Validation params. Max Iteratiions: ${lrModel.getMaxIter}  Regularization Parameters: ${lrModel.getRegParam}")
    println(s"Best linear regression Model by Cross Validation RMSE : ${lrModel.summary.rootMeanSquaredError}")

    //do prediction - print first k
    val predictions = cvModel.transform(test)
    predictions.select("features","label","prediction").take(5).foreach(println)

    // RMSE on Test Set
    println()
    println()
    val testEvaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = testEvaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

  }
}