package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.linalg.Vector
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}
import org.apache.spark.ml.param.ParamMap

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = sqlContext.read.textFile(filePath).toDF()
    val Array(trainingData, testData) = obsDF.randomSplit(Array(0.7, 0.3),1234)

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

    val pipeline =  new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLR))



    for(iter <-Array(10,50)) {
      for(reg <- Array(0.1,0.9)){
        val paramMap = ParamMap(myLR.maxIter->iter).put(myLR.regParam->reg)
        val pipelineModel: PipelineModel = pipeline.fit(trainingData,paramMap)
        val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

        println(s"HyperParameters: Max Iterations: ${lrModel.getMaxIter}. Regularization Parameter: ${lrModel.getRegParam}")
        //print rmse of our model
        val trainingSummary = lrModel.summary
        println(s"RMSE on training data: ${trainingSummary.rootMeanSquaredError}")

        //do prediction - print first k
        val predictions = pipelineModel.transform(testData)
        predictions.select( "features","label","prediction").show(5)

        //RMSE on TestData
        println()
        val testEvaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
        val rmse = testEvaluator.evaluate(predictions)
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)

        println()
        println("---------------------------------------------------------")
      }
    }



  }
}