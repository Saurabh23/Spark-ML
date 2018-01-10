package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{RegexTokenizer, VectorAssembler, VectorSlicer}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.PipelineModel
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

    val myLR = new MyLinearRegressionImpl()

    val lrStage = 6

    val pipeline =  new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLR))
    val pipelineModel: PipelineModel = pipeline.fit(trainingData)
    val myLRModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]

    //print rmse of our model
    val trainingError = myLRModel.trainingError(99)
    println(s"RMSE on training data: ${trainingError}")

    //do prediction - print first k
    val predictions = pipelineModel.transform(testData)
    predictions.select( "features","label","prediction").take(5).foreach(println)

    println()
    val testRMSE:Double = scala.math.pow(predictions.map(row=>scala.math.pow((row.getAs[Double]("label") - row.getAs[Double]("prediction")),2)).reduce((a,b)=> a+b)/predictions.count(),0.5)
    print(s"RMSE on test data: ${testRMSE}")

  }
}