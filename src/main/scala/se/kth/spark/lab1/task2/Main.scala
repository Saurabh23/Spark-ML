package se.kth.spark.lab1.task2

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.linalg.Vector

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.textFile(filePath)
    //rawDF.show(5)

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("tokens")
      .setPattern(",")


    //Step2: transform with tokenizer and show 5 rows
    //val tokenizedDF = regexTokenizer.transform(rawDF)
    //tokenizedDF.select("tokens").take(5).foreach(println)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
      .setInputCol("tokens")
      .setOutputCol("vectors")

    //    val vectorizedDF = arr2Vect.transform(tokenizedDF)
    //    vectorizedDF.select("vectors").take(5).foreach(println)


    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer()
      .setInputCol("vectors")
      .setOutputCol("vectorLabel")

    lSlicer.setIndices(Array(0))

    //    val labelledDF = lSlicer.transform(vectorizedDF)
    //    labelledDF.select("vectorLabel").take(5).foreach(println)


    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF((v:Vector) => v(0))
      .setInputCol("vectorLabel")
      .setOutputCol("doubleLabel")

    //    val numericlabelledDF = v2d.transform(labelledDF)
    //    numericlabelledDF.select("doubleLabel").take(5).foreach(println)

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    //min. year hardcoded right now. TODO implement Estimator to get minimum year
    val lShifter = new DoubleUDF((d:Double) => (d - 1922.0 ))
      .setInputCol("doubleLabel")
      .setOutputCol("label")

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("vectors")
      .setOutputCol("features")

    fSlicer.setIndices(Array(1,2,3))

    //    val featureDf = fSlicer.transform(numericlabelledDF)
    //    featureDf.select("features").take(5).foreach(println)

    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer))

    //Step9: generate m odel by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
    val transformedDF = pipelineModel.transform(rawDF)

    //Step11: drop all columns from the dataframe other than label and features
    val trainedDF = transformedDF.select("label","features")
    trainedDF.select("label","features").take(5).foreach(println)
  }
}