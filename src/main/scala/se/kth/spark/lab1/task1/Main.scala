package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

case class Song(year: Integer, feature1: Double, feature2: Double, feature3: Double)

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    //val rawDF = ???

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    rdd.take(5).foreach(println)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(row => row.split(","))


    //Step3: map each row into a Song object by using the year label and the first three features
    val songsRdd = recordsRdd.map(features => Song(features(0).substring(0,4).toInt,features(1).toDouble,features(2).toDouble,features(3).toDouble))

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF()

    //????????????      cache songsDf in memory . Should this be done after registering as a table of before ???
    songsDf.cache()

    songsDf.show(5)
    songsDf.createOrReplaceTempView("songs")

    //Q1 How many songs there are in the DataFrame
    println(songsDf.count())
    sqlContext.sql("select count(*) from songs").show()

    //Q2 How many songs were released between the years 1998 and 2000
    println(songsDf.filter($"year" >= 1998 && $"year" <= 2000 ).count())
    sqlContext.sql("select count(*) from songs where year >=1998 and year <= 2000").show()

    //Q3What is the min, max and mean value of the year column
    sqlContext.sql("select min(year),max(year),mean(year) from songs").show()

    val yearsDF = songsDf.map(song => song.getAs[Int]("year")).cache()

    println("minimum: "+yearsDF.reduce((year1,year2) => {
      if(year1<year2) year1
      else year2
    }  ))


    println("maximum: "+yearsDF.reduce((year1,year2) => {
      if(year1>year2) year1
      else year2
    }  ))

    // ????? better way to calculate mean
    println("mean: "+yearsDF.reduce((year1,year2) => year1+year2)/songsDf.count())


    //Q4: Show the number of songs per year between the years 2000 and 2010
    sqlContext.sql("select year,count(*) from songs where year>=2000 and year<=2010 group by year").show()
    //no reduceByKey for Data frame ??????????????
    //songsDf.map(song => (song(0),1)).groupByKey(0\
    songsDf.filter($"year">=2000 && $"year"<=2010).groupBy("year").count().show()

  }



}