package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    //TODO read about use of ( and { and fold functions. They may be difficult to parallelize
    //((v1.toArray).zip(v2.toArray)).map{ case(a,b)=>a*b}.foldLeft(0.0)((b,a)=>b+a)
    ((v1.toArray).zip(v2.toArray)).map{ case(a,b)=>a*b}.reduce((a,b)=>a+b)
  }

  def dot(v: Vector, s: Double): Vector = {
    Vectors.dense(v.toArray.map(value=>value*s))
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    Vectors.dense((v1.toArray).zip(v2.toArray).map{ case(a,b)=>a+b})
  }

  def fill(size: Int, fillVal: Double): Vector = {
    Vectors.dense(Array.fill(size){fillVal})
  }
}