package se.kth.spark.lab1

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.hack.VectorType
import scala.collection.mutable.WrappedArray

class Array2Vector(override val uid: String)
    extends UnaryTransformer[WrappedArray[String], Vector, Array2Vector] {

  def this() = this(Identifiable.randomUID("arrayToVector"))

  override protected def createTransformFunc: WrappedArray[String] => Vector = {
    (p1: WrappedArray[String]) => {Vectors.dense(p1.array.map { x => x.replaceAll("\"", "").toDouble })}
  }

  override protected def outputDataType: VectorType = {
    new VectorType
  }
}