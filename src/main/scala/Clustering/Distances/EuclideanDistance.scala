package Clustering.Distances

import Clustering.Feature
import Clustering._
import scala.math.sqrt
import scala.math.pow

object EuclideanDistance extends Feature[Point] with Serializable {
  override def distance(a: Point, b: Point): Double = {
    val coupledPoints = a.coordinates.zip(b.coordinates)
    val coordDistances = coupledPoints.map(pair => pair._1-pair._2).map(coordDiff => pow(coordDiff, 2))
    sqrt(coordDistances.sum)
  }

}
