package Clustering.Distances

import Clustering.{Feature, Point}

object ManhattanDistance  extends Feature[Point]{
  override def distance(a: Point, r: Point): Double = {
    (a.coordinates zip r.coordinates).map(xy=>xy._1-xy._2).map(x=>Math.abs(x)).sum
  }
}
