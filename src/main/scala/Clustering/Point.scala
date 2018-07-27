package Clustering


/**
 * Represents a point into an n-dimensional space
 * @param coordinates: array of the point coordinates. The number of coordinates determines the n-dimensional space that the point is placed into
 * @param id: an unique ID to identify the point
*/
// required to make the class serializable by KryoSerializer
@SerialVersionUID(100L)
class Point(val coordinates: Array[Double], val id: Long) extends Serializable {

  /**
   * Returns the point corresponding to the sum of two points
   * @param p: the point to sum to the point that the method is invoked onto
   * @return the point corresponding to the sum of the two points
   */
  def sum(p: Point): Point = {
    val sumCoordinates = coordinates.zip(p.coordinates).map(pair => pair._1 + pair._2)
    new Point(sumCoordinates, -1) //ID fittizio
  }

  /**
   * Returns the point corresponding to the difference of two points
   * @param p: the point to subtract from the point that the method is invoked onto
   * @return the point corresponding to the difference of the two points
   */
  def minus(p: Point): Point = {
    val min = coordinates.zip(p.coordinates).map(pair => pair._1 - pair._2)
    new Point(min, -1) //ID fittizio
  }

  /**
    * Returns the point corresponding to the moltiplication of the coordinates of a point by an integer
    * @param n: the integer to multiply the point coordinates by
    * @return the point obtained by multipliying the coordinates of the point that the method has been invoked onto by the given integer
    */
  def multiply(n: Double): Point = {
    new Point(coordinates.map(_ * n), -1) //ID fittizio
  }

  /**
  * Returns the point corresponding to the division of the coordinates of a point by an integer
  * @param n: the integer to divide the point coordinates by
  * @return the point obtained by dividing the coordinates of the point that the method has been invoked onto by the given integer
  */
  def divide(n: Int): Point = {
    new Point(coordinates.map(_ / n), -1) //ID fittizio
  }

  /**
   * @return an unique hash code for the point
  */
  override def hashCode(): Int = {
    coordinates.deep.hashCode()
  }

  /**
  * @return a string representation of the point
  */
  override def toString: String = {
    var s: String = "{"
    for (i <- coordinates) {
      s += i.toString + ","
    }
    s += "}"
    s
  }

  /**
   * Compares to points of an n-dimensional space
   * @param pointa: the point to compare with the point that the method has been invoked onto
   * @return true if the two points are equal, false otherwise
  */
  override def equals(pointa: Any): Boolean = {
    pointa match {
      case pointa: Point => pointa.coordinates.sameElements(this.coordinates)
      case _ => false
    }
  }

}
