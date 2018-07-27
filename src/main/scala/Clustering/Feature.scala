package Clustering

/**
 * Trait to make a genheric type behave as a feature, i.e. a vector that can be manipolated by the system
*/
trait Feature[T] {

  /**
   * Calculates the distances between two features. Choosing the most appropriate algorythm to calculate the distance is delegated to the implementation.
   * @param a: first point to calculate the distance between
   * @param r: second point to calculate the distance between
   * @return the distance between the two given points
  */
  def distance(a:T, r:T): Double

}
