package Clustering

import org.apache.spark.rdd.RDD


/**
 * Abstract class to encapsulate the logic for a clustering algorythm and make different algorythms swappable in the system
 */
abstract class Clustering_Method{
  /**
   * Clusterizes the given dataset. Choosing the most appropriate clustering algorythm is delegated to the implementation
   @param iter: the number of iterations to perform. If 0 the implementation should use other criteria to stop the algorythm
   * @param k: the number of clusters to create
   * @param distances: the Feature type to use for calculating points distances during the clustering process
   * @param RSSVariation: minimal RSS (Residual Sum of Squares) variation between two iterations that the algorythm should be stopped after
   * @param centroidSimilarity: desired centroid similarity to stop the algorythm when reached
   * @return the clusterized dataset
  */
 def clusterize(iter:Int,k:Int, distances: Feature[Point],RSSVariation:Double,centroidSimilarity:Double):RDD[(Point, List[Long])]

  /**
   * @return all points in the dataset
  */
 def getPoints : RDD[Point]

  /**
   * @return the list of centroids of the clusters in the dataset
  */
 def getCentroids : Array[Point]

}
