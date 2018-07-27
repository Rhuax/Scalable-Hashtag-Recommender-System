package Clustering



import breeze.linalg.max
import breeze.numerics.sqrt
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
/**
  * An implementation of the K-Means Clustering.
  *
  *
  *
  * @param points dataset
  * @param finalsCentroids unused
  */

class K_Means(points: RDD[Point],calculateSSI:Boolean=false,setMaster:String, var finalsCentroids: Array[Point] = null) extends Clustering_Method with Serializable {

  //points è un RDD che contiene la lista delle immagini featurizzate
  points.persist(StorageLevel.MEMORY_AND_DISK)

  def sparkConf = new SparkConf()
  sparkConf.setAppName("HashTagRecommender")
  sparkConf.setMaster(setMaster)
  sparkConf.set("spark-serializer", "org.apache.spark.serializer.KryoSerializer")
  sparkConf.set("spark.kryoserializer.buffer.max", "2047")
  sparkConf.registerKryoClasses(Array(classOf[Clustering.Point]))

  def sc = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate().sparkContext


  sc.setLogLevel("ERROR")
  /**
    * Main function to clusterize the dataset.
    *
    * @param iterations Number of iterations. If 0 then other parameters are used
    * @param k Number of clusters. If 0 a heuristic approach is used to estimate the optimal k
    * @param distances Function to calculate the distance between points
    * @param RSSVariation Residual Sum of Squares minimum variation between 2 iterations to stop the process of clustering
    * @param centroidSimilarity Minimum cosine similarity between the sets of centroids between 2 iteration to stop the process of clustering
    * @return Returns an RDD of couples. Every couple is formed by a found centroid and the list of points inside its cluster
    */

  override def clusterize(iterations: Int = 0, k: Int, distances: Feature[Point], RSSVariation: Double = 0.0, centroidSimilarity: Double = 0.0): RDD[(Point, List[Long])] = {

    //Check if K needs to be generate by thumb rule
    def newK = if (k == 0) sqrt(points.count() / 2).toInt else k
    println("K-Means using k="+newK)
    //Take a sample of unique random centroids from point
    def centroids = points.takeSample(withReplacement = false, newK)

    //Stopping criteria
    def returnc = {
      if (iterations > 0) {
        //I want to use iteration
        runByIteration(iterations, centroids, distances)
      }
      else if (RSSVariation > 0.0) {
        //I want to use RSS variation
        runByRSSVariation(centroids, RSSVariation, distances, Double.MaxValue)
      }
      else {
        //I want to use Cosine Similarity
        runCosine(centroidSimilarity, centroids, distances)
      }
    }



    this.finalsCentroids = returnc
    // Build dictionary with centroids and points associated
    val dictionary = buildFinalCluster(distances)

    dictionary

  }

  /**
    * Inner recursive function which interrupts when all the iterations have been done
    * @param iter Number of iterations to be done
    * @param centroids List of current found centroids
    * @param distances The distance metric measure to be used
    * @return Final found centroids
    */

  private def runByIteration(iter: Int, centroids: Array[Point], distances: Feature[Point]): Array[Point] = {
    println("RunIteration number "+iter)
    if (iter == 0) {
      centroids
    }
    else {
      val c = centroids

      //Find nearest centroid for each point
      val nearest: RDD[(Point, Point)] =points
        .map(x=>(c.reduce((centroid1,centroid2)=>if (distances.distance(x,centroid1)<distances.distance(x,centroid2)) centroid1 else centroid2),x))


      //Summation of each point for each cluster
      val newC = nearest
        .combineByKey(
          (x: Point) => (x, 1),
          (x: (Point, Int), y: Point) => (x._1 sum y, x._2 + 1),
          (x: (Point, Int), y: (Point, Int)) => (x._1 sum y._1, x._2 + y._2)
        )
        //Mean point for each cluster
        .mapValues(x => x._1 divide x._2)
        // Collecting the new centroids
        .map(x => x._2)
        .collect()

      runByIteration(iter - 1, newC, distances)
    }
  }

  /**
    * Inner recursive function which interrupts when the RSS is below the RSS we gave
    * @param centroids List of current found centroids
    * @param RSSVariation Residual sum of squares
    * @param distances The distance metric measure to be used
    * @return Final found centroids
    */

  private def runByRSSVariation(centroids: Array[Point], RSSVariation: Double, distances: Feature[Point], lastRSS: Double): Array[Point] = {
    println("RunRSSVariation lastRSS " + lastRSS +  "RSSVariation " + RSSVariation )

    // Compute RSS
    def computeRSS(centroid: Point, cluster_points: List[Point]): Double = {
      cluster_points.map(elem => Math.pow(distances.distance(centroid, elem), 2)).sum
    }

    //Parallelize centroids in RDD
    val c = sc.parallelize(centroids)



    //Find nearest centroid for each point
    def nearest=points
      .map(x=>(c.reduce((centroid1,centroid2)=>if (distances.distance(x,centroid1)<distances.distance(x,centroid2)) centroid1 else centroid2),x))

    //Summation of each point for each cluster
    val newC = nearest.combineByKey(
      (x: Point) => (x, 1),
      (x: (Point, Int), y: Point) => (x._1 sum y, x._2 + 1),
      (x: (Point, Int), y: (Point, Int)) => (x._1 sum y._1, x._2 + y._2)
    )
      //Mean point for each cluster
      .mapValues(x => x._1 divide x._2)
      // Collecting the new centroids
      .map(x => x._2)

    // Get new points
    val newPoints=points
      .map(x=>(newC.reduce((centroid1,centroid2)=>if (distances.distance(x,centroid1)<distances.distance(x,centroid2)) centroid1 else centroid2),x))

    // Create list of centroids and points
    val couple=newPoints.combineByKey(
      (x: Point) => List(x),
      (x: List[Point], y: Point) => y :: x,
      (x: List[Point], y: List[Point]) => x ::: y
    )

    // Evaluate RSS
    val newRSS = couple.map(x => computeRSS(x._1, x._2)).sum

    // Interrupt clustering if RSS variation is below the value gave by the user
    if (Math.abs(newRSS - lastRSS) <= RSSVariation) {
      newC.collect()
    }
    else {
      runByRSSVariation(newC.collect(), RSSVariation, distances, newRSS)
    }
  }

  /**
    * Inner recursive function which interrupts when the cosine similarity is stable based on a pre-determined threshold
    * @param threshold Value which determine the percentage of similarity to reach
    * @param centroids List of current found centroids
    * @param distances Function to calculate the distance between points
    * @return  Final found centroids
    */

  private def runCosine(threshold: Double, centroids: Array[Point], distances: Feature[Point]): Array[Point] = {

    println("RunCosine")

    /**
      * Gradient step for the centroid
      * @param oldCentroid
      * @param newCentroid
      * @return
      */
    def cosineSimilarity(oldCentroid: Point, newCentroid: Point): Double = {
      //Sum between product of centroids new and old coordinates
      def num: Double = (oldCentroid.coordinates zip newCentroid.coordinates).map(x => x._1 * x._2).sum
      //Squared root beetween the sum of squared centroids
      def den1: Double = math.sqrt(oldCentroid.coordinates.map(x => math.pow(x, 2)).sum)

      def den2: Double = math.sqrt(newCentroid.coordinates.map(x => math.pow(x, 2)).sum)

      num / (den1 * den2)
    }

    //Parallelize centroids in RDD
    def c = sc.parallelize(centroids)

    def nearest=points
      .map(x=>(c.reduce((centroid1,centroid2)=>if (distances.distance(x,centroid1)<distances.distance(x,centroid2)) centroid1 else centroid2),x))

    //Summation of each point for each cluster
    val newC = nearest.combineByKey(
        (x: Point) => (x, 1),
        (x: (Point, Int), y: Point) => (x._1 sum y, x._2 + 1),
        (x: (Point, Int), y: (Point, Int)) => (x._1 sum y._1, x._2 + y._2)
      )
      //Mean point for each cluster
      .mapValues(x => x._1 divide x._2)
      // Collecting the new centroids
      .map(x => x._2)
      // Transoform RDD in Array[Point]
      .collect()

    // Similarities calculate Cosine Similarity and return true if the value is still below the threshold
    def similarities = (centroids zip newC).map(x => cosineSimilarity(x._1, x._2)).exists(x => x < threshold)

    if (similarities) {
      runCosine(threshold, newC, distances)
    }
    else {
      newC
    }
  }

  /**
    * Function which associate every centroids with their list of points
    * @param distances Function to calculate the distance between points
    * @return Dictionary of centroids and points associated
    */

  private def buildFinalCluster(distances: Feature[Point]) = {
    // Nearest points from centroids
    val nearest: RDD[(Point, Point)] =points.map(x=>(this.finalsCentroids
      .reduce((centroid1,centroid2)=>if (distances.distance(x,centroid1)<distances.distance(x,centroid2)) centroid1 else centroid2),x))

    // Calculate mean Simple Silhouette Index
    if (calculateSSI) meanSSI(nearest,distances)

    // Dictionary of centroids and list of ID
    def dict: RDD[(Point, List[Long])] = nearest.combineByKey(
      (x: Point) => List[Long](x.id),
      (x: List[Long], y: Point) => y.id :: x,
      (x: List[Long], y: List[Long]) => x ::: y
    )

    dict
  }

  /** Function that calculate the mean Simple Silhouette Index to measure the quality of the clastering
    * @param newCouple RDD of centroid and point couple
    * @param distances distance method
    */

  private def meanSSI(newCouple:RDD[(Point,Point)],distances: Feature[Point]) = {

    //Dictionary of Centroids and list of Points
    def dict: RDD[(Point, List[Point])] = newCouple.combineByKey(
      (x: Point) => List[Point](x),
      (x: List[Point], y: Point) => y:: x,
      (x: List[Point], y: List[Point]) => x ::: y
    )

    val justC=dict.keys.collect()

    /** Calculate silhouette for each point
      * @param p Point to apply silhouette
      * @return Silhouette for each point
      */
    def calculateSilhouetteForPoint(p:Point):Double={
      val distancesFromAllCentroids = justC.map(x=>distances.distance(p,x))

      val nearestNeighbor=distancesFromAllCentroids.sortBy(identity).take(2)
      (nearestNeighbor(1)-nearestNeighbor(0))/ max(nearestNeighbor(0),nearestNeighbor(1))
    }

    // Sum every silhouette and keeping the number of points
    val silhouetteSum: RDD[(Double, Int)] =dict.map(x=>x._2.foldLeft((0.0,0))
    (
      (acc,newpoint)=> (acc._1+calculateSilhouetteForPoint(newpoint),acc._2+1)
    ))

    val totalssi: (Double, Int) = silhouetteSum.reduce((one, two)=>(one._1+two._1,one._2+two._2))
    val meanssi: Double = totalssi._1/totalssi._2

    println("MeanSSI : " + meanssi)

  }

  /**
    * Getter for Points
    * @return
    */

  override def getPoints: RDD[Point] = points

  /**
    * Getter for centroids
    * @return
    */

  override def getCentroids: Array[Point] = finalsCentroids

}
