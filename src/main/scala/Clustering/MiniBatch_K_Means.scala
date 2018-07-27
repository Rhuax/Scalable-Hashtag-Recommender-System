package Clustering

import breeze.numerics.sqrt
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import breeze.linalg.max
/**
  * An implementation of the Mini Batch K-Means Clustering.
  *  https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
  *
  *
  *
  * @param points dataset
  * @param batch_size batch size
  * @param finalsCentroids unused
  */
class MiniBatch_K_Means(points: RDD[Point],batch_size:Int,calculateSSI:Boolean=false,setMaster:String, var finalsCentroids: Array[Point] = null) extends Clustering_Method with Serializable {
  points.persist(StorageLevel.MEMORY_AND_DISK_SER)
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
  override def clusterize(iterations: Int = 0, k: Int, distances: Feature[Point], RSSVariation: Double = 0.0, centroidSimilarity: Double = 0.0) = {

    def newK = if (k == 0) sqrt(points.count() / 2).toInt else k
    println("K="+newK)
    def centroids = points.takeSample(withReplacement = false, newK)
    println("Starting centroids "+centroids.length)

    //Stopping criteria
    def returnc = {
      if (iterations > 0) {
        //I want to use iteration
        runByIteration(iterations, centroids, distances)
      }
      else if (RSSVariation > 0.0) {
        runByRSSVariation(centroids, RSSVariation, distances, Double.MaxValue)
      }
      else {
        runCosine(centroidSimilarity, centroids, distances)
      }
    }


    //println("returnC "+returnc.length)
    finalsCentroids = returnc
    val dictionary = buildFinalCluster(distances)
    dictionary

  }

  /**
    * Inner recursive function which implements an iteration of the algorithm
    * @param iterations Number of iterations to be done
    * @param centroids List of current found centroids
    * @param distances The distance metric measure to be used
    * @return Final found centroids
    */
  private def runByIteration(iterations: Int, centroids: Array[Point],distances:Feature[Point]): Array[Point] = {
    println("Iteration:"+iterations)
    if(iterations==0){
      centroids
    }
    else{
      //Mini batch
      val M=sc.parallelize(points.takeSample(withReplacement = false,batch_size))
      val c = centroids
      val nuovae=M
        .map(x=>(c.reduce((punto1,punto2)=>if (distances.distance(x,punto1)<distances.distance(x,punto2)) punto1 else punto2),x))

      val comb=nuovae.combineByKey(
        (x: Point) => x::Nil,
        (x: List[Point], y: Point) => x:::(y::Nil),
        (x: List[Point], y: List[Point]) => x:::y
      )

      //println("comb size "+comb.count())
      def update_centroid(centroid:Point,x:Point,acc:Int):Point={
        val n=1/(acc+1)
        centroid.multiply(1-n) sum x.multiply(n)
      }


      def updated_centroids=comb.map(coppia=>coppia._2.foldLeft((coppia._1,0))((acc,ele)=>(update_centroid(acc._1,ele,acc._2),acc._2+1)))
      //println("updated size:"+updated_centroids.count())

      def newcentroids=updated_centroids.map(x=>x._1).collect()
      runByIteration(iterations-1,newcentroids,distances)

    }
  }


  /**
    * Inner recursive function which interrupts when the RSS is below the given RSS
    * @param centroids List of current found centroids
    * @param RSSVariation Residual sum of squares
    * @param distances The distance metric measure to be used
    * @return Final found centroids
    */

  private def runByRSSVariation(centroids: Array[Point], RSSVariation: Double, distances: Feature[Point], lastRSS: Double): Array[Point] = {
    println("RunRSSVariation lastRSS " + lastRSS +  "RSSVariation " + RSSVariation )

    def computeRSS(centroid: Point, cluster_points: List[Point]): Double = {
      cluster_points.map(elem => Math.pow(distances.distance(centroid, elem), 2)).sum
    }

    //Do K-Means
    def c =centroids
    val M=sc.parallelize(points.takeSample(withReplacement = false,batch_size))

    //(centroide, punti)
    val nuovae=M
      .map(x=>(c.reduce((punto1,punto2)=>if (distances.distance(x,punto1)<distances.distance(x,punto2)) punto1 else punto2),x))
    //Nuovi centroidi
    def n = nuovae.combineByKey(
      (x: Point) => (x, 1),
      (x: (Point, Int), y: Point) => (x._1 sum y, x._2 + 1),
      (x: (Point, Int), y: (Point, Int)) => (x._1 sum y._1, x._2 + y._2)
    )
      .mapValues(x => x._1 divide x._2)
      .map(x => x._2)

    /*def prendipunti = nuovae.combineByKey(
      (x: Point) => List(x),
      (x: List[Point], y: Point) => y :: x,
      (x: List[Point], y: List[Point]) => x ::: y
    )//.map(x => x._2)*/

    def prendipuntiNuovi=points
      .map(x=>(n.reduce((punto1,punto2)=>if (distances.distance(x,punto1)<distances.distance(x,punto2)) punto1 else punto2),x))

    def coppie=prendipuntiNuovi.combineByKey(
      (x: Point) => List(x),
      (x: List[Point], y: Point) => y :: x,
      (x: List[Point], y: List[Point]) => x ::: y
    )

    def newRSS = coppie.map(x => computeRSS(x._1, x._2)).sum

    if (Math.abs(newRSS - lastRSS) <= RSSVariation) {
      n.collect()
    }
    else {
      runByRSSVariation(n.collect(), RSSVariation, distances, newRSS)
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

    def cosineSimilarity(oldCentroid: Point, newCentroid: Point): Double = {
      def num: Double = (oldCentroid.coordinates zip newCentroid.coordinates).map(x => x._1 * x._2).sum

      def den1: Double = math.sqrt(oldCentroid.coordinates.map(x => math.pow(x, 2)).sum)

      def den2 = math.sqrt(newCentroid.coordinates.map(x => math.pow(x, 2)).sum)

      num / (den1 * den2)
    }

    //Mini batch
    val M=sc.parallelize(points.takeSample(withReplacement = false,batch_size))//.map(_._2)
    val c = centroids
    def nuovae=points
      .map(x=>(c.reduce((punto1,punto2)=>if (distances.distance(x,punto1)<distances.distance(x,punto2)) punto1 else punto2),x))
      .combineByKey(
        (x: Point) => (x, 1),
        (x: (Point, Int), y: Point) => (x._1 sum y, x._2 + 1),
        (x: (Point, Int), y: (Point, Int)) => (x._1 sum y._1, x._2 + y._2)
      )
      .mapValues(x => x._1 divide x._2)
      .map(x => x._2)//nuovi centroidi

    def nuoviCentroidi=nuovae.collect()

    def similarities = (centroids zip nuoviCentroidi).map(x => cosineSimilarity(x._1, x._2)).exists(x => x < threshold)

    if (similarities) {
      runCosine(threshold, nuoviCentroidi, distances)
    }
    else {
      nuoviCentroidi
    }
  }




  /**
    * Builds the dictionary which holds the final centroids and its points
    *
    * @param distances distance metric to be used
    * @return
    */
  private def buildFinalCluster(distances: Feature[Point]): RDD[(Point, List[Long])]= {

    val nuovae: RDD[(Point, Point)] =points.map(x=>(this.finalsCentroids
      .reduce((punto1,punto2)=>if (distances.distance(x,punto1)<distances.distance(x,punto2)) punto1 else punto2),x))


    if (calculateSSI) meanSSI(nuovae,distances)

    def dict: RDD[(Point, List[Long])] = nuovae.combineByKey(
      (x: Point) => List[Long](x.id),
      (x: List[Long], y: Point) => y.id :: x,
      (x: List[Long], y: List[Long]) => x ::: y
    ) //(centroide,Lista punti che gli apaprtengono

    dict
  }






  private def meanSSI(coppieNuove:RDD[(Point,Point)],distances: Feature[Point]) = {

    def dict: RDD[(Point, List[Point])] = coppieNuove.combineByKey(
      (x: Point) => List[Point](x),
      (x: List[Point], y: Point) => y:: x,
      (x: List[Point], y: List[Point]) => x ::: y
    )

    val soloCentroidi=dict.keys.collect()

     def calculateSilhouetteForPoint(p:Point):Double={
       val distanzaConTuttiCentroidi=soloCentroidi.map(x=>distances.distance(p,x))

      val rrrrrr=distanzaConTuttiCentroidi.sortBy(identity).take(2)
      (rrrrrr(1)-rrrrrr(0))/ max(rrrrrr(0),rrrrrr(1))
    }



    val iola=dict.map(x=>x._2.foldLeft((0.0,0))
    (
      (acc,nuovopunto)=> (acc._1+calculateSilhouetteForPoint(nuovopunto),acc._2+1)
    ))

    val totalssi=iola.reduce((uno,due)=>(uno._1+due._1,uno._2+due._2))
    val meanssi=totalssi._1/totalssi._2

    if(meanssi <= 0) println("Hai fatto un casino, c'è qualche punto clusterizzato male!!!! MeanSSI: " + meanssi)
    else{
      if(meanssi <= 0.5) println("Insomma... si poteva fare di meglio. MeanSSI: " + meanssi)
      else println("Hai fatto un ottimo lavoro! MeanSSI: " + meanssi)
    }
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
