import Clustering.Distances._
import Clustering.{K_Means, MiniBatch_K_Means, Point}
import HashtagRecommender.HashtagRecommender
import MyConf.MyConf
import UtilityClass.DataLoader
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

object Main {

  def main(args: Array[String]) {
    val conf = new MyConf(args)
    val sparkConf = new SparkConf()
    val masterURL = conf.masterURL.apply()
    // Set up spark configuration
    sparkConf.setAppName("HashTagRecommender")
    sparkConf.setMaster(masterURL)
    sparkConf.set("spark-serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.set("spark.kryoserializer.buffer.max", "2047")
    sparkConf.registerKryoClasses(Array(classOf[Clustering.Point]))
    // Initialize a spark session built with the configuration given before
    val sc = SparkSession
      .builder()
      .config(sparkConf)
      .getOrCreate()
    sc.sparkContext.setLogLevel("ERROR")

    val featureFileName = conf.featuresFile.apply().getAbsolutePath
    val tagFileName = conf.tagFile.apply().getAbsolutePath
    // Create DataSet with key ID, Image Vector and HashTag associated
    def dataSet = DataLoader.load(featureFileName, tagFileName)
    // Val "points" gets all the point in the DataSet
    val points = dataSet.map(x=>x._2._1)
    // Check if the DataSet has already been stored
    val recommender=if(!conf.inClusterFile.supplied) {
      val k = conf.k.apply()
      // Check which clustering method has been choose
      val clustering = if (conf.clusteringMethod.apply().equalsIgnoreCase("minibatch")) {
        val batch = conf.batchSize.apply()
        new MiniBatch_K_Means(points, batch, conf.ssi.apply(), masterURL)
      } else {
        new K_Means(points, conf.ssi.apply(), masterURL)
      }
      val iter = conf.iterations.apply()
      val rss = conf.rssVariation.apply()
      val centroidSimilarity = conf.centroidSimilarity.apply()
      // Call "clusterize" method on our class
      val centroids = clustering.clusterize(iter, k, EuclideanDistance, rss, centroidSimilarity)
      centroids.persist(StorageLevel.MEMORY_AND_DISK)
      // Save the cluster file if the user gave an explicit path
      if (conf.outClusterFile.supplied) {
        DataLoader.saveRDDtoObjectFile(centroids, conf.outClusterFile.apply().getAbsolutePath)
      }

      new HashtagRecommender(centroids, EuclideanDistance, dataSet,masterURL)
    }
    else{
      // Load the cluster file
      def savedCentroids=DataLoader.loadRDDfromDisk[(Point,List[Long])](conf.inClusterFile.apply().getAbsolutePath)
      new HashtagRecommender(savedCentroids, EuclideanDistance, dataSet,masterURL)

    }

    val pathImage = conf.imagePath.apply()
    val similarImage = conf.closestImages.apply()
    // Evoke the method for the Hashtags recommendation
    recommender.getHashtagsFromImage(pathImage,similarImage)

  }

}
