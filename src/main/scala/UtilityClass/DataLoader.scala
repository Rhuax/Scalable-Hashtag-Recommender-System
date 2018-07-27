package UtilityClass

import java.text.SimpleDateFormat
import java.util.Date

import Clustering.Point
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.reflect.ClassTag


object DataLoader {


  val sparkConf = new SparkConf()
  sparkConf.setAppName("HashTagRecommender")
  sparkConf.set("spark-serializer", "org.apache.spark.serializer.KryoSerializer")
  sparkConf.set("spark.kryoserializer.buffer.max", "2047")
  sparkConf.registerKryoClasses(Array(classOf[Clustering.Point]))

  val sc = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate().sparkContext
  sc.setLogLevel("ERROR")
  // optimal number of partitions to split the data in, so that parallelization is tuned
val partitionsCount: Int = (1.75*Runtime.getRuntime.availableProcessors()*(sc.getExecutorStorageStatus.length-1)).toInt


  /**
   *   Function to load the dataset (image features and hashtag) from text file to SparkRDD
   *
   * @param images path for image features file
   * @param tag_list path for hashtag list file
   * @param numPart (optional, defaults to @partitionsCount) number of partition for the RDD
   * @return Return an RDD[(Long,Sample)] a PairRDD where the key is the id of the image (from 0 to N)
   *         and Sample contains the features and the hashtag for each image loaded
   */
  def load(images:String, tag_list:String, numPart:Int = partitionsCount )= {
    def pointsCoordinates = sc
      .textFile(images, minPartitions = numPart)
      .map(line => line.split(",")
      .map(coordStr => coordStr.toDouble))
      //The first
      .map(pointCoordinates => (pointCoordinates.head.toLong,new Point(pointCoordinates.drop(1),pointCoordinates.head.toLong)))

    def hashtags=sc
      .textFile(tag_list, minPartitions =  numPart)
      .map(x=> x.split(" "))
      .map(x => (x.head.toLong,x.drop(1)))


    def iola= pointsCoordinates.join(hashtags)
    iola


  }

  /**
    * Save an RDD object to disk
    * @param rdd RDD to store
    * @param path (optional. by default try to save to /tmp/timestamp) Path to specify, must be a nonexisting directory
    * @tparam T Type parameter for specifying generic RDD method
    */
  def saveRDDtoObjectFile[T](rdd:RDD[T] , path:String = null){
    try {
      val newPath: String = if (path == null) "/tmp/" +  new SimpleDateFormat("ddMMyyyy_HHmmss").format(new Date()) else path
      rdd.saveAsObjectFile(newPath)
    }catch{case e: Exception => e.printStackTrace()}
   }

  /**
    * Load an RDD previously saved to disk
    * @param path Path of the saved RDD
    * @param T Returning type of the RDD to load
    * @return RDD[T]
    */
  def loadRDDfromDisk[T:ClassTag](path: String): RDD[T] = {
    sc.objectFile[T](path)
  }
}
