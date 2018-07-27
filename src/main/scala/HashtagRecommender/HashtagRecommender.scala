package HashtagRecommender



import org.apache.spark.rdd.RDD
import Clustering._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import sys.process._
import scala.sys.process.{ProcessLogger, stderr}

/**
* Class to encapsulate the logic for recommending hashtags from an image
* @param centroidsResult: an array of pairs composed by a centroid and the list of the IDS of the dataset points associated to that centroid
 * @param feature: type of the feature to use for comparisons
* @param dataset: RDD containig the system dataset. It is made of pairs where the first item the the ID of a point and the second item is a pair consisting of the point and its associated hashtags
 * @param setMaster: URL of the master node used in the computation
*/
class HashtagRecommender(centroidsResult: RDD[(Point, List[Long])], feature: Feature[Point], dataset: RDD[(Long, (Point,Array[String]))], setMaster:String) extends Serializable {
  // for some reason, the standard methods to retrieve the Spark context don't work on our AWS cluster, so we have to instantiate it manually each time we need it
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
sc.setLogLevel("ERROR")//a

  /**
   * Executes a python script to convert an image into an unique vector that can be treated as a point in an n-dimensional space
   * @param path: the path of the image to convert
   * @return the point representing the image
   */
  private def callPython(path: String): Point = {
    val command = "python3 getimage.py " + path
    var result: String = ""
    val exec = command ! ProcessLogger(result += _, stderr append _)

    val point = new Point(result
      .split(",")
      .map(coordStr => coordStr.toDouble), -1)
    point
  }


  /**
   * Returns the recommended hashtags from an image
   * @param path: path of the image to recommend hashtags for
   * @param closestN: number of similar images in the dataset to consider for the hashtags extraction
   * @return a list of pairs made of an hashtag and its frequency in the considered closestN images
   */
  def getHashtagsFromImage(path: String,closestN:Int) = {
    val thePoint = callPython(path)
    getHashtags(thePoint,closestN)
  }


  /**
   * Prints the recommended hashtags from an image
   * @param imageVector: point in an n-dimensional space representing the image to recommend hashtags for
   * @param closestN: number of similar images in the dataset to consider for the hashtags extraction
   */
  private def getHashtags(imageVector: Point,closestN:Int) = {

    def closest_points = this.getClosestPoint(imageVector,closestN)
    println("Sorted tags by frequency:")
    extractSortByHashtag(closest_points).foreach(println)
    val closest_pointsColl=closest_points.collect()
    println("Closest image tags:")
    closest_pointsColl.take(1)(0)._2.foreach(println)
    println("Closest images ids:")
    closest_pointsColl.foreach(x => println(x._1.id))//aa


  }

  /**
   * Returns the points of the dataset that are closest to a given point
   * @param from: the point to get the closest (i.e. less distant) points from
   * @param closestN: the number of closest points to return
   * @return the closest points
    */
  private def getClosestPoint(from: Point,closestN:Int)= {
    val c=centroidsResult.keys
    val closestCentroid =c.reduce((centrmigliore,centr)=>if (feature.distance(centrmigliore,from)<feature.distance(centr,from)) centrmigliore else centr)

    val listeIDPunti = centroidsResult.lookup(closestCentroid).head

    val t=dataset.filter(x=>listeIDPunti.contains(x._1))

    val init=List[(Point,Array[String])]()



    def addToList(thelist:List[(Point,Array[String])],idPoint:(Point,Array[String]))={

      def ordered=idPoint::thelist.sortBy(x=>feature.distance(x._1,closestCentroid))
      ordered.take(closestN)
    }

    def mergeLists(listacc1:List[(Point,Array[String])],listacc2:List[(Point,Array[String])])={
      def ordered=(listacc1:::listacc2).sortBy(x=>feature.distance(x._1,closestCentroid))
      ordered.take(closestN)

    }

    def getPointFromId(id:Long)={
      dataset.lookup(id).head
    }






    val risul=t.aggregate(init)(
      (listacc,idPunto:(Long,(Point,Array[String])))=> listacc.length match{
        case `closestN` => addToList(listacc,idPunto._2)
        case _=> idPunto._2::listacc
      },
      (listacc1,listacc2)=>mergeLists(listacc1,listacc2)
    )
    sc.parallelize(risul)
  }


  /**
   * Returns the hashtags associated to a set of points sorted by their frequency
   * @param resultPoints: RDD of points to extract hashtags from. It is composed by pairs where the first item is a point of the dataset and the secondo ne is the array of hashtags associated to that point
   * @return the hashtags sorted by their frequency
  */
  private def extractSortByHashtag(resultPoints:RDD[(Point,Array[String])])={

    val r =resultPoints.map(x=>x._2)
    val m: Array[String] = r.reduce(_++_)
    val f: Map[String, Int] = m.groupBy(identity).mapValues(_.length)
    f.toList.sortBy(-_._2.toInt)
  }

}
