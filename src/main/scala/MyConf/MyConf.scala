package MyConf

import java.io.File
import org.rogach.scallop._


class MyConf(arguments: Seq[String]) extends ScallopConf(arguments) {

  val debug = opt[Boolean](
  descr = "Enables debug output, disabled by default.")


  val clusteringMethod = opt[String](
  argName = "Clustering Method",
  descr = "Name of the algorythm to use for clusterizing the dataset, can be either kmeans or minibatch",
  validate = (arg) => arg=="kmeans" || arg=="minibatch",
    default = Some("kmeans")
  )

  val batchSize = opt[Int](
    argName = "Batch Size",
    descr = "Number of sample taken after each iteration of minibatch kmeans",
    default = Some(1000)
  )

  val iterations = opt[Int](
  descr = "Number of iterations that the clustering algorythm will be run for. If 0 other criteria are used to determine when data are clasterized enough for the system.",
  validate = (arg) => arg>=0,
    default =  Some(0)
  )

  val k = opt[Int](
  argName = "clusters",
  descr = "Number of clusters to create. If 0 or not provided a heuristic approach is used to estimate the optimal value",
  validate = (arg) => arg>=0,
  default = Some(0)
  )

  val rssVariation = opt[Double](
  argName = "RSS Variation",
  descr = "RSS (Residual Sum of Squares) minimum variation between 2 iterations to stop the process of clustering.",
  default = Some(0)
  )

  val centroidSimilarity = opt[Double](
  descr = "Minimum cosine similarity between the sets of centroids between 2 iterations to stop the clusterization process.",
  default = Some(0.8)
)

  val featuresFile = opt[File](
  argName = "Images file",
  descr = "Path to a file containing image features to be used by the system for making hashtag predictions",
  required = true
  )
  validateFileExists(featuresFile)

  val tagFile = opt[File](
  argName = "Hashtag file",
  descr = "Path to a file containing a list of the hashtags that the system will be able to predict",
  required = true
  )
  validateFileExists(tagFile)

  val imagePath = opt[String](
  descr = "Path to an image to predict hashtags for. Can be a local path or a remote URL.",
  required = true
  )
  val inClusterFile = opt[File](
  descr = "File to load the clusterized dataset from. If provided, the clustering process won't be executed."
  )
  validateFileIsFile(inClusterFile)

  val outClusterFile = opt[File](
  descr = "Path to a file where the clusterized dataset will be saved to."
  )
  validateFileDoesNotExist(outClusterFile)

  val closestImages = opt[Int](
  descr = "Number of the most similar images to consider for predicting hashtags.",
  validate = (arg) => arg>0,
    default = Some(4)
  )

  val ssi = opt[Boolean](
  argName = "calculate SSI",
  descr = "Calculates the SSI (Silhouett Score Index) after clusterizing the dataset. Warning: computing the SSI slows down *a lot* the program execution.",
  default = Some(false)
  )

  val masterURL = opt[String](
    argName = "Master URL",
    descr = "Master URL",
    default = Some("local[*]")
  )



  banner("""
  Usage: hashtag-recommender-system [OPTIONS]
  Recommends appropriate hashtags for a given image.
  """.stripMargin)
  version("hashtag-recommender-system v0.1. Copyright © the Dream Team. Distributed under the MIT license.\n\n\n ████████╗██╗  ██╗███████╗    ██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗    ████████╗███████╗ █████╗ ███╗   ███╗\n╚══██╔══╝██║  ██║██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║    ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║\n   ██║   ███████║█████╗      ██║  ██║██████╔╝█████╗  ███████║██╔████╔██║       ██║   █████╗  ███████║██╔████╔██║\n   ██║   ██╔══██║██╔══╝      ██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║       ██║   ██╔══╝  ██╔══██║██║╚██╔╝██║\n   ██║   ██║  ██║███████╗    ██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║       ██║   ███████╗██║  ██║██║ ╚═╝ ██║\n   ╚═╝   ╚═╝  ╚═╝╚══════╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝       ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝\n                                                                                                                ")
  verify()
}
