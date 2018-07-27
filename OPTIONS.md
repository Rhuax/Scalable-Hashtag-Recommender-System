#hashtag-recommender-system v0.1. Copyright © the Dream Team. Distributed under the MIT license.


 ████████╗██╗  ██╗███████╗    ██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗    ████████╗███████╗ █████╗ ███╗   ███╗
╚══██╔══╝██║  ██║██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║    ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║
   ██║   ███████║█████╗      ██║  ██║██████╔╝█████╗  ███████║██╔████╔██║       ██║   █████╗  ███████║██╔████╔██║
   ██║   ██╔══██║██╔══╝      ██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║       ██║   ██╔══╝  ██╔══██║██║╚██╔╝██║
   ██║   ██║  ██║███████╗    ██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║       ██║   ███████╗██║  ██║██║ ╚═╝ ██║
   ╚═╝   ╚═╝  ╚═╝╚══════╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝       ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝


  Usage: hashtag-recommender-system [OPTIONS]
  Recommends appropriate hashtags for a given image.

  -b, --batch-size  <Batch Size>                 Number of sample taken after
                                                 each iteration of minibatch
                                                 kmeans
      --centroid-similarity  <arg>               Minimum cosine similarity
                                                 between the sets of centroids
                                                 between 2 iterations to stop
                                                 the clusterization process.
      --closest-images  <arg>                    Number of the most similar
                                                 images to consider for
                                                 predicting hashtags.
  -c, --clustering-method  <Clustering Method>   Name of the algorythm to use
                                                 for clusterizing the dataset,
                                                 can be either kmeans or
                                                 minibatch
  -d, --debug                                    Enables debug output, disabled
                                                 by default.
  -f, --features-file  <Images file>             Path to a file containing image
                                                 features to be used by the
                                                 system for making hashtag
                                                 predictions
      --image-path  <arg>                        Path to an image to predict
                                                 hashtags for. Can be a local
                                                 path or a remote URL.
      --in-cluster-file  <arg>                   File to load the clusterized
                                                 dataset from. If provided, the
                                                 clustering process won't be
                                                 executed.
  -i, --iterations  <arg>                        Number of iterations that the
                                                 clustering algorythm will be
                                                 run for. If 0 other criteria
                                                 are used to determine when data
                                                 are clasterized enough for the
                                                 system.
  -k, --k  <clusters>                            Number of clusters to create.
                                                 If 0 or not provided a
                                                 heuristic approach is used to
                                                 estimate the optimal value
  -m, --master-u-r-l  <Master URL>               Master URL
  -o, --out-cluster-file  <arg>                  Path to a file where the
                                                 clusterized dataset will be
                                                 saved to.
  -r, --rss-variation  <RSS Variation>           RSS (Residual Sum of Squares)
                                                 minimum variation between 2
                                                 iterations to stop the process
                                                 of clustering.
  -s, --ssi                                      Calculates the SSI (Silhouett
                                                 Score Index) after clusterizing
                                                 the dataset. Warning: computing
                                                 the SSI slows down *a lot* the
                                                 program execution.
  -t, --tag-file  <Hashtag file>                 Path to a file containing a
                                                 list of the hashtags that the
                                                 system will be able to predict
  -h, --help                                     Show help message
  -v, --version                                  Show version of this program
