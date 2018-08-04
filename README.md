# Scalable Hashtag Recommender System

Hashtag inference from a given image using k-means or fast k-means clustering,

  Usage: hashtag-recommender-system [OPTIONS]
  Recommends appropriate hashtags for a given image.

Check OPTIONS.md for the list of the possible parameters.


# Deploy on Amazon aws via Flintrock

- Run deploy.sh inside the "deploy" folder. The script is configured to upload another script on the master and on each slave machine. It copies the jar file on each machine. Finally it runs the previously uploaded script "nodesetup" on each machine in parallel. This script downloads and installs all the dependencies (e.g. python, python libraries ecc.)
N.B in order to let flintrock access aws machine you need to export the aws keys and id as system keys via:
export AWS_ACCESS_KEY_ID="yourawsid"
export AWS_SECRET_ACCESS_KEY="yourawskey"

- Conf.yaml is used by flintrock to configure the cluster properties. One can choose the OS, number of slaves ecc.

- Once installed all the dependencies, in order to run the jar one needs to login via ssh to the master slave. Then run a script like this:

spark-submit --master spark://ip-<masterip> --driver-memory 25G --executors-memory 28G 
--executor-core 8 --class "Main" --deploy-mode client  shrs.jar \
 -f file_path.csv -t tag_listc.txt --image-path "https://image.ibb.co/kYdbKT/IMG_20180725_194058_490.jpg" 
 -c minibatch -i 20 -b 10000  -m spark://ip-<masterip>:7077
  
  driver-memory, executors-memory and executor core are parameters of master and slaves machines. Spark default uses only 1 GB on each machine.
