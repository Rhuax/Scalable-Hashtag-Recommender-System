#!/bin/bash
sbt --error assembly &&
spark-submit \
  --class "Main" \
  target/scala-2.11/scalable-hashtag-recommender-system-assembly-0.1.jar "$@"
