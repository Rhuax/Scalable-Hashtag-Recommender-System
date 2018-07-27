#!/bin/bash

echo "Deploying cluster..."
flintrock --config conf.yaml launch test-cluster &&
flintrock --config conf.yaml copy-file test-cluster nodesetup.sh /home/ec2-user/nodesetup.sh &&
flintrock --config conf.yaml copy-file test-cluster "shrs.jar" "/home/ec2-user/shrs.jar" &&
flintrock --config conf.yaml run-command test-cluster "sh nodesetup.sh"
