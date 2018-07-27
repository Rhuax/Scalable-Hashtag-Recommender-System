#!/bin/bash
echo "Adding required packages to the node"
sudo yum -y install git python36 &&
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" &&
sudo python3 get-pip.py &&

sudo /usr/local/bin/pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl &&
sudo /usr/local/bin/pip3 install torchvision pretrainedmodels tqdm requests

curl --user-agent dreamteam --referer zavv30 "https://**************************/file_path.csv" > file_path.csv
curl --user-agent dreamteam --referer zavv30 "https://*************************/tag_listc.txt" > tag_listc.txt
curl --user-agent dreamteam --referer zavv30 "https://************************/getimage.py" > getimage.py



