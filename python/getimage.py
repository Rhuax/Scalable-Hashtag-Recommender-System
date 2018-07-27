from __future__ import print_function

import sys
import torch
import pretrainedmodels.utils as utils
import pretrainedmodels

from io import BytesIO
from PIL import Image

import requests

load_img = utils.LoadImage()
torch.set_printoptions(precision=20)

model_name = 'alexnet'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

tf_img = utils.TransformImage(model)


def get_vector(image_name):
    if image_name.startswith("http"):
        response = requests.get(image_name)
        input_img = Image.open(BytesIO(response.content))

    else:
        input_img = load_img(image_name)
    input_tensor = tf_img(input_img)
    input_tensor = input_tensor.unsqueeze(0)
    input = torch.autograd.Variable(input_tensor, requires_grad=False)
    output_features = model.features(input).view(4096)
    return output_features.data.numpy()


if len(sys.argv) < 2:
    print("no image given")
    exit()
else:
    image_name = sys.argv[1]

data = get_vector(image_name)
for i in data:
    print(str(i)+",")


