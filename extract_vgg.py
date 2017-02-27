import numpy as np 

import sys
#### Yourt path to caffe ########################
caffe_python_path = "/root/caffe/python" 
sys.path.append(caffe_python_path)
import caffe
import argparse

from  VGG16 import *

parser = argparse.ArgumentParser()

parser.add_argument("--caffemodel",help = "path to trained caffe model",type = str,default = "VGG_ILSVRC_16_layers.caffemodel")
parser.add_argument("--deploy",help =" path to  deploy.protxt",type = str,default = "VGG_ILSVRC_16_layers_deploy.prototxt")
parser.add_argument("--image_path",help = "path to image",type = str,default = "ak.png")
parser.add_argument("--mean_file",help = "path to mean image",type = str,default = "")
parser.add_argument("--output_file",help = "writing features to this file",type = str,default = "out.txt")
parser.add_argument("--extract_layer",help = "layer's name that extract features from",type = str,default = "fc6")

args = parser.parse_args()

vgg = VGG16(model_path = args.deploy,pretrained_path = args.caffemodel)

img = caffe.io.load_image(args.image_path)
features = vgg.extract_feature(img,blob = args.extract_layer)

print features


