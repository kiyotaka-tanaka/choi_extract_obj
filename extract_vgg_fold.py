import numpy as np 
import os
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
parser.add_argument("--image_folder",help = "path to image",type = str,default = "")
parser.add_argument("--with_fileName",help = "include imagename to feature if 1", default = 0)
parser.add_argument("--output_file",help = "writing features to this file",type = str,default = "out.txt")
parser.add_argument("--extract_layer",help = "layer's name that extract features from",type = str,default = "conv5_3")
parser.add_argument("--label", help = "label for your data ", type =str , default = "0\t1")

args = parser.parse_args()

vgg = VGG16(model_path = args.deploy,pretrained_path = args.caffemodel)

images = os.listdir(args.image_folder)

file1 = open(args.output_file,"w")

def array2txt(array):
    line = args.label
    for i in array:
        line =line + "\t" +str(i) 
        
    return line

for image in images:
    
    if image.endswith(".jpg") or image.endswith(".png") or image.endswith("jpeg") :
        path_to_img = args.image_folder + "/" + image
        #img = caffe.io.load_image(args.image_folder + "/" + image)
        img = caffe.io.load_image(path_to_img)
        features = vgg.extract_feature(img,blob = args.extract_layer)
        
        #features = np.resize(features,(1,len(features)))

        line = array2txt(np.array(features))
        #print line
        if args.with_fileName:
            line = path_to_img + "\t" + line
        file1.write(line)
        file1.write("\n")

        

file1.close()

'''
if args.output_file.endswith(".npy"):
    np.save(args.output_file,features)
elif args.output_file.endswith(".txt"):
    np.savetxt(args.output_file,features,delimiter = "\t")

#print type(features)

print features.shape
'''
