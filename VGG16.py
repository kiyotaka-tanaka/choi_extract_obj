###################################
import numpy as np
import sys
sys.path.append("/root/caffe/python")
import caffe

class VGG16:
    def __init__(self, model_path, pretrained_path):
        caffe.set_mode_gpu()
         
        # create mean array
        self.mean = np.zeros((3, 224, 224))
 
        self.mean[0] = 103.939
        self.mean[1] = 116.779
        self.mean[2] = 123.68
         
        # create network
        self.net = caffe.Net(model_path, pretrained_path, caffe.TEST)
        self.net.blobs["data"].reshape(1, 3, 224, 224)
         
        # create preprocessor (expect input: HxWxC(RGB))
        self.transformer = caffe.io.Transformer({"data": self.net.blobs["data"].data.shape})
        self.transformer.set_transpose("data", (2,0,1))
         
        self.transformer.set_mean("data", self.mean)
        self.transformer.set_raw_scale("data", 255)
        self.transformer.set_channel_swap("data", (2,1,0))
 
    def extract_feature(self, img, blob="fc7"):
        preprocessed_img = self.transformer.preprocess("data", img)
        out = self.net.forward_all(**{self.net.inputs[0]: preprocessed_img, "blobs": [blob]})
        feat = out[blob]
        feat = feat[0] 
        return feat
 
 ###################################



