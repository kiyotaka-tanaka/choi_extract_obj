caffe を　gpu01に入れました。

/root/caffe/ に置いてあります。

python　で caffeを使う時には、　
import sys
sys.path.append("/root/caffe/python")
import caffe

として使います。

VGG16.py は VGG_ILSVR_16_layers　の任意の層から特長量抽出するコードである。

使い方の例は

extract_vgg.py  である。

usage: extract_vgg.py [-h] [--caffemodel CAFFEMODEL]
                           [--deploy DEPLOY]
			   [--image_path IMAGE_PATH] 
                           [--output_file OUTPUT_FILE]
                           [--extract_layer EXTRACT_LAYER]

optional arguments:
  -h, --help				       show this help message and exit
  --caffemodel CAFFEMODEL                      path to trained caffe model
  --deploy DEPLOY                              path to deploy.protxt
  --image_path IMAGE_PATH                      path to image
  --output_file OUTPUT_FILE                    writing features to this file
  --extract_layer EXTRACT_LAYER                layer's name that extract features from

extrac_vgg_fold.py フォルダー指定するとフォルダー中の画像全てから特長量抽出し、ファイルへ書き出す。

使い方の例：
python extract_vgg_fold.py  --image_folder  /home/choi/extract_object/senzai_train --output_file senzai_train.txt --label $'1\t0'


usage: extract_vgg_fold.py [-h]  [--caffemodel CAFFEMODEL]
       			   	 [--deploy DEPLOY]
                           	 [--image_folder IMAGE_FOLDER]
                           	 [--with_fileName WITH_FILENAME]
                           	 [--output_file OUTPUT_FILE]
                           	 [--extract_layer EXTRACT_LAYER] [--label LABEL]

optional arguments:         -h, --help            show this help message and exit

     --caffemodel CAFFEMODEL			   path to trained caffe model
     --deploy DEPLOY       			   path to deploy.protxt
     --image_folder IMAGE_FOLDER		   path to image
     --with_fileName WITH_FILENAME		   include imagename to feature if 1
     --output_file OUTPUT_FILE			   writing features to this file
    --extract_layer EXTRACT_LAYER		   layer's name that extract features from
    --label LABEL         			   label for your data