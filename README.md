REQUIREMENTS:
 python2.7  
 torch7(image,cutorch,nn,cudnn,loadcaffe,paths,image,hdf5,xlua)


必要なファイル：

　 VGG_ILSVRC_16_layers.caffemodel 　VGG-16ネットワークをImageNetの1000類画像で学習されたモデルです。
   
   wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel　でダウンロード。
  
  
  VGG_ILSVRC_16_layers_deploy.prototxt VGG ネットワークの　caffe　様な定義ファイルです。



コードの説明：

extract.lua は指定されたフォルダーの中にある全ての画像に対して、特徴量を指定された特徴量用なフォルダーに　image_name.h5という形で書き込む。

extract.py は　パラメータで受け取った画像の特徴量をフォルダーから読み取ってtextファイルに書くようなコードになっています。

使い方： python extract.py image_name ＊ ：指定画像の特徴量ファイルに書き出す時　extract.shを使った方が良いです。

extract.sh ２つのパラメータを受け取る。image_name と　output_file_name

使い方： bash extract.sh image_path output_file

例：　bash extract.sh ak.png out.txt

物体の特報量を抽出は、　VGG_ILSVRC_16_layers.caffemodel　と　VGG_ILSVRC_16_layers_deploy.prototxt　を使う。
