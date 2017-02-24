rm -rf  image/*
cp  $1 image/
rm -rf features/*
th extract.lua
python extract.py $2
