#!/bin/bash
set -xe 

aria2c -x 16 'http://images.cocodataset.org/zips/train2014.zip'
aria2c -x 16 'http://images.cocodataset.org/zips/val2014.zip'

# if you don't have aria2c, please use following command to download the data
# aria2 is a tool that can perform parallel download
# wget 'http://images.cocodataset.org/zips/train2014.zip'
# wget 'http://images.cocodataset.org/zips/val2014.zip'

unzip train2014.zip
unzip val2014.zip

rm train2014.zip
rm val2014.zip
