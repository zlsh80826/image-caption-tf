#!/bin/bash
set -xe 

aria2c -x 16 'http://images.cocodataset.org/zips/train2014.zip'
aria2c -x 16 'http://images.cocodataset.org/zips/val2014.zip'

unzip train2014.zip
unzip val2014.zip

rm train2014.zip
rm val2014.zip
