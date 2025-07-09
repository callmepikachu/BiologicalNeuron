#!/bin/bash
# 自动下载RACE数据集并解压到data目录
mkdir -p data
cd data
if [ ! -f RACE.tar.gz ]; then
  echo "正在下载RACE数据集..."
  curl -O http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
fi
echo "解压RACE数据集..."
tar -xzvf RACE.tar.gz
cd ..
echo "RACE数据集已准备好。" 