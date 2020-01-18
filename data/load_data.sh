#!/bin/bash

## Div2k
if [ -d "./div2k" ]; then rm -Rf div2k; fi
if [ -d "./tmp" ]; then rm -Rf tmp; fi
mkdir tmp
mkdir div2k
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip tmp
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip tmp
unzip tmp/DIV2K_train_HR.zip -d div2k
unzip tmp/DIV2K_valid_HR.zip -d div2k
rm tmp/DIV2K_train_HR.zip
rm tmp/DIV2K_valid_HR.zip

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip tmp
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip tmp
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip tmp
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip tmp

unzip tmp/DIV2K_train_LR_bicubic_X2.zip -d div2k
unzip tmp/DIV2K_valid_LR_bicubic_X2.zip -d div2k
unzip tmp/DIV2K_train_LR_bicubic_X4.zip -d div2k
unzip tmp/DIV2K_valid_LR_bicubic_X4.zip -d div2k

rm -rf tmp
