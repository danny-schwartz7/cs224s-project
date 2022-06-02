#!/usr/bin/env bash
git submodule update --init --recursive
mkdir -p asr/trained_models/wavglow-styletransfer waveglow/checkpoints
cp asr/pretrained.ckpt asr/trained_models/wavglow-styletransfer/epoch=77-step=184938.ckpt
pushd waveglow/checkpoints
gdown 1_YoJ3CBXnAQmua9B47sLTSgVuxbo-pv-
mv waveglow_206000.zip waveglow_206000
popd