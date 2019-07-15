#!/bin/bash

if [ "$BASH_VERSION" = '' ]; then
    echo "warining: should run by bash"
    exit
fi

trap "exit" INT TERM # convert other temination signal to EXIT
trap "kill 0" EXIT # crtl-C stop the current & background script

cd ./re3

attention="hard"
bbox_encoding="mask"
fuse_type="spp"

lrn_rate="1e-5"
label_type="center"
label_norm="fix"
unroll_type="dynamic"
use_inference_prob="-1"

rand_seed="None"

run_val="False"
worker_num="1"
buffer_size="2"
use_parallel="True"
use_tfdataset="True"
display="False"

weight_prefix="preprocess/"
restore_dir=""
model_name="re3-${attention}_${bbox_encoding}"

python re3.py --lrn_rate $lrn_rate \
              --attention $attention \
              --fuse_type $fuse_type \
              --label_type $label_type \
              --label_norm $label_norm \
              --unroll_type $unroll_type \
              --bbox_encoding $bbox_encoding \
              --use_inference_prob $use_inference_prob \
              --buffer_size $buffer_size \
              --use_parallel $use_parallel \
              --use_tfdataset $use_tfdataset \
              --run_val $run_val \
              --display $display \
              --restore_dir "$restore_dir" \
              --model_name "$model_name" \
              --weight_prefix "$weight_prefix"
            # --model_name "re3-mask_lstm512_hard_2019_07_09_00_01_00"
                # --log ""
                #  &
                # --rand_seed ${rand} \
wait