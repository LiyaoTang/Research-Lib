#!/bin/bash

if [ "$BASH_VERSION" = '' ]; then
    echo "warining: should run by bash"
    exit
fi

trap "exit" INT TERM # convert other temination signal to EXIT
trap "kill 0" EXIT # crtl-C stop the current & background script

cd ./re3

lrn_rate="1e-5"
attention="hard"
fuse_type="spp"
label_type="center"
label_norm="fix"
unroll_type="dynamic"
bbox_encoding="mask"
use_inference_prob="-1"

max_step="1e6"
rand_seed="None"

run_val="False"
worker_num="1"
buffer_size="5"
use_parallel="True"
use_tfdataset="False"
display="False"

python re3.py --lrn_rate $lrn_rate \
              --attention $attention \
              --fuse_type $fuse_type \
              --label_type $label_type \
              --label_norm $label_norm \
              --unroll_type $unroll_type \
              --bbox_encoding $bbox_encoding \
              --use_inference_prob $use_inference_prob \
              --max_step $max_step \
              --buffer_size $buffer_size \
              --use_parallel $use_parallel \
              --use_tfdataset $use_tfdataset \
              --run_val $run_val \
              --display $display \
              --restore_dir $restore_dir \
            # --model_name "re3-mask_lstm512_hard_2019_07_09_00_01_00"
                # --log ""
                #  &
                # --rand_seed ${rand} \
wait