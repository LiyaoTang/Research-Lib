#!/bin/bash

if [ "$BASH_VERSION" = '' ]; then
    echo "warining: should run by bash"
    exit
fi

trap "exit" INT TERM  # convert other temination signal to EXIT
trap "kill 0" EXIT  # crtl-C stop the current & background script

cd ./re3

attention="soft"
bbox_encoding="mask"
fuse_type="spp"

lrn_rate="1e-5"
label_type="center"
label_norm="fix"
unroll_type="dynamic"
use_inference_prob="-1"

run_val="False"
worker_num="1"
buffer_size="2"
use_parallel="False"
use_tfdataset="True"
display="False"
rand_seed="None"

weight_prefix="preprocess/"
model_name="re3-${attention}_${bbox_encoding}"
restore_dir="./Model/${model_name}"

# while ps -e | grep 8132 2> /dev/null; do sleep 3600; done; # polling on process

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
              --restore_dir "${restore_dir}" \
              --weight_prefix "${weight_prefix}" \
              --model_name "${model_name}"
                #  &
                # --rand_seed ${rand} \

attention="hard"
model_name="re3-${attention}_${bbox_encoding}"
restore_dir="./Model/${model_name}"
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
              --restore_dir "${restore_dir}" \
              --weight_prefix "${weight_prefix}" \
              --model_name "${model_name}"
#                 # --log ""
#                 #  &
#                 # --rand_seed ${rand} \
wait