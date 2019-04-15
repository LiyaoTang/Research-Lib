#!/bin/bash

if [ "$BASH_VERSION" = '' ]; then
    echo "warining: should run by bash"
    exit
fi

trap "exit" INT TERM # convert other temination signal to EXIT
trap "kill 0" EXIT # crtl-C stop the current & background script

cd ./corner
train_path="../../Data/corner"

# model needs full receptive field; enough depth (4 hidden conv layer)
# basic pipe: 3-32;1-32|3-32;1-32|3   3-32;5-32|3-64;5-64|3
# larger max receptive field: 3-32;13-32|3-32;13-32|3 (27)   3-32;17-32|3-32;17-32|3   3-32;19-32|3-32;19-32|3 (39)
# larger_min_receptive_field="5-32;13-32|5-32;13-32|3   5-32;19-32|5-32;19-32|3"
# deeper_pipe: 3-32;1-32|3-32;1-32|3-32;1-32|3-32;1-32|3 (11) 3-32;9-32|3-32;9-32|3-32;9-32|3-32;9-32|3 (30) 3-32;13-32|3-32;13-32|3-32;13-32|3-32;13-32|3 (40+)
# expanding_pipe="3-32;13-32|3-64;13-64|3-128;13-128|3   5-32;13-32|5-64;13-64|5-128;13-128|3   1-32;13-32|1-64;13-64|1-128;13-128|3" # (40)
conv_list="3-32;9-32|3-32;9-32|3-32;9-32|3-32;9-32|3" # fixed for the moment

cols_list="2-3-4-5-6-7-8-9  2-3-4-5-6-8-9" # rcs no effect
norm_type_list=("m" "s") # ("" "m" "s")
reg_type_list=("" "L1" "L2")
reg_scale_list="1e-4 1e-5"
bn_list="0 1" # no effect with batch size 1
learn_rate_list="1e-3 1e-4 3e-5 1e-5 1e-6" # 3e-6 or 1e-6

weight_list=("" "bal") # use 'bal', otherwise will pred all neg
rand_seed_list="0 1 2 3 4 5 6 7 8 9"

conv="3-32;13-32|3-32;13-32|3"
cols="2-3-4-5-6-7-8-9"
learn_rate="1e-6"
weight="bal"
bn="0"

reg_type=""
reg_scale="0.001"
norm_type="m"


loss_type="xen"
reg_type="L2"
for reg_scale in ${reg_scale_list}; do
python tf-fcnpipe.py --conv ${conv} \
                    --select_cols ${cols} \
                    --norm_type "$norm_type" \
                    --reg_type "$reg_type" \
                    --reg_scale "$reg_scale" \
                    --batchnorm ${bn} \
                    --learning_rate ${learn_rate} \
                    --class_name "non-center;car-center" \
                    --weighted_loss "$weight" \
                    --train  ${train_path} \
                    --batch 1 \
                    --epoch 30 \
                    --loss_type ${loss_type}
                    # --log ""
                    #  &
                    # --rand_seed ${rand} \
done

# reg_type=""
# loss_type="crf"
# learn_rate="3e-5"
# python tf-fcnpipe.py --conv ${conv} \
#                     --select_cols ${cols} \
#                     --norm_type "$norm_type" \
#                     --reg_type "$reg_type" \
#                     --reg_scale "$reg_scale" \
#                     --batchnorm ${bn} \
#                     --learning_rate ${learn_rate} \
#                     --class_name "non-center;car-center" \
#                     --weighted_loss "$weight" \
#                     --train  ${train_path} \
#                     --batch 1 \
#                     --epoch 30 \
#                     --loss_type ${loss_type}
#                     # --log ""
#                     #  &
#                     # --rand_seed ${rand} \

wait