#!/bin/bash

cd ./corner
train_path="../../Data/corner"
tree_num_list="80 160" # "5 10 20 40 80 160 320 640 1280"
cols_list="2-3-4-5-6-7-8-9  2-3-4-5-6-8-9" #"3-6-7  4-6-7 0 1" #"2-4-6  2-6-9  3-5-7  3-5-6-7  3-5-6  3-4-5  3-4-5-7"  # "2-3-4-5-6-7-8-9  2-3-5-6-7-8  2-3-5-6-8  2-3-5-7  3-5-7  2-3-5  3-5" # "2-3-4-5-6-7-8-9  2-3-4-5-6-7-8-9-10" # "0-1  0-1-2-3-4-5-6-7-8-9"  "2 3 4 5 6 7 8 9"
rand_seed_list="0 1 2 3 4 5 6 7 8 9"

# for tree_num in ${tree_num_list}; do
# for cols in ${cols_list}; do
tree_num=40
cols="0-2-3-7-8-9"
# for cols in ${cols_list}; do
    python sk-randforest.py --class_name "other;car  " \
                            --tree_num ${tree_num} \
                            --cv_fold 5 \
                            --train  ${train_path} \
                            --select_cols ${cols} \
                            &
                            # --log_dir "" \
                            # --corner_only \
                            # --rand_seed ${rand} \
# done

wait