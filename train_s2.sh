#! /bin/bash

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_ep -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --use_pos --use_embedding --gpu_id 1
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_dth_noed -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0 --weighted_mv auto --dist_threshold 0.4 --gpu_id 1
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_noed -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0 --weighted_mv auto --gpu_id 1
sleep 5