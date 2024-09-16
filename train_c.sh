#! /bin/bash

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_dth -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
# sleep 5

# python main.py -lr 0.001 -wd 0.0005 -me 100 --scheduler_step_size -1 --saving_freq 2 --name C_Reduce -tl -1 --downsample 3 --merged_pred attention --temporal_type_local tr --temporal_type_global tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto -bs 128 --clip 1.5
# sleep 5

python main.py -lr 0.001 -wd 0.0005 -me 100 --scheduler_step_size -1 --saving_freq 2 --name C_bs1_s1 -tl -1 --downsample 3 --merged_pred attention --temporal_type_local tr --temporal_type_global tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto -bs 1 --clip 1.5 --coef_last 0 --coef_middle 1
sleep 5

python main.py -lr 0.001 -wd 0.0005 -me 100 --scheduler_step_size -1 --saving_freq 2 --name C_bs1_s12 -tl -1 --downsample 3 --merged_pred attention --temporal_type_local tr --temporal_type_global tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto -bs 1 --clip 1.5 --coef_last 0.5 --coef_middle 0.5 
sleep 5