#! /bin/bash

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_p -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --use_pos
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_edge_p -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --use_pos_edge
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_early -tl 10 --downsample 3 --merged_pred early --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --gpu_id 1
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_attn_dth -tl 10 --downsample 3 --merged_pred attention --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4 --gpu_id 1
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_acc_test -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto -bs 64 --num_acc 4 --gpu_id 1
# sleep 5

# python main.py --temporal_length -1 --downsample 3 --saving_freq 2 --edge_dropout 0.15 --merged_pred late --learning_rate 0.005 --weight_decay 0.0005 --scheduler_step_size 10 --weighted_mv auto --name Whole -me 50
# sleep 5

# python main.py --temporal_length -1 --downsample 3 --saving_freq 2 --edge_dropout 0.15 --merged_pred late --learning_rate 0.005 --weight_decay 0.0005 --scheduler_step_size -1 --weighted_mv auto --name W_Reduce -me 100
# sleep 5

# python main.py --temporal_length -1 --downsample 3 --saving_freq 2 --edge_dropout 0.15 --merged_pred late --learning_rate 0.005 --weight_decay 0.0005 --scheduler_step_size -1 --weighted_mv auto --name W_Red_Causal -me 100 --causal
# sleep 5

# python main.py --temporal_length -1 --downsample 3 --saving_freq 2 --edge_dropout 0.15 --merged_pred late --learning_rate 0.005 --weight_decay 0.0005 --scheduler_step_size -1 --weighted_mv auto --name W_Red_L1 -me 100 --num_of_temp_layers 4
# sleep 5
