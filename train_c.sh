#! /bin/bash

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_dth -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --saving_freq 2 --name Owl_late -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --gnn_type GATv2Conv
# sleep 5

# python cross_validation.py -lr 0.001 -me 40 --saving_freq 2 --name Owl_G2 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --gnn_type GATv2Conv
# sleep 5
 
# python cross_validation.py -lr 0.001 -me 40 --saving_freq 2 --name Owl_g3l_s -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --num_layers 3 --channels 16 32 64 --num_heads 2 2 2
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F_early -tl 10 --downsample 3 --merged_pred early --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F_late -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F_attn -tl 10 --downsample 3 --merged_pred attention --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F20_L -tl 20 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F25_L -tl 25 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_E -tl 10 --downsample 3 --merged_pred early --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_A -tl 10 --downsample 3 --merged_pred attention --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
# sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_L_lstm -tl 10 --downsample 3 --merged_pred late --temporal_type lstm --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_L_no_enc -tl 10 --downsample 3 --merged_pred late --temporal_type none --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
sleep 5