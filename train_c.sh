#! /bin/bash

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_dth -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --saving_freq 2 --name Owl_late -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --gnn_type GATv2Conv
# sleep 5

python cross_validation.py -lr 0.001 -me 40 --saving_freq 2 --name Owl_G2 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --gnn_type GATv2Conv
sleep 5
 
python cross_validation.py -lr 0.001 -me 40 --saving_freq 2 --name Owl_g3l_s -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --num_layers 3 --channels 16 32 64 --num_heads 2 2 2
sleep 5