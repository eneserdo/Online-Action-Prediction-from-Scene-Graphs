#! /bin/bash

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_dth -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
# sleep 5


# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Owl_attn -tl 10 --downsample 3 --merged_pred attention --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Owl_early -tl 10 --downsample 3 --merged_pred early --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Owl_late -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data
# sleep 5


python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Owl_Wi -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --weighted_loss inv
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Owl_Wsi -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --weighted_loss sqrt_inv
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Owl_We -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --weighted_loss effective
sleep 5