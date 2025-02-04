#! /bin/bash

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_ep -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --use_pos --use_embedding --gpu_id 1
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_dth_noed -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0 --weighted_mv auto --dist_threshold 0.4 --gpu_id 1
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_noed -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0 --weighted_mv auto --gpu_id 1
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_attn_dth -tl 10 --downsample 3 --merged_pred attention --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4 --gpu_id 1
# sleep 5

# python main.py --temporal_length -1 --downsample 3 --saving_freq 2 --merged_pred late --learning_rate 0.005 --scheduler_step_size 10 --weighted_mv auto --name W_noreg -me 50 --gpu_id 1
# sleep 5

# python main.py --temporal_length -1 --downsample 3 --saving_freq 2 --merged_pred late --learning_rate 0.01 --scheduler_step_size 10 --weighted_mv auto --name W_noreg_lrhigh -me 50 --gpu_id 1
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 50 --scheduler_step_size 8 --saving_freq 2 --name Alpha_dth_t30d1 -tl 30 --downsample 1 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4 --gpu_id 1
# sleep 5

# python main.py --temporal_length -1 --downsample 3 --saving_freq 2 --edge_dropout 0.15 --merged_pred late --learning_rate 0.005 --weight_decay 0.0005 --scheduler_step_size -1 --weighted_mv auto --name W_Red_L2 -me 100 --dim_feedforward 128 --gpu_id 1
# sleep 5

# python main.py --temporal_length -1 --downsample 3 --saving_freq 2 --edge_dropout 0.15 --merged_pred late --learning_rate 0.005 --weight_decay 0.0005 --scheduler_step_size -1 --weighted_mv auto --name W_Red_XL -me 100 --dim_feedforward 192 --num_of_temp_layers 4 --out_channels 96 --num_heads_tr 8 --gpu_id 1
# sleep 5

# python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_f -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto
# sleep 5

# python main.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_bs1024 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --gpu_id 1 -bs 1024
# sleep 5

# python main.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name Alpha_bs16 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --gpu_id 1 -bs 16
# sleep 5

# python cross_validation.py -lr 0.001 -me 40 --saving_freq 2 --name Owl_g_s -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --num_layers 2 --channels 24 64 --num_heads 2 2 --gpu_id 1
# sleep 5

# python cross_validation.py -lr 0.001 -me 40 --saving_freq 2 --name Owl_g_4h_nc -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --filtered_data --num_layers 2 --channels 64 128 --num_heads 4 4 --no_concat --gpu_id 1
# sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_L_NC_44 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4 --no_concat --num_heads 4 4 --gpu_id 1
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_E_NC_44 -tl 10 --downsample 3 --merged_pred early --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4 --no_concat --num_heads 4 4 --gpu_id 1
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_E_Small -tl 10 --downsample 3 --merged_pred early --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4 --channels 32 32 --gpu_id 1
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_L_NC_88 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4 --no_concat --num_heads 8 8 --gpu_id 1
sleep 5

python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F10_L_no_wmv -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv none --dist_threshold 0.4 --gpu_id 1
sleep 5