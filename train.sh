#! /bin/bash
# python main.py -g GATConv -lr 0.001 -me 15 --scheduler_step_size 15 --name GATConv_bn
# sleep 5 # to let the gpu become empty
# python main.py -g NNConv -lr 0.001 -me 15 --scheduler_step_size 15 --name NNConv_bn
# sleep 5  
# python main.py -g NNConv -lr 0.001 -me 15 -nbn --scheduler_step_size 15 --name NNConv_nobn
# sleep 5
# python main.py -g TransformerConv -lr 0.001 -me 15 --scheduler_step_size 15 --name TFConv_bn

# python main.py -g TransformerConv -lr 0.001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc 
# sleep 5
# python main.py -g TransformerConv -lr 0.001 -wd 0.001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd3
# sleep 5
# python main.py -g TransformerConv -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4
# sleep 5

# python main.py -g GATv2Conv -lr 0.001 -me 20 --scheduler_step_size 5 --saving_freq 4 --name GATv2Conv
# sleep 5
# python main.py -g TransformerConv -lr 0.00001 -me 20 --dropout 0.1 --scheduler_step_size 10 --saving_freq 4 --name TFConv_dropout1
# sleep 5

# python main.py -g TransformerConv -o AdamW -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_AdamW
# sleep 5
# python main.py -g TransformerConv -o NAdam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_NAdam
# sleep 5
# python main.py -g TransformerConv -o RMSprop -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_RMSprop
# sleep 5
# python main.py -g TransformerConv -o SGD -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_SGD
# sleep 5

# python main.py -g TransformerConv -o AdamW -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_AdamW_hc32_hs128 -hc 32 -hs 128
# sleep 5
# python main.py -g TransformerConv -o AdamW -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_AdamW_hc64_hs64 -hc 64 -hs 64
# sleep 5

# python main.py -g TransformerConv -o SGD -lr 0.1 -mn 0.9 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_SGD_mn_lr1
# sleep 5

# python main.py -g TransformerConv -o SGD -lr 0.01 -mn 0.9 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_SGD_mn_lr2
# sleep 5


# python main.py -g TransformerConv -o RMSprop -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_RMSprop -fih

# python main.py -g TransformerConv -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4 -fih

# python main.py -g TransformerConv -o RMSprop -lr 0.0001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_RMSprop_lr3_FIH -fih

# 

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name TFC_sc8_wd4_FIH -fih --gpu_id 1
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name TFC_sc8_wd4 --gpu_id 1
# sleep 5


# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name TFC_sc8_wd4_D1 --dropout 0.1 --gpu_id 1
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name TFC_sc8_wd4_ED1 --edge_dropout 0.1 --gpu_id 1
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name TFC_sc8_wd4_GP --use_global_pooling --gpu_id 1
# sleep 5

# python main.py -g TransformerConv -o RMSprop -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name TFC_lr3_sc8_wd4_rmsprop --gpu_id 1
# sleep 5

# python main.py -g TransformerConv -o RMSprop -lr 0.001 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 4 --name TFC_sc8_wd45_rmsprop --gpu_id 1
# sleep 5

# python main.py -g TransformerConv -o RMSprop -lr 0.001 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 4 --name TFC_sc8_wd45_rmsprop --gpu_id 1
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_tl_50 -tl 50 -bs 512
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_tl_30_hs_64 -tl 30 -bs 512 -hs 64
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_tl_30_oc_32 -tl 30 -bs 512 -oc 32
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_tl_30_lr_01 -tl 30 -bs 512
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_tl_30_slw_01 -tl 30 -bs 512 --label_loss_weight 0.01
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_tl_30_bi -tl 30 -bs 512 --temporal_type bi
# sleep 5

# python prepare_dataset.py -tl 30


# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_BI_ML -tl 30 --gpu_id 1 --temporal_type bi --merged_pred late
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_BI_ML_Perf -tl 30 --perfect_seg --gpu_id 1 --temporal_type bi --merged_pred late
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_BI_ML_Seg -tl 30 --label_loss_weight 0 --gpu_id 1 --temporal_type bi --merged_pred late
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_TR_ML_LN -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_TR_ML_IN -tl 30 --merged_pred late --temporal_type tr --norm_layer InstanceNorm
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_10_BI_ML -tl 10 --temporal_type bi --merged_pred late
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_TR_ML_Perf -tl 30 --perfect_seg --gpu_id 1 --temporal_type tr --merged_pred late
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_TR_ML_Seg -tl 30 --label_loss_weight 0 --gpu_id 1 --temporal_type tr --merged_pred late
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_50_TR_ML_Seg_sDTW_g01 -tl 50 --label_loss_weight 0 --temporal_type tr --merged_pred late --seg_loss sDTW --sdtw_gamma 0.1
# sleep 5


# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_10_TR_ML_LN -tl 10 --merged_pred late --temporal_type tr --norm_layer LayerNorm
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name OTO_50_TR_ML_GN -tl 50 --merged_pred late --temporal_type tr --norm_layer GraphNorm --task onetoone
# sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_10_TR_ML_LN_Seg -tl 10 --temporal_type tr --merged_pred late --norm_layer LayerNorm --task segonly
sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_TR_ML_LN_Seg -tl 30 --temporal_type tr --merged_pred late --norm_layer LayerNorm --task segonly
sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_50_TR_ML_LN_Seg -tl 50 --temporal_type tr --merged_pred late --norm_layer LayerNorm --task segonly
sleep 5
