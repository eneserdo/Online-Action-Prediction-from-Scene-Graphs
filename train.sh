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

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_10_TR_ML_LN_Seg -tl 10 --temporal_type tr --merged_pred late --norm_layer LayerNorm --task segonly
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_TR_ML_LN_Seg -tl 30 --temporal_type tr --merged_pred late --norm_layer LayerNorm --task segonly
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_50_TR_ML_LN_Seg -tl 50 --temporal_type tr --merged_pred late --norm_layer LayerNorm --task segonly
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_10_TR_ML_LN_Seg_R -tl 10 --temporal_type tr --merged_pred late --norm_layer LayerNorm --task segonly --save_segments --save_individual
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name SEG_30_TR_ML_LN_Seg_R -tl 30 --temporal_type tr --merged_pred late --norm_layer LayerNorm --task segonly --save_segments --save_individual
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_W -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --use_weighted_mv
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_RR -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_W2 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --use_weighted_mv
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_ED05 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.05
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_EDD05 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.05 --dropout 0.05
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_ED15_W2 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.15 --use_weighted_mv
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_ED20 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.2
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_ED25 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.25
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_ED30 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.30
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.00025 -me 40 --scheduler_step_size 8 --saving_freq 4 --name OTO_30_TR_ML_LN_ED15_wd25 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.15
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0005 -me 60 --scheduler_step_size 10 --saving_freq 4 --name Beta_L -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --hidden_channels 96 --out_channels 96 --lstm_hidden_size 192 --num_heads 4 --gpu_id 1
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name Beta_BS64 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --batch_size 64 --gpu_id 1
# sleep 5

# Beta = OTO_30_TR_ML_LN


# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name Beta_NoML_ED15 -tl 30 --merged_pred none --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.15
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name Beta_ED15 -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.15
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 4 --name Beta_ED15_MV -tl 30 --merged_pred late --temporal_type tr --norm_layer LayerNorm --task onetoone --disable_seg --edge_dropout 0.15 --use_weighted_mv
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name Beta1_15d2_L -tl 15 --downsample 2 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --hidden_channels 32 --out_channels 32 --num_heads1 4 --num_heads2 4 --dim_feedforward 192
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name Beta1_15d2_XL -tl 15 --downsample 2 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --hidden_channels 32 --out_channels 48 --num_heads1 4 --num_heads2 4 --dim_feedforward 192
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name Beta1_15d2_XXL -tl 15 --downsample 2 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --hidden_channels 48 --out_channels 48 --num_heads1 4 --num_heads2 4 --dim_feedforward 192
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0005 -me 30 --scheduler_step_size 8 --saving_freq 4 --name Beta1_15d2_3XL -tl 15 --downsample 2 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --hidden_channels 48 --out_channels 48 --num_heads1 6 --num_heads2 6 --dim_feedforward 256
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_VF_S_R -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --use_vf --hidden_channels 96 --out_channels 96 --num_heads1 4 --num_heads2 4 --dim_feedforward 256 --use_global_pooling -bs 128
# sleep 5

# python cross_validation.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0005 -me 15 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_VF_S_Reg1 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.2 --use_vf --hidden_channels 96 --out_channels 96 --num_heads1 4 --num_heads2 4 --dim_feedforward 256 --use_global_pooling -bs 128 --dropout 0.1
# sleep 5

# python cross_validation.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0005 -me 15 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_VF_XS -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --use_vf --hidden_channels 64 --out_channels 64 --num_heads1 4 --num_heads2 4 --dim_feedforward 128 --use_global_pooling -bs 128
# sleep 5

# python cross_validation.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0005 -me 15 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_VF_S_Reg2 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.25 --use_vf --hidden_channels 96 --out_channels 96 --num_heads1 4 --num_heads2 4 --dim_feedforward 256 --use_global_pooling -bs 128 --dropout 0.2
# sleep 5

# python cross_validation.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.001 -me 15 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_VF_S_Reg3 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.2 --use_vf --hidden_channels 96 --out_channels 96 --num_heads1 4 --num_heads2 4 --dim_feedforward 256 --use_global_pooling -bs 128
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0002 -wd 0 -me 20 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_Reg2 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.3 --dropout 0.4 --num_heads1 4 --num_heads2 4 --dropout_tr 0.4 --dim_feedforward 256
# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0002 -wd 0 -me 20 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_Reg1 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.3 --dropout 0.3 --num_heads1 4 --num_heads2 4 --dropout_tr 0.3 --dim_feedforward 256
# sleep 5


# python cross_validation.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0002 -me 15 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_VF_S_Reg4 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.4 --use_vf --hidden_channels 96 --out_channels 96 --num_heads1 4 --num_heads2 4 --dim_feedforward 256 --use_global_pooling -bs 128 --dropout 0.4
# sleep 5

# python cross_validation.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0002 -me 15 --scheduler_step_size 8 --saving_freq 2 --name Beta1_10d3_VF_S_Reg5 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.3 --use_vf --hidden_channels 96 --out_channels 96 --num_heads1 4 --num_heads2 4 --dim_feedforward 256 --use_global_pooling -bs 128 --dropout 0.5 --dropout_tr 0.5
# sleep 5

# python cross_validation.py --gpu_id 1 -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 6 --saving_freq 2 --name Beta1_10d3_VF_XL1 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --use_vf --hidden_channels 96 --out_channels 96 --num_heads1 4 --num_heads2 4 --dim_feedforward 256 --use_global_pooling -bs 32 --dropout 0.05 --num_heads_tr 8
# sleep 5

# python cross_validation.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 30 --scheduler_step_size 6 --saving_freq 2 --name Beta1_10d3_VF_XL2 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.2 --use_vf --hidden_channels 128 --out_channels 128 --num_heads1 4 --num_heads2 4 --dim_feedforward 512 --use_global_pooling -bs 32 --dropout 0.1 --num_heads_tr 8
# python cross_validation.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.001 -me 30 --scheduler_step_size 6 --saving_freq 2 --name Beta1_10d3_VF_XL3 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --use_vf --hidden_channels 128 --out_channels 128 --num_heads1 4 --num_heads2 4 --dim_feedforward 512 --use_global_pooling -bs 32 --dropout 0.05 --num_heads_tr 8
# python cross_validation.py -g TransformerConv -o Adam -lr 0.001 -wd 0.001 -me 30 --scheduler_step_size 4 --saving_freq 2 --name Beta1_10d3_VF_XL4 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --use_vf --hidden_channels 96 --out_channels 96 --num_heads1 8 --num_heads2 8 --dim_feedforward 256 --use_global_pooling -bs 32 --dropout 0.05 --num_heads_tr 8 --gpu_id 1


python cross_validation.py --batch_size 256 -tl 10 --downsample 3 -g TransformerConv --saving_freq 2 --edge_dropout 0.15 --dropout 0.0 --norm_layer LayerNorm --dropout_tr 0.1 --lstm_hidden_size 128 --temporal_type tr --merged_pred late -o Adam --learning_rate 0.0005 --momentum 0.9 --weight_decay 0.0005 --max_epochs 40 --name Gamma_fm_Red
sleep

python cross_validation.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --saving_freq 2 --name Gamma_Red_h8 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --gpu_id 0 -nh 8
sleep 5

python cross_validation.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 2 --name Gamma_R_h8_nl3 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --gpu_id 0 -nh 8 -nl 3
sleep 5

python cross_validation.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 2 --name Gamma_G_h8_l2 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --gpu_id 0 -nh 8 -nl 2 --monograph
sleep 5

python cross_validation.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 2 --name Gamma_G_h16_l2 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --gpu_id 0 -nh 16 -nl 2 --monograph
sleep 5

python cross_validation.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 2 --name Gamma_G_h16_l3 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --gpu_id 0 -nh 16 -nl 3 --monograph
sleep 5

python cross_validation.py -g TransformerConv -o Adam -lr 0.0005 -wd 0.0005 -me 20 --scheduler_step_size 8 --saving_freq 2 --name Gamma_G_h16_l4 -tl 10 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --gpu_id 0 -nh 16 -nl 4 --monograph
sleep 5