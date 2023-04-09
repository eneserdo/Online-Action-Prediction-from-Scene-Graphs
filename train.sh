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

python main.py -g TransformerConv -o NAdam -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_NAdam
sleep 5
python main.py -g TransformerConv -o RMSprop -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_RMSprop
sleep 5
python main.py -g TransformerConv -o SGD -lr 0.001 -wd 0.0001 -me 20 --scheduler_step_size 4 --saving_freq 4 --name TFConv_sc_wd4_SGD
sleep 5
