#! /bin/bash
python main.py -g GATConv -lr 0.001 -me 15 --scheduler_step_size 15 --name GATConv_bn
sleep 5 # to let the gpu become empty
python main.py -g NNConv -lr 0.001 -me 15 --scheduler_step_size 15 --name NNConv_bn
sleep 5  # to let the gpu become empty
python main.py -g NNConv -lr 0.001 -me 15 -nbn --scheduler_step_size 15 --name NNConv_nobn
sleep 5  # to let the gpu become empty
python main.py -g TransformerConv -lr 0.001 -me 15 --scheduler_step_size 15 --name TFConv_nobn