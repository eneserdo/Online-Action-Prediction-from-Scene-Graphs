#! /bin/bash
python main.py -g GATConv -lr 0.001 -me 15 --name GATConv_bn_feasibility2
sleep 5 # to let the gpu become empty
python main.py -g NNConv -lr 0.001 -me 15 --name NNConv_bn_feasibility2
sleep 5  # to let the gpu become empty
python main.py -g NNConv -lr 0.001 -me 15 -nbn --name NNConv__nobn_feasibility2
sleep 5  # to let the gpu become empty
python main.py -g NNConv -lr 0.001 -me 15 -nbn --name NNConv__nobn_feasibility2
