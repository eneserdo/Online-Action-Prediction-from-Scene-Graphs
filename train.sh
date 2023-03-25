#! /bin/bash
python main.py -g GATConv -lr 0.001 -me 10 --name GATConv_feasibility
sleep 1
python main.py -g NNConv -lr 0.001 -me 10 --name NNConv_feasibility
