#! /bin/bash
python main.py -g GATConv -lr 0.01 -scheduler_step_size 1 -me 5 -name GATConv_feasibility
sleep 1
python main.py -g NNConv -lr 0.01 -scheduler_step_size 1 -me 5 -name GATConv_feasibility
