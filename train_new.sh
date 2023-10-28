# python prepare_dataset.py -tl 10

# python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0001 -me 20 --scheduler_step_size 10 --saving_freq 4 --num_heads 3 --lstm_hidden_size 192 --name medium

# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00005 -me 20 --scheduler_step_size 10 --saving_freq 4  --weighted_loss inv --name TFC_inv

# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00005 -me 20 --scheduler_step_size 10 --saving_freq 4  --weighted_loss sqrt_inv --name TFC_sqrt_inv

# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00005 -me 20 --scheduler_step_size 10 --saving_freq 4 --name TFC_effective  --weighted_loss effective

# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00005 -me 20 --scheduler_step_size 10 --saving_freq 4 --name TFC_fih -fih

# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 30 --scheduler_step_size 10 --schedular_gamma 0.8 --use_cls_token --temporal_type tr --saving_freq 4 --name TFC_cls_token

# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 30 --scheduler_step_size 10 --schedular_gamma 0.8 --temporal_type tr --saving_freq 4 --name TFC_normal_tr

# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00008 -me 50 --scheduler_step_size 10 --schedular_gamma 0.8 --saving_freq 4 --name long_trained_lr4

# sleep 5

# python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.00008 -me 50 --scheduler_step_size 10 --schedular_gamma 0.7 --saving_freq 4 --name long_trained_lr3


python main.py -g TransformerConv -o Adam -lr 0.0001 -me 20 --scheduler_step_size 5 --scheduler_gamma 0.75 --name TC_S1 -hc 16 -nh 4 -oc 16 -hs 224 --gpu_id 1
sleep 5

python main.py -g NNConv -o Adam -lr 0.0001 -me 20 --scheduler_step_size 5 --scheduler_gamma 0.75 --name NN_S1 -hc 22 -nh 6 -oc 44 -hs 224 --gpu_id 1
sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0001 -me 20 --scheduler_step_size 5 --scheduler_gamma 0.75 --name TC_S2 -hc 18 -nh 4 -oc 22 -hs 224 --gpu_id 1
sleep 5

python main.py -g NNConv -o Adam -lr 0.0001 -me 20 --scheduler_step_size 5 --scheduler_gamma 0.75 --name NN_S2 -hc 28 -nh 8 -oc 56 -hs 224 --gpu_id 1
sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0001 -me 20 --scheduler_step_size 5 --scheduler_gamma 0.75 --name TC_S3 -hc 18 -nh 5 -oc 20 -hs 224 --gpu_id 1
sleep 5

python main.py -g NNConv -o Adam -lr 0.0001 -me 20 --scheduler_step_size 5 --scheduler_gamma 0.75 --name NN_S3 -hc 32 -nh 8 -oc 64 -hs 224 --gpu_id 1
sleep 5
