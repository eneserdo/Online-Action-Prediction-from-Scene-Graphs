python prepare_dataset.py -tl 10

python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.0001 -me 20 --scheduler_step_size 10 --saving_freq 4 --num_heads 3 --lstm_hidden_size 192 --name medium

sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00005 -me 20 --scheduler_step_size 10 --saving_freq 4  --weighted_loss inv --name TFC_inv

sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00005 -me 20 --scheduler_step_size 10 --saving_freq 4  --weighted_loss sqrt_inv --name TFC_sqrt_inv

sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00005 -me 20 --scheduler_step_size 10 --saving_freq 4 --name TFC_effective  --weighted_loss effective

sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00005 -me 20 --scheduler_step_size 10 --saving_freq 4 --name TFC_fih -fih

sleep 5

python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 30 --scheduler_step_size 10 --schedular_gamma 0.8 --use_cls_token --temporal_type tr --saving_freq 4 --name TFC_cls_token

sleep 5

python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.0001 -me 30 --scheduler_step_size 10 --schedular_gamma 0.8 --temporal_type tr --saving_freq 4 --name TFC_normal_tr

sleep 5

python main.py -g TransformerConv -o Adam -lr 0.0001 -wd 0.00008 -me 50 --scheduler_step_size 10 --schedular_gamma 0.8 --saving_freq 4 --name long_trained_lr4

sleep 5

python main.py -g TransformerConv -o Adam -lr 0.001 -wd 0.00008 -me 50 --scheduler_step_size 10 --schedular_gamma 0.7 --saving_freq 4 --name long_trained_lr3


