# Online-Action-Prediction-from-Scene-Graphs
Online Action Prediction from Scene Graphs

(Just a generic name, maybe we can change that)

## Environment

You can use following codes to create the required environments (may not be inclusive)

```
conda create -n myenv python=3.8

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge 

conda install pyg -c pyg    # it is expected to install pytorch geometric 2.2

pip install enlighten natsort wandb torchmetrics matplotlib seaborn
pip install functorch==0.2.0

```


## Training

When you run the code for the very first time, you are required to process the data. To do that run 

`python main.py -p`

for other runs omit the flag.


## Test

...
