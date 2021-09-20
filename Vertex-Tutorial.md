This `md` file is used to document a tutorial on how to use SPVCNN on vertexing dataset. 

# Toy model 

Prerequisite: 
1. a W&B account. W&B (Weights and Biases) is a tool to track and visualize ML projects. 
   1. sign up an account at https://wandb.ai
   2. Notify me that I can add you to our team where you can monitor your training results 
2. Google Colab with GPU runtime 

Following the Colab Notebook here 

https://colab.research.google.com/drive/13vASWwqvlcqnxJbHixZAFQiNa156bx96?usp=sharing

## Install dependencies 



## Prepare datasets
We will generate some simple datasets, let's call it `toy model`. 

```
/content/drive/MyDrive/ML-Vertexing/
```

The following codes are used to generate the 
``` python 
from calo_cluster.datasets.simple import SimpleDataModule
from pathlib import Path
# %%
SimpleDataModule.generate(Path('/content/simple'), n_events=10000)
# %%
```

## Triaining 

```
!python train.py dataset=simple ~semantic_criterion train=single_gpu wandb.name="simple_test_colab" model.cr=0.5 train.num_epochs=5 | tee log.vertex.test
```