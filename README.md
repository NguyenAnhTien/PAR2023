# PAR2023

This repository is the official implementation for the contest [International Contest on Pedestrian Attribute Recognition](https://mivia.unisa.it/par2023/#contest).

We achieved the top 4 leading performances.

## prerequisites
1. Create Wandb account at https://wandb.ai/site

2. Install pytorch lightning
    pip install pytorch-lightning

## Train
### GPU
`python main.py --train True --dev_mode False --gpu 1`

### CPU
`python main.py --train True --dev_mode False --gpu -1`

## Prepare data
1. Get in `/path/to/PAR2023` folder

2. `mkdir -p data/PAR2023`

=======

## prerequisites
1. Create Wandb account at https://wandb.ai/site

2. Install pytorch lightning
    pip install pytorch-lightning

## Train

### GPU
`python main.py --train True --dev_mode False --gpu <GPU ID (ex. 1)>`

### CPU
`python main.py --train True --dev_mode False --gpu -1`

## DEV MODE
Only use very small portion of data with one epoch.

### GPU 
`python main.py --train True --dev_mode True --gpu <GPU ID (ex. 1)>`

### CPU
`python main.py --train True --dev_mode True --gpu -1`

## Prepare data
1. Get in `/path/to/PAR2023` folder

2. `mkdir -p data/PAR2023`

# Unittest
`python3 -m unittest`
