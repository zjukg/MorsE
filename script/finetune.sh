#!/bin/sh

dataset='fb237'
version=1
kge='TransE'
gpu=0

python main.py --data_name ${dataset}_v${version} --name ${dataset,,}_v${version}_${kge,,}_finetune \
        --metatrain_state ./state/${dataset,,}_v${version}_${kge,,}/${dataset,,}_v${version}_${kge,,}.best \
        --step fine_tune --kge ${kge} --gpu cuda:${gpu}