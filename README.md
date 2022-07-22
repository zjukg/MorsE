# MorsE

This repository contains the experimental code for our SIGIR 2022 paper: [Meta-Knowledge Transfer for Inductive Knowledge Graph Embedding](https://arxiv.org/abs/2110.14170). In this paper, to achieve inductive knowledge graph embedding, we propose a model **MorsE**, which does not learn embeddings for entities but learns transferable **meta-knowledge** that can be used to produce entity embeddings. Such meta-knowledge is modeled by entity-independent modules and learned by meta-learning.

![](./fig/method.png)

## Requirements

We run our code mainly based on ```PyTorch 1.7.1``` and ```DGL 0.6.1``` with CUDA. You can install cooresponding version based on your GPU resources. Furthermore, we also need ```lmdb``` to store the sampled sub-KGs for meta-training and ```argparse``` to parse command lines.

## Dataset

For inductive link prediction tasks, we use datasets proposed in the paper [Inductive Relation Prediction by Subgraph Reasoning](https://arxiv.org/abs/1911.06962). Before training our MorsE, please download the data from [GraIL](https://github.com/kkteru/grail/tree/master/data), and directly put these datasets in the ```data``` folder in our repository.


## Train MorsE

We provide example scripts for explaining the usage of our code. For example, for meta-training MorsE on ```fb237_v1``` based on ```TransE```, you can try the following command line:

```bash
bash script/metatrain.sh
```

The training process, validation results, and final test results will be printed and saved in the corresponding log file. After training, you can find training logs in the ```log``` folder and the tensorboad logs are saved in the ```tb_log``` folder.

You can also fine-tune the meta-trained model as described in our paper. After finishing meta-training, you can try:

```bash
bash script/finetune.sh
```

We make some important args as variables in these scripts, and you can try different datasets and KGE methods by rewriting them. The details of argument list can be found in the ```main.py```.

## Citation

If you use or extend our work, please cite the following paper:

```
@inproceedings{MorsE,
author = {Chen, Mingyang and Zhang, Wen and Zhu, Yushan and Zhou, Hongting and Yuan, Zonggang and Xu, Changliang and Chen, Huajun},
title = {Meta-Knowledge Transfer for Inductive Knowledge Graph Embedding},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531757},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {927–937},
numpages = {11},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```
