import argparse
from utils import init_dir, set_seed, get_num_rel
from meta_trainer import MetaTrainer
from post_trainer import PostTrainer
import os
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', default='fb237_v1')
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--db_path', default=None)

    parser.add_argument('--num_rel', default=None, type=int)

    parser.add_argument('--task_name', default='fb237_v1', type=str)

    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)

    parser.add_argument('--num_train_subgraph', default=10000)
    parser.add_argument('--num_valid_subgraph', default=200)
    parser.add_argument('--num_sample_for_estimate_size', default=50)
    parser.add_argument('--rw_param', default=[10, 10, 5], type=list)
    parser.add_argument('--num_sample_cand', default=5, type=int)
    parser.add_argument('--subgraph_drop', default=1, type=float)

    parser.add_argument('--metatrain_num_neg', default=32)
    parser.add_argument('--metatrain_num_epoch', default=10)
    parser.add_argument('--metatrain_bs', default=64, type=int)
    parser.add_argument('--metatrain_lr', default=0.01, type=float)
    parser.add_argument('--metatrain_check_per_step', default=10, type=int)

    parser.add_argument('--posttrain_num_neg', default=64, type=int)
    parser.add_argument('--posttrain_bs', default=512, type=int)
    parser.add_argument('--posttrain_lr', default=0.001, type=int)
    parser.add_argument('--posttrain_num_epoch', default=1000, type=int)
    parser.add_argument('--posttrain_check_per_epoch', default=5, type=int)
    parser.add_argument('--posttrain_early_stop_patience', default=30, type=int)
    parser.add_argument('--eval_bs', default=512, type=int)

    # params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)

    # params for KGE
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--double_entity_embedding', default=False, type=bool)
    parser.add_argument('--double_relation_embedding', default=False, type=bool)
    parser.add_argument('--adv_temp', default=1.0, type=float)

    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()
    init_dir(args)

    args.data_path = f'./data/{args.data_name}.pkl'
    args.db_path = f'./data/{args.data_name}_subgraph'

    args.num_rel = get_num_rel(args)

    for suffix in ['final_all', 'final_50']:
        with open(os.path.join(args.log_dir, f"{args.task_name}_{suffix}.csv"), "a") as rstfile:
            rst_writer = csv.writer(rstfile)
            rst_writer.writerow(["name", "mrr", "hits@1", "hits@5", "hits@10"])

    for seed in range(5):
        args.seed = seed
        set_seed(args.seed)
        args.name = args.task_name + f'_seed{args.seed}'

        # meta train
        args.run_mode = 'meta_train'
        meta_trainer = MetaTrainer(args)
        meta_trainer.train()

        # fine tune
        args.run_mode = 'post_train'
        args.metatrain_state = f'./state/{args.name}_meta_train/{args.name}_meta_train.best'
        post_trainer = PostTrainer(args)
        post_trainer.train()

        del meta_trainer
        del post_trainer


