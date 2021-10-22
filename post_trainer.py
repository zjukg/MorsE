import os
import torch
from torch import optim
import numpy as np
from utils import get_posttrain_train_valid_dataset
from torch.utils.data import DataLoader
from datasets import KGETrainDataset, KGEEvalDataset
from trainer import Trainer


class PostTrainer(Trainer):
    def __init__(self, args):
        super(PostTrainer, self).__init__(args)
        self.args = args
        self.load_metatrain()

        # dataloader
        train_dataset, valid_dataset = get_posttrain_train_valid_dataset(args)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.posttrain_bs,
                                      collate_fn=KGETrainDataset.collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_bs,
                                      collate_fn=KGEEvalDataset.collate_fn)


        # get ent emb
        self.ent_init(self.indtest_train_g)
        ent_feat = self.indtest_train_g.ndata['feat']

        self.ent_feat = ent_feat.detach().requires_grad_()

        # optim
        self.optimizer = optim.Adam([self.ent_feat] + list(self.rgcn.parameters())
                                    + list(self.kge_model.parameters()), lr=self.args.posttrain_lr)

    def save_checkpoint(self, e):
        state = {'ent_feat': self.ent_feat,
                 'rgcn': self.rgcn.state_dict(),
                 'kge_model': self.kge_model.state_dict()}

        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(e) + '.ckpt'))

    def load_ckpt(self):
        state = torch.load(self.args.posttrain_ckpt, map_location=self.args.gpu)
        self.ent_feat = state['ent_feat']
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.ent_feat = state['ent_feat']
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    def load_metatrain(self):
        state = torch.load(self.args.metatrain_state, map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    def get_ent_emb(self):
        self.indtest_train_g.ndata['feat'] = self.ent_feat
        ent_emb = self.rgcn(self.indtest_train_g)
        return ent_emb

    def train(self):
        best_epoch = 0
        best_eval_rst = {'mrr': 0, 'hits@1': 0, 'hits@5': 0, 'hits@10': 0}
        bad_count = 0
        self.logger.info('start post-training')

        self.evaluate_indtest_test_triples()
        self.evaluate_indtest_test_triples(num_cand=50)

        eval_res = self.evaluate_indtest_valid_triples(num_cand=50)
        self.write_evaluation_result(eval_res, 0)

        for i in range(1, self.args.posttrain_num_epoch + 1):
            losses = []
            for batch in self.train_dataloader:
                pos_triple, neg_tail_ent, neg_head_ent = [b.to(self.args.gpu) for b in batch]

                ent_emb = self.get_ent_emb()
                # kge loss
                loss = self.get_loss(pos_triple, neg_tail_ent, neg_head_ent, ent_emb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            self.logger.info('epoch: {} | loss: {:.4f}'.format(i, np.mean(losses)))

            if i % self.args.posttrain_check_per_epoch == 0:
                eval_res = self.evaluate_indtest_valid_triples(num_cand=50)
                self.write_evaluation_result(eval_res, i)

                if eval_res['hits@10'] > best_eval_rst['hits@10']:
                    best_eval_rst = eval_res
                    best_epoch = i
                    self.logger.info('best model | hits@10 {:.4f}'.format(best_eval_rst['hits@10']))
                    self.save_checkpoint(i)
                    bad_count = 0
                else:
                    bad_count += 1
                    self.logger.info('best model is at epoch {0}, hits@10 {1:.4f}, bad count {2}'.format(
                        best_epoch, best_eval_rst['hits@10'], bad_count))

            if bad_count >= self.args.posttrain_early_stop_patience:
                self.logger.info('early stop at epoch {}'.format(i))
                break

        self.logger.info('finish post-training')
        self.logger.info('save best model')
        self.save_model(best_epoch)

        self.logger.info('best validation | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            best_eval_rst['mrr'], best_eval_rst['hits@1'],
            best_eval_rst['hits@5'], best_eval_rst['hits@10']))

        self.before_test_load()
        rst_all = self.evaluate_indtest_test_triples()
        rst_50 = self.evaluate_indtest_test_triples(num_cand=50)

        self.write_rst_csv({'final_all': rst_all, 'final_50': rst_50})

    def evaluate_indtest_valid_triples(self, num_cand='all'):
        ent_emb = self.get_ent_emb()

        results = self.evaluate(ent_emb, self.valid_dataloader, num_cand)

        self.logger.info('valid on ind-test-graph')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results

    def evaluate_indtest_test_triples(self, num_cand='all'):
        """do evaluation on test triples of ind-test-graph"""
        ent_emb = self.get_ent_emb()

        results = self.evaluate(ent_emb, self.indtest_test_dataloader, num_cand=num_cand)

        self.logger.info(f'test on ind-test-graph, sample {num_cand}')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
