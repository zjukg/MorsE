import torch.nn as nn
import torch
import dgl


class EntInit(nn.Module):
    def __init__(self, args):
        super(EntInit, self).__init__()
        self.args = args

        if args.double_entity_embedding:
            self.rel_head_emb = nn.Parameter(torch.Tensor(args.num_rel, args.emb_dim * 2))
            self.rel_tail_emb = nn.Parameter(torch.Tensor(args.num_rel, args.emb_dim * 2))
        else:
            self.rel_head_emb = nn.Parameter(torch.Tensor(args.num_rel, args.emb_dim))
            self.rel_tail_emb = nn.Parameter(torch.Tensor(args.num_rel, args.emb_dim))

        nn.init.xavier_normal_(self.rel_head_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_tail_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, g_bidir):
        num_edge = g_bidir.num_edges()
        etypes = g_bidir.edata['type']
        g_bidir.edata['ent_e'] = torch.zeros(num_edge, self.args.emb_dim).to(self.args.gpu)
        rh_idx = etypes < self.args.num_rel
        rt_idx = etypes >= self.args.num_rel
        g_bidir.edata['ent_e'][rh_idx] = self.rel_head_emb[etypes[rh_idx]]
        g_bidir.edata['ent_e'][rt_idx] = self.rel_tail_emb[etypes[rt_idx] - self.args.num_rel]

        message_func = dgl.function.copy_e('ent_e', 'msg')
        reduce_func = dgl.function.mean('msg', 'feat')
        g_bidir.update_all(message_func, reduce_func)
        g_bidir.edata.pop('ent_e')