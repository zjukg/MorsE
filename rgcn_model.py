import torch
from torch import nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases=None, has_bias=False, activation=None,
                 is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels

        # for msg_func
        self.rel_weight = None
        self.input_ = None

        self.has_bias = has_bias
        self.activation = activation

        self.is_input_layer = is_input_layer

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels*2, self.num_bases))
        self.self_loop_weight = nn.Parameter(torch.Tensor(self.in_dim, self.out_dim))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))

        self.aggregator = Aggregator()

        # bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_dim))
            nn.init.zeros_(self.bias)

    def msg_func(self, edges):
        w = self.rel_weight.index_select(0, edges.data['type'])
        msg = torch.bmm(edges.src[self.input_].unsqueeze(1), w).squeeze(1)
        curr_emb = torch.mm(edges.dst[self.input_], self.self_loop_weight)  # (B, F)
        a = 1 / edges.dst['in_d'].to(torch.float32).to(device=w.device).reshape(-1, 1)

        return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

    def apply_node_func(self, nodes):
        node_repr = nodes.data['h']

        if self.has_bias:
            node_repr = node_repr + self.bias

        if self.activation:
            node_repr = self.activation(node_repr)

        return {'h': node_repr}

    def forward(self, g):
        # generate all relations' weight from bases
        weight = self.weight.view(self.num_bases, self.in_dim * self.out_dim)
        self.rel_weight = torch.matmul(self.w_comp, weight).view(
            self.num_rels*2, self.in_dim, self.out_dim)

        # normalization constant
        g.dstdata['in_d'] = g.in_degrees()

        self.input_ = 'feat' if self.is_input_layer else 'h'

        g.update_all(self.msg_func, self.aggregator, self.apply_node_func)


        if self.is_input_layer:
            g.ndata['repr'] = torch.cat([g.ndata['feat'], g.ndata['h']], dim=1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h']], dim=1)


class RGCN(nn.Module):
    def __init__(self, args):
        super(RGCN, self).__init__()

        self.emb_dim = args.ent_dim
        self.num_rel = args.num_rel
        self.num_bases = args.num_bases
        self.num_layers = args.num_layers
        self.device = args.gpu

        # create rgcn layers
        self.layers = nn.ModuleList()
        self.build_model()

        self.jk_linear = nn.Linear(self.emb_dim*(self.num_layers+1), self.emb_dim)

    def build_model(self):
        # i2h
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # h2h
        for idx in range(self.num_layers - 1):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNLayer(self.emb_dim,
                         self.emb_dim,
                         self.num_rel,
                         self.num_bases,
                         has_bias=True,
                         activation=F.relu,
                         is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.emb_dim,
                         self.emb_dim,
                         self.num_rel,
                         self.num_bases,
                         has_bias=True,
                         activation=F.relu)

    def forward(self, g):
        for idx, layer in enumerate(self.layers):
            layer(g)

        g.ndata['h'] = self.jk_linear(g.ndata['repr'])
        return g.ndata['h']
