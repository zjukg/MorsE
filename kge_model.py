import torch.nn as nn
import torch


class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.args = args
        self.nrelation = args.num_rel
        self.emb_dim = args.emb_dim
        self.epsilon = 2.0

        self.gamma = torch.Tensor([args.gamma])

        self.embedding_range = torch.Tensor([(self.gamma.item() + self.epsilon) / args.emb_dim])
        self.relation_dim = self.emb_dim * 2 if args.double_relation_embedding else args.emb_dim
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def forward(self, sample, ent_emb, mode='single'):
        self.entity_embedding = ent_emb

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            if head_part != None:
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            if head_part == None:
                head = self.entity_embedding.unsqueeze(0)
            else:
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            if tail_part != None:
                try:
                    batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                except IndexError:
                    print(tail_part)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if tail_part == None:
                tail = self.entity_embedding.unsqueeze(0)
            else:
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        elif mode == 'rel-batch':
            head_part, tail_part = sample
            if tail_part != None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 2]
            ).unsqueeze(1)

            if tail_part == None:
                relation = self.relation_embedding.unsqueeze(0)
            else:
                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.score_func(head, relation, tail, mode)

        return score

    def score_func(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
