import argparse
import torch
import torch.nn as nn
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, as_edge_prediction_sampler
from dgl.dataloading.negative_sampler import Uniform
import pandas as pd
import itertools
import tqdm
import dgl.function as fn
import torch
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
import tqdm
import torch as th
import dgl.nn as dglnn
import gc
gc.collect()

parser = argparse.ArgumentParser(description='Link prediction')
parser.add_argument('-f', '--file_path',
                    default='data/graph_data.bin', type=str, help='dgl图文件路径')
parser.add_argument('-d', '--device', default=None,
                    type=str, help='设备, cpu或cuda')
parser.add_argument('-e', '--epoch', default=20, type=int, help='运行回合数')
parser.add_argument('--begin_seed', default=42, type=int, help='起始种子')
parser.add_argument('--end_seed', default=52, type=int, help='结束种子')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = device if args.device == None else args.device
hetero_graph = dgl.load_graphs(args.file_path)[0][0]

print(f'device: {device}')
print('图结构: \n', hetero_graph)


class RelGraphConvLayer(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        # 这个地方只是起到计算的作用, 不保存数据
        self.conv = HeteroGraphConv({
            # graph conv 里面有模型参数weight,如果外边不传进去的话,里面新建
            # 相当于模型加了一层全链接, 对每一种类型的边计算卷积
            rel: GraphConv(in_feat, out_feat, norm='right',
                           weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                # 每个关系,又一个weight,全连接层
                self.weight = nn.Parameter(
                    th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):

        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            # 这每个关系对应一个权重矩阵对应输入维度和输出维度
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(
                k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        # 多类型的边结点卷积完成后的输出
        # 输入的是blocks 和 embeding
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        #
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(self,
                 g,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            embed = nn.Parameter(torch.Tensor(
                g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self, block=None):

        return self.embeds


class EntityClassify(nn.Module):
    def __init__(self,
                 g,
                 h_dim, out_dim,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(EntityClassify, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.embed_layer = RelGraphEmbed(g, self.h_dim)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False))

        # h2h , 这里不添加隐层,只用2层卷积
        # for i in range(self.num_hidden_layers):
        #    self.layers.append(RelGraphConvLayer(
        #        self.h_dim, self.h_dim, self.rel_names,
        #        self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
        #        dropout=self.dropout))
        # h2o

        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    # 输入 blocks,embeding
    def forward(self, h=None, blocks=None):
        if h is None:
            # full graph training
            h = self.embed_layer()
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            # 输入 blocks,embeding
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h

    def inference(self, g, batch_size, device=device, num_workers=0, x=None):

        if x is None:
            x = self.embed_layer()

        for l, layer in enumerate(self.layers):
            y = {
                k: th.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim)
                for k in g.ntypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {k: th.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # print(input_nodes)
                block = blocks[0].to(device)

                h = {k: x[k][input_nodes[k]].to(device)
                     for k in input_nodes.keys()}
                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y

# 根据节点类型和节点ID抽取embeding 参与模型训练更新


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb

# Define a Heterograph Conv model


class Model(nn.Module):

    def __init__(self, graph, hidden_feat_dim, out_feat_dim):
        super().__init__()
        self.rgcn = EntityClassify(graph,
                                   hidden_feat_dim,
                                   out_feat_dim)
        self.pred = HeteroDotProductPredictor()

    def forward(self, h, pos_g, neg_g, blocks, etype):
        h = self.rgcn(h, blocks)
        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype)


class MarginLoss(nn.Module):

    def forward(self, pos_score, neg_score):
        # 求损失的平均值 , view 改变tensor 的形状
        # 1- pos_score + neg_score ,应该是 -pos 符号越大变成越小  +neg_score 越小越好
        return (1 - pos_score + neg_score.view(pos_score.shape[0], -1)).clamp(min=0).mean()


class HeteroDotProductPredictor(nn.Module):

    def forward(self, graph, h, etype):
        # 在计算之外更新h,保存为全局可用
        # h contains the node representations for each edge type computed from node_clf_hetero.py
        with graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def train_etype_one_epoch(etype, spec_dataloader):
    losses = []
    #  input nodes 为采样的subgraph中的所有的节点的集合
    for input_nodes, pos_g, neg_g, blocks in tqdm.tqdm(spec_dataloader):
        emb = extract_embed(all_node_embed, input_nodes)
        pos_score, neg_score = model(emb, pos_g, neg_g, blocks, etype)
        loss = loss_func(pos_score, neg_score)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('{:s} Epoch {:d} | Loss {:.4f}'.format(
        etype, seed, sum(losses) / len(losses)))
    return losses


# 执行训练
for seed in range(args.begin_seed, args.end_seed):
    torch.manual_seed(seed)
    dgl.seed(seed)
    # 采样定义
    neg_sample_count = 1
    batch_size = 11892915 // 200
    # 采样2层全部节点
    sampler = MultiLayerFullNeighborSampler(2)
    # 边的条数,数目比顶点个数多很多.
    # 这是 EdgeDataLoader 数据加载器

    hetero_graph.edges['order'].data['train_mask'] = torch.zeros(11892915, dtype=torch.bool).bernoulli(1.0)
    train_item_eids = hetero_graph.edges['order'].data['train_mask'].nonzero(as_tuple=True)[0]

    sampler = as_edge_prediction_sampler(sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg_sample_count))

    item_dataloader = dgl.dataloading.DataLoader(
        hetero_graph, {'order': train_item_eids}, sampler,
        batch_size=batch_size, shuffle=True)

    hidden_feat_dim = 12  # 客户特征长度
    out_feat_dim = 12

    embed_layer = RelGraphEmbed(hetero_graph, hidden_feat_dim)
    all_node_embed = embed_layer()

    model = Model(hetero_graph, hidden_feat_dim, out_feat_dim)
    # 优化模型所有参数, 主要是weight以及输入的embeding参数
    all_params = itertools.chain(model.parameters(), embed_layer.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.01, weight_decay=0)

    loss_func = MarginLoss()

    loss_table = {}
    print("start epoch:", seed)
    model.train()
    losses = train_etype_one_epoch('order', item_dataloader)
    loss_table[seed] = losses

loss_table = pd.DataFrame(loss_table)
loss_table.to_csv(f'data/result.csv')