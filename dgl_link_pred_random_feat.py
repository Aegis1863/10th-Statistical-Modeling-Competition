import argparse
import torch
import torch.nn as nn
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, as_edge_prediction_sampler
import pandas as pd
import itertools
import tqdm
import dgl.function as fn
import torch
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
import tqdm
import dgl.nn as dglnn
import gc
import warnings

warnings.filterwarnings('ignore')
gc.collect()

parser = argparse.ArgumentParser(description='Link prediction')
parser.add_argument('-f', '--file_patorch',
                    default='data/model/dgl/graph_data.bin', type=str, help='dgl图文件路径')
parser.add_argument('-d', '--device', default=None,
                    type=str, help='设备, cpu或cuda')
parser.add_argument('-e', '--epoch', default=20, type=int, help='运行回合数')
parser.add_argument('--begin_seed', default=42, type=int, help='起始种子')
parser.add_argument('--end_seed', default=43, type=int, help='结束种子')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = device if args.device == None else args.device
hetero_graph = dgl.load_graphs(args.file_patorch)[0][0]

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
                    torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):

        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            # 这每个关系对应一个权重矩阵对应输入维度和输出维度
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
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
        # inputs = {key: value.to(device) for key, value in inputs.items()}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

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

    def inference(self, g, batch_size, device="cpu", num_workers=0, x=None):

        if x is None:
            x = self.embed_layer()

        for l, layer in enumerate(self.layers):
            y = {
                k: torch.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim)
                for k in g.ntypes
            }

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # print(input_nodes)
                block = blocks[0]  # .to(device)
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
        emb[ntype] = node_embed[ntype][nid.to('cpu')]
    return emb


class Model(nn.Module):

    def __init__(self, graph, hidden_feat_dim, out_feat_dim):
        super().__init__()
        self.rgcn = EntityClassify(graph,
                                   hidden_feat_dim,
                                   out_feat_dim)
        self.pred = HeteroDotProductPredictor()

    def forward(self, h, pos_g, neg_g, blocks, etype):
        h = self.rgcn(h, blocks)  # h 是客户的维度12的特征矩阵，输出是转换后的嵌入
        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype)


class MarginLoss(nn.Module):

    def forward(self, pos_score, neg_score):
        # 求损失的平均值 , view 改变tensor 的形状
        # 1- pos_score + neg_score ,应该是 -pos 符号越大变成越小  +neg_score 越小越好
        return (1 - pos_score + neg_score.view(pos_score.shape[0], -1)).clamp(min=0).mean()


class HeteroDotProductPredictor(nn.Module):

    def forward(self, graph, h, etype):
        # 在计算之外更新h, 保存为全局可用
        # h contains torche node representations for each edge type computed from node_clf_hetero.py
        witorch graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'),
                              etype=etype)  # * 在这里给出分数
            return graph.edges[etype].data['score']


def train_etype_one_epoch(etype, train_dataloader, test_dataloader):
    train_losses = []
    val_losses = []
    #  input nodes 为采样的subgraph中的所有的节点的集合
    for input_nodes, pos_g, neg_g, blocks in tqdm.tqdm(train_dataloader):
        model.train()
        emb = extract_embed(all_node_embed, input_nodes)
        pos_score, neg_score = model(emb, pos_g, neg_g, blocks, etype)
        loss = loss_func(pos_score, neg_score)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        val_loss = test(etype, test_dataloader)
        val_losses.append(val_loss)
        print('\n {:s} Seed {:d} | Train_avg_loss {:.3f} | Val_avg_loss {:.3f}'.format(
            etype, seed, sum(train_losses) / len(train_losses), val_loss))
    return train_losses, val_losses

def test(etype, spec_dataloader):
    model.eval()
    val_losses = []
    for input_nodes, pos_g, neg_g, blocks in tqdm.tqdm(spec_dataloader):
        emb = extract_embed(all_node_embed, input_nodes)
        pos_score, neg_score = model(emb, pos_g, neg_g, blocks, etype)
        loss = loss_func(pos_score, neg_score)
        val_losses.append(loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss

# 执行训练
train_loss_table = {}
val_loss_table = {}
for seed in range(args.begin_seed, args.end_seed):
    torch.manual_seed(seed)
    dgl.seed(seed)
    # 采样定义
    # neg_sample_count = (hetero_graph['order'].num_edges() // hetero_graph.nodes['customer'][0]['index'].shape[0]) * 1000
    neg_sample_count = 2000
    batch_size = hetero_graph['order'].num_edges() // 250
    # 采样2层全部节点
    sampler = MultiLayerFullNeighborSampler(2)
    # 边的条数,数目比顶点个数多很多.
    # 这是 EdgeDataLoader 数据加载器
    total_edges = hetero_graph['order'].num_edges()
    # 训练集占 70%
    train_size = int(0.7 * total_edges)
    val_size = total_edges - train_size
    random_indices = torch.randperm(total_edges)  # 随机索引
    hetero_graph.edges['order'].data['train_mask'] = torch.zeros(total_edges, dtype=torch.bool)
    hetero_graph.edges['order'].data['train_mask'][random_indices[:train_size]] = True  # 标记训练集
    hetero_graph.edges['order'].data['val_mask'] = ~hetero_graph.edges['order'].data['train_mask']  # 标记验证集
    # 获取训练集和验证集的边索引
    train_item_eids = hetero_graph.edges['order'].data['train_mask'].nonzero(as_tuple=True)[0]
    val_item_eids = hetero_graph.edges['order'].data['val_mask'].nonzero(as_tuple=True)[0]

    sampler = as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg_sample_count))

    train_dataloader = dgl.dataloading.DataLoader(
        hetero_graph, {'order': train_item_eids}, sampler,
        batch_size=batch_size, shuffle=True)
    val_dataloader = dgl.dataloading.DataLoader(
        hetero_graph, {'order': val_item_eids}, sampler,
        batch_size=batch_size, shuffle=False)

    hidden_feat_dim = 12  # 客户特征长度
    out_feat_dim = 12

    embed_layer = RelGraphEmbed(hetero_graph, hidden_feat_dim)
    all_node_embed = embed_layer()

    model = Model(hetero_graph, hidden_feat_dim, out_feat_dim)
    # 优化模型所有参数, 主要是weight以及输入的embeding参数
    all_params = itertools.chain(model.parameters(), embed_layer.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.01, weight_decay=0)

    loss_func = MarginLoss()

    print("start seed:", seed)
    
    train_losses, val_losses = train_etype_one_epoch('order', train_dataloader, val_dataloader)
    train_loss_table[seed] = train_losses
    val_loss_table[seed] = val_losses
    # 保存模型
    # torch.save(model.state_dict(), f'ckpt/model_params_{seed}.pt')

# 保存训练结果
train_loss_table = pd.DataFrame(train_loss_table)
val_loss_table = pd.DataFrame(val_loss_table)
train_loss_table.to_csv(f'data/result/dgl/train_loss.csv', index=False, encoding='utf-8-sig')
val_loss_table.to_csv(f'data/result/dgl/val_loss.csv', index=False, encoding='utf-8-sig')