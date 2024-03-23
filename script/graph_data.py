import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd

cus = pd.read_csv('../data/customers_m1.csv')
com = pd.read_csv('../data/commodity_m1.csv')
order_m = pd.read_csv('../data/orders_m1.csv', nrows=10000)

# 按客户id和商品编码分组，并筛选出需要合并的订单
groups = order_m.groupby(['客户id', '商品编码'])
orders_to_merge = []
for id_list, group in groups:
    if group.shape[0] > 1:  # 同一客户、同一商品但订单号不一样的订单需要合并
        merged_order = group.iloc[0].copy()  # 复制第一个订单作为合并后的订单
        merged_order[0:3] = merged_order[0:3].astype(str)
        for col in group.columns[3:6]:  # 求和
            merged_order[col] = group[col].sum()
        for col in group.columns[6:8]:  # 求均值
            merged_order[col] = group[col].mean()
        for col in group.columns[8:]:  # 求任意
            merged_order[col] = group[col].any()
        order_m.drop(order_m.loc[(order_m['客户id']==id_list[0]) & (order_m['商品编码']==id_list[1])].index, inplace=True)  # 删除所有重复订单
        order_m = order_m._append(merged_order, ignore_index=True)  # 增加被合并的订单

order_m = order_m.reset_index()

# 构建客户节点映射字典
cus_map = {id: i for i, id in enumerate(cus['客户id'].unique())}

# 构建商品节点映射字典
com_map = {id: i for i, id in enumerate(com['商品编码'].unique())}

# 重新编码订单边索引
order_src, order_dst, origin_cus, origin_prod = [], [], [], []
for cus_id, prod_id in zip(order_m['客户id'], order_m['商品编码']):
    if cus_id in cus_map and prod_id in com_map:
        order_src.append(cus_map[cus_id])
        order_dst.append(com_map[prod_id])
        origin_cus.append(cus_id)
        origin_prod.append(prod_id)

order_src = torch.tensor(order_src)
order_dst = torch.tensor(order_dst)

# 反查清理订单表
order_m = order_m.loc[order_m['客户id'].isin(origin_cus) & order_m['商品编码'].isin(origin_prod)]

# 构建特征
cus_features = torch.tensor(cus.drop(columns=['客户id']).astype(float).astype(float).values, dtype=torch.float)
com_features = torch.tensor(com.drop(columns=['商品编码']).astype(float).astype(float).values, dtype=torch.float)
edge_features = torch.tensor(order_m.drop(columns=['订单号', '客户id', '商品编码']).astype(float).values, dtype=torch.float)

data = HeteroData()

# 添加节点
data['customer'].x = cus_features
data['customer'].num_nodes = len(cus_map)
data['product'].x = com_features 
data['product'].num_nodes = len(com_map)

# 添加边
data['customer', 'order', 'product'].edge_index = torch.stack([order_src, order_dst])
data['customer', 'order', 'product'].edge_attr = edge_features

data = T.ToUndirected()(data)  # 转无向图，原本边方向是客户到产品，现在改成双向，否则商品信息无法到客户

print('data结构：')
print(data)
print('=========')

torch.save(data, '../data/hetero_graph_data.pt')