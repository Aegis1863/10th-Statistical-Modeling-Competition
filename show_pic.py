import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams['font.sans-serif'] = 'Microsoft Yahei'

parser = argparse.ArgumentParser(description='绘制训练结果')
parser.add_argument('-n', '--file_name', default='link_pre_Random_feat.csv', type=str, help='文件名')
args = parser.parse_args()

data = pd.read_csv(f'data/result/pyg/{args.file_name}')
data.plot()
plt.title('训练结果')
plt.xlabel('Epoch')
plt.show()