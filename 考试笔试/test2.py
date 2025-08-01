# 3、请使用Python和PyTorch实现一个自定义的Dataset类。
# 该Dataset类需要从给定的CSV文件（包含两列：feature 和 label；feature 列包含特征数据，label 列包含对应的标签。）中读取数据，
# 并返回每个样本的特征和标签。
import pandas as pd
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, 0].values
        self.labels = self.data.iloc[:, 1].values
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    