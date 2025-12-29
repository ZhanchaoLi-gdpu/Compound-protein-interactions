import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph
from dhg.models import HGNNP
from torch.optim import Adam
import dhg
import pandas as pd
import numpy as np
import scipy as sp


class OptimizedHypergraphVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, n_vertices):
        super().__init__()
        # 编码器：HGNNP需要单独初始化
        self.gnn_encoder = HGNNP(in_dim, hidden_dim, hidden_dim)
        self.encoder_fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        # 解码器结构重构部分
        self.decoder_struct = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_vertices)
        )

        # 解码器特征重构部分
        self.decoder_feat = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, in_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).clamp(min=1e-4)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, X, hg):
        # 正确的HGNNP调用方式
        gnn_out = self.gnn_encoder(X, hg)
        return self.encoder_fc(gnn_out)

    def forward(self, X, hg):
        # 编码过程
        enc_out = self.encode(X, hg)
        mu, logvar = torch.chunk(enc_out, 2, dim=-1)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 结构重构
        struct_recon = torch.sigmoid(self.decoder_struct(z))

        # 特征重构
        feat_recon = self.decoder_feat(z)

        return struct_recon, feat_recon, mu, logvar


def improved_loss(recon_x, x, recon_feat, feat, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    FEAT_LOSS = F.mse_loss(recon_feat, feat, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + FEAT_LOSS + KLD


if __name__ == "__main__":
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    
    
    # 超参数设置
    in_dim = 881
    hidden_dim = 256  # 增大隐藏层维度
    latent_dim = 8  # 增大潜在空间维度
    n_vertices = 8362

    # 初始化模型
    model = OptimizedHypergraphVAE(in_dim, hidden_dim, latent_dim, n_vertices).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 更高效的邻接矩阵构建
    # hyperedges = [[0, 1, 2], [1, 2, 3], [3, 4], [0, 4]]
    # 读取文件并将每行转换为元组列表
    with open(r'I:\HyperGraphCompoundProteinInteractions\DrugBank.V5.1.13\DrugBank_ProteinEdgeList.txt',
              'r') as file:  # 替换为你的文件路径
        hyperedges = [
            tuple(map(int, line.strip().split()))
            for line in file
            if line.strip()  # 跳过空行
        ]
    hg = dhg.Hypergraph(n_vertices, hyperedges)

    X = pd.read_csv(r'I:\HyperGraphCompoundProteinInteractions\DrugBank.V5.1.13\DrugBank_Drug.csv',header=None,na_values=['NA', 'N/A', 'null', 'NaN', ''])
    X = X.fillna(0)
    X = np.array(X)
    X = torch.from_numpy(X)  # 将Numpy数组转换为Pytorch张量
    X = X.type(torch.float32)
    # 数据准备并移动到GPU
    X = X.to(device)
    
    # 获取超图邻接矩阵（修正部分）
    adj = torch.zeros(n_vertices, n_vertices)
    for e in hg.e[0]:
        for i in e:
            for j in e:
                adj[i, j] = 1
    # 数据准备并移动到GPU
    adj = adj.to(device)
    
    # 训练循环
    for epoch in range(20000):
        optimizer.zero_grad()
        struct_recon, feat_recon, mu, logvar = model(X, hg)
        loss = improved_loss(struct_recon, adj, feat_recon, X, mu, logvar)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#1. 直接获取编码器原始输出‌（未分割的均值和对数方差）
# 获取编码器原始输出（维度为latent_dim*2）
enc_output = model.encode(X, hg)  # 直接调用编码器模块
print("Encoder raw output shape:", enc_output.shape)  # 应显示[10000, 64*2]


#2. 获取分割后的均值和方差‌：
# 在forward方法外部复现分割逻辑
mu, logvar = torch.chunk(enc_output, 2, dim=-1)
print("Mu shape:", mu.shape)      # 应显示[10000, 64]
print("Logvar shape:", logvar.shape)  # 应显示[10000, 64]
mudrug = mu.cpu().detach().numpy()
# 将数组保存为.npy文件
np.save(r'I:\HyperGraphCompoundProteinInteractions\DrugBank.V5.1.13\mudrug.npy', mudrug)
sp.io.savemat(r'I:\HyperGraphCompoundProteinInteractions\DrugBank.V5.1.13\mudrug.mat', {'mudrug': mudrug})

#3. 完整示例：提取并可视化潜在空间‌：
import matplotlib.pyplot as plt

# 获取潜在变量
with torch.no_grad():
    enc_output = model.encoder(X, hg)
    mu, _ = torch.chunk(enc_output, 2, dim=-1)

# 可视化前两个维度
plt.scatter(mu[:, 0].numpy(), mu[:, 1].numpy(), alpha=0.5)
plt.title("Latent Space Distribution")
plt.xlabel("z1")
plt.ylabel("z2")
plt.show()

