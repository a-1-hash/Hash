import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    计算点云的 K 近邻
    输入 x: (Batch_size, 3, Num_points)
    输出 idx: (Batch_size, Num_points, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # 获取距离最近的 k 个点的索引 (包含自身)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=16, idx=None):
    """
    提取局部几何特征 (类似 PointNet++ 的局部感知和 EdgeConv)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()

    # 提取邻居点的特征
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # 拼接：(局部特征 - 中心特征) 和 (中心特征)
    # 这让网络既知道局部的"起伏形变"，又知道点在全局的具体位置
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class SurfaceFeatureEncoder(nn.Module):
    """
    用于提取人体表面 (SMPL Vertices) 局部几何特征的独立模块
    """

    def __init__(self, k=16, out_dim=128):
        super(SurfaceFeatureEncoder, self).__init__()
        self.k = k
        self.out_dim = out_dim

        # 局部特征提取层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 全局融合层
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        """
        输入 x: 人体表面点云，形状为 (Batch_size, Num_points, 3) -> 通常 Num_points 是 6890 (SMPL)
        输出 : 表面每个点的高维特征，形状为 (Batch_size, Num_points, out_dim)
        """
        # 转换为通道优先 (B, 3, N)
        x = x.permute(0, 2, 1).contiguous()

        # 第一次局部特征提取
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]  # 局部最大池化

        # 第二次局部特征提取 (感受野扩大)
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        # 整合为最终特征
        out = self.conv3(x2)

        # 转回 (Batch_size, Num_points, out_dim) 方便后续插值
        return out.permute(0, 2, 1).contiguous()


def knn_interpolate(query_pts, surface_pts, surface_feats, k=16, chunk_size=10000, sigma=0.03):
    """
    利用 KNN 将体表特征插值到内部查询点 (引入全局距离指数衰减)

    参数:
        sigma (float): 控制特征衰减范围的超参数。假设坐标空间是米 (m)，
                       sigma=0.03 表示距离表面 3cm 后，特征强度将发生显著衰减。
                       你可以根据渲染出来的 Mesh 效果微调这个值 (例如 0.02 ~ 0.05)。
    """
    B, N, _ = query_pts.shape
    _, M, D = surface_feats.shape

    # 预先分配好特征存放空间
    interpolated_feats = torch.zeros(B, N, D, device=query_pts.device, dtype=surface_feats.dtype)
    flat_feats = surface_feats.view(B * M, D)
    batch_offsets = torch.arange(B, device=query_pts.device).view(B, 1, 1) * M

    # 分块处理内部查询点，防止显存溢出
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        chunk_query = query_pts[:, i:end_i, :]

        # 1. 计算距离矩阵
        dist = torch.cdist(chunk_query, surface_pts)

        # 2. 找到距离最近的 K 个表面点
        dist_k, idx_k = torch.topk(dist, k, dim=-1, largest=False)

        # 3. 计算相对权重 (用于局部 K 个点之间的特征融合)
        # 使用 1/d 确保越近的表面点对该查询点的特征贡献越大
        dist_k_clamp = torch.clamp(dist_k, min=1e-10)
        weights_relative = 1.0 / dist_k_clamp
        weights_normalized = weights_relative / torch.sum(weights_relative, dim=-1, keepdim=True)

        # 4. 提取邻居特征并加权融合 (此时特征还是 100% 强度)
        flat_idx = (idx_k + batch_offsets).view(-1)
        gathered_feats = flat_feats[flat_idx].view(B, end_i - i, k, D)
        chunk_interp = torch.sum(gathered_feats * weights_normalized.unsqueeze(-1), dim=2)

        # 5. 【核心新增】计算全局距离衰减掩码 (基于最近邻点的距离)
        # 使用高斯函数进行平滑衰减: exp(-d^2 / (2*sigma^2))
        min_dist = dist_k[:, :, 0:1]  # 获取距离该查询点最近的一个表面点的绝对距离
        global_decay = torch.exp(- (min_dist ** 2) / (2 * sigma ** 2))

        # 6. 应用全局衰减
        # 离表面越远，global_decay 越接近 0，特征平滑地褪去；
        # 在体表之外的空气中，特征将完全归零，交由 lambda_outcan 损失函数接管。
        interpolated_feats[:, i:end_i, :] = chunk_interp * global_decay

    return interpolated_feats