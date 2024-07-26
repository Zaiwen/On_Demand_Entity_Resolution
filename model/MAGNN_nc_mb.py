import torch
import torch.nn as nn
import numpy as np

from model.base_MAGNN import MAGNN_ctr_ntype_specific


# support for mini-batched forward
# only support one layer for one ctr_ntype
class MAGNN_nc_mb_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_mb_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.ctr_ntype_layer = MAGNN_ctr_ntype_specific(num_metapaths,
                                                        etypes_list,
                                                        in_dim,
                                                        num_heads,
                                                        attn_vec_dim,
                                                        rnn_type,
                                                        r_vec,
                                                        attn_drop,
                                                        use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        # ctr_ntype-specific layers
        h = self.ctr_ntype_layer(inputs)

        h_fc = self.fc(h)
        return h_fc, h


class MAGNN_nc_mb(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(MAGNN_nc_mb, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_nc_mb layers
        self.layer1 = MAGNN_nc_mb_layer(num_metapaths,
                                        num_edge_type,
                                        etypes_list,
                                        hidden_dim,
                                        out_dim,
                                        num_heads,
                                        attn_vec_dim,
                                        rnn_type,
                                        attn_drop=dropout_rate)

    def forward(self, inputs):
        # 解析输入参数
        g_list, features_list, type_mask, edge_metapath_indices_list, target_idx_list = inputs

        # 初始化转换后的特征矩阵为全零矩阵，大小为 (节点类型数量, 隐藏层维度)，并移动到与 features_list[0] 相同的设备上
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)

        # 对每个节点类型进行特征转换
        for i, fc in enumerate(self.fc_list):
            # 获取当前节点类型对应的节点索引
            node_indices = np.where(type_mask == i)[0]
            # print("Node indices shape:", node_indices.shape)
            # print("Node indices:", node_indices)
            # print("Transformed features shape:", transformed_features.shape)
            # print("Features shape for type", i, ":", features_list[i].shape)

            # 更新节点类型 i 对应的节点特征
            transformed_features[node_indices] = fc(features_list[i])

        # 对转换后的特征进行 Dropout 操作
        transformed_features = self.feat_drop(transformed_features)

        # 将转换后的特征作为输入，通过隐藏层进行处理，得到预测结果 logits 和中间隐藏层表示 h
        logits, h = self.layer1((g_list, transformed_features, type_mask, edge_metapath_indices_list, target_idx_list))

        return logits, h

    # def forward(self, inputs):
    #     # 解析输入参数
    #     g_list, features_list, type_mask, edge_metapath_indices_list, target_idx_list = inputs
    #
    #     # 初始化转换后的特征矩阵为稀疏矩阵，大小为 (节点类型数量, 隐藏层维度)，并移动到与 features_list[0] 相同的设备上
    #     updated_features = torch.sparse.FloatTensor(torch.Size([features_list[0].shape[0], self.hidden_dim])).to(
    #         features_list[0].device)

        # 对每个节点类型进行特征转换
        # for i, fc in enumerate(self.fc_list):
        #     # 获取当前节点类型对应的节点索引
        #     node_indices = torch.LongTensor(np.where(type_mask == i)[0])
        #
        #     # 创建一个与 features_list[i] 相同大小的稀疏张量
        #     updated_features_i = torch.sparse.FloatTensor(torch.Size([features_list[i].shape[0], self.hidden_dim])).to(
        #         features_list[i].device)
        #
        #     # 将 features_list[i] 中的特征复制到 updated_features_i 中对应位置
        #     node_indices_sparse = torch.LongTensor(node_indices).reshape(-1, 1)
        #
        #     values = features_list[i][node_indices]
        #     updated_features_i = updated_features_i.clone().coalesce().indices().copy_(
        #         node_indices_sparse).values().copy_(values)
        #
        #     # 更新 updated_features
        #     updated_features = updated_features + updated_features_i
        #
        # # 对转换后的特征进行 Dropout 操作
        # updated_features = self.feat_drop(updated_features.to_dense())
        #
        # # 将转换后的特征作为输入，通过隐藏层进行处理，得到预测结果 logits 和中间隐藏层表示 h
        # logits, h = self.layer1((g_list, updated_features, type_mask, edge_metapath_indices_list, target_idx_list))
        #
        # return logits, h