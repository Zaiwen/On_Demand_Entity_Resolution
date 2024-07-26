import time
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

import torch
print(torch.__version__)

import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('.\\utils1')
from utils1.pytorchtools import EarlyStopping

from utils1.data import load_DBLP_data
from utils1.tools import index_generator, evaluate_results_nc, parse_minibatch
from model import MAGNN_nc_mb


# 定义模型输出维度
# out_dim 表示模型的输出维度，通常用于设置最终输出层的神经元数量
out_dim = 4

# 定义丢弃率
# dropout_rate 表示在训练过程中随机丢弃一部分神经元的概率，防止过拟合
dropout_rate = 0.5

# 定义学习率
# lr (learning rate) 表示优化器在每次更新参数时步长的大小，控制学习速度
lr = 0.005

# 定义权重衰减
# weight_decay 表示在优化过程中对模型权重施加的惩罚项，用于防止过拟合
weight_decay = 0.001

# 定义边类型列表
# etypes_list 表示不同路径的边类型序列，每个子列表表示一种路径
# 例如，第一个子列表 [0, 1] 表示从类型 0 到类型 1 的路径
# 第二个子列表 [0, 2, 3, 1] 表示从类型 0 到类型 2 再到类型 3 最后到类型 1 的路径
# 第三个子列表 [0, 4, 5, 1] 表示从类型 0 到类型 4 再到类型 5 最后到类型 1 的路径
etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]


def run_model_DBLP(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    # 加载DBLP数据集，包括邻接表、元路径索引列表、特征列表、邻接矩阵、类型掩码、标签和训练/验证/测试索引
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data()

    # print(type_mask)

    # for i in range(0, len(features_list)):
        # print(len(features_list[i]))
    # 判断是否使用GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 将特征列表转换为张量，并转移到指定设备上
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]

    # 根据特征类型选择输入维度
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    # 将标签转换为张量，并转移到指定设备上
    labels = torch.LongTensor(labels).to(device)

    # 获取训练、验证和测试索引，并排序
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    all_idx = np.arange(len(labels))  # 全数据集的索引

    # 定义结果存储列表
    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []

    # 重复多次实验
    for _ in range(repeat):
        # 定义MAGNN模型并转移到指定设备上
        net = MAGNN_nc_mb(3, 6, etypes_list, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type,
                          dropout_rate)
        net.to(device)

        # 定义优化器
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # 训练循环
        net.train()

        save_postfix = 'test'

        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))

        dur1 = []
        dur2 = []
        dur3 = []
        train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)

        for epoch in range(num_epochs):
            t_start = time.time()

            # 训练模式
            net.train()
            for iteration in range(train_idx_generator.num_iterations()):
                # 前向传播
                t0 = time.time()
                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)
                t1 = time.time()
                dur1.append(t1 - t0)

                logits, embeddings = net(
                    (train_g_list, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx_batch])
                t2 = time.time()
                dur2.append(t2 - t1)

                # 反向传播
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                t3 = time.time()
                dur3.append(t3 - t2)

                # 打印训练信息
                if iteration % 50 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

            # 验证模式
            net.eval()
            val_logp = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # 前向传播
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                    logits, embeddings = net(
                        (val_g_list, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                    logp = F.log_softmax(logits, 1)
                    val_logp.append(logp)
                val_loss = F.nll_loss(torch.cat(val_logp, 0), labels[val_idx])
            t_end = time.time()
            # # 打印验证信息
            # print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            #     epoch, val_loss.item(), t_end - t_start))
            # 早停
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # 使用 evaluate_results_nc 进行测试
        all_idx_generator = index_generator(batch_size=batch_size, indices=all_idx, shuffle=False)

        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        # 在进入测试阶段之前初始化all_embeddings
        all_embeddings = []

        # 使用 evaluate_results_nc 进行测试
        all_idx_generator = index_generator(batch_size=batch_size, indices=all_idx, shuffle=False)

        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()

        # 打开文件用于写入嵌入向量和节点ID
        with open('output_file/embeddings.txt', 'w') as f:
            with torch.no_grad():
                for iteration in range(all_idx_generator.num_iterations()):
                    all_idx_batch = all_idx_generator.next()
                    all_g_list, all_indices_list, all_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_list, all_idx_batch, device, neighbor_samples)

                    logits, embeddings = net(
                        (all_g_list, features_list, type_mask, all_indices_list, all_idx_batch_mapped_list))
                    all_embeddings.append(embeddings)

                    # Write each embedding and node ID to file
                    for idx, embedding in zip(all_idx_batch, embeddings):
                        embedding_str = ' '.join(map(str, embedding.cpu().numpy()))
                        node_type = type_mask[idx]
                        if node_type == 0:
                            node_type_str = "author"
                        elif node_type == 1:
                            node_type_str = "paper"
                        elif node_type == 2:
                            node_type_str = "term"
                        elif node_type == 3:
                            node_type_str = "conf"
                        else:
                            node_type_str = "unknown"

                        f.write(f"{idx} {embedding_str}\n")

                        # 打印节点ID、节点类型和嵌入向量
                        # print(f"Node ID: {idx}, Node Type: {node_type_str}, Embedding: {embedding_str}")

            test_embeddings = torch.cat(all_embeddings, 0)  # 确保使用all_embeddings
            print("labels", labels)
            print("labels[test_idx].cpu().numpy()", labels[test_idx].cpu().numpy())
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                test_embeddings.cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)

        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        nmi_mean_list.append(nmi_mean)
        nmi_std_list.append(nmi_std)
        ari_mean_list.append(ari_mean)
        ari_std_list.append(ari_std)

    # print out a summary of the evaluations
    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)
    print('----------------------------------------------------------------')
    print('SVM tests summary')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means tests summary')
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_std_list.mean()))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_std_list.mean()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=5, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=100, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='DBLP', help='Postfix for the saved model and result. Default is DBLP.')

    args = ap.parse_args()
    run_model_DBLP(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                   args.epoch, args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)
