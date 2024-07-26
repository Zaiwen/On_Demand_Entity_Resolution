import numpy as np
# np.seterr(divide='ignore',invalid='ignore')
import pyflann
import timeit
import math

from convert import transformation

import numpy as np
from pyflann import FLANN
import timeit

import numpy as np
from pyflann import FLANN
import timeit

def get_blocks(embedding, convert_embedding, blocks_path, num_neighbors, distance_threshold):

    ary = []
    with open(embedding, encoding='utf-8') as f:
        for line in f:
            toks = line.strip().split(' ')
            ary.append(''.join(filter(str.isdigit, toks[0])))

    # print("ary:", ary)  # 调试输出

    dataset_file = convert_embedding

    # 读取数据集
    dataset = np.load(dataset_file)
    # print("Original dataset shape:", dataset.shape)  # 调试输出

    # 确保数据类型为float32
    dataset = dataset.astype(np.float32)
    assert dataset.dtype == np.float32

    # 归一化所有向量的长度，因为我们关心的是余弦相似度
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    # print("Normalized dataset shape:", dataset.shape)  # 调试输出

    # 中心化数据集和查询向量，这能显著提高LSH的性能
    center = np.mean(dataset, axis=0)
    dataset -= center
    # print("Centered dataset shape:", dataset.shape)  # 调试输出

    flann = FLANN()
    t1 = timeit.default_timer()
    flann.build_index(dataset, algorithm='kmeans', target_precision=0.9, log_level="info")
    t2 = timeit.default_timer()
    # print(f'Construction time: {t2 - t1} seconds')

    outfile = open(blocks_path, 'w', encoding='utf-8')
    for i, query in enumerate(dataset):
        # print(f"Processing query {i}")  # 调试输出
        # # 查找相似实体并获取其距离
        neighbors, dists = flann.nn_index(query, num_neighbors)
        # print(f"Neighbors for query {i}: {neighbors}")  # 调试输出
        # print(f"Distances for query {i}: {dists}")  # 调试输出

        # 展平 neighbors 和 dists 数组
        neighbors = neighbors.flatten()
        dists = dists.flatten()
        # print(f"Flattened neighbors for query {i}: {neighbors}")  # 调试输出
        # print(f"Flattened distances for query {i}: {dists}")  # 调试输出

        # 根据距离阈值进行过滤
        filtered_neighbors = [neighbor for neighbor, dist in zip(neighbors, dists) if dist <= distance_threshold]
        # print(f"Filtered neighbors for query {i} within distance {distance_threshold}: {filtered_neighbors}")  # 调试输出

        outfile.write("target entity: ")
        outfile.write(str(ary[i]))
        outfile.write("\n")
        outfile.write("similar entities: ")
        for neighbor in filtered_neighbors:
            # print(f"Processing neighbor {neighbor} for query {i}")  # 调试输出
            if str(ary[i]) != str(ary[neighbor]):
                outfile.write(str(ary[neighbor]))
                outfile.write(" ")
        outfile.write("\n")

    outfile.close()


if __name__ == "__main__":
    embedding = "output_file/embeddings"
    # Vector_transformation (txt->npy)
    dimension = 150 # structural_embedding_dimension512
    convert_e = transformation(embedding, dimension)
    print("--vector transformation done")
    # 运行过了，故不运行

    print(convert_e)
    # Blocks_generation
    # euclidean_distance = 0.1  # distance_threshold_in_the_blocking_phase
    blocks_path = "output_file/blocks.txt"
    get_blocks('output_file/embeddings.txt', convert_e, blocks_path, 100,0.2)