import MinkowskiEngine as ME
import numpy as np
import torch

mat = np.zeros([1024, 1200, 1200])
print(mat.shape)
mat = np.transpose(mat, (1, 2, 0))
print(mat.shape)


# 生成点云
points = np.asarray([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [2, 2, 0],
                     [1, 1, 1]])
feats = np.asarray([[1],
                    [2],
                    [3],
                    [4],
                    [5]]).astype(np.float32)
map_resolution = 1
semantic_map_size = torch.zeros([1, 3, 4, 4]).shape
device = torch.device("cuda")
points = torch.from_numpy(points).to(device)
feats = torch.from_numpy(feats).to(device)

coordinates, features = ME.utils.batch_sparse_collate([(points[:, [1, 0]] / map_resolution, feats)], device=device)
print(coordinates.shape)
print(features.shape)
keep_idx = (coordinates[:, 0] >= 0) * (coordinates[:, 0] < semantic_map_size[-2]) * \
           (coordinates[:, 1] >= 0) * (coordinates[:, 1] < semantic_map_size[-1])
coordinates, features = coordinates[keep_idx], features[keep_idx]
if len(coordinates) > 0:
    semantic_feature_map_sparse = ME.TensorField(coordinates=coordinates, features=features,
        quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL).sparse()
    semantic_feature_map = semantic_feature_map_sparse.dense(shape=semantic_map_size, min_coordinate=torch.IntTensor([0, 0]))[0]
    print(semantic_feature_map.shape)
    print(type(semantic_feature_map))
    print(semantic_feature_map)
# print(type(semantic_feature_map.numpy()))
# print(semantic_feature_map.numpy().dtype)
