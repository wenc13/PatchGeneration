import os
import math
import torch
import numpy as np
import pickle
import plyfile
import warnings

color_palette = [(60,  179, 113), (239, 177,  24), (70,  130, 180), (255, 127,  80), (0,   191, 255), (138,  43, 226),
                 (225,  40,  40), (40,  225,  40), (40,   40, 225), (225, 225,   0), (64,  224, 208), (225,   0, 225),
                 (192, 192, 192), (128,   0,   0), (128, 128,   0), (0,   128,   0), (128,   0, 128), (0,   128, 128),
                 (0,     0, 128), (165,  42,  42), (100, 149, 237), (233, 150, 122), (255, 140,   0), (240, 197, 117),
                 (189, 183, 107), (154, 205,  50), (85,  107,  47), (34,  139,  34), (152, 251, 152), (0,   250, 154),
                 (75,    0, 130), (218, 112, 214), (0,   225, 225), (139,  69,  19), (188, 143, 143), (112, 128, 144)]


class Logger:
    """logger of the network loss """

    def __init__(self):
        self.history = []
        self.data = []

    def add(self, val):
        self.data.append(val)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    def mean(self):
        m = np.mean(np.array(self.data))
        return m

    def reset(self):
        if self.data:
            self.history.append(np.mean(np.array(self.data)))
            self.data = []


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_ply(points, filename, normals=None, colors=None):
    vertex = np.core.records.fromarrays(np.array(points).transpose(), names='x, y, z', formats='f4, f4, f4')
    desc = vertex.dtype.descr

    if normals is not None:
        assert len(normals) == len(points)
        vertex_normal = np.core.records.fromarrays(np.array(normals).transpose(), names='nx, ny, nz', formats='f4, f4, f4')
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == len(points)
        vertex_color = np.core.records.fromarrays(np.array(colors).transpose() * 255.0, names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(len(points), dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_patches(points, filename, normals=None, colors=None):
    vertex = np.core.records.fromarrays(np.concatenate(points, axis=0).transpose(), names='x, y, z', formats='f4, f4, f4')
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = []
        for i in range(points.shape[0]):
            assert len(normals[i]) == len(points[i])
            vertex_normal.append(normals[i])
        vertex_normal = np.concatenate(vertex_normal, axis=0)
        vertex_normal = np.core.records.fromarrays(np.array(vertex_normal).transpose(), names='nx, ny, nz', formats='f4, f4, f4')
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = []
        for i in range(points.shape[0]):
            assert len(colors[i]) == len(points[i])
            vertex_color.append(colors[i])
        vertex_color = np.concatenate(vertex_color, axis=0)
        vertex_color = np.core.records.fromarrays(np.array(vertex_color).transpose() * 255.0, names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(len(np.concatenate(points, axis=0)), dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_batch(points_batch, file_path, step=0, patch=False):
    batch_size = points_batch.shape[0]

    # if type(file_path) != list:
    #     basename = os.path.splitext(file_path)[0]
    #     ext = '.ply'

    if patch:
        colors = []
        for i in range(points_batch.shape[1]):
            color_i = np.repeat([color_palette[i]], points_batch.shape[2], axis=0)
            colors.append(color_i)
        colors = np.array(colors) / 255.0

    for batch_idx in range(batch_size):
        if patch:
            save_ply_patches(points_batch[batch_idx],
                             os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                             '%04d_patches.ply' % (step * batch_size + batch_idx)), colors=colors)
        else:
            save_ply(points_batch[batch_idx],
                     os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                  '%04d.ply' % (step * batch_size + batch_idx)))

        # if type(file_path) == list:
        #     save_ply(points_batch[batch_idx], os.path.join(file_path[batch_idx], '%04d.ply' % (step * batch_size + batch_idx)))
        # else:
        #     save_ply(points_batch[batch_idx], os.path.join(file_path, '%04d.ply' % (step * batch_size + batch_idx)))


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points: indexed points data, [B, S, C]
    """
    device = points.device
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(points.shape[0], dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sampling(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_indices_gpu(qrs, pts, k, sort=True):
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Input:
        qrs: Query point cloud.
        pts: Reference point cloud.
        k: Number of nearest neighbors to collect.
        dilation: Kernel dilation
        sort: whether to return the elements in sorted order.
    Return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
        is the set k-nearest neighbors for the representative points in pts[n].
    """

    A, B = qrs, pts

    r_A = torch.sum(A * A, dim=2, keepdim=True)
    r_B = torch.sum(B * B, dim=2, keepdim=True)
    m = torch.bmm(A, torch.transpose(B, 1, 2))
    D = r_A - 2 * m + torch.transpose(r_B, 1, 2)

    distances, indices = torch.topk(D, k + 1, dim=2, largest=False, sorted=sort)

    return indices


def sample_and_group(npoint, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape

    fps_idx = farthest_point_sampling(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    idx = knn_indices_gpu(new_xyz, xyz, nsample)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def no_grad_trunc_normal(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return no_grad_trunc_normal(tensor, mean, std, a, b)


def sphere_normalization(points):
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    z_mean = np.mean(points[:, 2])

    points = np.array(points) - np.array([x_mean, y_mean, z_mean])
    dist = np.sqrt(np.power(points[:, 0], 2) + np.power(points[:, 1], 2) + np.power(points[:, 2], 2))
    dist_max = max(dist)

    points = np.array(points) / dist_max

    return points


def cube_normalization(points):
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    z_mean = np.mean(points[:, 2])

    x_min_max = [min(points[:, 0]), max(points[:, 0])]
    y_min_max = [min(points[:, 1]), max(points[:, 1])]
    z_min_max = [min(points[:, 2]), max(points[:, 2])]

    scale = 1.0
    x_len = scale * (x_min_max[1] - x_min_max[0])
    y_len = scale * (y_min_max[1] - y_min_max[0])
    z_len = scale * (z_min_max[1] - z_min_max[0])

    bb_len = max([x_len, y_len, z_len])

    points = np.array(points) - np.array([x_mean, y_mean, z_mean])
    points = np.array(points) / bb_len

    return points
