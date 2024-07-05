import torch
import torch.nn as nn
import torch.nn.functional as F

from aux.utils import trunc_normal


class Discriminator(nn.Module):
    """ Discriminator """

    def __init__(self, options):
        super(Discriminator, self).__init__()

        self.nlatent = options.nlatent
        self.cont_dim = options.cont_dim

        # Feature extraction
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, self.nlatent, 1)

        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.bn3 = nn.Identity()

        # Discriminative
        self.fc1 = nn.Linear(self.nlatent, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.fc_bn1 = nn.Identity()
        self.fc_bn2 = nn.Identity()

        # Contrastive
        self.cont_head1 = nn.Linear(128, self.cont_dim)
        self.cont_head2 = nn.Linear(256, self.cont_dim)
        self.cont_head3 = nn.Linear(self.nlatent, self.cont_dim)
        self.cont_heads = [self.cont_head1, self.cont_head2, self.cont_head3]

    def forward(self, x, mode='dis'):
        x1 = F.leaky_relu(self.bn1(self.conv1(x.transpose(2, 1).contiguous())))
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        x4 = torch.max(x3, 2, keepdim=True)[0]
        x4 = x4.view(-1, self.nlatent)

        if mode == 'dis':
            # Discriminative
            y1 = F.leaky_relu(self.fc_bn1(self.fc1(x4)))
            y2 = F.leaky_relu(self.fc_bn2(self.fc2(y1)))
            y3 = self.fc3(y2)

            return F.sigmoid(y3), y3
        elif mode == 'cont':
            # Contrastive
            c1 = torch.max(x1, 2, keepdim=True)[0]
            c1 = c1.view(x.size(0), -1)
            c1 = self.cont_head1(c1)
            c2 = torch.max(x2, 2, keepdim=True)[0]
            c2 = c2.view(x.size(0), -1)
            c2 = self.cont_head2(c2)
            c3 = torch.max(x3, 2, keepdim=True)[0]
            c3 = c3.view(x.size(0), -1)
            c3 = self.cont_head3(c3)

            return [c1, c2, c3]
        else:
            raise ValueError("Invalid mode!")


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, options):
        super(Encoder, self).__init__()

        self.nlatent = options.nlatent
        self.z_dim = options.z_dim

        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, self.nlatent, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(self.nlatent)

        self.mu = nn.Sequential(nn.Linear(self.nlatent, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                nn.Linear(128, self.z_dim))
        self.var = nn.Sequential(nn.Linear(self.nlatent, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Linear(128, self.z_dim))

    def reparameterize_gaussian(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)

        return mean + std * eps

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x.transpose(2, 1).contiguous())))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.nlatent)

        m = self.mu(x)
        v = self.var(x)

        # reparameterize
        z = self.reparameterize_gaussian(m, v)

        return m, v, z


# ========================================================= Generator MLP ========================================================
# class MLP_Ori(nn.Module):
#     def __init__(self, options):
#         super(MLP_Ori, self).__init__()
#
#         self.npoint = options.npoint
#         self.z_dim = options.z_dim
#
#         self.linear1 = torch.nn.Linear(self.z_dim, 256)
#         self.linear2 = torch.nn.Linear(256, 512)
#         self.linear3 = torch.nn.Linear(512, 1024)
#         self.linear4 = torch.nn.Linear(1024, self.npoint * 3)
#
#         self.th = nn.Tanh()
#         self.bn1 = torch.nn.BatchNorm1d(256)
#         self.bn2 = torch.nn.BatchNorm1d(512)
#         self.bn3 = torch.nn.BatchNorm1d(1024)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.linear1(x)))
#         x = F.relu(self.bn2(self.linear2(x)))
#         x = F.relu(self.bn3(self.linear3(x)))
#         x = self.th(self.linear4(x)).reshape(-1, self.npoint, 3)
#         return x, x


class MLP_G(nn.Module):
    def __init__(self, nlatent=1024):
        super(MLP_G, self).__init__()

        self.nlatent = nlatent
        self.conv1 = torch.nn.Conv1d(self.nlatent, self.nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.nlatent, self.nlatent // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.nlatent // 2, self.nlatent // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.nlatent // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent)
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent // 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class MLP_Generator(nn.Module):
    """ MLP-G """

    def __init__(self, options):
        super(MLP_Generator, self).__init__()

        self.npatch = options.npatch
        self.npatch_point = options.npatch_point
        self.patchDim = options.patchDim
        self.z_dim = options.z_dim

        self.decoder = nn.ModuleList([MLP_G(nlatent=self.patchDim + self.z_dim) for i in range(0, self.npatch)])
        self.grid = torch.nn.Parameter(torch.FloatTensor(self.npatch, self.patchDim, self.npatch_point))

    def forward(self, x):
        outs = []
        patches = []
        for i in range(0, self.npatch):
            # random planar patch
            # ==========================================================================
            rand_grid = self.grid[i].unsqueeze(0).expand(x.size(0), -1, -1)
            patches.append(rand_grid[0].transpose(1, 0))
            # ==========================================================================

            # cat with latent vector and decode
            # ==========================================================================
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
            # ==========================================================================

        return torch.cat(outs, 2).transpose(2, 1).contiguous(), torch.stack(outs, 1).transpose(3, 2).contiguous()


# ====================================================== Helper Classes ==========================================================
class MLP_P(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, qk_scale=None, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


# ==================================================== Generator PointTrans ======================================================
class BlockPoint(nn.Module):
    def __init__(self, dim_in, dim_out, heads, mlp_ratio=4.,
                 qk_scale=None, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, se=0):
        super().__init__()

        self.res = dim_in == dim_out

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, heads=heads, head_dim=dim_in // heads, qk_scale=qk_scale,
                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim_in)
        self.mlp = MLP_P(in_features=dim_in, hidden_features=int(dim_in * mlp_ratio),
                         out_features=dim_out, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, tokens):
        tokens = tokens + self.drop_path(self.attn(self.norm1(tokens)))  # B*N, k*k, c
        if self.res:
            tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))  # B*N, k*k, c
        else:
            tokens = self.drop_path(self.mlp(self.norm2(tokens)))  # B*N, k*k, c

        return tokens


class MLP_Attention(nn.Module):
    def __init__(self, nlatent=1024, nwords=128):
        super(MLP_Attention, self).__init__()

        self.nlatent = nlatent
        self.nwords = nwords

        self.conv1 = torch.nn.Conv1d(self.nlatent, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.block1 = torch.nn.Sequential(BlockPoint(dim_in=128, dim_out=128, heads=4),
                                          BlockPoint(dim_in=128, dim_out=128, heads=4))
        self.block2 = torch.nn.Sequential(BlockPoint(dim_in=64, dim_out=64, heads=4),
                                          BlockPoint(dim_in=64, dim_out=64, heads=4))

        self.pos_1 = nn.Parameter(torch.zeros(1, self.nwords, 128))
        self.pos_2 = nn.Parameter(torch.zeros(1, self.nwords, 64))

        trunc_normal(self.pos_1, std=.02)
        trunc_normal(self.pos_2, std=.02)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x.transpose(1, 2) + self.pos_1).transpose(1, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block2(x.transpose(1, 2) + self.pos_2).transpose(1, 2)
        x = self.th(self.conv3(x))
        return x


class PointTrans_Generator(nn.Module):
    """ PointTrans-G """

    def __init__(self, options):
        super(PointTrans_Generator, self).__init__()

        self.npatch = options.npatch
        self.npatch_point = options.npatch_point
        self.patchDim = options.patchDim
        self.z_dim = options.z_dim

        self.decoder = nn.ModuleList([MLP_Attention(nlatent=self.patchDim + self.z_dim, nwords=self.npatch_point)
                                      for i in range(0, self.npatch)])
        self.grid = torch.nn.Parameter(torch.FloatTensor(self.npatch, self.patchDim, self.npatch_point))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        outs = []
        patches = []
        for i in range(0, self.npatch):
            # random planar patch
            # ==========================================================================
            rand_grid = self.grid[i].unsqueeze(0).expand(x.size(0), -1, -1)
            patches.append(rand_grid[0].transpose(1, 0))
            # ==========================================================================

            # cat with latent vector and decode
            # ==========================================================================
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()

            outs.append(self.decoder[i](y))
            # ==========================================================================

        return torch.cat(outs, 2).transpose(2, 1).contiguous(), torch.stack(outs, 1).transpose(3, 2).contiguous()


# ====================================================== Generator DualTrans =====================================================
class BlockDual(nn.Module):
    def __init__(self, patch_dim_in, patch_dim_out, point_dim_in, point_dim_out, patch_heads, point_heads,
                 num_words, mlp_ratio=4., qk_scale=None, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, se=0):
        super().__init__()

        self.has_point = point_dim_in > 0
        self.res_patch = patch_dim_in == patch_dim_out
        self.res_point = point_dim_in == point_dim_out

        if self.has_point:
            # --------------------------------------------- Point ---------------------------------------------
            self.point_norm1 = norm_layer(point_dim_in)
            self.point_attn = Attention(point_dim_in, heads=point_heads, head_dim=point_dim_in // point_heads,  qk_scale=qk_scale,
                                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.point_norm2 = norm_layer(point_dim_in)
            self.point_mlp = MLP_P(in_features=point_dim_in, hidden_features=int(point_dim_in * mlp_ratio),
                                   out_features=point_dim_out, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer(num_words * point_dim_out)
            self.proj = nn.Linear(num_words * point_dim_out, patch_dim_in, bias=False)
            self.proj_norm2 = norm_layer(patch_dim_in)

        # --------------------------------------------- Patch ---------------------------------------------
        self.patch_norm1 = norm_layer(patch_dim_in)
        self.patch_attn = Attention(patch_dim_in, heads=patch_heads, head_dim=patch_dim_in // patch_heads, qk_scale=qk_scale,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.patch_norm2 = norm_layer(patch_dim_in)
        self.patch_mlp = MLP_P(in_features=patch_dim_in, hidden_features=int(patch_dim_in * mlp_ratio),
                               out_features=patch_dim_out, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, point_tokens, patch_tokens):
        if self.has_point:
            point_tokens = point_tokens + self.drop_path(self.point_attn(self.point_norm1(point_tokens)))  # B*N, k*k, c
            if self.res_point:
                point_tokens = point_tokens + self.drop_path(self.point_mlp(self.point_norm2(point_tokens)))  # B*N, k*k, c
            else:
                point_tokens = self.drop_path(self.point_mlp(self.point_norm2(point_tokens)))  # B*N, k*k, c

            B, N, C = patch_tokens.size()
            patch_tokens = patch_tokens + self.proj_norm2(self.proj(self.proj_norm1(point_tokens.reshape(B, N, -1))))

        patch_tokens = patch_tokens + self.drop_path(self.patch_attn(self.patch_norm1(patch_tokens)))
        if self.res_patch:
            patch_tokens = patch_tokens + self.drop_path(self.patch_mlp(self.patch_norm2(patch_tokens)))
        else:
            patch_tokens = self.drop_path(self.patch_mlp(self.patch_norm2(patch_tokens)))

        return point_tokens, patch_tokens


class DualTrans_Generator(nn.Module):
    """ DualTrans-G """

    def __init__(self, options):
        super(DualTrans_Generator, self).__init__()

        self.npoint = options.npoint
        self.npatch = options.npatch
        self.nwords = options.nwords
        self.npatch_point = options.npatch_point
        self.patchDim = options.patchDim
        self.z_dim = options.z_dim

        patch_dim_list, point_dim_list = [1024, 1024, 1024, 1024, 1024, (self.npoint//self.npatch)*3], [512, 512, 512, 512, 512, 512]
        patch_dim, point_dim = patch_dim_list[0], point_dim_list[0]
        patch_heads, point_heads = 8, 8
        drop_rate, drop_path_rate = 0., 0
        depth = len(patch_dim_list)-1

        assert self.npoint % (self.npatch * self.nwords) == 0
        self.grid = torch.nn.Parameter(torch.FloatTensor(self.npatch, self.npatch_point, self.patchDim))

        self.proj_norm1_i = nn.LayerNorm((self.npoint // (self.npatch * self.nwords)) * (self.z_dim + self.patchDim))
        self.proj_i = nn.Linear((self.npoint // (self.npatch * self.nwords)) * (self.z_dim + self.patchDim), point_dim)
        self.proj_norm2_i = nn.LayerNorm(point_dim)

        self.proj_norm1_o = nn.LayerNorm(self.nwords * point_dim)
        self.proj_o = nn.Linear(self.nwords * point_dim, patch_dim)
        self.proj_norm2_o = nn.LayerNorm(patch_dim)

        self.point_pos = nn.Parameter(torch.zeros(1, self.nwords, point_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, self.npatch, patch_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for i in range(depth):
            patch_dim_in, patch_dim_out = patch_dim_list[i], patch_dim_list[i+1]
            point_dim_in, point_dim_out = point_dim_list[i], point_dim_list[i+1]
            blocks.append(BlockDual(
                patch_dim_in=patch_dim_in, patch_dim_out=patch_dim_out,
                point_dim_in=point_dim_in, point_dim_out=point_dim_out,
                patch_heads=patch_heads, point_heads=point_heads,
                num_words=self.nwords, drop=drop_rate, drop_path=dpr[i]))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(patch_dim)

        trunc_normal(self.patch_pos, std=.02)
        trunc_normal(self.point_pos, std=.02)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        y = x.view(x.shape[0], 1, 1, x.shape[1]).expand(-1, self.grid.shape[0], self.grid.shape[1], -1)
        y = torch.cat((y, self.grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)), -1).contiguous()

        point_tokens = y.reshape(y.shape[0] * self.npatch, self.nwords, -1)
        point_tokens = self.proj_norm2_i(self.proj_i(self.proj_norm1_i(point_tokens)))
        point_tokens = point_tokens + self.point_pos

        patch_tokens = point_tokens.reshape(x.shape[0], self.npatch, -1)
        patch_tokens = self.proj_norm2_o(self.proj_o(self.proj_norm1_o(patch_tokens)))
        patch_tokens = patch_tokens + self.patch_pos

        patch_tokens = self.pos_drop(patch_tokens)

        for blk in self.blocks:
            point_tokens, patch_tokens = blk(point_tokens, patch_tokens)

        patch_tokens = patch_tokens.reshape(x.shape[0], self.npatch, self.npoint // self.npatch, -1).permute(1, 0, 3, 2)
        outs, patches = [], []
        for i in range(0, self.npatch):
            outs.append(patch_tokens[i])

        return torch.cat(outs, 2).transpose(2, 1).contiguous(), torch.stack(outs, 1).transpose(3, 2).contiguous()
