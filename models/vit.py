import torch
import torch.nn as nn
from functools import partial

class DQN_Conv(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden1: int = 32, hidden2: int = 64, hidden3: int = 100):
        super(DQN_Conv, self).__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        embed_dim = 64
        self.patch_embed = PatchEmbed(patch_size=1, embed_dim=embed_dim).to(device)
        # exp7
        #self.patch_embed = PatchEmbed(patch_size=4, in_chans=4, embed_dim=embed_dim).to(device)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 7*7 + 1, embed_dim))
        # exp7
        #self.pos_embed = nn.Parameter(torch.zeros(1, 10*6 + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=4) for _ in range(1)
            ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, n_actions)

    def forward(self, x):
        out = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(out.shape[0], -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out = out + self.pos_embed

        attns = list()
        for blk in self.blocks:
            out, attn = blk(out)
            attns.append(attn)

        out = self.norm(out)
        out = out[:, 0]

        out = self.fc1(out)

        return out, attns


class Mlp(nn.Module):
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


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=7, patch_size=1, in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        out, attn = self.attn(self.norm1(x))
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x, attn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn