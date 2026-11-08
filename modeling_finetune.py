from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange, repeat


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # me: support window mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



"""
adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""
# support cross attention
class GeneralAttention(nn.Module):
    def __init__(
            self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim if context_dim is None else context_dim, all_head_dim * 2, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, T1, C = x.shape
        q_bias, kv_bias = self.q_bias, None
        if self.q_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, T1, self.num_heads, -1).transpose(1,2) # me: (B, H, T1, C//H)
        kv = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)
        _, T2, _ = kv.shape
        kv = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple), me： (B, H, T2, C//H)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # me: (B, H, T1, T2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T1, -1) # (B, H, T1, C//H) -> (B, T1, H, C//H) -> (B, T1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, num_tokens, bottleneck_dim=64, drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 temporal_seq_len=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=bottleneck_dim, out_features=num_tokens,
                       act_layer=act_layer, drop=drop)

        self.temporal_seq_len = temporal_seq_len

    def forward(self, x):
        attn = self.mlp(self.norm(x)) # (b, t*n1, n2), n2=num_tokens
        attn = torch.softmax(rearrange(attn, 'b (t n1) n2 -> (b t) n2 n1', t=self.temporal_seq_len), dim=-1)
        x = rearrange(x, 'b (t n1) c -> (b t) n1 c', t=self.temporal_seq_len)
        x = torch.matmul(attn, x)
        x = rearrange(x, '(b t) n2 c -> b (t n2) c', t=self.temporal_seq_len)
        return x


class CSBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads, num_cross_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, cross_attn_head_dim=None):
        super().__init__()

        self.cross_attn = GeneralAttention(
            dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim)
        self.cross_norm1 = norm_layer(dim)
        self.cross_norm2 = norm_layer(context_dim)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None

    def forward(self, x, context):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_0 * self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # me: for more attention types
        self.temporal_seq_len = num_frames // self.tubelet_size
        self.spatial_num_patches = num_patches // self.temporal_seq_len
        self.input_token_size = (num_frames // self.tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


# Temporal Pyramid and Spatial Bottleneck Transformer (TPSBT) in SVFAP
class TPSBT(nn.Module):
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 # depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 keep_temporal_dim=False, # do not perform temporal pooling, has higher priority than 'use_mean_pooling'
                 head_activation_func=None, # activation function after head fc, mainly for the regression task
                 stage_depths=(12, 6, 3), # number of blocks in each stage
                 num_bottleneck_tokens=(8,), # number of bottleneck tokens in spatial bottleneck transformer
                 temporal_kernel_size=2, # kernel size for 1D temporal conv
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        # model architecture: TPSBT
        depth = sum(stage_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.stage_depths = stage_depths

        ## Temporal Pyramid: kernel size of 1D conv
        self.temporal_kernel_size = temporal_kernel_size
        ## Spatial Bottleneck Transformer: number of bottleneck tokens for summarizing important information in spatial attention
        if len(num_bottleneck_tokens) == 1:
            self.num_bottleneck_tokens = num_bottleneck_tokens * (len(stage_depths) - 1)  # repeat
        else:
            assert len(num_bottleneck_tokens) == (len(stage_depths) - 1)
            # support different numbers of bottleneck tokens in different stages
            self.num_bottleneck_tokens = num_bottleneck_tokens

        self.blocks = nn.ModuleList([])
        self.temporal_downsamples = nn.ModuleList([])
        self.spatial_downsamples = nn.ModuleList([])

        block_idx = 0
        for stage_idx, stage_depth in enumerate(stage_depths):
            # 1st stage (standard Transformer block)
            if stage_idx == 0:
                for _ in range(stage_depth):
                    self.blocks.append(
                        Block(
                            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], norm_layer=norm_layer,
                            init_values=init_values)
                    )
                    block_idx += 1
            # 2nd, 3rd,... stage (TPSBT block)
            else:
                # Temporal Pyramid (TP)
                self.temporal_downsamples.append(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=temporal_kernel_size, stride=temporal_kernel_size)
                )
                # Spatial Bottleneck Transformer (SBT)
                cur_temporal_seq_len = self.patch_embed.temporal_seq_len // (temporal_kernel_size ** stage_idx)
                ## spatial attention
                self.spatial_downsamples.append(
                    SpatialAttention(
                        dim=embed_dim, num_tokens=self.num_bottleneck_tokens[stage_idx - 1],
                        bottleneck_dim=128, drop=drop_rate, norm_layer=norm_layer, temporal_seq_len=cur_temporal_seq_len
                    )
                )
                ## cross self block
                for _ in range(stage_depth):
                    self.blocks.append(
                        CSBlock(dim=embed_dim, context_dim=embed_dim, num_heads=num_heads,
                                num_cross_heads=num_heads,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[block_idx], norm_layer=norm_layer, init_values=init_values)
                    )
                    block_idx += 1

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # me: add frame-level prediction support
        self.keep_temporal_dim = keep_temporal_dim

        # me: add head activation function support for regression task
        if head_activation_func is not None:
            if head_activation_func == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else: # default
            self.head_activation_func = nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        if not isinstance(self.head, nn.Identity):
            trunc_normal_(self.head.weight, std=.02)

            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        # model architecture
        block_idx = 0
        for stage_idx, stage_depth in enumerate(self.stage_depths):
            # 1st stage (standard Transformer block)
            if stage_idx == 0:
                for _ in range(stage_depth):
                    x = self.blocks[block_idx](x)
                    block_idx += 1
            # 2nd, 3rd,... stage (TPSBT)
            else:
                # Temporal Pyramid (TP)
                cur_temporal_seq_len = self.patch_embed.temporal_seq_len // (self.temporal_kernel_size ** (stage_idx - 1))
                x = rearrange(x, 'b (t hw) c -> (b hw) c t', t=cur_temporal_seq_len)
                x = self.temporal_downsamples[stage_idx - 1](x)
                x = rearrange(x, '(b hw) c t -> b (t hw) c', b=B)

                # Spatial Bottleneck Transformer (SBT)
                x_before_spatial_downsample = x
                # spatial attention
                x = self.spatial_downsamples[stage_idx - 1](x)
                # cross self block
                for _ in range(stage_depth - 1):
                    x = self.blocks[block_idx](x, context=x_before_spatial_downsample)
                    block_idx += 1
                # reverse cross self block for spatial resolution recovering
                x = self.blocks[block_idx](x_before_spatial_downsample, context=x)
                block_idx += 1


        x = self.norm(x)
        if self.fc_norm is not None:
            # me: add frame-level prediction support
            if self.keep_temporal_dim:
                x = rearrange(x, 'b (t hw) c -> b c t hw',
                              t=self.patch_embed.temporal_seq_len,
                              hw=self.patch_embed.spatial_num_patches)
                # spatial mean pooling
                x = x.mean(-1) # (B, C, T)
                # temporal upsample: 8 -> 16, for patch embedding reduction
                x = torch.nn.functional.interpolate(
                    x, scale_factor=self.patch_embed.tubelet_size,
                    mode='linear'
                )
                x = rearrange(x, 'b c t -> b t c')
                return self.fc_norm(x)
            else:
                return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x, save_feature=False):
        x = self.forward_features(x)
        if save_feature:
            feature = x
        x = self.head(x)
        # me: add head activation function support
        x = self.head_activation_func(x)
        # me: add frame-level prediction support
        if self.keep_temporal_dim:
            x = x.view(x.size(0), -1) # (B,T,C) -> (B,T*C)
        if save_feature:
            return x, feature
        else:
            return x



#-----------------------------------------------------
#
#
# ----------- Fusion Model ---------------------------
#
#
#-----------------------------------------------------



class LandmarkStream(nn.Module):
    """
    Receives per-frame landmark coordinates (B, T, landmark_dim) and learns
    temporal relationships using a standard Transformer.

    landmark_dim  : raw input dim, e.g. 68×2 = 136
    embed_dim     : internal model dimension
    num_heads     : attention heads
    num_layers    : depth of the Transformer
    num_frames    : number of frames (= temporal sequence length T)
    dropout       : dropout probability
    """

    def __init__(
        self,
        landmark_dim: int = 156,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        num_frames: int = 16,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Project raw landmark coords to embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(landmark_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Learnable temporal positional encoding  (T, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=dropout)

        # Standard Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,       # (B, T, C) convention
            norm_first=True,        # pre-norm (same style as TPSBT)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, landmark_dim)
        returns : (B, T, embed_dim)   -- per-frame landmark features
        """
        B, T, N, C = x.shape
        x = x.reshape(B, T, N* C)
        x = self.input_proj(x)                          # (B, T, embed_dim)
        x = x + self.pos_embed[:, :T, :]               # add temporal pos enc
        x = self.pos_drop(x)
        x = self.transformer(x)                         # (B, T, embed_dim)
        x = self.norm(x)
        return x  


class StreamFusion(nn.Module):
    """
    Fuses two streams:
      1. Concatenates along the feature (channel) dimension.
      2. Projects back to a common dim.
      3. Runs cross-attention layers so streams can interact.

    rgb_dim       : feature dim coming from TPSBT stream
    lm_dim        : feature dim coming from LandmarkStream
    fused_dim     : hidden dim after projection (= rgb_dim by default)
    num_heads     : attention heads in fusion cross-attention
    num_layers    : depth of cross-attention fusion
    dropout       : dropout probability
    """

    def __init__(
        self,
        rgb_dim: int,
        lm_dim: int,
        fused_dim: int = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        fused_dim = fused_dim or rgb_dim

        # After concat the dim is (rgb_dim + lm_dim); project to fused_dim
        self.input_proj = nn.Linear(rgb_dim + lm_dim, fused_dim)
        self.norm_in = norm_layer(fused_dim)

        # Stack of cross-attention + self-attention blocks (CSBlock style)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=fused_dim,
                nhead=num_heads,
                dim_feedforward=int(fused_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = norm_layer(fused_dim)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        lm_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        rgb_feat : (B, D_rgb)  -- pooled RGB feature vector
        lm_feat  : (B, D_lm)  -- pooled landmark feature vector
        returns  : (B, fused_dim)
        """
        # Concat along feature axis  -->  (B, D_rgb + D_lm)
        x = torch.cat([rgb_feat, lm_feat], dim=-1)

        # Project and add a sequence dim so Transformer can operate
        x = self.input_proj(x)          # (B, fused_dim)
        x = x.unsqueeze(1)              # (B, 1, fused_dim)  -- treat as 1-token seq

        for layer in self.layers:
            x = layer(x)               # (B, 1, fused_dim)

        x = self.norm_out(x)
        x = x.squeeze(1)               # (B, fused_dim)
        return x



class DualStreamTPSBT(nn.Module):
    """
    Dual-stream multimodal model.

    Stream 1 : RGB frames  --[TPSBT]--> rgb_feat   (B, rgb_embed_dim)
    Stream 2 : Landmarks   --[LandmarkStream]--> lm_feat  (B, lm_embed_dim)
    Fusion   : Concat --> cross-attention --> FC --> prediction
    """

    def __init__(
        self,
        # ── RGB stream (TPSBT) ──────────────────────────────────────────────
        img_size: int = 160,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        all_frames: int = 16,
        tubelet_size: int = 2,
        stage_depths: tuple = (12, 6, 3),
        num_bottleneck_tokens: tuple = (8,),
        temporal_kernel_size: int = 2,
        # ── Landmark stream ──────────────────────────────────────────────────
        landmark_dim: int = 156,        # raw landmark feature dim
        landmark_embed_dim: int = 256,  # internal landmark stream dim
        landmark_num_heads: int = 4,
        landmark_num_layers: int = 4,
        landmark_num_frames: int = 16,
        # ── Fusion ───────────────────────────────────────────────────────────
        fusion_num_heads: int = 8,
        fusion_num_layers: int = 2,
        dropout: float = 0.1,
        # ── Head ─────────────────────────────────────────────────────────────
        num_classes: int = 400,
        head_activation_func=None,
        use_mean_pooling = True,
        init_scale=0     # 'sigmoid', 'relu', 'tanh', or None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.landmark_embed_dim = landmark_embed_dim

        # ── Stream 1: RGB (TPSBT backbone, NO classification head) ───────────
        self.rgb_stream = TPSBT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,              # remove the head; we'll add our own
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            all_frames=all_frames,
            tubelet_size=tubelet_size,
            use_mean_pooling=use_mean_pooling,
            keep_temporal_dim=False,
            stage_depths=stage_depths,
            num_bottleneck_tokens=num_bottleneck_tokens,
            temporal_kernel_size=temporal_kernel_size,
            init_values=0.0,
        )
        # rgb_stream.forward_features returns (B, embed_dim)
        self.patch_embed = self.rgb_stream.patch_embed 

        # ── Stream 2: Landmark ───────────────────────────────────────────────
        self.landmark_stream = LandmarkStream(
            landmark_dim=landmark_dim,
            embed_dim=landmark_embed_dim,
            num_heads=landmark_num_heads,
            num_layers=landmark_num_layers,
            num_frames=landmark_num_frames,
            dropout=dropout,
        )
        # We pool the temporal output to get (B, landmark_embed_dim)
        self.landmark_pool = nn.Sequential(
            nn.LayerNorm(landmark_embed_dim),
            # mean-pool over T is done in forward()
        )

        # ── Fusion ───────────────────────────────────────────────────────────
        self.fusion = StreamFusion(
            rgb_dim=embed_dim,
            lm_dim=landmark_embed_dim,
            fused_dim=embed_dim,        # project back to embed_dim
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            dropout=dropout,
        )

        # ── Classification head ──────────────────────────────────────────────
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if head_activation_func == "sigmoid":
            self.head_act = nn.Sigmoid()
        elif head_activation_func == "relu":
            self.head_act = nn.ReLU()
        elif head_activation_func == "tanh":
            self.head_act = nn.Tanh()
        else:
            self.head_act = nn.Identity()

        # Weight init for the new modules (rgb_stream handles its own init)
        self._init_new_weights()

    def _init_new_weights(self):
        for m in [self.landmark_stream, self.fusion, self.head]:
            for module in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(module, nn.Linear):
                    trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.constant_(module.bias, 0)
                    nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Pass through no_weight_decay hint for both streams."""
        return {
            'rgb_stream.pos_embed',
            'rgb_stream.cls_token',
            'landmark_stream.pos_embed',
        }
    

    def get_num_layers(self):
        """Return number of layers from RGB stream for compatibility."""
        return self.rgb_stream.get_num_layers()
    
    def forward(
        self,
        rgb: torch.Tensor,
        landmarks: torch.Tensor,
        save_feature=False
    ) -> torch.Tensor:
        """
        rgb       : (B, C, T, H, W)      -- video tensor
        landmarks : (B, T, landmark_dim) -- per-frame keypoints
        returns   : (B, num_classes)
        """
        # ── Stream 1: RGB features ───────────────────────────────────────────
        # forward_features returns pooled (B, embed_dim)
        rgb_feat = self.rgb_stream.forward_features(rgb)    # (B, D_rgb)

        # ── Stream 2: Landmark features ──────────────────────────────────────
        lm_feat = self.landmark_stream(landmarks)            # (B, T, D_lm)
        # Apply norm then mean-pool over time
        lm_feat = self.landmark_pool[0](lm_feat)            # LayerNorm
        lm_feat = lm_feat.mean(dim=1)                       # (B, D_lm)

        # ── Fusion ───────────────────────────────────────────────────────────
        fused = self.fusion(rgb_feat, lm_feat)              # (B, D_fused)

        # ── Prediction head ──────────────────────────────────────────────────
        out = self.head(fused)                              # (B, num_classes)
        out = self.head_act(out)
        return out

    def load_rgb_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load a pretrained TPSBT checkpoint into the RGB stream only.
        Missing / unexpected keys (fusion, landmark) are safely ignored.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)

        # Prefix keys to match rgb_stream.*
        prefixed = {f"rgb_stream.{k}": v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(prefixed, strict=False)
        print(f"[load_rgb_pretrained] missing={len(missing)}  unexpected={len(unexpected)}")
        return self



@register_model
def dual_stream_base_patch16_160(pretrained=False, **kwargs):
    # [timm-compat patch] naye timm ki extra kwargs hatao
    for _k in ('pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir'):
        kwargs.pop(_k, None)
    """
    Default config matching svfap_base_patch16_160 for the RGB stream,
    with a 68-keypoint (136-dim) landmark stream.
    """
    # Remove all kwargs that conflict with explicit parameters
    kwargs.pop('all_frames', None)
    kwargs.pop('stage_depths', None)
    kwargs.pop('temporal_kernel_size', None)
    kwargs.pop('num_bottleneck_tokens', None)
    kwargs.pop('drop_path', None)
    kwargs.pop('attn_drop_rate', None)
    kwargs.pop('drop', None)
    kwargs.pop('drop_rate', None)


    model = DualStreamTPSBT(
        img_size=160,
        patch_size=16,
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        all_frames=16,
        stage_depths=(12, 6, 3),
        num_bottleneck_tokens=(8,),
        temporal_kernel_size=2,
        landmark_dim=156,
        landmark_embed_dim=256,
        landmark_num_heads=4,
        landmark_num_layers=4,
        landmark_num_frames=16,
        fusion_num_heads=8,
        fusion_num_layers=2,
        dropout=0.1,
        **kwargs,
    )
    return model

@register_model
def svfap_base_patch16_160(pretrained=False, **kwargs):
    # [timm-compat patch] naye timm ki extra kwargs hatao
    for _k in ('pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir'):
        kwargs.pop(_k, None)
    model = TPSBT(
        img_size=160,
        patch_size=16, embed_dim=512, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model