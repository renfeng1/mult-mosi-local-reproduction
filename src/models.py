import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


def make_temporal_projection(in_dim, out_dim, kernel_size):
    if in_dim == out_dim and kernel_size == 1:
        return nn.Identity()
    return nn.Conv1d(
        in_dim,
        out_dim,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=False,
    )


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        super().__init__()
        self.orig_d_l = hyp_params.orig_d_l
        self.orig_d_a = hyp_params.orig_d_a
        self.orig_d_v = hyp_params.orig_d_v

        self.d_l = hyp_params.proj_dim
        self.d_a = hyp_params.proj_dim
        self.d_v = hyp_params.proj_dim

        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l + self.d_a + self.d_v
        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        output_dim = hyp_params.output_dim

        self.proj_l = make_temporal_projection(self.orig_d_l, self.d_l, hyp_params.kernel_l)
        self.proj_a = make_temporal_projection(self.orig_d_a, self.d_a, hyp_params.kernel_a)
        self.proj_v = make_temporal_projection(self.orig_d_v, self.d_v, hyp_params.kernel_v)

        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type="la")
            self.trans_l_with_v = self.get_network(self_type="lv")
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type="al")
            self.trans_a_with_v = self.get_network(self_type="av")
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type="vl")
            self.trans_v_with_a = self.get_network(self_type="va")

        self.trans_l_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type="l", layers=-1):
        if self_type in ["l", "al", "vl"]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ["a", "la", "va"]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ["v", "lv", "av"]:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == "l_mem":
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == "a_mem":
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == "v_mem":
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def forward(self, x_l, x_a, x_v):
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        proj_x_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        if self.lonly:
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if isinstance(h_ls, tuple):
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]

        if self.aonly:
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if isinstance(h_as, tuple):
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if isinstance(h_vs, tuple):
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training)
        )
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs
