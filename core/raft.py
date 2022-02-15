import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout) # g_θ in paper
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout) # g_θ in paper
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H_8, W_8 = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H_8, W_8)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1) # 8倍して高解像度軸に値を合わせた flow 配列を，3x3 で畳み込み的に切り取っていく
        up_flow = up_flow.view(N, 2, 9, 1, 1, H_8, W_8) # (N,2,1,3,3,H/8,W/8) を (N,2,9,1,1,H/8,W/8) にした

        up_flow = torch.sum(mask * up_flow, dim=2) # up_flow の 4,5 次元目は 8,8 にブロードキャストされると思われる。sumによって3次元目の要素数は1に。
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H_8, 8*W_8)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous() # メモリの連続した領域へのアクセスとなるように再配置（メモリコピーが発生する）
        image2 = image2.contiguous() # 参考：https://ohke.hateblo.jp/entry/2019/11/30/230000

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision): # mixed precisionは演算子によって float32 と float16 を切り替えることで計算の高速化を図る。
            fmap1, fmap2 = self.fnet([image1, image2]) # 特徴量マップを画像毎に取得
        
        fmap1 = fmap1.float() # float32 にする
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius) # この時点では fmap の pyramid を作るだけ
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius) # corr 計算は事前に行われる。

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1) # 画像１の context を取得。構造は特徴量マップとほぼ変わらない。
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net) # GRU cell への入力になる
            inp = torch.relu(inp) # GRU cell への入力になる

        coords0, coords1 = self.initialize_flow(image1) # flow = coords1 (frame2での位置) - coords0 (frame1での位置)

        if flow_init is not None: # warm-start
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach() # 計算グラフを切る。これにより coords1 の重みの勾配は計算されない
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0 # 1/8 resolutions
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
