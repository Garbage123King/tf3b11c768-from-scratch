#!/usr/bin/env python3
"""Export network structure + per-layer activations of
kata1-tf3-b11c768-s11001M-d5973M.bin.gz to netvis.json for 3D visualization.

Reuses the exact inference semantics of ../main.py (validated against the C
implementation to <2e-6 relerr). Records for every operator:
  - weight statistics, output statistics
  - spatial RMS map  (361 values, 19x19)
  - per-channel RMS   (C values)
Plus detail data for transformer sub-blocks:
  - attention softmax maps (head 0 of every attention, uint8-quantized)
  - RoPE frequencies
  - FFN intermediate activations (statistics only)
Run:  python export_vis.py   (needs ../model.bin and ../input.bin)
Out:  netvis.json
"""
import base64
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
XY, N = 19, 361
C_TRUNK, C_MID, NH, QD, NP_ = 768, 384, 12, 32, 16
FFN_DIM, N_BLOCKS, N_SUB = 1152, 11, 6


# ------------------------------------------------------------------ loader
class Loader:
    def __init__(self, data: bytes):
        self.d = data
        self.pos = 0

    def token(self) -> str:
        d, i = self.d, self.pos
        while i < len(d) and d[i] in b" \t\r\n":
            i += 1
        j = i
        while j < len(d) and d[j] not in b" \t\r\n":
            j += 1
        self.pos = j
        return d[i:j].decode("ascii")

    def int(self):
        return int(self.token())

    def flt(self):
        return float(self.token())

    def floats(self, n) -> np.ndarray:
        i = self.pos
        while self.d[i] != 0x40:
            i += 1
        assert self.d[i : i + 5] == b"@BIN@"
        i += 5
        arr = np.frombuffer(self.d, dtype="<f4", count=n, offset=i)
        self.pos = i + 4 * n
        return arr


class Model:
    def __init__(self):
        self.conv = {}   # name -> (cy,cx,cic,coc,w)
        self.mat = {}    # name -> (ic,oc,w)
        self.bn = {}     # name -> (ms,mb)
        self.rms = {}    # name -> (w,eps)
        self.rope = {}   # name -> (h,pairs,w)
        self.biases = {} # name -> b


def load_model(path="model.bin"):
    p = Loader(open(path, "rb").read())
    m = Model()
    m.name = p.token()
    version = p.int()
    nbin = p.int()
    nglb = p.int()
    for _ in range(7):
        p.flt()
    p.int()
    for _ in range(7):
        p.int()

    t = p.token()
    assert t == "trunk"
    numBlocks = p.int()
    trunkC = p.int()
    midC = p.int()
    p.int()
    p.int()
    p.int()
    p.int()
    for _ in range(5):
        p.int()

    def read_conv():
        name = p.token()
        cy, cx, cic, coc = p.int(), p.int(), p.int(), p.int()
        p.int()
        p.int()
        w = p.floats(cy * cx * cic * coc)
        m.conv[name] = (cy, cx, cic, coc, w)

    def read_bn():
        name = p.token()
        c = p.int()
        eps = p.flt()
        hs = p.int()
        hb = p.int()
        mean = p.floats(c)
        var = p.floats(c)
        scale = p.floats(c) if hs else np.ones(c, dtype=np.float32)
        bias = p.floats(c) if hb else np.zeros(c, dtype=np.float32)
        ms = scale / np.sqrt(var + eps)
        mb = bias - ms * mean
        m.bn[name] = (ms.astype(np.float32), mb.astype(np.float32))

    def read_mat():
        name = p.token()
        ic, oc = p.int(), p.int()
        m.mat[name] = (ic, oc, p.floats(ic * oc))

    def read_rms():
        name = p.token()
        c = p.int()
        eps = p.flt()
        m.rms[name] = (p.floats(c), eps)

    def read_rope():
        name = p.token()
        h = p.int()
        pairs = p.int()
        two = p.int()
        assert two == 2
        m.rope[name] = (h, pairs, p.floats(h * pairs * 2))

    def read_bias():
        name = p.token()
        c = p.int()
        m.biases[name] = p.floats(c)

    def read_act():
        p.token()
        assert p.token() == "ACTIVATION_SILU"

    def read_attn(prefix):
        name = p.token()
        h, kvh, qd, vd, ur, lr = (p.int() for _ in range(6))
        assert ur == 1 and lr == 1
        read_rms()
        for _ in range(4):
            read_mat()
        rn = p.token()
        rh, rp, r2 = p.int(), p.int(), p.int()
        m.rope[rn] = (rh, rp, p.floats(rh * rp * r2))

    def read_ffn():
        name = p.token()
        c, fd, sw = p.int(), p.int(), p.int()
        assert sw == 1
        read_rms()
        read_mat()
        read_mat()
        read_mat()

    def read_nbt():
        kind = p.token()
        assert kind == "nested_bottleneck_block"
        bname = p.token()
        nsub = p.int()
        read_bn()
        read_act()
        read_conv()
        for si in range(nsub):
            skind = p.token()
            if skind == "transformer_attention_block":
                read_attn(bname + ".blockstack." + str(si))
            else:
                assert skind == "transformer_ffn_block"
                read_ffn()
        read_bn()
        read_act()
        read_conv()

    cn = p.token()
    cy, cx, cic, coc = p.int(), p.int(), p.int(), p.int()
    p.int()
    p.int()
    m.conv[cn] = (cy, cx, cic, coc, p.floats(cy * cx * cic * coc))
    read_mat()
    for _ in range(numBlocks):
        read_nbt()
    read_bn()
    read_act()

    ph = p.token()
    assert ph == "model.policy_head"
    p.int()
    p.int()
    p.int()
    p.int()
    read_conv()
    read_conv()
    read_bn()
    read_act()
    read_mat()
    read_bn()
    read_act()
    read_conv()
    read_mat()
    read_bias()
    read_act()
    read_mat()

    vh = p.token()
    assert vh == "model.value_head"
    p.int()
    p.int()
    p.int()
    read_conv()
    read_bn()
    read_act()
    read_mat()
    read_bias()
    read_act()
    read_mat()
    read_bias()
    read_mat()
    read_bias()
    read_conv()
    print("loaded", m.name, "v", version, "- convs", len(m.conv), "mats", len(m.mat),
          "bns", len(m.bn), "rms", len(m.rms), "ropes", len(m.rope))
    return m, version, nbin, nglb


# ------------------------------------------------------------------ helpers
def T(a):
    return torch.from_numpy(np.array(a, dtype=np.float32))


def tstats(v):
    v = v.detach().float()
    n = max(1, v.numel())
    return dict(
        rms=round(float(v.pow(2).mean().sqrt()), 4),
        mean=round(float(v.mean()), 4),
        std=round(float(v.float().std()), 4) if n > 1 else 0.0,
        min=round(float(v.min()), 4),
        max=round(float(v.max()), 4),
    )


def r4(a, k=4):
    return [round(float(x), k) for x in a]


def spatial_rms(v):
    # (C,N) -> (N,)
    return v.detach().float().pow(2).mean(0).sqrt()


def channel_rms(v):
    return v.detach().float().pow(2).mean(1).sqrt()


def quant_attn(pmap):
    """softmax prob map (N,N) -> uint8 b64, log scale."""
    x = pmap.detach().float().clamp_min(1e-12)
    v = (x * N).clamp_min(1e-12).log() / math.log(N)
    q = (v.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return base64.b64encode(q.tobytes()).decode("ascii")


# ------------------------------------------------------------------ export
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    os.chdir(root)
    if not os.path.exists("model.bin"):
        import gzip
        with gzip.open("kata1-tf3-b11c768-s11001M-d5973M.bin.gz", "rb") as f, open("model.bin", "wb") as g:
            g.write(f.read())

    m, version, nbin, nglb = load_model("model.bin")
    m.tconv = {k: (v[0], v[1], v[2], v[3], T(v[4])) for k, v in m.conv.items()}
    m.tmat = {k: (v[0], v[1], T(v[2])) for k, v in m.mat.items()}
    m.tbn = {k: (T(v[0]), T(v[1])) for k, v in m.bn.items()}
    m.trms = {k: (T(v[0]), v[1]) for k, v in m.rms.items()}
    m.trope = {k: (v[0], v[1], T(v[2])) for k, v in m.rope.items()}
    m.tbias = {k: T(v) for k, v in m.biases.items()}

    inp = torch.from_numpy(np.fromfile("input.bin", dtype="<f4")).view(22, N)
    glb = torch.from_numpy(np.fromfile("input_global.bin", dtype="<f4")).view(19)
    mask = inp[0]
    S = mask.sum().item()
    d = math.sqrt(S) - 14.0

    def mm(name, x):
        ic, oc, w = m.tmat[name]
        return w.view(ic, oc).t() @ x

    def conv1x1(name, x):
        if name in m.tmat:
            ic, oc, w = m.tmat[name]
            return w.view(ic, oc).t() @ x
        cy, cx, cic, coc, w = m.tconv[name]
        assert cy == 1 and cx == 1
        return w.view(cic, coc).t() @ x

    def bn_act_mask(x, name, k=None):
        ms, mb = m.tbn[name]
        y = F.silu(x * ms[:, None] + mb[:, None]) * mask[None, :]
        return y

    nodes = []
    edges = []
    attn_maps = {}
    rope_data = {}
    wstats_cache = {}

    def wstats(name):
        if name in wstats_cache:
            return wstats_cache[name]
        w = None
        if name in m.tmat:
            w = m.tmat[name][2]
        elif name in m.tconv:
            w = m.tconv[name][4]
        elif name in m.tbn:
            w = m.tbn[name][0]
        elif name in m.trms:
            w = m.trms[name][0]
        elif name in m.tbias:
            w = m.tbias[name]
        s = tstats(w) if w is not None else None
        wstats_cache[name] = s
        return s

    def add_node(nid, label, kind, wname=None, shape=None, nparams=0,
                 out=None, sp=None, ch=None, parent=None, note=None):
        nd = dict(id=nid, label=label, kind=kind,
                  w=wstats(wname) if wname else None,
                  shape=shape, params=int(nparams), out=out, parent=parent,
                  note=note)
        if sp is not None:
            nd["sp"] = r4(spatial_rms(sp) if isinstance(sp, torch.Tensor) and sp.dim() == 2 else sp)
        if ch is not None:
            nd["ch"] = r4(channel_rms(ch) if isinstance(ch, torch.Tensor) and ch.dim() == 2 else ch)
        nodes.append(nd)
        return nd

    def edge(s, t, kind="main"):
        edges.append(dict(s=s, t=t, k=kind))

    # ---------------- input ----------------
    add_node("input", "input planes", "input", shape=[22, N], out=tstats(inp),
             sp=None, ch=inp, note="22 binary feature planes")
    add_node("input_global", "global features", "input", shape=[19], out=tstats(glb),
             note="19 global scalars")

    # ---------------- initial ----------------
    cy, cx, cic, coc, wc = m.tconv["model.conv_spatial"]
    trunk = F.conv2d(inp.view(1, 22, XY, XY), wc.view(cy, cx, cic, coc).permute(3, 2, 0, 1), padding=cy // 2).view(coc, N)
    add_node("conv_spatial", "conv 3x3", "conv", "model.conv_spatial", [C_TRUNK, N],
             cic * coc * cy * cx, tstats(trunk), sp=trunk, ch=trunk)
    g = mm("model.linear_global", glb[:, None]).squeeze(1)
    add_node("linear_global", "linear 19→768", "matmul", "model.linear_global", [C_TRUNK],
             19 * C_TRUNK, tstats(g), note="broadcast onto board")
    trunk = trunk + g[:, None]
    add_node("init_add", "trunk init", "add", shape=[C_TRUNK, N], out=tstats(trunk), sp=trunk, ch=trunk)
    edge("input", "conv_spatial")
    edge("input_global", "linear_global")
    edge("conv_spatial", "init_add")
    edge("linear_global", "init_add")

    def attn_block(prefix, mid, rec, head_full=0):
        """one transformer_attention_block; rec: dict to fill extra detail"""
        norm_name = prefix + ".norm1"
        xn = mid * m.trms[norm_name][0][:, None] / rms_denom(mid, norm_name)
        xn = xn * mask[None, :]
        q = mm(prefix + ".q_proj", xn)
        k = mm(prefix + ".k_proj", xn)
        v = mm(prefix + ".v_proj", xn)
        kh, pairs, rf = m.trope[prefix + ".rope_freqs"]
        rf = rf.view(kh, pairs, 2)
        pos = torch.arange(N)
        py = (pos // XY).float()
        px = (pos % XY).float()
        ang = px[None, None, :] * rf[:, :, 0:1] + py[None, None, :] * rf[:, :, 1:2]

        def rot(x):
            x3 = x.view(NH, NP_, 2, N)
            x0 = x3[:, :, 0]
            x1 = x3[:, :, 1]
            c = torch.cos(ang)
            s = torch.sin(ang)
            return torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=2).view(C_MID, N)

        qr = rot(q)
        kr = rot(k)
        qh = qr.view(NH, QD, N)
        kh2 = kr.view(NH, QD, N)
        vh = v.view(NH, QD, N)
        sc = torch.einsum("hdn,hdm->hnm", qh, kh2) / math.sqrt(QD)
        sc = sc + (mask[None, None, :] - 1.0) * 1e30
        pr = torch.softmax(sc, dim=2)
        out = torch.einsum("hnm,hdm->hdn", pr, vh).view(C_MID, N)
        out = mm(prefix + ".out_proj", out)
        out = out * mask[None, :]
        rec["norm"] = tstats(xn)
        rec["q"] = tstats(qr)
        rec["k"] = tstats(kr)
        rec["v"] = tstats(v)
        rec["out_proj"] = tstats(out)
        rec["map"] = pr[head_full]
        rope_data[prefix] = r4(rf.reshape(-1), 6)
        return out, pr

    def rms_denom(x, name):
        w, eps = m.trms[name]
        ss = (x * x).sum(0) / C_MID
        return torch.sqrt(ss + eps)

    def ffn_block(prefix, mid, rec):
        norm_name = prefix + ".norm"
        w_, eps = m.trms[norm_name]
        xn = mid * w_[:, None] / torch.sqrt((mid * mid).sum(0) / C_MID + eps)
        xn = xn * mask[None, :]
        a = mm(prefix + ".ffn_linear1", xn)
        g2 = mm(prefix + ".ffn_linear_gate", xn)
        hh = F.silu(a) * g2
        out = mm(prefix + ".ffn_linear2", hh)
        out = out * mask[None, :]
        rec["norm"] = tstats(xn)
        rec["l1"] = tstats(a)
        rec["gate"] = tstats(g2)
        rec["act"] = tstats(hh)
        rec["l2"] = tstats(out)
        rec["act_sp"] = r4(spatial_rms(hh))
        return out

    # ---------------- blocks ----------------
    for b in range(N_BLOCKS):
        bp = "model.blocks.%d" % b
        bid = "blk%d" % b
        pre_bn = bp + ".normactconvp.norm"
        pre_conv = bp + ".normactconvp.conv"
        post_bn = bp + ".normactconvq.norm"
        post_conv = bp + ".normactconvq.conv"

        t1 = bn_act_mask(trunk, pre_bn)
        add_node(bid + ".preBN", "BN+SiLU", "bn", pre_bn, [C_TRUNK, N],
                 4 * C_TRUNK, tstats(t1), sp=t1, ch=t1, parent=bid)
        mid = conv1x1(pre_conv, t1)
        add_node(bid + ".preConv", "conv 1x1 768→384", "conv", pre_conv, [C_MID, N],
                 C_TRUNK * C_MID, tstats(mid), sp=mid, ch=mid, parent=bid)
        edge(("blk%d.out" % (b - 1)) if b else "init_add", bid + ".preBN")
        edge(bid + ".preBN", bid + ".preConv")

        sub_ids = []
        for si in range(N_SUB):
            pname = bp + ".blockstack.%d" % si
            sid = bid + (".a%d" % (si // 2) if si % 2 == 0 else ".f%d" % ((si - 1) // 2))
            if si % 2 == 0:
                rec = {}
                out, pr = attn_block(pname, mid, rec)
                add_node(sid, "attention", "attn", pname + ".norm1", [C_MID, N],
                         4 * C_MID * C_MID + C_MID, tstats(out), sp=out, ch=out, parent=bid,
                         note="12 heads, RoPE, softmax over board")
                attn_maps[sid] = quant_attn(rec["map"])
                nodes[-1]["detail"] = {k2: v2 for k2, v2 in rec.items() if k2 != "map"}
                sub_ids.append(sid)
            else:
                rec = {}
                out = ffn_block(pname, mid, rec)
                add_node(sid, "FFN", "ffn", pname + ".norm", [C_MID, N],
                         3 * C_MID * FFN_DIM, tstats(out), sp=out, ch=out, parent=bid,
                         note="SwiGLU 384→1152→384")
                nodes[-1]["detail"] = rec
                sub_ids.append(sid)
            edge(bid + ".preConv" if si == 0 else sub_ids[si - 1], sid, "main")
            mid = mid + out
        add_node(bid + ".stack_out", "stack out", "add", shape=[C_MID, N],
                 out=tstats(mid), sp=mid, ch=mid, parent=bid)
        edge(sub_ids[-1], bid + ".stack_out")

        t2 = bn_act_mask(mid, post_bn)
        add_node(bid + ".postBN", "BN+SiLU", "bn", post_bn, [C_MID, N],
                 4 * C_MID, tstats(t2), sp=t2, ch=t2, parent=bid)
        resid = conv1x1(post_conv, t2)
        add_node(bid + ".postConv", "conv 1x1 384→768", "conv", post_conv, [C_TRUNK, N],
                 C_MID * C_TRUNK, tstats(resid), sp=resid, ch=resid, parent=bid)
        edge(bid + ".stack_out", bid + ".postBN")
        edge(bid + ".postBN", bid + ".postConv")
        trunk = trunk + resid
        add_node(bid + ".out", "block out (+residual)", "add", shape=[C_TRUNK, N],
                 out=tstats(trunk), sp=trunk, ch=trunk, parent=bid)
        edge(bid + ".postConv", bid + ".out")
        edge(("blk%d" % (b - 1) + ".out") if b else "init_add", bid + ".out", "res")

    # ---------------- tip ----------------
    t3 = bn_act_mask(trunk, "model.norm_trunkfinal")
    add_node("tip", "trunk final BN+SiLU", "bn", "model.norm_trunkfinal", [C_TRUNK, N],
             4 * C_TRUNK, tstats(t3), sp=t3, ch=t3)
    trunk = t3
    edge("blk10.out", "tip")

    # ---------------- policy head ----------------
    p1 = conv1x1("model.policy_head.conv1p", trunk)
    add_node("pol.conv1p", "conv 1x1 768→96", "conv", "model.policy_head.conv1p", [96, N],
             C_TRUNK * 96, tstats(p1), sp=p1, ch=p1)
    g1 = conv1x1("model.policy_head.conv1g", trunk)
    add_node("pol.conv1g", "conv 1x1 768→96", "conv", "model.policy_head.conv1g", [96, N],
             C_TRUNK * 96, tstats(g1), sp=g1, ch=g1)
    edge("tip", "pol.conv1p")
    edge("tip", "pol.conv1g")
    g1 = bn_act_mask(g1, "model.policy_head.biasg")
    add_node("pol.biasg", "BN+SiLU", "bn", "model.policy_head.biasg", [96, N],
             3 * 96, tstats(g1), sp=g1, ch=g1)
    edge("pol.conv1g", "pol.biasg")
    mean = g1.sum(1) / S
    p2 = mean * (d / 10.0)
    p3 = (g1 + (mask - 1.0)[None, :]).max(1).values
    gp = torch.cat((mean, p2, p3))
    add_node("pol.gpool", "gpool concat 288", "gpool", shape=[288], out=tstats(gp),
             note="[mean, mean*d/10, max] pooling")
    edge("pol.biasg", "pol.gpool")
    gbias = mm("model.policy_head.linear_g", gp[:, None]).squeeze(1)
    add_node("pol.linear_g", "linear 288→96", "matmul", "model.policy_head.linear_g", [96],
             288 * 96, tstats(gbias))
    edge("pol.gpool", "pol.linear_g")
    p1 = p1 + gbias[:, None]
    add_node("pol.add_g", "add gpool bias", "add", shape=[96, N], out=tstats(p1), sp=p1, ch=p1)
    edge("pol.conv1p", "pol.add_g")
    edge("pol.linear_g", "pol.add_g")
    p1 = bn_act_mask(p1, "model.policy_head.bias2")
    add_node("pol.bias2", "BN+SiLU", "bn", "model.policy_head.bias2", [96, N],
             3 * 96, tstats(p1), sp=p1, ch=p1)
    edge("pol.add_g", "pol.bias2")
    policy = conv1x1("model.policy_head.conv2p", p1)
    add_node("pol.out", "policy logits 2×361", "head_out", "model.policy_head.conv2p", [2, N],
             96 * 2, tstats(policy), sp=policy, ch=policy)
    edge("pol.bias2", "pol.out")
    pp = mm("model.policy_head.linear_pass", gp[:, None]).squeeze(1)
    add_node("pol.linear_pass", "linear 288→96", "matmul", "model.policy_head.linear_pass", [96],
             288 * 96, tstats(pp))
    edge("pol.gpool", "pol.linear_pass")
    pp = F.silu(pp + m.tbias["model.policy_head.linear_pass_bias"])
    add_node("pol.pass_silu", "SiLU+bias", "silu", shape=[96], out=tstats(pp))
    edge("pol.linear_pass", "pol.pass_silu")
    passv = mm("model.policy_head.linear_pass2", pp[:, None]).squeeze(1)
    add_node("pol.pass_out", "pass logits", "head_out", "model.policy_head.linear_pass2", [2],
             96 * 2, tstats(passv))
    edge("pol.pass_silu", "pol.pass_out")

    # ---------------- value head ----------------
    v1 = conv1x1("model.value_head.conv1", trunk)
    add_node("val.conv1", "conv 1x1 768→192", "conv", "model.value_head.conv1", [192, N],
             C_TRUNK * 192, tstats(v1), sp=v1, ch=v1)
    edge("tip", "val.conv1")
    v1 = bn_act_mask(v1, "model.value_head.bias1")
    add_node("val.bias1", "BN+SiLU", "bn", "model.value_head.bias1", [192, N],
             3 * 192, tstats(v1), sp=v1, ch=v1)
    edge("val.conv1", "val.bias1")
    vmean = v1.sum(1) / S
    vg = torch.cat((vmean, vmean * (d / 10.0), vmean * (d * d / 100.0 - 0.1)))
    add_node("val.gpool", "gpool concat 576", "gpool", shape=[576], out=tstats(vg),
             note="[mean, mean*d/10, mean*(d^2/100-0.1)]")
    edge("val.bias1", "val.gpool")
    v2 = mm("model.value_head.linear2", vg[:, None]).squeeze(1)
    v2 = F.silu(v2 + m.tbias["model.value_head.bias2"])
    add_node("val.linear2", "linear 576→192 +SiLU", "matmul", "model.value_head.linear2", [192],
             576 * 192, tstats(v2))
    edge("val.gpool", "val.linear2")
    value = mm("model.value_head.linear_valuehead", v2[:, None]).squeeze(1) + m.tbias["model.value_head.bias_valuehead"]
    add_node("val.out", "value 3", "head_out", "model.value_head.linear_valuehead", [3],
             192 * 3, tstats(value), note="win/loss/draw scores")
    edge("val.linear2", "val.out")
    misc = mm("model.value_head.linear_miscvaluehead", v2[:, None]).squeeze(1) + m.tbias["model.value_head.bias_miscvaluehead"]
    add_node("val.misc", "miscvalue 6", "head_out", "model.value_head.linear_miscvaluehead", [6],
             192 * 6, tstats(misc), note="score belief parameters")
    edge("val.linear2", "val.misc")
    own = conv1x1("model.value_head.conv_ownership", v1)
    add_node("val.own", "ownership 1×361", "head_out", "model.value_head.conv_ownership", [1, N],
             192, tstats(own), sp=own, ch=own)
    edge("val.bias1", "val.own")

    # top-level block summary nodes
    for b in range(N_BLOCKS):
        bid = "blk%d" % b
        sub = [x for x in nodes if x.get("parent") == bid]
        npar = sum(x["params"] for x in sub)
        last = sub[-1] if sub else None
        add_node(bid, "Block %d" % b, "block", shape=[C_TRUNK, N], nparams=npar,
                 out={k2: last["out"][k2] for k2 in ("rms", "mean", "std", "min", "max")} if last else None,
                 parent=None, note="nested bottleneck, 6 sub-blocks")

    vis = dict(
        meta=dict(
            model=m.name,
            version=version,
            binfeat=22,
            glbfeat=19,
            trunk=C_TRUNK,
            mid=C_MID,
            heads=NH,
            qd=QD,
            ffn=FFN_DIM,
            blocks=N_BLOCKS,
            nsub=N_SUB,
            xy=XY,
            value=r4(value, 4),
            misc=r4(misc, 4),
            policy_pass=r4(passv, 4),
            top5=[],
        ),
        nodes=nodes,
        edges=edges,
        attn_maps=attn_maps,
        rope=rope_data,
    )
    # top5
    probs = torch.softmax(policy[0], 0)
    top = probs.topk(5)
    vis["meta"]["top5"] = [[int(i), round(float(p), 4)] for i, p in zip(top.indices, top.values)]

    out_path = os.path.join(here, "netvis.json")
    with open(out_path, "w") as f:
        json.dump(vis, f, separators=(",", ":"))
    with open(os.path.join(here, "netvis.js"), "w") as f:
        f.write("window.NETVIS_DATA=")
        json.dump(vis, f, separators=(",", ":"))
        f.write(";\n")
    print("nodes:", len(nodes), "edges:", len(edges), "attn_maps:", len(attn_maps))
    print("wrote", out_path, "%.1f MB" % (os.path.getsize(out_path) / 1e6))


if __name__ == "__main__":
    main()
