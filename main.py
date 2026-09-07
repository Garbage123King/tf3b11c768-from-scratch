# kata1-tf3-b11c768-s11001M-d5973M.bin.gz 模型的独立推理实现 (PyTorch, float32)
#
# 模型: b11c768h12nbt3tflrs-fson-silu, version=17
#   输入: 22 个二值特征 (19x19), 19 个全局特征
#   trunk: conv3x3 22->768 + linear_global 19->768
#          11 x nested_bottleneck_block:
#              preBN(+SiLU+mask) -> conv1x1 768->384
#              6 x (transformer_attention / transformer_ffn 交替, 各带残差)
#              postBN(+SiLU+mask) -> conv1x1 384->768 (累加回 trunk)
#          norm_trunkfinal(BN) + SiLU
#   policy head / value head 见下方 forward
#
# 语义与 KataGo C++ (eigenbackend.cpp / desc.cpp) 完全一致:
#   * BN: y = mask ? silu(x*mergedScale + mergedBias) : 0
#         mergedScale = scale/sqrt(var+eps), mergedBias = bias - mergedScale*mean
#   * RMSNorm: y = x / sqrt(mean(x^2)+eps) * w, 棋盘外置 0
#   * matmul 权重文件布局 (ic,oc): y[oc] = sum_ic w[ic,oc]*x[ic]
#   * conv 权重文件布局 (y,x,ic,oc)
#   * RoPE: 每对 (2p,2p+1) 旋转, angle = x*freqX + y*freqY (可学习, 每 kv head)
#   * attention: score = q.k/sqrt(qHeadDim), key 在棋盘外被屏蔽, 查询残差乘 mask
#   * gpool: [mean, mean*(sqrt(S)-14)/10, max], value gpool 第三项换成 mean*((d^2)/100-0.1)
#
# 用法:
#   python main.py           # 生成输入 input.bin/input_global.bin, 推理, 输出 out_python.bin
#   python main.py compare   # 对比 out_python.bin 与 out_c.bin

import gzip
import math
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

DIR = os.path.dirname(os.path.abspath(__file__))
GZFILE = os.path.join(DIR, "kata1-tf3-b11c768-s11001M-d5973M.bin.gz")
BINFILE = os.path.join(DIR, "model.bin")

XY = 19
N = XY * XY
C_TRUNK = 768
C_MID = 384
N_HEADS = 12
Q_HEAD_DIM = 32
V_HEAD_DIM = 32
N_PAIRS = Q_HEAD_DIM // 2
FFN_DIM = 1152
N_BLOCKS = 11
N_SUB = 6

# ---------------------------------------------------------------------------
# 模型文件解析 (desc.cpp 语法, 只读必要字段, 权重直接 frombuffer 零拷贝)


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

    def ints(self, n):
        return [int(self.token()) for _ in range(n)]

    def int(self):
        return int(self.token())

    def flt(self):
        return float(self.token())

    def floats(self, n) -> np.ndarray:
        d, i = self.d, self.pos
        while d[i] != 0x40:  # '@'
            i += 1
        assert d[i : i + 5] == b"@BIN@", "expected @BIN@ marker"
        i += 5
        arr = np.frombuffer(d, dtype="<f4", count=n, offset=i)
        self.pos = i + 4 * n
        return arr


class Model:
    pass


def load_model(path=BINFILE) -> Model:
    with open(path, "rb") as f:
        data = f.read()
    p = Loader(data)
    m = Model()

    m.name = p.token()
    m.version = p.int()
    m.nbin = p.int()
    m.nglb = p.int()
    assert m.version == 17 and m.nbin == 22 and m.nglb == 19
    m.mults = [p.flt() for _ in range(7)]
    m.metaenc = p.int()
    m.flags = p.ints(7)

    # trunk
    t = p.token()
    assert t == "trunk"
    (numBlocks, trunkC, midC, regC, gpoolC, gpoolC2) = p.ints(6)
    assert (numBlocks, trunkC, midC) == (N_BLOCKS, C_TRUNK, C_MID)
    normkind = p.int()
    unused = p.ints(5)

    m.conv = {}   # name -> (cy,cx,ic,oc, np.ndarray) 文件布局 (y,x,ic,oc)
    m.mat = {}    # name -> (ic,oc, np.ndarray)     文件布局 (ic,oc)
    m.bn = {}     # name -> (ms, mb) float32 [C]
    m.rms = {}    # name -> (w, eps) float32 [C]
    m.rope = {}   # name -> (kv_heads, pairs, np.ndarray [kv,pairs,2])
    m.bias = {}   # name -> np.ndarray [C]

    def read_conv():
        name = p.token()
        cy, cx, cic, coc, dy, dx = p.ints(6)
        w = p.floats(cy * cx * cic * coc)
        m.conv[name] = (cy, cx, cic, coc, w)

    def read_mat():
        name = p.token()
        ic, oc = p.ints(2)
        w = p.floats(ic * oc)
        m.mat[name] = (ic, oc, w)

    def read_bn(expected=None):
        # BatchNormLayerDesc: name c eps hasScale hasBias [mean var scale bias]
        name = p.token()
        if expected is not None:
            assert name == expected, (name, expected)
        try:
            c = int(p.token())
        except ValueError:
            ctx = p.d[p.pos : p.pos + 80]
            raise ValueError(
                f"read_bn({name}): bad token near {p.pos!r}: {ctx!r}"
            )
        eps = p.flt()  # 可能是 1e-20 这类科学计数法
        hs = p.int()
        hb = p.int()
        mean = p.floats(c)
        var = p.floats(c)
        scale = p.floats(c) if hs else np.ones(c, dtype=np.float32)
        bias = p.floats(c) if hb else np.zeros(c, dtype=np.float32)
        ms = scale / np.sqrt(var + eps)
        mb = bias - ms * mean
        m.bn[name] = (ms.astype(np.float32), mb.astype(np.float32))

    def read_rms():
        name = p.token()
        c = p.int()
        eps = p.flt()
        w = p.floats(c)
        m.rms[name] = (w, eps)

    def read_rope():
        name = p.token()
        h, pairs, two = p.ints(3)
        assert two == 2
        w = p.floats(h * pairs * 2)
        m.rope[name] = (h, pairs, w)

    def read_matbias():
        name = p.token()
        c = p.int()
        m.bias[name] = p.floats(c)

    def read_activation():
        aname = p.token()
        act = p.token()
        assert act == "ACTIVATION_SILU", (aname, act)

    def read_attn(prefix):
        # transformer_attention_block: name h kv qd vd useRope learnableRope
        #   preLN q k v out [rope_freqs]
        name = p.token()
        h, kvh, qd, vd, ur, lr = p.ints(6)
        read_rms()
        read_mat()
        read_mat()
        read_mat()
        read_mat()
        if ur:
            assert lr == 1
            read_rope()
        m.attn.setdefault("meta", {})[prefix] = (h, kvh, qd, vd)

    def read_ffn():
        name = p.token()
        c, ffn, sw = p.ints(3)
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
        assert nsub == N_SUB
        read_bn()          # normactconvp.norm
        read_activation()  # normactconvp act
        read_conv()        # normactconvp.conv
        for si in range(nsub):
            skind = p.token()
            if skind == "transformer_attention_block":
                read_attn(bname + ".blockstack." + str(si))
            elif skind == "transformer_ffn_block":
                read_ffn()
            else:
                raise ValueError(skind)
        read_bn()          # normactconvq.norm
        read_activation()
        read_conv()        # normactconvq.conv

    m.attn = {}
    # 初始 conv + 全局 matmul
    read_conv()  # model.conv_spatial
    read_mat()   # model.linear_global
    for _ in range(numBlocks):
        read_nbt()
    # trunk 末端
    read_bn("model.norm_trunkfinal")
    read_activation()  # model.act_trunkfinal

    # policy head
    ph = p.token()
    assert ph == "model.policy_head"
    poc = p.int()
    m.policyOutChannels = poc
    p.ints(3)
    read_conv()   # conv1p
    read_conv()   # conv1g
    read_bn()     # biasg
    read_activation()  # actg
    read_mat()    # linear_g
    read_bn()     # bias2
    read_activation()  # act2
    read_conv()   # conv2p
    read_mat()    # linear_pass
    read_matbias() # linear_pass_bias
    read_activation()  # act_pass
    read_mat()    # linear_pass2

    # value head
    vh = p.token()
    assert vh == "model.value_head"
    p.ints(3)
    read_conv()   # conv1
    read_bn()     # bias1
    read_activation()  # act1
    read_mat()    # linear2
    read_matbias() # bias2
    read_activation()  # act2
    read_mat()    # linear_valuehead
    read_matbias() # bias_valuehead
    read_mat()    # linear_miscvaluehead
    read_matbias() # bias_miscvaluehead
    read_conv()   # conv_ownership

    # 权重转 torch (拷贝出可写内存后可释放文件缓冲)
    def T(a):
        return torch.from_numpy(np.array(a, dtype=np.float32))

    m.tconv = {k: (v[0], v[1], v[2], v[3], T(v[4])) for k, v in m.conv.items()}
    m.tmat = {k: (v[0], v[1], T(v[2])) for k, v in m.mat.items()}
    m.tbn = {k: (T(v[0]), T(v[1])) for k, v in m.bn.items()}
    m.trms = {k: (T(v[0]), v[1]) for k, v in m.rms.items()}
    m.trope = {k: (v[0], v[1], T(v[2])) for k, v in m.rope.items()}
    m.tbias = {k: T(v) for k, v in m.bias.items()}
    print(
        f"model loaded: {m.name} v{m.version}, "
        f"convs={len(m.tconv)} mats={len(m.tmat)} bns={len(m.tbn)} rms={len(m.trms)} "
        f"ropes={len(m.trope)} biases={len(m.tbias)}"
    )
    return m


# ---------------------------------------------------------------------------
# 前向计算


def silu(x):
    return x * torch.sigmoid(x)


def make_input():
    rng = np.random.RandomState(20240907)
    inp = rng.randint(0, 2, size=(22, XY, XY)).astype(np.float32)
    inp[0] = 1.0  # 通道0是 mask, 19x19 全在棋盘上
    glb = rng.uniform(-1, 1, size=19).astype(np.float32)
    return inp, glb


def forward(m: Model, inp: np.ndarray, glb: np.ndarray):
    inp_t = torch.from_numpy(np.ascontiguousarray(inp)).view(22, N)
    glb_t = torch.from_numpy(np.ascontiguousarray(glb))
    mask = inp_t[0].clone()  # (N,)
    S = float(mask.sum().item())
    d = math.sqrt(S) - 14.0

    def mm(name, x):
        # 文件布局 (ic,oc): y[oc,n] = sum_ic w[ic,oc]*x[ic,n]
        ic, oc, w = m.tmat[name]
        return w.view(ic, oc).t() @ x

    def conv1x1(name, x):
        if name in m.tmat:
            ic, oc, w = m.tmat[name]
            return w.view(ic, oc).t() @ x
        cy, cx, cic, coc, w = m.tconv[name]
        assert cy == 1 and cx == 1
        return w.view(cic, coc).t() @ x

    def bn_act_mask(x, name):
        ms, mb = m.tbn[name]
        y = x * ms[:, None] + mb[:, None]
        y = silu(y)
        return y * mask[None, :]

    def rmsnorm(x, name):
        w, eps = m.trms[name]
        c = x.shape[0]
        ss = (x * x).sum(dim=0, keepdim=True)  # (1,N)
        inv = torch.rsqrt(ss / c + eps)
        y = x * inv * w[:, None]
        return y * mask[None, :]

    snaps = {}

    # --- 初始层 ---
    cy, cx, cic, coc, wc = m.tconv["model.conv_spatial"]
    wc = wc.view(cy, cx, cic, coc).permute(3, 2, 0, 1).contiguous()
    trunk = F.conv2d(inp_t.view(1, 22, XY, XY), wc, padding=cy // 2).view(C_TRUNK, N)
    g = mm("model.linear_global", glb_t[:, None]).squeeze(1)  # (768,)
    trunk = trunk + g[:, None]
    snaps["trunk_after_initial"] = trunk.clone()

    # --- RoPE 位置 ---
    pos = torch.arange(N)
    py = (pos // XY).float()
    px = (pos % XY).float()

    def attn_block(mid, blk, sub):
        p = f"model.blocks.{blk}.blockstack.{sub}"
        h, kvh, qd, vd = m.attn["meta"][p]
        xn = rmsnorm(mid, p + ".norm1")
        q = mm(p + ".q_proj", xn)  # (384,N)
        k = mm(p + ".k_proj", xn)
        v = mm(p + ".v_proj", xn)
        # RoPE
        rh, rp, rf = m.trope[p + ".rope_freqs"]
        rf = rf.view(rh, rp, 2)
        ang = px[None, None, :] * rf[:, :, 0:1] + py[None, None, :] * rf[:, :, 1:2]
        cos = ang.cos()  # (kv,pairs,N)
        sin = ang.sin()

        def rot(x, nheads):
            x3 = x.view(nheads, N_PAIRS, 2, N)
            x0 = x3[:, :, 0, :]
            x1 = x3[:, :, 1, :]
            r0 = x0 * cos - x1 * sin
            r1 = x0 * sin + x1 * cos
            return torch.stack((r0, r1), dim=2).view(nheads * Q_HEAD_DIM, N)

        q = rot(q, h)
        k = rot(k, kvh)
        # attention
        qh = q.view(h, qd, N)
        kh = k.view(kvh, qd, N)
        vh = v.view(kvh, vd, N)
        scale = 1.0 / math.sqrt(qd)
        sc = torch.einsum("hdn,hdm->hnm", qh, kh) * scale  # (h,N,N)
        # 屏蔽棋盘外的 key
        keymask = torch.where(mask > 0, torch.zeros(N), torch.full((N,), float("-inf")))
        sc = sc + keymask[None, None, :]
        pr = torch.softmax(sc, dim=2)
        out = torch.einsum("hnm,hdm->hdn", pr, vh).view(h * vd, N)
        out = mm(p + ".out_proj", out)
        return out * mask[None, :]

    def ffn_block(mid, blk, sub):
        p = f"model.blocks.{blk}.blockstack.{sub}"
        xn = rmsnorm(mid, p + ".norm")
        a = mm(p + ".ffn_linear1", xn)
        g = mm(p + ".ffn_linear_gate", xn)
        hh = silu(a) * g
        out = mm(p + ".ffn_linear2", hh)
        return out * mask[None, :]

    for blk in range(N_BLOCKS):
        bp = f"model.blocks.{blk}"
        mid = conv1x1(
            bp + ".normactconvp.conv", bn_act_mask(trunk, bp + ".normactconvp.norm")
        )
        if blk == 0:
            snaps["blk0_mid_after_pre"] = mid.clone()
        for sub in range(N_SUB):
            if sub % 2 == 0:
                mid = mid + attn_block(mid, blk, sub)
            else:
                mid = mid + ffn_block(mid, blk, sub)
            if blk == 0 and sub == 0:
                snaps["blk0_mid_after_attn0"] = mid.clone()
            if blk == 0 and sub == 1:
                snaps["blk0_mid_after_ffn0"] = mid.clone()
        if blk == 0:
            snaps["blk0_mid_after_stack"] = mid.clone()
        resid = conv1x1(
            bp + ".normactconvq.conv", bn_act_mask(mid, bp + ".normactconvq.norm")
        )
        trunk = trunk + resid
        snaps[f"trunk_after_block{blk}"] = trunk.clone()

    # --- trunk 末端 ---
    trunk = bn_act_mask(trunk, "model.norm_trunkfinal")
    snaps["trunk_after_tip"] = trunk.clone()

    # --- policy head ---
    p1 = conv1x1("model.policy_head.conv1p", trunk)  # (96,N)
    g1 = conv1x1("model.policy_head.conv1g", trunk)  # (96,N)
    g1 = bn_act_mask(g1, "model.policy_head.biasg")
    mean = g1.sum(dim=1) / S  # (96,)
    p2_ = mean * (d / 10.0)
    p3_ = (g1 + (mask - 1.0)[None, :]).max(dim=1).values
    gp = torch.cat((mean, p2_, p3_))  # (288,)
    g1bias = mm("model.policy_head.linear_g", gp[:, None]).squeeze(1)  # (96,)
    p1 = p1 + g1bias[:, None]
    p1 = bn_act_mask(p1, "model.policy_head.bias2")
    policy = conv1x1("model.policy_head.conv2p", p1)  # (2,N)
    pp = mm("model.policy_head.linear_pass", gp[:, None]).squeeze(1)
    pp = pp + m.tbias["model.policy_head.linear_pass_bias"]
    pp = silu(pp)
    policy_pass = mm("model.policy_head.linear_pass2", pp[:, None]).squeeze(1)  # (2,)

    # --- value head ---
    v1 = conv1x1("model.value_head.conv1", trunk)  # (192,N)
    v1 = bn_act_mask(v1, "model.value_head.bias1")
    vmean = v1.sum(dim=1) / S
    vmean = torch.cat(
        (vmean, vmean * (d / 10.0), vmean * (d * d / 100.0 - 0.1))
    )  # (576,)
    v2 = mm("model.value_head.linear2", vmean[:, None]).squeeze(1)
    v2 = silu(v2 + m.tbias["model.value_head.bias2"])  # (192,)
    value = mm("model.value_head.linear_valuehead", v2[:, None]).squeeze(1)
    value = value + m.tbias["model.value_head.bias_valuehead"]  # (3,)
    misc = mm("model.value_head.linear_miscvaluehead", v2[:, None]).squeeze(1)
    misc = misc + m.tbias["model.value_head.bias_miscvaluehead"]  # (6,)
    ownership = conv1x1("model.value_head.conv_ownership", v1)  # (1,N)

    snaps["policy"] = policy
    snaps["policy_pass"] = policy_pass
    snaps["value"] = value
    snaps["miscvalue"] = misc
    snaps["ownership"] = ownership
    return snaps, S


# ---------------------------------------------------------------------------
# 输出布局 (main.c 按同样顺序写 out_c.bin)

def sections():
    secs = [
        ("trunk_after_initial", C_TRUNK * N),
        ("blk0_mid_after_pre", C_MID * N),
        ("blk0_mid_after_attn0", C_MID * N),
        ("blk0_mid_after_ffn0", C_MID * N),
        ("blk0_mid_after_stack", C_MID * N),
    ]
    for b in range(N_BLOCKS):
        secs.append((f"trunk_after_block{b}", C_TRUNK * N))
    secs += [
        ("trunk_after_tip", C_TRUNK * N),
        ("policy", 2 * N),
        ("policy_pass", 2),
        ("value", 3),
        ("miscvalue", 6),
        ("ownership", N),
    ]
    return secs


def main():
    if not os.path.exists(BINFILE):
        print("decompressing model...")
        with gzip.open(GZFILE, "rb") as fin, open(BINFILE, "wb") as fout:
            shutil.copyfileobj(fin, fout)

    m = load_model()

    inp_path = os.path.join(DIR, "input.bin")
    glb_path = os.path.join(DIR, "input_global.bin")
    if os.path.exists(inp_path) and os.path.exists(glb_path):
        inp = np.fromfile(inp_path, dtype=np.float32).reshape(22, XY, XY)
        glb = np.fromfile(glb_path, dtype=np.float32)
        print("loaded existing input.bin / input_global.bin")
    else:
        inp, glb = make_input()
        inp.tofile(inp_path)
        glb.tofile(glb_path)
        print("generated input.bin / input_global.bin (seed=20240907)")

    snaps, S = forward(m, inp, glb)

    out = []
    for name, cnt in sections():
        arr = snaps[name].reshape(-1).numpy()
        assert arr.size == cnt, (name, arr.size, cnt)
        out.append(arr.astype(np.float32))
    allout = np.concatenate(out)
    outpath = os.path.join(DIR, "out_python.bin")
    allout.tofile(outpath)
    print(f"wrote {outpath} ({allout.size} floats)")

    # 摘要
    pol = snaps["policy"]
    logits = pol[0] - pol[0].max()
    probs = np.exp(logits.numpy())
    probs /= probs.sum()
    top = np.argsort(probs)[::-1][:5]
    print("value:", np.round(snaps["value"].numpy(), 4))
    print("miscvalue:", np.round(snaps["miscvalue"].numpy(), 4))
    print("policy_pass:", np.round(snaps["policy_pass"].numpy(), 4))
    print("ownership mean/max/min: %.4f %.4f %.4f" % (
        snaps["ownership"].mean(), snaps["ownership"].max(), snaps["ownership"].min()))
    print("top5 moves (pos, prob):", [(int(t), round(float(probs[t]), 4)) for t in top])
    print("trunk_after_tip  abs mean %.4f  max %.4f" % (
        abs(snaps["trunk_after_tip"]).mean(), abs(snaps["trunk_after_tip"]).max()))


def compare():
    a = np.fromfile(os.path.join(DIR, "out_python.bin"), dtype=np.float32)
    b = np.fromfile(os.path.join(DIR, "out_c.bin"), dtype=np.float32)
    assert a.size == b.size, (a.size, b.size)
    off = 0
    print(f"{'section':24s} {'n':>9s} {'maxabs':>12s} {'relmax':>12s}")
    for name, cnt in sections():
        x = a[off : off + cnt]
        y = b[off : off + cnt]
        off += cnt
        d = np.abs(x - y)
        denom = max(np.abs(x).max(), 1e-9)
        print(f"{name:24s} {cnt:9d} {d.max():12.3e} {d.max()/denom:12.3e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    else:
        main()
