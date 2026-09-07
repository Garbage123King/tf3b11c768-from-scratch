#!/usr/bin/env python3
"""Sequential parser for KataGo .bin model files (desc.cpp grammar).
Dumps a clean structure summary with exact bin offsets/lengths."""
import struct, sys, io

class Parser:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.log = []

    def token(self):
        # skip whitespace
        while self.pos < len(self.data):
            c = self.data[self.pos]
            if c in b' \t\r\n':
                self.pos += 1
            else:
                break
        start = self.pos
        while self.pos < len(self.data):
            c = self.data[self.pos]
            if c in b' \t\r\n':
                break
            self.pos += 1
        return self.data[start:self.pos].decode('ascii')

    def int(self):
        return int(self.token())

    def float(self):
        return float(self.token())

    def floats(self, n, desc):
        # skip to '@'
        nskip = 0
        while self.data[self.pos] != ord('@'):
            self.pos += 1
            nskip += 1
            assert nskip <= 100, "too much whitespace before @BIN@"
        hdr = self.data[self.pos:self.pos+5]
        assert hdr == b'@BIN@', f"expected @BIN@ got {hdr}"
        self.pos += 5
        raw = self.data[self.pos:self.pos+4*n]
        assert len(raw) == 4*n, f"not enough data for {desc}: want {n} floats"
        self.pos += 4*n
        vals = struct.unpack(f'<{n}f', raw)
        self.log.append((desc, self.pos - 4*n, n, vals))
        return vals

    def peek_nonws(self, k=40):
        p = self.pos
        out = []
        while p < len(self.data) and len(out) < k:
            c = self.data[p]
            if c in b' \t\r\n':
                if out and out[-1] == ' ':
                    p += 1
                    continue
                out.append(' ')
                p += 1
                continue
            if 32 <= c < 127:
                out.append(chr(c))
            else:
                out.append('.')
            p += 1
        return ''.join(out())

    def context(self, k=60):
        p = self.pos
        out = []
        while p < len(self.data) and len(out) < k:
            c = self.data[p]
            if 32 <= c < 127 or c in (10,):
                out.append(chr(c))
            else:
                out.append('.')
            p += 1
        return ''.join(out)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'model.bin'
    with open(path, 'rb') as f:
        data = f.read()
    p = Parser(data)

    name = p.token()
    version = p.int()
    nbin = p.int()
    nglb = p.int()
    print(f"MODEL: {name} version={version} binfeat={nbin} glbfeat={nglb}")
    if version >= 13:
        mults = [p.float() for _ in range(7)]
        print(f"multipliers: {mults}")
    if version >= 15:
        metaenc = p.int()
        v17flags = [p.int() for _ in range(7)]
        print(f"metaenc={metaenc} flags={v17flags}")

    # trunk ("trunk" itself is the name token)
    t = p.token(); assert t == 'trunk', t
    numBlocks = p.int(); trunkC = p.int(); midC = p.int(); regC = p.int(); gpoolC = p.int(); gpoolC2 = p.int()
    print(f"trunk: blocks={numBlocks} c_trunk={trunkC} c_mid={midC} reg={regC} gpool={gpoolC},{gpoolC2}")
    if version >= 15:
        normkind = p.int(); unused = [p.int() for _ in range(5)]
        print(f"  trunkNormKind={normkind} unused={unused}")
    # initial conv
    cname = p.token(); cy = p.int(); cx = p.int(); cic = p.int(); coc = p.int(); dy = p.int(); dx = p.int()
    w = p.floats(cy*cx*cic*coc, f"{cname} conv {cy}x{cx} {cic}->{coc}")
    print(f"  {cname}: conv {cy}x{cx} {cic}->{coc} dil={dy},{dx} ({len(w)} floats)")
    # initial matmul
    mname = p.token(); mic = p.int(); moc = p.int()
    w = p.floats(mic*moc, f"{mname} matmul {mic}->{moc}")
    print(f"  {mname}: matmul {mic}->{moc}")

    for bi in range(numBlocks):
        kind = p.token()
        if kind == 'nested_bottleneck_block':
            bname = p.token(); nsub = p.int()
            print(f"  BLOCK {bi}: {kind} {bname} nsub={nsub}")
            # preBN (4 bins)
            nn = p.token(); assert nn.endswith('.norm'), nn
            nch = p.int(); eps = p.float(); hs = p.int(); hb = p.int()
            bins = [p.floats(nch, f"{nn}[{i}]") for i in range(2 + hs + hb)]
            print(f"    {nn}: c={nch} eps={eps} hasScale={hs} hasBias={hb} ({len(bins)} bins)")
            an = p.token(); act = p.token()
            cn = p.token(); cy=p.int(); cx=p.int(); cic=p.int(); coc=p.int(); dy=p.int(); dx=p.int()
            w = p.floats(cy*cx*cic*coc, f"{cn} conv")
            print(f"    {cn}: conv {cy}x{cx} {cic}->{coc}")
            for si in range(nsub):
                skind = p.token()
                if skind == 'transformer_attention_block':
                    an2 = p.token(); nh=p.int(); nkv=p.int(); qd=p.int(); vd=p.int(); ur=p.int(); lr=p.int()
                    nn1 = p.token(); nc = p.int(); neps = p.float()
                    w1 = p.floats(nc, f"{nn1} rmsnorm w")
                    print(f"    ATTN {an2}: h={nh} kv={nkv} qd={qd} vd={vd} rope={ur} learnable={lr}")
                    print(f"      {nn1}: rmsnorm c={nc} eps={neps}")
                    for proj in ('q','k','v','out'):
                        pn = p.token(); pi = p.int(); po = p.int()
                        w = p.floats(pi*po, f"{pn} matmul")
                        print(f"      {pn}: {pi}->{po}")
                    rn = p.token(); rh=p.int(); rp=p.int(); r2=p.int()
                    w = p.floats(rh*rp*r2, f"{rn} ropefreqs")
                    print(f"      {rn}: rope_freqs {rh}x{rp}x{r2}")
                elif skind == 'transformer_ffn_block':
                    fn = p.token(); fc = p.int(); ff = p.int(); sw = p.int()
                    nn2 = p.token(); nc = p.int(); neps = p.float()
                    w1 = p.floats(nc, f"{nn2} rmsnorm w")
                    print(f"    FFN {fn}: c={fc} ffn={ff} swiglu={sw}")
                    print(f"      {nn2}: rmsnorm c={nc} eps={neps}")
                    for ln in ('ffn_linear1','ffn_linear_gate','ffn_linear2'):
                        pn = p.token(); pi = p.int(); po = p.int()
                        w = p.floats(pi*po, f"{pn} matmul")
                        print(f"      {pn}: {pi}->{po}")
                else:
                    raise ValueError(f"unknown subblock kind {skind} at {p.context()}")
            # postBN
            nn = p.token(); assert nn.endswith('.norm'), nn
            nch = p.int(); eps = p.float(); hs = p.int(); hb = p.int()
            bins = [p.floats(nch, f"{nn}[{i}]") for i in range(2 + hs + hb)]
            print(f"    {nn}: c={nch} eps={eps} hasScale={hs} hasBias={hb} ({len(bins)} bins)")
            an = p.token(); act = p.token()
            cn = p.token(); cy=p.int(); cx=p.int(); cic=p.int(); coc=p.int(); dy=p.int(); dx=p.int()
            w = p.floats(cy*cx*cic*coc, f"{cn} conv")
            print(f"    {cn}: conv {cy}x{cx} {cic}->{coc}")
        else:
            raise ValueError(f"unknown block kind {kind} at {p.context()}")

    # trunk tip
    nn = p.token()
    if nn == 'model.norm_trunkfinal':
        nch = p.int(); eps = p.float(); hs = p.int(); hb = p.int()
        bins = [p.floats(nch, f"{nn}[{i}]") for i in range(2 + hs + hb)]
        print(f"  {nn}: c={nch} eps={eps} hasScale={hs} hasBias={hb} ({len(bins)} bins)")
        # stats
        m = bins[0]; v = bins[1]
        print(f"    mean[:5]={list(m[:5])} var[:5]={list(v[:5])}")
        an = p.token(); act = p.token()
        print(f"  {an}: {act}")
    else:
        raise ValueError(f"unexpected trunk tip {nn}")

    # policy head
    ph = p.token(); assert ph == 'model.policy_head', ph
    if version >= 17:
        poc = p.int(); unused = [p.int() for _ in range(3)]
        print(f"POLICY: outChannels={poc} unused={unused}")
    cn = p.token(); cy=p.int(); cx=p.int(); cic=p.int(); coc=p.int(); dy=p.int(); dx=p.int()
    w = p.floats(cy*cx*cic*coc, f"{cn}")
    print(f"  {cn}: conv {cy}x{cx} {cic}->{coc}")
    cn = p.token(); cy=p.int(); cx=p.int(); cic=p.int(); coc=p.int(); dy=p.int(); dx=p.int()
    w = p.floats(cy*cx*cic*coc, f"{cn}")
    print(f"  {cn}: conv {cy}x{cx} {cic}->{coc}")
    for bname in ('model.policy_head.biasg',):
        nn = p.token(); nch=p.int(); eps=p.float(); hs=p.int(); hb=p.int()
        bins = [p.floats(nch, f"{nn}[{i}]") for i in range(2+hs+hb)]
        print(f"  {nn}: c={nch} eps={eps} hasScale={hs} hasBias={hb}")
    an = p.token(); act = p.token(); print(f"  {an}: {act}")
    pn = p.token(); pi=p.int(); po=p.int(); w = p.floats(pi*po, f"{pn}")
    print(f"  {pn}: matmul {pi}->{po}")
    nn = p.token(); nch=p.int(); eps=p.float(); hs=p.int(); hb=p.int()
    bins = [p.floats(nch, f"{nn}[{i}]") for i in range(2+hs+hb)]
    print(f"  {nn}: c={nch} eps={eps} hasScale={hs} hasBias={hb}")
    an = p.token(); act = p.token(); print(f"  {an}: {act}")
    cn = p.token(); cy=p.int(); cx=p.int(); cic=p.int(); coc=p.int(); dy=p.int(); dx=p.int()
    w = p.floats(cy*cx*cic*coc, f"{cn}")
    print(f"  {cn}: conv {cy}x{cx} {cic}->{coc} (policyOutChannels={coc})")
    pn = p.token(); pi=p.int(); po=p.int(); w = p.floats(pi*po, f"{pn}")
    print(f"  {pn}: matmul {pi}->{po}")
    pn = p.token(); po=p.int(); w = p.floats(po, f"{pn}")
    print(f"  {pn}: bias {po}")
    an = p.token(); act = p.token(); print(f"  {an}: {act}")
    pn = p.token(); pi=p.int(); po=p.int(); w = p.floats(pi*po, f"{pn}")
    print(f"  {pn}: matmul {pi}->{po}")

    # value head
    vh = p.token(); assert vh == 'model.value_head', vh
    if version >= 17:
        unused = [p.int() for _ in range(3)]
        print(f"VALUE: unused={unused}")
    cn = p.token(); cy=p.int(); cx=p.int(); cic=p.int(); coc=p.int(); dy=p.int(); dx=p.int()
    w = p.floats(cy*cx*cic*coc, f"{cn}")
    print(f"  {cn}: conv {cy}x{cx} {cic}->{coc}")
    nn = p.token(); nch=p.int(); eps=p.float(); hs=p.int(); hb=p.int()
    bins = [p.floats(nch, f"{nn}[{i}]") for i in range(2+hs+hb)]
    print(f"  {nn}: c={nch} eps={eps} hasScale={hs} hasBias={hb}")
    an = p.token(); act = p.token(); print(f"  {an}: {act}")
    pn = p.token(); pi=p.int(); po=p.int(); w = p.floats(pi*po, f"{pn}")
    print(f"  {pn}: matmul {pi}->{po}")
    pn = p.token(); po=p.int(); w = p.floats(po, f"{pn}")
    print(f"  {pn}: bias {po}")
    an = p.token(); act = p.token(); print(f"  {an}: {act}")
    for ln in ('model.value_head.linear_valuehead','model.value_head.bias_valuehead',
               'model.value_head.linear_miscvaluehead','model.value_head.bias_miscvaluehead'):
        pn = p.token()
        if 'bias' in pn:
            b=p.int()
            w = p.floats(b, f"{pn}")
            print(f"  {pn}: bias {b}")
        else:
            a=p.int(); b=p.int()
            w = p.floats(a*b, f"{pn}")
            print(f"  {pn}: matmul {a}->{b}")
    cn = p.token(); cy=p.int(); cx=p.int(); cic=p.int(); coc=p.int(); dy=p.int(); dx=p.int()
    w = p.floats(cy*cx*cic*coc, f"{cn}")
    print(f"  {cn}: conv {cy}x{cx} {cic}->{coc}")

    print(f"\nParsed OK. pos={p.pos} filelen={len(data)} leftover={len(data)-p.pos}")
    # save log
    with open('bins.txt','w') as f:
        for desc, off, n, vals in p.log:
            f.write(f"{off}\t{n}\t{desc}\n")
    print(f"Wrote {len(p.log)} bins to bins.txt")

if __name__ == '__main__':
    main()
