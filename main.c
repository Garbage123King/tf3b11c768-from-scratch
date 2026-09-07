/* Standalone C inference (float32) for kata1-tf3-b11c768-s11001M-d5973M.bin.gz
 * Semantics identical to main.py (KataGo desc.cpp/eigenbackend.cpp).
 * Build: gcc -O2 -fopenmp -o main.exe main.c -lz
 * Run  : ./main.exe  (needs input.bin/input_global.bin + model.bin,
 *        writes out_c.bin with same section layout as out_python.bin)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define XY 19
#define N (XY * XY)
#define C_TRUNK 768
#define C_MID 384
#define N_HEADS 12
#define QD 32
#define NPAIRS (QD / 2)
#define FFN_DIM 1152
#define N_BLOCKS 11
#define N_SUB 6

static void die(const char *msg) {
  fprintf(stderr, "FATAL: %s\n", msg);
  exit(1);
}

static float *xfmalloc(size_t n) {
  float *p = (float *)malloc(sizeof(float) * n);
  if (!p) die("malloc failed");
  return p;
}

typedef struct { int ic, oc; float *w; } Mat;
typedef struct { float *ms, *mb; } BN;
typedef struct { float *w; double eps; } RMS;
typedef struct { float *b; } BiasV;
typedef struct { RMS norm; Mat q, k, v, out; float *rope; } Attn;
typedef struct { RMS norm; Mat l1, lg, l2; } FFN;
typedef struct {
  BN preBN; Mat preConv;
  Attn attn[3]; FFN ffn[3];
  BN postBN; Mat postConv;
} Block;
typedef struct {
  float *conv_spatial; Mat linear_global;
  Block blocks[N_BLOCKS];
  BN trunkfinal;
  Mat conv1p, conv1g; BN biasg; Mat linear_g; BN bias2; Mat conv2p;
  Mat linear_pass; BiasV linear_pass_bias; Mat linear_pass2;
  Mat conv1; BN bias1; Mat linear2; BiasV bias2v;
  Mat linear_valuehead; BiasV bias_valuehead;
  Mat linear_miscvaluehead; BiasV bias_miscvaluehead;
  Mat conv_ownership;
} Model;
/* ------------- model file parser (same as main.py Loader) ------------- */
static unsigned char *g_data = NULL;
static long g_len = 0, g_pos = 0;

static void skip_ws(void) {
  while (g_pos < g_len) {
    unsigned char c = g_data[g_pos];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') g_pos++;
    else break;
  }
}

static void token(char *buf, int cap) {
  skip_ws();
  int i = 0;
  while (g_pos < g_len) {
    unsigned char c = g_data[g_pos];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') break;
    if (i < cap - 1) buf[i++] = (char)c;
    g_pos++;
  }
  buf[i] = 0;
}

static void tok_expect(const char *expect) {
  char b[256];
  token(b, sizeof b);
  if (strcmp(b, expect) != 0) {
    fprintf(stderr, "FATAL: expect token ");
    fprintf(stderr, "%s", expect);
    fprintf(stderr, " got ");
    fprintf(stderr, "%s\n", b);
    exit(1);
  }
}

static long tok_int(void) {
  char b[128]; token(b, sizeof b); return strtol(b, NULL, 10);
}

static double tok_flt(void) {
  char b[128]; token(b, sizeof b); return strtod(b, NULL);
}

/* read @BIN@ then n floats, copy into aligned malloc */
static float *tok_floats(long n) {
  while (g_pos < g_len && g_data[g_pos] != '@') g_pos++;
  if (g_pos + 5 > g_len || memcmp(g_data + g_pos, "@BIN@", 5) != 0)
    die("expected @BIN@ marker");
  g_pos += 5;
  if (g_pos + 4 * n > g_len) die("not enough binary data");
  float *p = xfmalloc((size_t)n);
  memcpy(p, g_data + g_pos, sizeof(float) * (size_t)n);
  g_pos += 4 * n;
  return p;
}
static void read_bn(BN *b) {
  char name[256]; token(name, sizeof name);
  long c = tok_int();
  double eps = tok_flt();
  long hs = tok_int(), hb = tok_int();
  float *mean = tok_floats(c);
  float *var = tok_floats(c);
  float *scale = hs ? tok_floats(c) : NULL;
  float *bias = hb ? tok_floats(c) : NULL;
  b->ms = xfmalloc((size_t)c);
  b->mb = xfmalloc((size_t)c);
  for (long i = 0; i < c; i++) {
    float sc = hs ? scale[i] : 1.0f;
    float bi = hb ? bias[i] : 0.0f;
    float ms = sc / sqrtf(var[i] + (float)eps);
    b->ms[i] = ms;
    b->mb[i] = bi - ms * mean[i];
  }
  free(mean); free(var); free(scale); free(bias);
}

static void read_mat(Mat *m) {
  char name[256]; token(name, sizeof name);
  m->ic = (int)tok_int();
  m->oc = (int)tok_int();
  m->w = tok_floats((long)m->ic * m->oc);
}

static void read_rms(RMS *r) {
  char name[256]; token(name, sizeof name);
  long c = tok_int();
  r->eps = tok_flt();
  r->w = tok_floats(c);
}

static void read_bias(BiasV *v) {
  char name[256]; token(name, sizeof name);
  long c = tok_int();
  v->b = tok_floats(c);
}

static void read_act(void) {
  char n1[256], n2[64];
  token(n1, sizeof n1); token(n2, sizeof n2);
  if (strcmp(n2, "ACTIVATION_SILU") != 0) die("unexpected activation");
}

static void read_conv(Mat *m) {
  char name[256]; token(name, sizeof name);
  long cy = tok_int(), cx = tok_int(), cic = tok_int(), coc = tok_int();
  tok_int(); tok_int();
  m->ic = (int)(cy * cx * cic);
  m->oc = (int)coc;
  m->w = tok_floats(cy * cx * cic * coc);
}

static void read_attn(Attn *a) {
  char name[256]; token(name, sizeof name);
  long h = tok_int(), kv = tok_int(), qd = tok_int(), vd = tok_int();
  long ur = tok_int(), lr = tok_int();
  if (h != N_HEADS || kv != N_HEADS || qd != QD || vd != QD || ur != 1 || lr != 1)
    die("unexpected attention params");
  read_rms(&a->norm);
  read_mat(&a->q); read_mat(&a->k); read_mat(&a->v); read_mat(&a->out);
  char rname[256]; token(rname, sizeof rname);
  long rh = tok_int(), rp = tok_int(), r2 = tok_int();
  if (rh != N_HEADS || rp != NPAIRS || r2 != 2) die("unexpected rope freqs shape");
  a->rope = tok_floats(rh * rp * r2);
}

static void read_ffn(FFN *f) {
  char name[256]; token(name, sizeof name);
  long c = tok_int(), fd = tok_int(), sw = tok_int();
  if (c != C_MID || fd != FFN_DIM || sw != 1) die("unexpected ffn params");
  read_rms(&f->norm);
  read_mat(&f->l1); read_mat(&f->lg); read_mat(&f->l2);
}

static void read_block(Block *blk) {
  char kind[64], bname[256];
  token(kind, sizeof kind);
  if (strcmp(kind, "nested_bottleneck_block") != 0) die("unexpected block kind");
  token(bname, sizeof bname);
  if (tok_int() != N_SUB) die("unexpected nsub");
  read_bn(&blk->preBN);
  read_act();
  read_conv(&blk->preConv);
  for (int s = 0; s < N_SUB; s++) {
    char skind[64]; token(skind, sizeof skind);
    if (strcmp(skind, "transformer_attention_block") == 0) read_attn(&blk->attn[s / 2]);
    else if (strcmp(skind, "transformer_ffn_block") == 0) read_ffn(&blk->ffn[(s - 1) / 2]);
    else die("unexpected subblock kind");
  }
  read_bn(&blk->postBN);
  read_act();
  read_conv(&blk->postConv);
}
static void load_model(Model *m, const char *binfile) {
  FILE *f = fopen(binfile, "rb");
  if (!f) die("cannot open model.bin");
  fseek(f, 0, SEEK_END);
  g_len = ftell(f);
  fseek(f, 0, SEEK_SET);
  g_data = (unsigned char *)malloc((size_t)g_len);
  if (!g_data) die("malloc failed for model file");
  if (fread(g_data, 1, (size_t)g_len, f) != (size_t)g_len) die("short read");
  fclose(f);
  g_pos = 0;
  char name[256];
  token(name, sizeof name);
  long version = tok_int(), nbin = tok_int(), nglb = tok_int();
  if (version != 17 || nbin != 22 || nglb != 19) die("unexpected model header");
  for (int i = 0; i < 7; i++) tok_flt();
  tok_int();
  for (int i = 0; i < 7; i++) tok_int();
  tok_expect("trunk");
  long numBlocks = tok_int(), trunkC = tok_int(), midC = tok_int();
  tok_int(); tok_int(); tok_int();
  if (numBlocks != N_BLOCKS || trunkC != C_TRUNK || midC != C_MID) die("unexpected trunk dims");
  tok_int();
  for (int i = 0; i < 5; i++) tok_int();
  {
    char cname[256]; token(cname, sizeof cname);
    long cy = tok_int(), cx = tok_int(), cic = tok_int(), coc = tok_int();
    tok_int(); tok_int();
    if (cy != 3 || cx != 3 || cic != 22 || coc != C_TRUNK) die("unexpected conv_spatial");
    m->conv_spatial = tok_floats(cy * cx * cic * coc);
  }
  read_mat(&m->linear_global);
  for (int b = 0; b < N_BLOCKS; b++) read_block(&m->blocks[b]);
  read_bn(&m->trunkfinal);
  read_act();
  tok_expect("model.policy_head");
  if (tok_int() != 2) die("unexpected policyOutChannels");
  for (int i = 0; i < 3; i++) tok_int();
  read_conv(&m->conv1p);
  read_conv(&m->conv1g);
  read_bn(&m->biasg);
  read_act();
  read_mat(&m->linear_g);
  read_bn(&m->bias2);
  read_act();
  read_conv(&m->conv2p);
  read_mat(&m->linear_pass);
  read_bias(&m->linear_pass_bias);
  read_act();
  read_mat(&m->linear_pass2);
  tok_expect("model.value_head");
  for (int i = 0; i < 3; i++) tok_int();
  read_conv(&m->conv1);
  read_bn(&m->bias1);
  read_act();
  read_mat(&m->linear2);
  read_bias(&m->bias2v);
  read_act();
  read_mat(&m->linear_valuehead);
  read_bias(&m->bias_valuehead);
  read_mat(&m->linear_miscvaluehead);
  read_bias(&m->bias_miscvaluehead);
  read_conv(&m->conv_ownership);
  free(g_data);
  g_data = NULL;
  printf("model loaded: %s v%ld\n", name, version);
}
/* ------------------------- forward operators ------------------------- */
static float silu(float t) { return t * (1.0f / (1.0f + expf(-t))); }

/* y[oc][j] = sum_ic w[ic][oc] * x[ic][j] */
static void matmul(const float *w, const float *x, float *y, int nic, int noc, int n) {
#pragma omp parallel for schedule(static)
  for (int o = 0; o < noc; o++) {
    float *yr = y + (size_t)o * n;
    memset(yr, 0, sizeof(float) * (size_t)n);
    for (int i = 0; i < nic; i++) {
      float wv = w[(size_t)i * noc + o];
      const float *xi = x + (size_t)i * n;
      for (int j = 0; j < n; j++)
        yr[j] += wv * xi[j];
    }
  }
}

static void bn_silu_mask(const float *x, const BN *bn, const float *mask, float *y, int c, int n) {
  for (int ci = 0; ci < c; ci++) {
    float ms = bn->ms[ci], mb = bn->mb[ci];
    const float *xr = x + (size_t)ci * n;
    float *yr = y + (size_t)ci * n;
    for (int j = 0; j < n; j++) {
      float t = xr[j] * ms + mb;
      yr[j] = silu(t) * mask[j];
    }
  }
}

static void rmsnorm(const float *x, const RMS *r, const float *mask, float *y, int c, int n) {
  for (int j = 0; j < n; j++) {
    if (mask[j] == 0.0f) {
      for (int ci = 0; ci < c; ci++) y[(size_t)ci * n + j] = 0.0f;
      continue;
    }
    float ss = 0.0f;
    for (int ci = 0; ci < c; ci++) {
      float v = x[(size_t)ci * n + j];
      ss += v * v;
    }
    float inv = 1.0f / sqrtf(ss / (float)c + (float)r->eps);
    for (int ci = 0; ci < c; ci++)
      y[(size_t)ci * n + j] = x[(size_t)ci * n + j] * inv * r->w[ci];
  }
}
typedef struct {
  float *xn, *q, *k, *v, *att, *out, *sc;
} AttnScratch;

static void attn_block(const Attn *a, const float *mid, const float *px, const float *py,
                       const float *mask, float *out, AttnScratch *s) {
  rmsnorm(mid, &a->norm, mask, s->xn, C_MID, N);
  matmul(a->q.w, s->xn, s->q, C_MID, C_MID, N);
  matmul(a->k.w, s->xn, s->k, C_MID, C_MID, N);
  matmul(a->v.w, s->xn, s->v, C_MID, C_MID, N);

  /* RoPE: pairs (2p,2p+1) rotated; angle = px*fx + py*fy */
  for (int h = 0; h < N_HEADS; h++) {
    for (int p = 0; p < NPAIRS; p++) {
      float fx = a->rope[(h * NPAIRS + p) * 2 + 0];
      float fy = a->rope[(h * NPAIRS + p) * 2 + 1];
      int c0 = h * QD + p * 2;
      for (int j = 0; j < N; j++) {
        float ang = px[j] * fx + py[j] * fy;
        float cs = cosf(ang), sn = sinf(ang);
        float q0 = s->q[(size_t)c0 * N + j], q1 = s->q[(size_t)(c0 + 1) * N + j];
        float k0 = s->k[(size_t)c0 * N + j], k1 = s->k[(size_t)(c0 + 1) * N + j];
        s->q[(size_t)c0 * N + j] = q0 * cs - q1 * sn;
        s->q[(size_t)(c0 + 1) * N + j] = q0 * sn + q1 * cs;
        s->k[(size_t)c0 * N + j] = k0 * cs - k1 * sn;
        s->k[(size_t)(c0 + 1) * N + j] = k0 * sn + k1 * cs;
      }
    }
  }

  const float scale = 1.0f / sqrtf((float)QD);
  for (int h = 0; h < N_HEADS; h++) {
    /* sc[nq][m] = sum_d q[h][d][nq]*k[h][d][m] */
    memset(s->sc, 0, sizeof(float) * (size_t)N * N);
    for (int d = 0; d < QD; d++) {
      const float *qr = s->q + (size_t)(h * QD + d) * N;
      const float *kr = s->k + (size_t)(h * QD + d) * N;
      for (int nq = 0; nq < N; nq++) {
        float qv = qr[nq];
        float *scr = s->sc + (size_t)nq * N;
        for (int m = 0; m < N; m++)
          scr[m] += qv * kr[m];
      }
    }
    for (int nq = 0; nq < N; nq++) {
      float *scr = s->sc + (size_t)nq * N;
      float mx = -INFINITY;
      for (int m = 0; m < N; m++) {
        float v = scr[m] * scale;
        if (mask[m] <= 0.0f) v = -INFINITY;
        scr[m] = v;
        if (v > mx) mx = v;
      }
      float sum = 0.0f;
      for (int m = 0; m < N; m++) {
        float e = expf(scr[m] - mx);
        scr[m] = e;
        sum += e;
      }
      float inv = 1.0f / sum;
      for (int m = 0; m < N; m++) scr[m] *= inv;
    }
    /* att[h][d][nq] = sum_m pr[nq][m]*v[h][d][m] */
    for (int d = 0; d < QD; d++) {
      const float *vr = s->v + (size_t)(h * QD + d) * N;
      float *orr = s->att + (size_t)(h * QD + d) * N;
      for (int nq = 0; nq < N; nq++) {
        const float *prr = s->sc + (size_t)nq * N;
        float acc = 0.0f;
        for (int m = 0; m < N; m++) acc += prr[m] * vr[m];
        orr[nq] = acc;
      }
    }
  }

  matmul(a->out.w, s->att, s->out, C_MID, C_MID, N);
  for (size_t i = 0; i < (size_t)C_MID * N; i++)
    out[i] = s->out[i] * mask[i % N];
}

static void ffn_block(const FFN *f, const float *mid, const float *mask, float *out,
                      float *xn, float *a, float *g) {
  rmsnorm(mid, &f->norm, mask, xn, C_MID, N);
  matmul(f->l1.w, xn, a, C_MID, FFN_DIM, N);
  matmul(f->lg.w, xn, g, C_MID, FFN_DIM, N);
  for (size_t i = 0; i < (size_t)FFN_DIM * N; i++)
    a[i] = silu(a[i]) * g[i];
  matmul(f->l2.w, a, out, FFN_DIM, C_MID, N);
  for (size_t i = 0; i < (size_t)C_MID * N; i++)
    out[i] *= mask[i % N];
}
static void decompress_gz(const char *gzfile, const char *binfile) {
  FILE *fout = fopen(binfile, "wb");
  if (!fout) die("cannot create model.bin");
  gzFile in = gzopen(gzfile, "rb");
  if (!in) die("cannot open model .gz");
  char buf[65536];
  int bytes;
  while ((bytes = gzread(in, buf, sizeof buf)) > 0)
    fwrite(buf, 1, bytes, fout);
  gzclose(in);
  fclose(fout);
  printf("decompressed %s -> %s\n", gzfile, binfile);
}

int main(void) {
  const char *gzfile = "kata1-tf3-b11c768-s11001M-d5973M.bin.gz";
  const char *binfile = "model.bin";

  FILE *test = fopen(binfile, "rb");
  if (!test) decompress_gz(gzfile, binfile);
  else fclose(test);

  Model *m = (Model *)malloc(sizeof(Model));
  if (!m) die("malloc model failed");
  load_model(m, binfile);

  float *input = xfmalloc((size_t)22 * N);
  float *glb = xfmalloc(19);
  {
    FILE *fi = fopen("input.bin", "rb");
    FILE *fg = fopen("input_global.bin", "rb");
    if (!fi || !fg) die("cannot open input.bin / input_global.bin (run python main.py first)");
    if (fread(input, 4, 22 * N, fi) != 22 * N) die("short read input.bin");
    if (fread(glb, 4, 19, fg) != 19) die("short read input_global.bin");
    fclose(fi);
    fclose(fg);
  }
  float *mask = xfmalloc(N);
  for (int j = 0; j < N; j++) mask[j] = input[0 * N + j];
  double S = 0.0;
  for (int j = 0; j < N; j++) S += (double)mask[j];
  double dd = sqrt(S) - 14.0;
  float d10 = (float)(dd / 10.0);
  float d2c = (float)(dd * dd / 100.0 - 0.1);

  float *trunk = xfmalloc((size_t)C_TRUNK * N);
  float *trunk2 = xfmalloc((size_t)C_TRUNK * N);
  float *mid = xfmalloc((size_t)C_MID * N);
  float *mid2 = xfmalloc((size_t)C_MID * N);
  float *subout = xfmalloc((size_t)C_MID * N);
  float *ffn_a = xfmalloc((size_t)FFN_DIM * N);
  float *ffn_g = xfmalloc((size_t)FFN_DIM * N);
  AttnScratch as;
  as.xn = xfmalloc((size_t)C_MID * N);
  as.q = xfmalloc((size_t)C_MID * N);
  as.k = xfmalloc((size_t)C_MID * N);
  as.v = xfmalloc((size_t)C_MID * N);
  as.att = xfmalloc((size_t)C_MID * N);
  as.out = xfmalloc((size_t)C_MID * N);
  as.sc = xfmalloc((size_t)N * N);
  float px[N], py[N];
  for (int j = 0; j < N; j++) {
    py[j] = (float)(j / XY);
    px[j] = (float)(j % XY);
  }

  FILE *fo = fopen("out_c.bin", "wb");
  if (!fo) die("cannot create out_c.bin");
  /* initial: conv3x3 22->768 (padding 1) + linear_global */
  memset(trunk, 0, sizeof(float) * (size_t)C_TRUNK * N);
  for (int cy = 0; cy < 3; cy++) {
    for (int cx = 0; cx < 3; cx++) {
      for (int ic = 0; ic < 22; ic++) {
        const float *wr = m->conv_spatial + (((size_t)(cy * 3 + cx) * 22 + ic) * C_TRUNK);
        for (int oc = 0; oc < C_TRUNK; oc++) {
          float wv = wr[oc];
          float *yr = trunk + (size_t)oc * N;
          const float *xr = input + (size_t)ic * N;
          for (int y = 0; y < XY; y++) {
            int iy = y + cy - 1;
            if (iy < 0 || iy >= XY) continue;
            for (int x = 0; x < XY; x++) {
              int ix = x + cx - 1;
              if (ix < 0 || ix >= XY) continue;
              yr[y * XY + x] += wv * xr[iy * XY + ix];
            }
          }
        }
      }
    }
  }
  {
    float g[C_TRUNK];
    for (int oc = 0; oc < C_TRUNK; oc++) {
      float s = 0.0f;
      for (int ic = 0; ic < 19; ic++)
        s += m->linear_global.w[(size_t)ic * C_TRUNK + oc] * glb[ic];
      g[oc] = s;
    }
    for (int oc = 0; oc < C_TRUNK; oc++) {
      float *yr = trunk + (size_t)oc * N;
      float gv = g[oc];
      for (int j = 0; j < N; j++) yr[j] += gv;
    }
  }
  fwrite(trunk, 4, (size_t)C_TRUNK * N, fo);

  /* 11 nested bottleneck blocks */
  for (int blk = 0; blk < N_BLOCKS; blk++) {
    Block *B = &m->blocks[blk];
    bn_silu_mask(trunk, &B->preBN, mask, trunk2, C_TRUNK, N);
    matmul(B->preConv.w, trunk2, mid, C_TRUNK, C_MID, N);
    if (blk == 0) fwrite(mid, 4, (size_t)C_MID * N, fo);
    for (int sub = 0; sub < N_SUB; sub++) {
      if (sub % 2 == 0)
        attn_block(&B->attn[sub / 2], mid, px, py, mask, subout, &as);
      else
        ffn_block(&B->ffn[(sub - 1) / 2], mid, mask, subout, as.xn, ffn_a, ffn_g);
      for (size_t i = 0; i < (size_t)C_MID * N; i++) mid[i] += subout[i];
      if (blk == 0 && (sub == 0 || sub == 1 || sub == N_SUB - 1))
        fwrite(mid, 4, (size_t)C_MID * N, fo);
    }
    bn_silu_mask(mid, &B->postBN, mask, mid2, C_MID, N);
    matmul(B->postConv.w, mid2, subout, C_MID, C_TRUNK, N);
    for (size_t i = 0; i < (size_t)C_TRUNK * N; i++) trunk[i] += subout[i];
    fwrite(trunk, 4, (size_t)C_TRUNK * N, fo);
  }

  /* trunk tip */
  bn_silu_mask(trunk, &m->trunkfinal, mask, trunk2, C_TRUNK, N);
  memcpy(trunk, trunk2, sizeof(float) * (size_t)C_TRUNK * N);
  fwrite(trunk, 4, (size_t)C_TRUNK * N, fo);
  /* policy head */
  float *p1 = xfmalloc((size_t)96 * N);
  float *g1 = xfmalloc((size_t)96 * N);
  float *p1b = xfmalloc((size_t)96 * N);
  matmul(m->conv1p.w, trunk, p1, C_TRUNK, 96, N);
  matmul(m->conv1g.w, trunk, g1, C_TRUNK, 96, N);
  bn_silu_mask(g1, &m->biasg, mask, g1, 96, N);
  float mean[96], gp[288];
  for (int c = 0; c < 96; c++) {
    const float *gr = g1 + (size_t)c * N;
    float s = 0.0f;
    float mx = -1.0e30f;
    for (int j = 0; j < N; j++) {
      s += gr[j];
      float t = gr[j] + (mask[j] - 1.0f);
      if (t > mx) mx = t;
    }
    mean[c] = s / (float)S;
    gp[c] = mean[c];
    gp[96 + c] = mean[c] * d10;
    gp[192 + c] = mx;
  }
  {
    float gb[96];
    for (int oc = 0; oc < 96; oc++) {
      float s = 0.0f;
      for (int ic = 0; ic < 288; ic++)
        s += m->linear_g.w[(size_t)ic * 96 + oc] * gp[ic];
      gb[oc] = s;
    }
    for (int c = 0; c < 96; c++) {
      float *pr = p1 + (size_t)c * N;
      float gv = gb[c];
      for (int j = 0; j < N; j++) pr[j] += gv;
    }
  }
  bn_silu_mask(p1, &m->bias2, mask, p1b, 96, N);
  float *policy = xfmalloc((size_t)2 * N);
  matmul(m->conv2p.w, p1b, policy, 96, 2, N);
  float pp[96], passv[2];
  for (int oc = 0; oc < 96; oc++) {
    float s = 0.0f;
    for (int ic = 0; ic < 288; ic++)
      s += m->linear_pass.w[(size_t)ic * 96 + oc] * gp[ic];
    pp[oc] = silu(s + m->linear_pass_bias.b[oc]);
  }
  for (int oc = 0; oc < 2; oc++) {
    float s = 0.0f;
    for (int ic = 0; ic < 96; ic++)
      s += m->linear_pass2.w[(size_t)ic * 2 + oc] * pp[ic];
    passv[oc] = s;
  }
  /* value head */
  float *v1 = xfmalloc((size_t)192 * N);
  float vmean[192], vg[576], v2[192], value[3], misc[6];
  matmul(m->conv1.w, trunk, v1, C_TRUNK, 192, N);
  bn_silu_mask(v1, &m->bias1, mask, v1, 192, N);
  for (int c = 0; c < 192; c++) {
    const float *vr = v1 + (size_t)c * N;
    float s = 0.0f;
    for (int j = 0; j < N; j++) s += vr[j];
    vmean[c] = s / (float)S;
    vg[c] = vmean[c];
    vg[192 + c] = vmean[c] * d10;
    vg[384 + c] = vmean[c] * d2c;
  }
  for (int oc = 0; oc < 192; oc++) {
    float s = 0.0f;
    for (int ic = 0; ic < 576; ic++)
      s += m->linear2.w[(size_t)ic * 192 + oc] * vg[ic];
    v2[oc] = silu(s + m->bias2v.b[oc]);
  }
  for (int oc = 0; oc < 3; oc++) {
    float s = 0.0f;
    for (int ic = 0; ic < 192; ic++)
      s += m->linear_valuehead.w[(size_t)ic * 3 + oc] * v2[ic];
    value[oc] = s + m->bias_valuehead.b[oc];
  }
  for (int oc = 0; oc < 6; oc++) {
    float s = 0.0f;
    for (int ic = 0; ic < 192; ic++)
      s += m->linear_miscvaluehead.w[(size_t)ic * 6 + oc] * v2[ic];
    misc[oc] = s + m->bias_miscvaluehead.b[oc];
  }
  float *ownership = xfmalloc((size_t)1 * N);
  matmul(m->conv_ownership.w, v1, ownership, 192, 1, N);

  fwrite(policy, 4, 2 * N, fo);
  fwrite(passv, 4, 2, fo);
  fwrite(value, 4, 3, fo);
  fwrite(misc, 4, 6, fo);
  fwrite(ownership, 4, N, fo);
  fclose(fo);

  printf("wrote out_c.bin\n");
  printf("value:       %.4f %.4f %.4f\n", value[0], value[1], value[2]);
  printf("miscvalue:   %.4f %.4f %.4f %.4f %.4f %.4f\n", misc[0], misc[1], misc[2], misc[3], misc[4], misc[5]);
  printf("policy_pass: %.4f %.4f\n", passv[0], passv[1]);
  {
    float mx = policy[0];
    for (int j = 0; j < N; j++) if (policy[j] > mx) mx = policy[j];
    float sum = 0.0f;
    for (int j = 0; j < N; j++) sum += expf(policy[j] - mx);
    printf("top5 moves:  ");
    for (int k = 0; k < 5; k++) {
      int best = 0;
      float bv = -1.0e30f;
      for (int j = 0; j < N; j++) if (policy[j] > bv) { bv = policy[j]; best = j; }
      printf("(%d, %.4f) ", best, expf(policy[best] - mx) / sum);
      policy[best] = -1.0e30f;
    }
    printf("\n");
  }
  {
    float mn = 1e30f, mx = -1e30f, s = 0.0f;
    for (int j = 0; j < N; j++) {
      s += ownership[j];
      if (ownership[j] < mn) mn = ownership[j];
      if (ownership[j] > mx) mx = ownership[j];
    }
    printf("ownership:   mean %.4f max %.4f min %.4f\n", s / N, mx, mn);
  }
  return 0;
}
