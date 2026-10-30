"""Confidence-adaptive histogram shifting (ACE), exactly reversible.

The embedding order is the ENC complexity order (other-half pixels only).  In this
module the thresholds [Tn_k, Tp_k] of the expansion rule are chosen per class k of
the embedding order (K=1 gives one global pair, K=3 gives three classes of ranks).
Class boundaries and thresholds are written in the header, so the decoder knows them,
and the class of a pixel depends only on its rank in the order, which the decoder
recomputes from the other half.  No quantity that depends on the pixel being changed
is used to choose a class.
"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdh_fixed as F
from rdh_fixed import CapacityError

K12 = [(-1,0),(1,0),(0,-1),(0,1),(-1,2),(-1,-2),(1,2),(1,-2),(-2,1),(-2,-1),(2,1),(2,-1)]
TW = 4                                   # bits per stored threshold magnitude
OPT = [(tn, tp) for tn in (-3,-2,-1,0) for tp in (0,1,2,3)]
WIDE = [(tn, tp) for tn in (-8,-5,-3) for tp in (3,5,8) if (tn, tp) not in [(-3,3)]]   # only tried if the narrow set cannot hold the payload
BND = [6000, 12000, 20000, 32000, 48000, 72000]

ORDER = 'enc'                            # 'enc' (proposed) or 'c0' (CNNP baseline order)

def enc_map(img):
    M, N = img.shape
    p = np.pad(img, 2, mode='edge')
    sh = lambda a, b: p[2+a:2+a+M, 2+b:2+b+N]
    L, U, Rr, D = sh(0,-1), sh(-1,0), sh(0,1), sh(1,0)
    c0 = (np.abs(L-U) + np.abs(U-Rr) + np.abs(Rr-D) + np.abs(D-L)) / 4.0
    if ORDER == 'c0': return c0
    return c0 + 0.5 * np.stack([sh(a, b) for a, b in K12]).std(axis=0)

def order(img, num):
    M, N = img.shape
    ii, jj = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    place = ((ii+jj) % 2 == num) & (ii >= 1) & (ii <= M-2) & (jj >= 1) & (jj <= N-2)
    idx = np.flatnonzero(place.T)                        # column-major, as in the repo
    ci, cj = idx % M, idx // M
    o = np.argsort(enc_map(img)[ci, cj], kind='stable')
    return ci[o], cj[o]

def bits2int(b): return int(''.join(str(int(x)) for x in b), 2) if len(b) else 0

def _tables(e, u, opts):
    caps, costs = [], []
    for tn, tp in opts:
        inr = u & (e >= tn) & (e <= tp)
        c = np.where(inr, e*e + e + 0.5, np.where(e > tp, (tp+1)**2, tn*tn)).astype(float)
        c = np.where(u, c, 0.0)
        caps.append(np.concatenate([[0], np.cumsum(inr)]))
        costs.append(np.concatenate([[0.0], np.cumsum(c)]))
    return caps, costs

def _search(caps, costs, S, need, K, nopt):
    best = (None, None, None)        # cost, (bounds, opts), stop rank
    B = sorted(set(min(x, S) for x in BND))
    bsets = [()] if K == 1 else list(itertools.combinations_with_replacement(B, K-1))
    for bnd in bsets:
        edges = [0] + list(bnd) + [S]
        for combo in itertools.product(range(nopt), repeat=K):
            run = 0; cost = 0.0; stop = None
            for k, o in enumerate(combo):
                a, b = edges[k], edges[k+1]
                cc = caps[o]; cs = costs[o]
                seg = cc[b] - cc[a]
                if run + seg >= need:
                    p = int(np.searchsorted(cc, cc[a] + (need - run), side='left'))
                    cost += cs[p] - cs[a]; stop = p - 1; break
                run += seg; cost += cs[b] - cs[a]
            if stop is not None and (best[0] is None or cost < best[0]):
                best = (cost, (bnd, combo), stop)
    return best

def _core(img, pred, wm, num, K, extra_flags):
    img = img.astype(np.float64).copy(); M, N = img.shape
    mn = int(np.ceil(np.log2(M*N)))
    wm = [int(x) for x in wm]; Lw = len(wm)
    si, sj = order(img, num); n = len(si)
    hdr_fixed = 4*mn + (K-1)*mn + 2*K*TW
    x = img[si, sj]; xp = pred[si, sj]
    e = (x - xp).astype(int)
    val = 2*x - xp
    flag_full = (val > 254) | (val < 0)
    if extra_flags:
        rank = -np.ones((M, N), dtype=int); rank[si, sj] = np.arange(n)
        for (i, j) in extra_flags:
            r = rank[i, j]
            if r >= 0: flag_full[r] = True
    S = min(n, 2*Lw + 4*mn)
    opts = OPT
    while True:
        fl = flag_full[:S]
        loc = np.zeros((M, N), dtype=int); loc[si[:S][fl], sj[:S][fl]] = 1
        cbits, n0, n1 = F.compress_location_map(loc.T.ravel())
        n_extract = len(cbits) + hdr_fixed
        need = Lw + n_extract
        if S > n - n_extract:
            S = n - n_extract
            if S <= 0: raise CapacityError('embedding region overlaps the header area')
            continue
        caps, costs = _tables(e[:S], ~fl, opts)
        cost, sol, stop = _search(caps, costs, S, need, K, len(opts))
        if sol is None:
            if S >= n - n_extract:
                if opts is OPT:
                    opts = OPT + WIDE; continue
                raise CapacityError('payload does not fit in the embedding region')
            S = min(n - n_extract, int(S*1.3) + 1000); continue
        if stop + 1 < S:
            S = stop + 1; continue        # fewer flags and a shorter map for a smaller region; same solution stays feasible
        break
    bnd, combo = sol
    TP = [opts[o][1] for o in combo]; TN = [opts[o][0] for o in combo]
    edges = [0] + list(bnd) + [n+1]
    cls = lambda r: next(k for k in range(K) if r < edges[k+1])
    # header (stored in the last pixels of the order) and the original bits it overwrites
    tail = [(si[n-1-t], sj[n-1-t]) for t in range(n_extract)]
    orig = [int(img[i, j]) & 1 for (i, j) in tail]
    msg = orig + wm
    hdr = F.num2bitlist(len(cbits), mn) + list(cbits) + F.num2bitlist(len(msg), mn)
    for k in range(K): hdr += F.num2bitlist(TP[k], TW) + F.num2bitlist(-TN[k], TW)
    for b in bnd: hdr += F.num2bitlist(b, mn)
    hdr += F.num2bitlist(n1, mn) + F.num2bitlist(n0, mn)
    assert len(hdr) == n_extract
    out = img.copy()
    for t, (i, j) in enumerate(tail):
        out[i, j] = int(img[i, j]) - (int(img[i, j]) & 1) + hdr[t]
    idx = 0
    fl = flag_full[:S]
    for r in range(S):
        if fl[r]: continue
        k = cls(r); d = int(e[r]); i, j = si[r], sj[r]
        if d > TP[k]: dd = d + TP[k] + 1
        elif d < TN[k]: dd = d + TN[k]
        else:
            dd = 2*d + msg[idx]; idx += 1
        out[i, j] = xp[r] + dd
        if idx >= len(msg): break
    if idx < len(msg): raise CapacityError('payload does not fit in the embedding region')
    return out

def embed_pass(img, pred, wm, num, K):
    flags = set()
    for _ in range(40):
        w = _core(img, pred, wm, num, K, flags)
        bad = np.argwhere((w < 0) | (w > 255))
        if len(bad) == 0: return w
        flags |= {(int(a), int(b)) for a, b in bad}
    raise CapacityError('overflow handling did not converge')

def decode_pass(img_w, pred, num, K):
    img = img_w.copy(); M, N = img.shape
    mn = int(np.ceil(np.log2(M*N)))
    si, sj = order(img, num); n = len(si)
    rd = [int(img[si[n-1-t], sj[n-1-t]]) & 1 for t in range(min(n, 400000))]
    p = 0
    len_map = bits2int(rd[p:p+mn]); p += mn
    cmap = rd[p:p+len_map]; p += len_map
    msg_len = bits2int(rd[p:p+mn]); p += mn
    TP, TN = [], []
    for k in range(K):
        TP.append(bits2int(rd[p:p+TW])); p += TW
        TN.append(-bits2int(rd[p:p+TW])); p += TW
    bnd = []
    for _ in range(K-1):
        bnd.append(bits2int(rd[p:p+mn])); p += mn
    n1 = bits2int(rd[p:p+mn]); p += mn
    n0 = bits2int(rd[p:p+mn]); p += mn
    n_extract = len_map + 4*mn + (K-1)*mn + 2*K*TW
    loc = np.zeros(M*N, dtype=int) if n1 == 0 else np.array(F.range_decode(cmap, [n0, n1], n0+n1), dtype=int)
    location_map = loc.reshape(N, M).T
    edges = [0] + bnd + [n+1]
    bits = []
    for r in range(n):
        if len(bits) >= msg_len: break
        i, j = si[r], sj[r]
        if location_map[i, j] == 0:
            k = next(kk for kk in range(K) if r < edges[kk+1])
            y = img[i, j]; xp = pred[i, j]; D = y - xp
            if D >= 2*TP[k] + 2: img[i, j] = xp + D - TP[k] - 1
            elif D <= 2*TN[k] - 1: img[i, j] = xp + D - TN[k]
            else:
                d = np.floor(D/2); bits.append(int(D - 2*d)); img[i, j] = xp + d
    orig = bits[:n_extract]; wm = bits[n_extract:]
    for t in range(n_extract):
        i, j = si[n-1-t], sj[n-1-t]
        v = int(img[i, j]); img[i, j] = v - (v & 1) + orig[t]
    return img, wm

def embed(g, msg, K):
    import run_exp as X
    half = len(msg) // 2; cur = g.astype(np.float64)
    for num in range(2):
        pn = X.predict(cur, num, False)
        cur = embed_pass(cur, pn, msg[num*half:(num+1)*half], num, K)
    return cur

def roundtrip(g, msg, K):
    import run_exp as X
    w = embed(g, msg, K)
    half = len(msg) // 2; cur = w.copy(); out = [None, None]
    for num in (1, 0):
        pn = X.predict(cur, num, False)
        cur, wmb = decode_pass(cur, pn, num, K); out[num] = wmb
    ok_img = bool(np.array_equal(cur, g))
    got = np.array(out[0] + out[1]); want = np.asarray(msg[:2*half])
    ok_msg = bool(len(got) == len(want) and np.array_equal(got, want))
    return ok_img, ok_msg, w
