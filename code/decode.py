import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X
import diag
R = X.R

def bits2int(b): return int(''.join(str(int(x)) for x in b), 2) if len(b) else 0

def decode_pass(img_w, pred, num, complexity_fn):
    """Invert one histogram-shifting pass. Returns (restored image, extracted watermark bits)."""
    img = img_w.copy(); M, N = img.shape
    mn = int(np.ceil(np.log2(M*N)))
    sc = diag.order(img, num, complexity_fn)          # same order the embedder used (needs other-half pixels only)
    rd = [int(img[i, j]) & 1 for (i, j) in reversed(sc)]  # header bits, bit_idx 0 = last coord
    p = 0
    len_map = bits2int(rd[p:p+mn]); p += mn
    cmap = rd[p:p+len_map]; p += len_map
    msg_len = bits2int(rd[p:p+mn]); p += mn
    Tp = bits2int(rd[p:p+mn]); p += mn
    Tn = -bits2int(rd[p:p+mn]); p += mn
    n1 = bits2int(rd[p:p+mn]); p += mn
    n0 = bits2int(rd[p:p+mn]); p += mn
    n_extract = len_map + 6*mn
    if n1 == 0:
        loc = np.zeros(M*N, dtype=int)
    else:
        loc = np.array(R.range_decode(cmap, [n0, n1], n0+n1), dtype=int)
    location_map = loc.reshape(N, M).T            # inverse of location_map.T.ravel()
    bits = []; idx = 0
    for (i, j) in sc:
        if idx >= msg_len: break
        if location_map[i, j] == 0:
            y = img[i, j]; xp = pred[i, j]; D = y - xp
            if D >= 2*Tp + 2:
                img[i, j] = xp + D - Tp - 1
            elif D <= 2*Tn - 1:
                img[i, j] = xp + D - Tn
            else:
                d = np.floor(D/2); b = int(D - 2*d)
                bits.append(b); idx += 1
                img[i, j] = xp + d
    # restore the LSBs that were overwritten by the header
    orig_lsb = bits[:n_extract]; wm = bits[n_extract:]
    for t, (i, j) in enumerate(reversed(sc[-n_extract:] if n_extract else [])):
        v = int(img[i, j]); img[i, j] = v - (v & 1) + orig_lsb[t]
    return img, wm

def full_roundtrip(g, msg, kw, complexity_fn, use_ocnnp=False):
    w = X.embed(g.copy(), msg, **kw)
    half = int(len(msg)/2); cur = w.copy(); out = [None, None]
    for num in (1, 0):                               # undo pass 1 first, then pass 0
        pred = X.predict(cur, num, use_ocnnp)
        cur, wm = decode_pass(cur, pred, num, complexity_fn)
        out[num] = wm
    ok_img = bool(np.array_equal(cur, g))
    got = np.array(out[0] + out[1]); want = np.concatenate([msg[:half], msg[half:2*half]])
    ok_msg = bool(len(got) == len(want) and np.array_equal(got, want))
    return ok_img, ok_msg, w

if __name__ == '__main__':
    path, L = sys.argv[1], int(sys.argv[2])
    g = cv2.cvtColor(cv2.resize(cv2.imread(path), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
    msg = np.random.RandomState(99).randint(0, 2, [L])
    ok_i, ok_m, w = full_roundtrip(g, msg, X.METHODS['CNNP'], R.calculate_complexity)
    print(os.path.basename(path), L, 'CNNP  image restored exactly:', ok_i, '| message recovered exactly:', ok_m, '| PSNR', round(X.psnr(w, g), 2))
