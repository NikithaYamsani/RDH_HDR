import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, decode as D
R = X.R
def spread3(img, i, j):            # std of left, below, right (all other-half pixels -> decoder can recompute)
    return float(np.std([img[i, j-1], img[i+1, j], img[i, j+1]]))
def spread4(img, i, j):            # std of the 4 plain neighbours
    return float(np.std([img[i, j-1], img[i-1, j], img[i, j+1], img[i+1, j]]))
def mean3(img, i, j):
    return float(np.mean([img[i, j-1], img[i+1, j], img[i, j+1]]))
orig_spread, orig_mean = R.calculate_lsbc_spread, R.calculate_lsbc
path, L = sys.argv[1], int(sys.argv[2])
g = cv2.cvtColor(cv2.resize(cv2.imread(path), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
msg = np.random.RandomState(99).randint(0, 2, [L])
cases = [('LSBC(mean) as implemented', 'mean', orig_mean, orig_mean),
         ('LSBC-spread as implemented', 'spread', orig_spread, orig_spread),
         ('LSBC-spread, 3 other-half nbrs', 'spread', spread3, spread3),
         ('LSBC-spread, 4 nbrs', 'spread', spread4, spread4)]
for name, mode, fn, dec_fn in cases:
    R.calculate_lsbc_spread = fn if mode == 'spread' else orig_spread
    R.calculate_lsbc = fn if mode == 'mean' else orig_mean
    kw = dict(use_ocnnp=False, use_lsbc=True, lsbc_mode=mode)
    try:
        ok_i, ok_m, w = D.full_roundtrip(g, msg, kw, dec_fn)
        print(f'{os.path.basename(path)} {L} {name}: image exact={ok_i} message exact={ok_m} PSNR={X.psnr(w, g):.2f}', flush=True)
    except Exception as e:
        print(f'{os.path.basename(path)} {L} {name}: DECODE FAILED ({type(e).__name__}: {str(e)[:80]})', flush=True)
