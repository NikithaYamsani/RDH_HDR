import sys, os, json, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, decode as D
from skimage.metrics import structural_similarity as ssim
R = X.R
K = [(-1,0),(1,0),(0,-1),(0,1),(-1,2),(-1,-2),(1,2),(1,-2),(-2,1),(-2,-1),(2,1),(2,-1)]   # 4 direct + 8 knight-move neighbours, all in the OTHER pixel half
base4 = R.calculate_complexity
def std12(img, i, j):
    M, N = img.shape
    return float(np.std([img[min(max(i+a,0),M-1), min(max(j+b,0),N-1)] for a, b in K]))
def cand(img, i, j): return base4(img, i, j) + 0.5*std12(img, i, j)       # decoder-computable ("reversible") complexity
imgdir, out = sys.argv[1], sys.argv[2]
names = sys.argv[3].split(',')
lens = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [10000, 50000, 100000]
rows = []
for fn in names:
    g = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(imgdir, fn)), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
    for L in lens:
        msg = np.random.RandomState(1234+L).randint(0, 2, [L]); t0 = time.time()
        try:
            wb = X.embed(g.copy(), msg, **X.METHODS['CNNP'])
            R.calculate_complexity = cand
            ok_i, ok_m, wc = D.full_roundtrip(g, msg, X.METHODS['CNNP'], cand)     # embeds with `cand` as complexity, then decodes
            R.calculate_complexity = base4
            row = dict(image=fn, bits=L, cnnp_psnr=round(X.psnr(wb, g), 4), cnnp_ssim=round(float(ssim(g, wb, data_range=255)), 6),
                       cand_psnr=round(X.psnr(wc, g), 4), cand_ssim=round(float(ssim(g, wc, data_range=255)), 6),
                       image_exact=ok_i, message_exact=ok_m, out_of_range=int(((wc < 0) | (wc > 255)).sum()), sec=round(time.time()-t0, 1))
        except Exception as e:
            R.calculate_complexity = base4
            row = dict(image=fn, bits=L, error=repr(e)[:120])
        print(json.dumps(row), flush=True); rows.append(row)
json.dump(rows, open(out, 'w'), indent=1)
