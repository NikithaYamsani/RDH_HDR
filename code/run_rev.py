import sys, os, json, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, decode as D
from skimage.metrics import structural_similarity as ssim
R = X.R
def spread4(img, i, j):   # std of the 4 plain neighbours (all in the other pixel half -> decoder can recompute)
    return float(np.std([img[i, j-1], img[i-1, j], img[i, j+1], img[i+1, j]]))
R.calculate_lsbc_spread = spread4
CASES = {'CNNP': (dict(use_ocnnp=False, use_lsbc=False, lsbc_mode='mean'), R.calculate_complexity),
         'CNNP+spread4 (reversible)': (dict(use_ocnnp=False, use_lsbc=True, lsbc_mode='spread'), spread4)}
imgdir, out = sys.argv[1], sys.argv[2]
names = sys.argv[3].split(',') if len(sys.argv) > 3 else sorted(os.listdir(imgdir))
rows = []
for fn in names:
    g = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(imgdir, fn)), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
    for L in (10000, 50000, 100000):
        msg = np.random.RandomState(1234+L).randint(0, 2, [L])
        for m, (kw, fnc) in CASES.items():
            t0 = time.time()
            try:
                ok_i, ok_m, w = D.full_roundtrip(g, msg, kw, fnc)
                row = dict(image=fn, bits=L, method=m, psnr=round(X.psnr(w, g), 4), ssim=round(float(ssim(g, w, data_range=255)), 6),
                           image_exact=ok_i, message_exact=ok_m, sec=round(time.time()-t0, 1))
            except Exception as e:
                row = dict(image=fn, bits=L, method=m, error=repr(e)[:120])
            print(json.dumps(row), flush=True); rows.append(row)
json.dump(rows, open(out, 'w'), indent=1)
