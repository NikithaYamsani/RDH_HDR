import sys, os, json, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, decode as D, rdh_fixed as F
from skimage.metrics import structural_similarity as ssim
X.R.cnn_histogram_shifting = F.cnn_histogram_shifting          # fixed embedder (overflow + capacity handling)
STD = '/home/claude/nikithayamsani/rdh_hdr/standard_test_images'
PATH = {'Lena':STD+'/Lena.bmp','Barbara':STD+'/Barbara.bmp','Peppers':STD+'/Peppers.bmp','Yacht':STD+'/yacht.bmp',
        'camera':'extra/camera.png','moon':'extra/moon.png','coffee':'extra/coffee.png',
        'chelsea':'heldout/chelsea.png','rocket':'heldout/rocket.png','coins':'heldout/coins.png','astronaut':'heldout/astronaut.png',
        'china':'heldout/china.png','flower':'heldout/flower.png'}
K = [(-1,0),(1,0),(0,-1),(0,1),(-1,2),(-1,-2),(1,2),(1,-2),(-2,1),(-2,-1),(2,1),(2,-1)]
base4 = F.calculate_complexity
def std12(img, i, j):
    M, N = img.shape
    return float(np.std([img[min(max(i+a,0),M-1), min(max(j+b,0),N-1)] for a, b in K]))
def cand(img, i, j): return base4(img, i, j) + 0.5*std12(img, i, j)
names = sys.argv[1].split(','); out = sys.argv[2]
rows = []
for nm in names:
    g = cv2.cvtColor(cv2.resize(cv2.imread(PATH[nm]), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
    for L in (10000, 50000, 100000):
        msg = np.random.RandomState(1234+L).randint(0, 2, [L])
        for method in ('CNNP', 'Proposed', 'OCNNP'):
            t0 = time.time(); row = dict(image=nm, bits=L, method=method)
            try:
                if method == 'OCNNP':
                    F.calculate_complexity = base4
                    w = X.embed(g.copy(), msg, **X.METHODS['OCNNP'])
                    row.update(exact=None)
                else:
                    fn = cand if method == 'Proposed' else base4
                    F.calculate_complexity = fn
                    ok_i, ok_m, w = D.full_roundtrip(g, msg, X.METHODS['CNNP'], fn)
                    row.update(exact=bool(ok_i and ok_m))
                row.update(psnr=round(X.psnr(w, g), 4), ssim=round(float(ssim(g, w, data_range=255)), 6), out_of_range=int(((w < 0) | (w > 255)).sum()))
            except F.CapacityError as e:
                row.update(capacity_error=str(e))
            except Exception as e:
                row.update(error=repr(e)[:100])
            finally:
                F.calculate_complexity = base4
            row['sec'] = round(time.time()-t0, 1)
            print(json.dumps(row), flush=True); rows.append(row)
json.dump(rows, open(out, 'w'), indent=1)
