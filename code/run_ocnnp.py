import sys, os, json, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, decode as D, rdh_fixed as F
from skimage.metrics import structural_similarity as ssim
X.R.cnn_histogram_shifting = F.cnn_histogram_shifting
exec(open('run_all.py').read().split("K = [")[0].split("STD = ")[1].join(["STD = ",""]) if False else "")
STD = 'standard_test_images'
PATH = {'Lena':STD+'/Lena.bmp','Barbara':STD+'/Barbara.bmp','Peppers':STD+'/Peppers.bmp','Yacht':STD+'/yacht.bmp','camera':'code/extra/camera.png','moon':'code/extra/moon.png','coffee':'code/extra/coffee.png','chelsea':'code/heldout/chelsea.png','rocket':'code/heldout/rocket.png','coins':'code/heldout/coins.png','astronaut':'code/heldout/astronaut.png','china':'code/heldout/china.png','flower':'code/heldout/flower.png'}
names = sys.argv[1].split(','); out = sys.argv[2]; rows = []
for nm in names:
    g = cv2.cvtColor(cv2.resize(cv2.imread(PATH[nm]), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
    for L in (10000, 50000, 100000):
        msg = np.random.RandomState(1234+L).randint(0, 2, [L]); t0 = time.time(); row = dict(image=nm, bits=L, method='OCNNP(rounded)')
        try:
            ok_i, ok_m, w = D.full_roundtrip(g, msg, X.METHODS['OCNNP'], F.calculate_complexity, use_ocnnp=True)
            row.update(exact=bool(ok_i and ok_m), psnr=round(X.psnr(w, g), 4), ssim=round(float(ssim(g, w, data_range=255)), 6), out_of_range=int(((w < 0) | (w > 255)).sum()), non_integer=int((w != np.round(w)).sum()))
        except F.CapacityError as e: row.update(capacity_error=str(e))
        except Exception as e: row.update(error=repr(e)[:100])
        row['sec'] = round(time.time()-t0, 1); print(json.dumps(row), flush=True); rows.append(row)
json.dump(rows, open(out, 'w'), indent=1)
