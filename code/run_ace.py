import sys, os, json, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, rdh_ace as A
from skimage.metrics import structural_similarity as ssim
STD = 'standard_test_images'
PATH = {'Lena':STD+'/Lena.bmp','Barbara':STD+'/Barbara.bmp','Peppers':STD+'/Peppers.bmp','Yacht':STD+'/yacht.bmp',
        'camera':'extra/camera.png','moon':'extra/moon.png','coffee':'extra/coffee.png',
        'chelsea':'heldout/chelsea.png','rocket':'heldout/rocket.png','coins':'heldout/coins.png','astronaut':'heldout/astronaut.png',
        'china':'heldout/china.png','flower':'heldout/flower.png',
        'cat':'extra2/cat.png','cell':'extra2/cell.png','hubble':'extra2/hubble_deep_field.png','immuno':'extra2/immunohistochemistry.png','retina':'extra2/retina.png'}
names = sys.argv[1].split(','); out = sys.argv[2]; Ks = [int(k) for k in sys.argv[3].split(',')]
rows = []
for nm in names:
    g = cv2.cvtColor(cv2.resize(cv2.imread(PATH[nm]), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
    for L in (10000, 50000, 100000):
        msg = np.random.RandomState(1234+L).randint(0, 2, [L])
        for K in Ks:
            t0 = time.time(); row = dict(image=nm, bits=L, method='ACE-K%d' % K)
            try:
                ok_i, ok_m, w = A.roundtrip(g, msg, K)
                row.update(exact=bool(ok_i and ok_m), psnr=round(X.psnr(w, g), 4), ssim=round(float(ssim(g, w, data_range=255)), 6),
                           out_of_range=int(((w < 0) | (w > 255)).sum()))
            except A.CapacityError as e:
                row.update(capacity_error=str(e))
            except Exception as e:
                row.update(error=repr(e)[:100])
            row['sec'] = round(time.time()-t0, 1)
            print(json.dumps(row), flush=True); rows.append(row)
json.dump(rows, open(out, 'w'), indent=1)
