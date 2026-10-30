import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X
from skimage.metrics import structural_similarity as ssim
R = X.R
K = [(-1,0),(1,0),(0,-1),(0,1),(-1,2),(-1,-2),(1,2),(1,-2),(-2,1),(-2,-1),(2,1),(2,-1)]  # 4 direct + 8 knight moves: all in the OTHER pixel half
def nb12(img, i, j):
    M, N = img.shape
    return [img[min(max(i+a,0),M-1), min(max(j+b,0),N-1)] for a, b in K]
def std12(img, i, j): return float(np.std(nb12(img, i, j)))
def range4(img, i, j):
    v = [img[i,j-1], img[i-1,j], img[i,j+1], img[i+1,j]]; return float(max(v)-min(v))
base4 = R.calculate_complexity
def base_plus_std12(img, i, j): return base4(img, i, j) + 0.5*std12(img, i, j)
CANDS = {'std12': std12, 'range4': range4, 'base+0.5std12': base_plus_std12}
imgdir_map = {'Lena.bmp':'/home/claude/nikithayamsani/rdh_hdr/standard_test_images','Barbara.bmp':'/home/claude/nikithayamsani/rdh_hdr/standard_test_images','Peppers.bmp':'/home/claude/nikithayamsani/rdh_hdr/standard_test_images','yacht.bmp':'/home/claude/nikithayamsani/rdh_hdr/standard_test_images','camera.png':'extra','moon.png':'extra','coffee.png':'extra'}
rows = []
for fn, d in imgdir_map.items():
    g = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(d, fn)), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
    for L in (50000, 100000):
        msg = np.random.RandomState(1234+L).randint(0, 2, [L])
        base_w = X.embed(g.copy(), msg, **X.METHODS['CNNP']); pb = X.psnr(base_w, g)
        for name, f in CANDS.items():
            R.calculate_complexity = f                       # swap the default complexity used when use_lsbc=False
            w = X.embed(g.copy(), msg, **X.METHODS['CNNP']); R.calculate_complexity = base4
            row = dict(image=fn, bits=L, cand=name, psnr=round(X.psnr(w, g), 4), base=round(pb, 4), gain=round(X.psnr(w, g)-pb, 4))
            print(json.dumps(row), flush=True); rows.append(row)
json.dump(rows, open('cand.json', 'w'), indent=1)
