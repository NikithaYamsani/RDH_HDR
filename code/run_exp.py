import sys, os, json, math, time
import numpy as np, cv2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import rdh_embed as R
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnp
from skimage.metrics import structural_similarity as ssim

def parse_sample(img_gray, num=0):
    source = np.zeros(img_gray.shape)
    for i in range(1, img_gray.shape[0]-1):
        for j in range(1, img_gray.shape[1]-1):
            if (i+j) % 2 == num:
                source[i-1, j] = img_gray[i-1, j]; source[i+1, j] = img_gray[i+1, j]
                source[i, j-1] = img_gray[i, j-1]; source[i, j+1] = img_gray[i, j+1]
    if num == 1:
        source = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
    return source[:, :, None]

def predict(img_gray, num, use_ocnnp):
    x = parse_sample(img_gray, num)
    p = nnp.forward(x[:, :, 0])
    p = np.around(p)
    if num == 1:
        p = cv2.rotate(p, cv2.ROTATE_90_COUNTERCLOCKWISE)
    pn = np.zeros(img_gray.shape); pn[1:-1, 1:-1] = p
    if use_ocnnp:
        pn = np.around(R.ocnnp_optimize(pn, target_parity=num))   # round: the repo's optimizer returns fractional predictions, giving non-integer watermarked pixels
    return pn

def embed(img_gray, msg, use_ocnnp, use_lsbc, lsbc_mode):
    half = int(len(msg)/2)
    for num in range(2):
        pn = predict(img_gray, num, use_ocnnp)
        w = R.cnn_histogram_shifting(img_gray, pn, msg[num*half:(num+1)*half], num, use_lsbc=use_lsbc, lsbc_mode=lsbc_mode)
        img_gray = np.array(w, dtype=np.float64)
    return img_gray

def psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 100.0 if mse < 1e-10 else 10*math.log10(255.0**2/mse)

METHODS = {
  'CNNP':               dict(use_ocnnp=False, use_lsbc=False, lsbc_mode='mean'),
  'CNNP+LSBC(mean)':    dict(use_ocnnp=False, use_lsbc=True,  lsbc_mode='mean'),
  'CNNP+LSBC-spread':   dict(use_ocnnp=False, use_lsbc=True,  lsbc_mode='spread'),
  'OCNNP':              dict(use_ocnnp=True,  use_lsbc=False, lsbc_mode='mean'),
}
if __name__ == '__main__':
    imgdir, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else sorted(os.listdir(imgdir))
    lengths = [10000, 50000, 100000]
    rows = []
    for fn in names:
        img = cv2.imread(os.path.join(imgdir, fn))
        g = cv2.cvtColor(cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
        for L in lengths:
            msg = np.random.RandomState(1234+L).randint(0, 2, [L])
            for m, kw in METHODS.items():
                t0 = time.time()
                try:
                    w = embed(g.copy(), msg, **kw)
                    p = psnr(w, g); s = ssim(g, w, data_range=255)
                    row = dict(image=fn, bits=L, method=m, psnr=round(p, 4), ssim=round(float(s), 6), sec=round(time.time()-t0, 1))
                except Exception as e:
                    row = dict(image=fn, bits=L, method=m, error=repr(e))
                print(json.dumps(row), flush=True)
                rows.append(row)
    json.dump(rows, open(out, 'w'), indent=1)
