import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X
R = X.R
def order(img, num, fn):
    M, N = img.shape
    place = np.zeros((M, N), dtype=int); comp = np.zeros((M, N))
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (i+j) % 2 == num:
                place[i, j] = 1; comp[i, j] = fn(img, i, j)
    idx = np.flatnonzero(place.T); coords = [(k % M, k // M) for k in idx]
    vals = [comp[i, j] for (i, j) in coords]
    o = np.argsort(vals, kind='stable')
    return [coords[k] for k in o]
FN = {'base': R.calculate_complexity, 'lsbc_mean': R.calculate_lsbc, 'lsbc_spread': R.calculate_lsbc_spread}
if __name__ == '__main__':
    path = sys.argv[1]; L = 50000
    img = cv2.imread(path)
    g = cv2.cvtColor(cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
    msg = np.random.RandomState(7).randint(0, 2, [L//2])
    pn = X.predict(g, 0, False)
    for name, fn in FN.items():
        w = R.cnn_histogram_shifting(g, pn, msg, 0, use_lsbc=(name != 'base'), lsbc_mode='spread' if name == 'lsbc_spread' else 'mean')
        o1 = order(g, 0, fn); o2 = order(np.array(w, dtype=np.float64), 0, fn)
        n = L//2 + 6*19 + 3000   # embedding region (payload + header slack)
        same_full = (o1 == o2)
        same_set = len(set(o1[:n]) & set(o2[:n])) / n
        pos_match = np.mean([a == b for a, b in zip(o1[:n], o2[:n])])
        print(f'{os.path.basename(path)} {name}: identical order={same_full}  top-{n} same-set={same_set:.4f}  same-position={pos_match:.4f}')
