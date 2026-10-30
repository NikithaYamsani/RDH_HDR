import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, decode as D, diag
R = X.R
# one pass; decoder is GIVEN the true (original-image) order -- proves the decoder is correct and the order is the only obstacle
g = cv2.cvtColor(cv2.resize(cv2.imread(sys.argv[1]), (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float64)
msg = np.random.RandomState(5).randint(0, 2, [5000]); pn = X.predict(g, 0, False)
for name, mode, fn in [('LSBC(mean)', 'mean', R.calculate_lsbc), ('LSBC-spread', 'spread', R.calculate_lsbc_spread)]:
    w = R.cnn_histogram_shifting(g, pn, msg, 0, use_lsbc=True, lsbc_mode=mode)
    true_order = diag.order(g, 0, fn)
    real = diag.order
    diag.order = lambda img, num, f, _o=true_order: _o       # oracle order
    r, wm = D.decode_pass(np.array(w, dtype=np.float64), pn, 0, fn)
    diag.order = real
    print(name, 'with oracle order -> image exact:', bool(np.array_equal(r, g)), '| bits exact:', bool(np.array_equal(np.array(wm), msg)))
