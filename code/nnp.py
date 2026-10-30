import os
import numpy as np
W = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnnp_weights.npz'))
def conv(x, w, b, pad):
    # x: (Cin,H,W) float32 ; w: (Cout,Cin,k,k)
    Cin, H, Wd = x.shape; Cout, _, k, _ = w.shape
    if pad: x = np.pad(x, ((0,0),(pad,pad),(pad,pad)))
    Ho, Wo = x.shape[1]-k+1, x.shape[2]-k+1
    cols = np.empty((Cin*k*k, Ho*Wo), dtype=np.float32); r = 0
    for c in range(Cin):
        for dy in range(k):
            for dx in range(k):
                cols[r] = x[c, dy:dy+Ho, dx:dx+Wo].reshape(-1); r += 1
    y = w.reshape(Cout, -1) @ cols + b[:, None]
    return y.reshape(Cout, Ho, Wo)
lrelu = lambda x: np.where(x > 0, x, 0.01*x).astype(np.float32)
def seq(x, p, pad0, pad2):
    x = lrelu(conv(x, W[p+'.0.weight'], W[p+'.0.bias'], pad0))
    return conv(x, W[p+'.2.weight'], W[p+'.2.bias'], pad2)
def forward(img):               # img: (H,W) float
    x = img[None].astype(np.float32)
    o1 = seq(x, 'conv1', 0, 1); o2 = seq(x, 'conv2', 1, 1); o3 = seq(x, 'conv3', 2, 1)
    s = o1+o2+o3
    o4 = seq(s, 'conv4', 1, 1)
    o5 = seq(s+o4, 'conv5', 1, 1)
    return o5[0]
