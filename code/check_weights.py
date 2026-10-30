import torch, numpy as np
ck = torch.load('model_parameter/model_state.pth', map_location='cpu')
sd = ck['network']
w = np.load('code/cnnp_weights.npz')
print("keys only in npz:", [k for k in w.files if k not in sd])
print("mismatched values:", [k for k in w.files if k in sd and not np.allclose(sd[k].numpy(), w[k])])