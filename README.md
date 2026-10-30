# Reversible Data Hiding with a CNN Predictor: Reproduction and a Decodable Embedding Order

This repository reproduces CNN-prediction-based reversible data hiding (RDH) in Python, adds the extraction (decoding) procedure, and makes the complexity-based embedding order reversible.

It started from the official code of Hu and Xiang (CNNP, 2021), which only computed PSNR and had no decoder. It was ported from MATLAB to Python, extended with OCNNP (Luo et al., 2024), and then extended with the contributions below.

**Scope:** 8-bit grayscale images, 512x512. HDR RGB support and confidence-from-network embedding are **future work, not implemented**.

## What is in this repository

| Part | Where | Status |
|---|---|---|
| CNNP predictor, chessboard split, expansion and histogram-shifting embedding | root (`rdh_embed.py`, `utils.py`, `main.py`, `model/`) | Reproduction of Hu and Xiang (2021); embedding ported from MATLAB to Python, with a custom range coder replacing MATLAB's `arithenco` |
| OCNNP diagonal-neighbour refinement (`ocnnp_optimize`) | `rdh_embed.py` | Reproduction of Luo et al. (2024); did not help in our tests |
| LSBC and LSBC-spread ordering | `rdh_embed.py` | Implemented as described; **cannot be decoded** (see below) |
| Extraction / decoder, exact-recovery checks | `code/decode.py`, `code/rev_test.py` | New |
| ENC: a complexity order the decoder can recompute | `code/run_final.py`, `code/rdh_ace.py` | New |
| Distortion-based threshold search (GT: global thresholds, AT: three confidence classes) | `code/rdh_ace.py` | New |
| Experiment scripts, logs, raw results | `code/`, `logs/`, `results/` | New |

## Findings

1. **LSBC as defined cannot be inverted.** Its neighbourhood includes pixels from the half that the embedder modifies, so the decoder cannot rebuild the embedding order. With a real decoder, LSBC-spread recovered 0 of 12 runs (`results/spread.json`). Measuring PSNR without a decoder cannot show that a method is reversible.
2. **ENC fixes this.** The complexity of a pixel is computed only from the 12 surrounding pixels of the other half (4 direct plus 8 knight-move neighbours), which the decoder can recompute.
3. **Most of the gain comes from the threshold search**, not from the order. The order alone adds only a few tenths of a dB.
4. **The confidence classes (AT) give a smaller and less certain gain than the global search (GT).** On held-out images AT was no better than GT.
5. **OCNNP did not improve on CNNP** in our reproduction. Its predictions must be rounded to integers first, otherwise the watermarked pixels are not integers.

## Results

PSNR in dB, average over the four standard images (Lena, Barbara, Peppers, Yacht), produced by the scripts in this repository:

| Payload | CNNP | ENC | ENC+GT | ENC+AT | ENC+AT gain over CNNP |
|---|---|---|---|---|---|
| 10,000 bits | 55.47 | 55.97 | 56.99 | 57.01 | +1.54 |
| 50,000 bits | 46.81 | 47.05 | 48.38 | 48.76 | +1.96 |
| 100,000 bits | 42.54 | 42.65 | 44.12 | 44.49 | +1.96 |

Over **all 13 test images** (4 standard plus 9 further images, 6 of them held out during design), ENC+AT is **0.84, 1.21 and 1.47 dB** above CNNP at the three payloads. Use these as the headline numbers; the standard-image figures above are higher because those images were also used to design the method.

ENC alone has a slightly lower SSIM than CNNP at 50,000 and 100,000 bits. With the threshold search, SSIM is higher than CNNP on the standard images (ENC+AT: 0.9980, 0.9887 and 0.9780, against 0.9975, 0.9843 and 0.9651 for CNNP).

Recovery: of 39 embeddings per method, 37 fit; all 37 recovered both the image and the message exactly. The other two (astronaut at 50,000 and 100,000 bits) raise a capacity error for every method.

Per-image PSNR and SSIM, and the raw outputs, are in `results/` and `logs/`. `results/old.json` is an earlier draft and should not be quoted.

## Setup

```
pip install numpy opencv-python scikit-image
pip install torch torchvision      # only needed for main.py and code/check_weights.py
```

Run all commands from the repository root.

## Reproducing the results

Check that the NumPy weights used by the experiments are identical to the pretrained model:
```
python code/check_weights.py
```
Expected: both lists empty.

Decoder and exact recovery on one image (the four ordering variants; the first two are expected to fail decoding):
```
python code/rev_test.py standard_test_images/Lena.bmp 10000
```

CNNP versus ENC, with exact-recovery flags:
```
python code/run_final.py standard_test_images results_std.json Lena.bmp,Barbara.bmp,Peppers.bmp,yacht.bmp
```

ENC with threshold search (`1` = global thresholds GT, `3` = three confidence classes AT):
```
python code/run_ace.py Lena,Barbara,Peppers,Yacht results_ace.json 1,3
```
Each embedding takes roughly 10 to 40 seconds on a CPU. The other `run_*.py` scripts produced the other files in `results/`; their output files are named after the script.

## Limitations

- Grayscale 8-bit images only. HDR support (reversible float handling, per-channel prediction) and confidence taken from the network are not implemented.
- The CNN predictor uses the pretrained CNNP weights and was not retrained.
- Nine of the 13 test images come from scikit-image and scikit-learn sample sets, which are not a standard RDH benchmark.
- The PSNR gain is measured against CNNP; we did not compare with other RDH families.

## References

- R. Hu and S. Xiang, "CNN Prediction Based Reversible Data Hiding," IEEE Signal Processing Letters, vol. 28, pp. 464-468, 2021. doi: 10.1109/LSP.2021.3059202
- Y. Luo et al., “Reversible data hiding based on optimized CNN predictor and prediction error expansion with lower surround background complexity,” Comput. Electr. Eng., vol. 119, 2024.

The CNNP network, weights and original MATLAB files are from Hu and Xiang's public release.
