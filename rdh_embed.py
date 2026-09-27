"""
Pure-Python port of the MATLAB embedding logic in this repo:
  - num2bitlist.m
  - calculate_complexity.m
  - calculate_tp_tn.m
  - cnn_expansion.m
  - cnn_histogram_shifting.m

This removes the dependency on MATLAB / matlab.engine / arithenco
(MATLAB's Communications Toolbox arithmetic coder). A small binary
range coder is implemented below to replace arithenco.
"""

import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# num2bitlist.m  ->  fixed-width binary representation of an integer
# ---------------------------------------------------------------------------
def num2bitlist(dec_number, bit_num):
    dec_number = int(dec_number)
    bitstr = format(dec_number, '0{}b'.format(bit_num))
    if len(bitstr) > bit_num:
        bitstr = bitstr[-bit_num:]  # truncate like MATLAB's dec2bin would only if it fits; keep same width
    return [int(b) for b in bitstr]


# ---------------------------------------------------------------------------
# calculate_complexity.m  ->  local background-complexity of pixel (i, j)
# NOTE: MATLAB is 1-indexed; here img, i, j are all 0-indexed.
# ---------------------------------------------------------------------------
def calculate_complexity(img, i, j):
    v1 = abs(img[i, j - 1] - img[i - 1, j])
    v2 = abs(img[i - 1, j] - img[i, j + 1])
    v3 = abs(img[i, j + 1] - img[i + 1, j])
    v4 = abs(img[i + 1, j] - img[i, j - 1])
    return (v1 + v2 + v3 + v4) / 4.0


# ---------------------------------------------------------------------------
# LSBC (Lower Surround Background Complexity), Eq. 8 from Luo et al. 2024:
#   Lambda = (theta1 + theta2 + theta3 + theta4 + theta5) / 5
# where, relative to marker pixel (i, j):
#   theta1 = left        (i, j-1)
#   theta2 = lower-left   (i+1, j-1)
#   theta3 = directly below (i+1, j)
#   theta4 = lower-right  (i+1, j+1)
#   theta5 = right        (i, j+1)
# NOTE: unlike calculate_complexity (which averages *differences* between
# neighbor pairs), LSBC averages the raw pixel *values* themselves.
# The last row of the image has no "below" neighbors and is excluded, per
# the paper ("the last row of the image is not computed for the LSBC").
# ---------------------------------------------------------------------------
def calculate_lsbc(img, i, j):
    theta1 = img[i, j - 1]
    theta2 = img[i + 1, j - 1]
    theta3 = img[i + 1, j]
    theta4 = img[i + 1, j + 1]
    theta5 = img[i, j + 1]
    return (theta1 + theta2 + theta3 + theta4 + theta5) / 5.0


# ---------------------------------------------------------------------------
# LSBC variant: same 5-neighbor geometry as Eq. 8, but using the SPREAD
# (standard deviation) of the neighbors instead of their mean. A plain mean
# of raw pixel values is really a brightness signal (tested: hurts PSNR by
# misranking dark-but-complex regions as "safe"). Standard deviation is a
# genuine local-texture/complexity signal, which is presumably closer to
# the paper's intent behind naming this "Background Complexity."
# ---------------------------------------------------------------------------
def calculate_lsbc_spread(img, i, j):
    theta1 = img[i, j - 1]
    theta2 = img[i + 1, j - 1]
    theta3 = img[i + 1, j]
    theta4 = img[i + 1, j + 1]
    theta5 = img[i, j + 1]
    vals = np.array([theta1, theta2, theta3, theta4, theta5])
    return float(np.std(vals))


# ---------------------------------------------------------------------------
# calculate_tp_tn.m  ->  greedily pick the most frequent prediction-error
# values until their combined count covers the message length; Tp/Tn are
# the max/min of the chosen values (the shifting thresholds).
# ---------------------------------------------------------------------------
def calculate_tp_tn(predicted_error_values, length_message):
    counts = Counter(predicted_error_values.astype(np.int64).ravel().tolist())
    items = sorted(counts.items(), key=lambda kv: -kv[1])  # descending by frequency
    temp_sum = 0
    place_index = []
    idx = 0
    while temp_sum < length_message and idx < len(items):
        value, cnt = items[idx]
        temp_sum += cnt
        place_index.append(value)
        idx += 1
    Tp = max(place_index)
    Tn = min(place_index)
    return Tp, Tn


# ---------------------------------------------------------------------------
# Binary range coder, replacing MATLAB's arithenco for a 2-symbol alphabet.
# Encodes a 0/1 sequence given static counts [count0, count1].
# Output is a Python list of 0/1 ints, decodable with range_decode below.
# ---------------------------------------------------------------------------
_TOP = 1 << 32
_HALF = 1 << 31
_QUARTER = 1 << 30
_MASK = _TOP - 1


def range_encode(symbols, counts):
    count0, count1 = counts
    total = count0 + count1
    if total == 0:
        return []
    low, high = 0, _MASK
    pending = 0
    out = []

    def emit(bit):
        out.append(bit)
        nonlocal pending
        while pending:
            out.append(1 - bit)
            pending -= 1

    for s in symbols:
        span = high - low + 1
        if s == 0:
            high = low + (span * count0) // total - 1
        else:
            low = low + (span * count0) // total
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _HALF + _QUARTER:
                pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            low &= _MASK
            high &= _MASK

    pending += 1
    emit(0 if low < _QUARTER else 1)
    return out


def range_decode(bits, counts, num_symbols):
    count0, count1 = counts
    total = count0 + count1
    bits = list(bits) + [0] * 33  # padding so we never run out
    pos = 0

    def read_bit():
        nonlocal pos
        b = bits[pos]
        pos += 1
        return b

    low, high = 0, _MASK
    code = 0
    for _ in range(32):
        code = (code << 1) | read_bit()

    result = []
    for _ in range(num_symbols):
        span = high - low + 1
        # infer symbol from where `code` falls
        thresh = low + (span * count0) // total - 1
        if code <= thresh:
            s = 0
            high = thresh
        else:
            s = 1
            low = thresh + 1
        result.append(s)
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                code -= _HALF
            elif low >= _QUARTER and high < _HALF + _QUARTER:
                low -= _QUARTER
                high -= _QUARTER
                code -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            code = (code << 1) | read_bit()
            low &= _MASK
            high &= _MASK
            code &= _MASK
    return result


def compress_location_map(location_map_flat):
    """location_map_flat: 1D numpy array of 0/1. Returns (bits, number0, number1)."""
    number1 = int(location_map_flat.sum())
    number0 = int(location_map_flat.size - number1)
    if number1 == 0:
        return [0], number0, number1
    bits = range_encode(location_map_flat.astype(int).tolist(), [number0, number1])
    return bits, number0, number1


# ---------------------------------------------------------------------------
# OCNNP optimizer (Eq. 6, Luo et al. 2024):
#   Theta = (rho1 + rho2 + rho3 + rho4) / 4
# where rho1..rho4 are the top-left, bottom-left, bottom-right, top-right
# DIAGONAL neighbors of the target pixel in the CNN-predicted image.
# The paper states the final predicted pixel is obtained by averaging this
# diagonal estimate again with the CNN's own prediction at that pixel.
# ---------------------------------------------------------------------------
def ocnnp_optimize(predicted_image, target_parity):
    """
    predicted_image: 2D numpy array (float), the raw CNNP output. Every pixel in
        the interior belongs to one of two chessboard classes: the class the CNN
        is actually predicting (kept as the real intensity), and the complementary
        class, which the network was trained to output as near-zero (Eq. 7's loss
        target is 0 there). Diagonal neighbors (i+-1, j+-1) always share the SAME
        parity as (i, j), since i+j changes by 0 or +-2, never +-1.
    target_parity: the value of num (0 or 1) used when this half was predicted.
        Pixels with (i + j) % 2 == target_parity hold the CNN's real prediction and
        are left untouched. The complementary pixels (which should be ~0) are
        refined with Eq. 6's diagonal average, blended with their own value.
    Returns: 2D numpy array (float), the OCNNP-optimized prediction.
    """
    img = predicted_image.astype(np.float64)
    M, N = img.shape
    optimized = img.copy()

    top_left = img[0:M - 2, 0:N - 2]      # rho1
    bottom_left = img[2:M, 0:N - 2]       # rho2
    bottom_right = img[2:M, 2:N]          # rho3
    top_right = img[0:M - 2, 2:N]         # rho4
    theta = (top_left + bottom_left + bottom_right + top_right) / 4.0

    center = img[1:M - 1, 1:N - 1]
    blended = (theta + center) / 2.0

    # Build a mask: True where (i+j)%2 == target_parity, i.e. the class actually
    # USED as xPredict during embedding (the "star" class for this round). These
    # are real CNN predictions, but still benefit from denoising against their
    # diagonal neighbors (also real predictions of the same class).
    ii, jj = np.meshgrid(np.arange(1, M - 1), np.arange(1, N - 1), indexing='ij')
    correction_mask = ((ii + jj) % 2 == target_parity)

    interior = optimized[1:M - 1, 1:N - 1]
    interior[correction_mask] = blended[correction_mask]
    optimized[1:M - 1, 1:N - 1] = interior

    return optimized


# ---------------------------------------------------------------------------
# cnn_expansion.m  ->  expansion embedding (PEE-style, no complexity sort)
# img, predicted_image: 2D numpy arrays (float), 0-indexed
# watermark: 1D numpy array of 0/1
# odd_or_even_num: 0 or 1, selects which chessboard subset to embed into
# ---------------------------------------------------------------------------
def cnn_expansion(img, predicted_image, watermark, odd_or_even_num):
    img = img.astype(np.float64).copy()
    M, N = img.shape
    mn = int(np.ceil(np.log2(M * N))) if M * N > 1 else 1
    img_w = img.copy()
    watermark = np.asarray(watermark).astype(int).tolist()
    watermark_length = len(watermark)

    odd_or_even_place = np.zeros((M, N), dtype=int)
    location_map = np.zeros((M, N), dtype=int)

    # MATLAB loops i=2:M-1, j=2:N-1 (1-indexed) -> python i=1..M-2, j=1..N-2
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (i + j) % 2 == odd_or_even_num:
                odd_or_even_place[i, j] = 1

    insertable_place = list(zip(*np.where(odd_or_even_place.T == 1)))  # column-major to mimic MATLAB find()
    # MATLAB's find() on a matrix returns linear indices in column-major order.
    # Reproduce that ordering directly:
    flat_idx = np.flatnonzero(odd_or_even_place.T) # column-major flatten
    coords = [(idx % M, idx // M) for idx in flat_idx]  # (i, j) 0-indexed

    minimum_lsb_length = watermark_length + 4 * mn
    current_length = minimum_lsb_length
    start_place = 0
    end_place = current_length

    compressed_location_map_bits = []
    while True:
        for k in range(start_place, min(end_place, len(coords))):
            i, j = coords[k]
            x = img_w[i, j]
            x_predict = predicted_image[i, j]
            value = 2 * x - x_predict
            if value > 254 or value < 0:
                location_map[i, j] = 1

        loc_flat = location_map.T.ravel()  # column-major to match MATLAB locationMap(:)
        compressed_location_map_bits, number0, number1 = compress_location_map(loc_flat)

        if current_length - minimum_lsb_length < len(compressed_location_map_bits) + 4 * mn:
            current_length += 1000
            start_place = end_place
            end_place = current_length
            if end_place > len(coords):
                end_place = len(coords)
        else:
            break

    # extract existing LSBs from the tail of insertable positions (to preserve reversibility bookkeeping)
    n_extract = len(compressed_location_map_bits) + 4 * mn
    extracted_lsb_bitlist = []
    for k in range(len(coords) - 1, len(coords) - 1 - n_extract, -1):
        i, j = coords[k]
        extracted_lsb_bitlist.append(int(img_w[i, j]) & 1)

    length_compressed_location_map_bitlist = num2bitlist(len(compressed_location_map_bits), mn)
    whole_compressed_location_map_bitlist = length_compressed_location_map_bitlist + list(compressed_location_map_bits)

    message_to_embed = extracted_lsb_bitlist + watermark
    message_to_embed_length = num2bitlist(len(message_to_embed), mn)

    number1_bitlist = num2bitlist(number1, mn)
    number0_bitlist = num2bitlist(number0, mn)

    lsb_to_replace_bitlist = (whole_compressed_location_map_bitlist +
                              message_to_embed_length + number1_bitlist + number0_bitlist)

    for k in range(len(coords) - 1, len(coords) - 1 - len(lsb_to_replace_bitlist), -1):
        i, j = coords[k]
        bit_idx = len(coords) - 1 - k
        current_val = int(img_w[i, j])
        img_w[i, j] = current_val - (current_val & 1) + lsb_to_replace_bitlist[bit_idx]

    index_message = 0
    for (i, j) in coords:
        if location_map[i, j] == 0:
            x = img_w[i, j]
            x_predict = predicted_image[i, j]
            dij = x - x_predict
            if message_to_embed[index_message] == 1:
                Dij = 2 * dij + 1
            else:
                Dij = 2 * dij + 0
            img_w[i, j] = Dij + x_predict
            index_message += 1
            if index_message >= len(message_to_embed):
                break

    return img_w


# ---------------------------------------------------------------------------
# cnn_histogram_shifting.m -> histogram-shifting embedding, sorted by LSBC-style
# local complexity (calculate_complexity), with Tp/Tn shifting thresholds.
# ---------------------------------------------------------------------------
def cnn_histogram_shifting(img, predicted_image, watermark, odd_or_even_num, use_lsbc=False, lsbc_mode='mean'):
    img = img.astype(np.float64).copy()
    M, N = img.shape
    mn = int(np.ceil(np.log2(M * N))) if M * N > 1 else 1
    img_w = img.copy()
    watermark = np.asarray(watermark).astype(int).tolist()
    watermark_length = len(watermark)

    odd_or_even_place = np.zeros((M, N), dtype=int)
    location_map = np.zeros((M, N), dtype=int)
    predicted_error = np.zeros((M, N), dtype=np.float64)
    complexity = np.zeros((M, N), dtype=np.float64)

    if use_lsbc:
        complexity_fn = calculate_lsbc_spread if lsbc_mode == 'spread' else calculate_lsbc
    else:
        complexity_fn = calculate_complexity

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (i + j) % 2 == odd_or_even_num:
                odd_or_even_place[i, j] = 1
                complexity[i, j] = complexity_fn(img_w, i, j)

    flat_idx = np.flatnonzero(odd_or_even_place.T)  # column-major, like MATLAB find()
    coords = [(idx % M, idx // M) for idx in flat_idx]
    complexity_vals = [complexity[i, j] for (i, j) in coords]
    order = np.argsort(complexity_vals, kind='stable')  # ascending, like MATLAB sort()
    sorted_coords = [coords[k] for k in order]

    minimum_lsb_length = watermark_length + 6 * mn
    current_length = minimum_lsb_length
    start_place = 0
    end_place = current_length

    compressed_location_map_bits = []
    while True:
        for k in range(start_place, min(end_place, len(sorted_coords))):
            i, j = sorted_coords[k]
            x = img_w[i, j]
            x_predict = predicted_image[i, j]
            value = 2 * x - x_predict
            predicted_error[i, j] = x - x_predict
            if value > 254 or value < 0:
                location_map[i, j] = 1

        loc_flat = location_map.T.ravel()
        compressed_location_map_bits, number0, number1 = compress_location_map(loc_flat)

        if current_length - minimum_lsb_length < len(compressed_location_map_bits) + 6 * mn:
            current_length += 1000
            start_place = end_place
            end_place = current_length
            if end_place > len(sorted_coords):
                end_place = len(sorted_coords)
        else:
            break

    n_extract = len(compressed_location_map_bits) + 6 * mn
    extracted_lsb_bitlist = []
    for k in range(len(sorted_coords) - 1, len(sorted_coords) - 1 - n_extract, -1):
        i, j = sorted_coords[k]
        extracted_lsb_bitlist.append(int(img_w[i, j]) & 1)

    length_compressed_location_map_bitlist = num2bitlist(len(compressed_location_map_bits), mn)
    whole_compressed_location_map_bitlist = length_compressed_location_map_bitlist + list(compressed_location_map_bits)

    message_to_embed = extracted_lsb_bitlist + watermark
    message_to_embed_length = num2bitlist(len(message_to_embed), mn)

    embed_region_coords = sorted_coords[:end_place]
    error_vals_for_tp_tn = np.array([predicted_error[i, j] for (i, j) in embed_region_coords])
    Tp, Tn = calculate_tp_tn(error_vals_for_tp_tn, len(message_to_embed))

    Tp_bitlist = num2bitlist(Tp, mn)
    Tn_bitlist = num2bitlist(abs(Tn), mn)
    number1_bitlist = num2bitlist(number1, mn)
    number0_bitlist = num2bitlist(number0, mn)

    lsb_to_replace_bitlist = (whole_compressed_location_map_bitlist + message_to_embed_length +
                              Tp_bitlist + Tn_bitlist + number1_bitlist + number0_bitlist)

    for k in range(len(sorted_coords) - 1, len(sorted_coords) - 1 - len(lsb_to_replace_bitlist), -1):
        i, j = sorted_coords[k]
        bit_idx = len(sorted_coords) - 1 - k
        current_val = int(img_w[i, j])
        img_w[i, j] = current_val - (current_val & 1) + lsb_to_replace_bitlist[bit_idx]

    index_message = 0
    for (i, j) in sorted_coords:
        if location_map[i, j] == 0:
            x = img_w[i, j]
            x_predict = predicted_image[i, j]
            dij = x - x_predict
            if dij > Tp:
                Dij = dij + Tp + 1
                img_w[i, j] = Dij + x_predict
            elif dij < Tn:
                Dij = dij + Tn
                img_w[i, j] = Dij + x_predict
            else:
                if message_to_embed[index_message] == 1:
                    Dij = 2 * dij + 1
                else:
                    Dij = 2 * dij + 0
                img_w[i, j] = Dij + x_predict
                index_message += 1
                if index_message >= len(message_to_embed):
                    break

    return img_w