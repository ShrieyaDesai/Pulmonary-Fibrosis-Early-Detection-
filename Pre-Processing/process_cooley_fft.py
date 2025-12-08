# process_cooley_fft.py
"""
Usage:
    # Process entire dataset (default settings)
    python process_cooley_fft.py

    # Process dataset but use a shorter window, or custom output names:
    python process_cooley_fft.py --root Clean_Lung_Sounds --sr 16000 --window_s 2.0 --hop_s 1.0 --n_fft 4096 --out features_cooley.npz --csv_out features_cooley.csv

What it does:
 - Scans ROOT (default "Clean_Lung_Sounds") for subfolders (each subfolder is a class)
 - Loads each .wav, resamples to sr (default 16000)
 - Applies bandpass_filter(y, 100, 2500)
 - Splits into windows (window_s, hop_s)
 - For each window: computes Cooley-Tukey radix-2 FFT (one-sided magnitude)
 - Saves:
     - compressed NPZ with X (num_windows x n_bins), y (labels), meta (list of {file,start,label})
     - CSV with feature-like columns (magnitudes) and label (optionally large)
"""
import os, argparse, math
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
import csv
from scipy.signal import butter, filtfilt

# ----------------------------
# Cooley-Tukey (radix-2) FFT
# ----------------------------
def _next_pow2(n):
    return 1 << ((n - 1).bit_length())

#makes all n as positive 
#If n is already a power of two, it returns n.
# if n<=0 ; compute the next power of 2 ≥ n.
#1 << k → left-shift the number 1 by k bits, which is equivalent to computing 2^k.
#(n - 1).bit_length() → gives the number of bits required to represent n-1 in binary.
#It returns 2^(bit_length(n-1))

############################################

#The number of bits needed to represent n-1 is exactly the exponent of the smallest power of 2 ≥ n
#Shifting 1 left by that many bits constructs that power of 2.
#radix-2 FFT requires lengths that are powers of two and  _next_pow2 pads input to satisfy that.

##############################################

def _bit_reversed_indices(n): #computes bit-reversal indices for in-place reordering.
    #returns an np array and takes input of int 
    bits = n.bit_length() - 1
    rev = np.zeros(n, dtype=int)
    for i in range(n):
        x = i
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev





def cooley_tukey_fft(x): #iterative radix-2 FFT
    x = np.asarray(x, dtype=complex)
    n0 = x.shape[0]
    if n0 == 0:
        return x
    n = _next_pow2(n0)
    if n != n0:
        x = np.pad(x, (0, n - n0))
    rev = _bit_reversed_indices(n)
    x = x[rev]
    m = 2
    while m <= n:
        half = m // 2
        w_m = np.exp(-2j * np.pi / m)
        for k in range(0, n, m):
            w = 1.0 + 0j
            for j in range(half):
                t = w * x[k + j + half]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + half] = u - t
                w *= w_m
        m *= 2
    return x

def rfft_mag_cooley(x, n_fft=None):
    x = np.asarray(x, dtype=float)
    if n_fft is None:
        n_fft = _next_pow2(len(x))
    X = cooley_tukey_fft(x[:n_fft])
    half = n_fft // 2 + 1
    return np.abs(X[:half])

# ----------------------------
# Filtering (zero-phase Butterworth)
# ----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.9999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(y, lowcut=100.0, highcut=2500.0, fs=16000, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # filtfilt may fail on very short signals; caller ensures adequate length
    y_f = filtfilt(b, a, y)
    return y_f

# ----------------------------
# Windowing helper
# ----------------------------
def sliding_windows(y, sr, window_s=2.0, hop_s=1.0):
    wlen = int(window_s * sr)
    hop = int(hop_s * sr)
    if wlen <= 0:
        raise ValueError("window_s too small or sr too small")
    starts = list(range(0, max(1, len(y) - wlen + 1), hop))
    windows = [y[s:s + wlen] for s in starts]
    return windows, starts

# ----------------------------
# Main processing: dataset -> cooley-fft mags
# ----------------------------
FOLDER_LABEL_MAP = {
    "Fibrosis_Positive": 1,
    "Fibrosis_Negative": 0,
    "positive": 1,
    "negative": 0,
    # add other exact folder names as needed
}


def infer_label_from_folder(folder_name):
    low = folder_name.lower()
    # exact match first
    for k, v in FOLDER_LABEL_MAP.items():
        if low == k:
            return v
    # contains match next
    for k, v in FOLDER_LABEL_MAP.items():
        if k in low:
            return v
    # fallback
    if any(k in low for k in ("Fibrosis_Positive", "Positive", "Fibrosis")): return 0
    if any(k in low for k in ("Fibrosis_Negative", "Negative", "Fibrosis")): return 1
    return None

def process_dataset(root="Clean_Lung_Sounds", sr=16000, window_s=2.0, hop_s=1.0,
                    n_fft=4096, lowcut=100.0, highcut=2500.0,
                    out_npz="features_cooley.npz", csv_out="features_cooley.csv"):
    X_list = []
    meta = []
    labels = []
    file_counter = 0
    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        label = infer_label_from_folder(folder)
        if label is None:
            print("Skipping folder (unknown label):", folder)
            continue
        wavs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.flac', '.mp3'))]
        if len(wavs) == 0:
            continue
        for wname in tqdm(wavs, desc=f"Processing {folder}"):
            path = os.path.join(folder_path, wname)
            try:
                # load with librosa to preserve variety of formats; returns float32
                y, orig_sr = librosa.load(path, sr=None, mono=True)
            except Exception as e:
                print("Failed to load", path, "->", e)
                continue
            # resample if needed
            if orig_sr != sr:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

            # normalize (optional but recommended)
            if np.max(np.abs(y)) > 0:
                y = y / (np.max(np.abs(y)) + 1e-9)
            # apply bandpass filtering (zero-phase)
            try:
                y_f = bandpass_filter(y, lowcut=lowcut, highcut=highcut, fs=sr)
            except Exception as e:
                # fallback: if filtfilt fails (too short signal), skip filtering
                print(f"Filtering failed for {path} (len={len(y)}). Skipping filter. Err:", e)
                y_f = y
            # window the filtered signal
            windows, starts = sliding_windows(y_f, sr, window_s=window_s, hop_s=hop_s)
            if len(windows) == 0:
                # if file shorter than window length, pad and do one window
                pad_len = int(window_s * sr) - len(y_f)
                if pad_len > 0:
                    y_pad = np.pad(y_f, (0, pad_len))
                else:
                    y_pad = y_f
                windows = [y_pad]
                starts = [0]
            # compute Cooley-Tukey rfft magnitude on each window
            for i, win in enumerate(windows):
                mag = rfft_mag_cooley(win, n_fft=n_fft)  # this is the Cooley-Tukey call
                X_list.append(mag)
                labels.append(label)
                meta.append({"file": path, "start_sample": starts[i], "label": label})
            file_counter += 1

    if len(X_list) == 0:
        raise RuntimeError("No windows processed. Check dataset path and file formats.")
    # pad/truncate mags to same length (should be same since n_fft fixed)
    X = np.vstack([x for x in X_list])  # shape (num_windows, n_bins)
    y = np.array(labels)
    # save npz
    np.savez_compressed(out_npz, X=X, y=y, meta=np.array(meta, dtype=object))
    print("Saved NPZ:", out_npz, "X shape:", X.shape, "y shape:", y.shape)
    # also write CSV metadata and a small CSV of first N magnitude bins + label (be careful: large file)
    
    with open(csv_out, "w", newline="") as fh:
        writer = csv.writer(fh)
        n_bins = X.shape[1]
        header = [f"mag_{i}" for i in range(n_bins)] + ["label", "file", "start_sample"]
        writer.writerow(header)
        for row_idx in range(X.shape[0]):
            writer.writerow(list(X[row_idx].tolist()) + [int(y[row_idx]), meta[row_idx]["file"], meta[row_idx]["start_sample"]])
    print("Saved CSV:", csv_out)
    return out_npz, csv_out

# simple recorder to capture a test sample and process it
def record_test(out_path="recordings/test_record.wav", duration=5.0, sr=16000):
    import sounddevice as sd
    import soundfile as sf
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Recording {duration}s to {out_path} (Speak/inhale/exhale)...")
    data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(out_path, data, sr)
    print("Recorded:", out_path)
    return out_path

if __name__ == "__main__":
    from tqdm import tqdm  # ensure tqdm imported for progress bars
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Clean_Lung_Sounds")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--window_s", type=float, default=2.0)
    parser.add_argument("--hop_s", type=float, default=1.0)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--lowcut", type=float, default=100.0)
    parser.add_argument("--highcut", type=float, default=2500.0)
    parser.add_argument("--out_npz", default="features_cooley.npz")
    parser.add_argument("--csv_out", default="features_cooley.csv")
    parser.add_argument("--record_test", action="store_true", help="Record a short test wav to recordings/test_record.wav before processing")
    args = parser.parse_args()

    if args.record_test:
        rec_path = record_test(duration=5.0, sr=args.sr)
        # You can manually move rec_path into a folder under Clean_Lung_Sounds/<class>/ to include in batch processing
    print("Starting dataset processing (Cooley-Tukey FFT on windows)...")
    process_dataset(root=args.root, sr=args.sr, window_s=args.window_s, hop_s=args.hop_s,
                    n_fft=args.n_fft, lowcut=args.lowcut, highcut=args.highcut,
                    out_npz=args.out_npz, csv_out=args.csv_out)
