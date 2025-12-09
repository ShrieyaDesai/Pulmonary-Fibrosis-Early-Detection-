Explanation of process_cooley_fft.py:
This script converts lung sounds into spectral vectors suitable for machine learning.
First, resampled each recording to a common sampling rate so frequency bins line up. 
Normalized to remove amplitude differences. 
Applied a zero-phase bandpass from 100 to 2500 Hz to focus on the clinically relevant band and remove low-frequency breathing motion and high-frequency noise.
The signal is split into overlapping windows (2s length, 1s hop). 
For each window we compute the one-sided magnitude spectrum using a radix-2 FFT — this gives us a fixed-length vector of spectral magnitudes per window.
We save all windows and metadata in a compressed NPZ; this is the feature matrix for downstream ML. 
Key choices (sr, window size, n_fft, bandpass) trade off time vs frequency resolution and noise robustness; we tuned these based on literature and domain intuition.

I also found two issues in the original code — label mapping and FFT padding — and fixed them to ensure correct labels and no truncation

Cooley-Tukey FFT Algorithm:
This algorithm is an FFT Algorithm used in signal processings

The FFT (Fast Fourier Transform) turns a time signal (like a cough or breath) into its frequencies (how much of each pitch is present).

A direct method (naive DFT) checks every frequency against every sample → takes O(n²) time (very slow when n is large).

Cooley–Tukey reorganizes the work using divide-and-conquer so it runs in O(n log n) time — hugely faster for big n.

Key features:

Works best when the number of samples is a power of two (2, 4, 8, 16, ...).

Uses “butterfly” combine steps: small FFT results get merged into bigger ones.

Cooley–Tukey FFT is a clever split-and-merge trick that finds frequencies much faster than checking everything one-by-one 
It is widely used because it’s fast, simple, and works great when the data length is a power of two.

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

In the Cooley–Tukey radix-2 FFT, the algorithm expects the input array to be arranged in bit-reversed order before doing the butterfly operations.

Reason 1: Makes the FFT iterative: The FFT is normally recursive (divide-and-conquer).
To make it iterative, Cooley–Tukey rearranges the input so that the butterflies operate on the correct pairs.

Reason 2: Makes butterfly patterns line up perfectly

Reason 3: Makes radix-2 FFT run in O(N log N) time
For every number i from 0 to n−1:

Take its binary representation.

Reverse its bits.

Store the reversed value in rev[i].

Example:
i = 3 (binary 011)
Reversed → 110 = 6
So rev[3] = 6.


Returns complex numbers (magnitude + phase). For many ML tasks we just take magnitude.




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

m is the size of the DFT-subproblem being computed in this stage. It starts at 2 and doubles every stage: 2, 4, 8, ..., n.

half is number of pairs (butterflies) per block: for m=8, half=4.

w_m is the primitive m-th complex root of unity 𝑒^(−2𝜋𝑖/𝑚)




def rfft_mag_cooley(x, n_fft=None):
    x = np.asarray(x, dtype=float)
    if n_fft is None:
        n_fft = _next_pow2(len(x))
    X = cooley_tukey_fft(x[:n_fft])
    half = n_fft // 2 + 1
    return np.abs(X[:half])
This function does:

Take the sound.

Run your Cooley–Tukey FFT on it (that finds all frequencies).

Keep only the first half (because real sounds have mirrored frequencies—you don’t need the second half).

Take only the strength (magnitude), not direction/phase.

Return that list of numbers.

So it turns sound → a row of frequency energies


: x[:n_fft] truncates the signal if it's longer than n_fft.

For real input signals, the FFT is symmetric:

Frequency bins above Nyquist are mirror images.

So you only keep the first half → “one-sided spectrum”.

Most ML models learn mainly from how much energy is in each frequency.

Phase rarely helps for lung-sound classification.

Magnitude gives cleaner, stable features.

rfft_mag_cooley converts each audio window into its one-sided magnitude spectrum using our custom FFT.
Only the first half of the FFT is taken because the input is real-valued and the second half is redundant.
Magnitude is used instead of phase because it captures the energy distribution of breath sounds, which is key for detecting fibrosis.”

A bigger n_fft = more detail in the frequency picture.A bigger n_fft = more detail in the frequency picture.


Powers of w_m produce the twiddle factors used inside butterflies.
