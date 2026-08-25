import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- Load & preprocess ----------
def load_gray(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=float)

def frob_error(A, Ak):
    return np.linalg.norm(A - Ak, 'fro')

# ---------- Display helper ----------
def show_row(images, titles, cmap='gray'):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap=cmap)
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.show()  # Close the popup window to resume script execution

# ================= SQUARE IMAGE: EVD + SVD =================
A = load_gray('cat_21.png')
n = A.shape[0]

# --- Precompute EVD and SVD ONCE ---
eigvals, Q = np.linalg.eig(A)
idx = np.argsort(-np.abs(eigvals))
eigvals, Q = eigvals[idx], Q[:, idx]
Q_inv = np.linalg.inv(Q)

U, S, Vt = np.linalg.svd(A, full_matrices=False)

def fast_evd_reconstruct(k):
    Lk = np.diag(eigvals[:k])
    Ak = Q[:, :k] @ Lk @ Q_inv[:k, :]
    return np.real(Ak)

def fast_svd_reconstruct(k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

k_values = [10, 30, 60]

for k in k_values:
    Ak_evd = fast_evd_reconstruct(k)
    Ak_svd = fast_svd_reconstruct(k)

    show_row([A, Ak_evd, np.abs(A - Ak_evd)],
             ['Original', f'EVD Reconstructed (k={k})', 'Error |A-Ak|'])
    show_row([A, Ak_svd, np.abs(A - Ak_svd)],
             ['Original', f'SVD Reconstructed (k={k})', 'Error |A-Ak|'])

    print(f"k={k}: EVD error = {frob_error(A, Ak_evd):.2f}, "
          f"SVD error = {frob_error(A, Ak_svd):.2f}")

# ---------- Error vs k plot (Fast) ----------
evd_errors, svd_errors = [], []
for k in range(1, n + 1):
    evd_errors.append(frob_error(A, fast_evd_reconstruct(k)))
    svd_errors.append(frob_error(A, fast_svd_reconstruct(k)))

plt.figure()
plt.plot(range(1, n + 1), evd_errors, label='EVD error')
plt.plot(range(1, n + 1), svd_errors, label='SVD error')
plt.xlabel('k (retained components)')
plt.ylabel('Frobenius norm error')
plt.title('Reconstruction Error vs k')
plt.legend()
plt.show()

# ================= RECTANGULAR IMAGE: SVD only =================
B = load_gray('cat_21(1).png')
UB, SB, VtB = np.linalg.svd(B, full_matrices=False)

def fast_svd_rect(k):
    return UB[:, :k] @ np.diag(SB[:k]) @ VtB[:k, :]

k_values_rect = [10, 30, 60]

for k in k_values_rect:
    Bk = fast_svd_rect(k)
    show_row([B, Bk, np.abs(B - Bk)],
             ['Original', f'SVD Reconstructed (k={k})', 'Error |B-Bk|'])
    print(f"k={k}: SVD error = {frob_error(B, Bk):.2f}")


