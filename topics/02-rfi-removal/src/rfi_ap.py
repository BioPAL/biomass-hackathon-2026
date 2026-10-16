"""Self-contained alternating-projection (AP) RFI suppression for SAR single-look-complex data.

Dependencies: numpy, scipy only.

    from rfi_ap import ap_filter
    clean, mask = ap_filter(slc)                    # slc: (rows, cols) or (n_pol, rows, cols) complex, rank by MDL
    clean, mask = ap_filter(slc, rank=30)           # fixed rank of the RFI matrix

Algorithm: Y. Huang, Z. Chen, C. Wen, J. Li, X.-G. Xia, W. Hong, "An Efficient Radio Frequency Interference
Mitigation Algorithm in Real Synthetic Aperture Radar Data", IEEE TGRS 60, 2022. The SLC image is taken to its 2-D
frequency domain, Y_f = F(Y), where the RFI is a low-rank matrix R_f while the scene is sparse in the image
domain. The problem   min rank(R_f) + lam*||X||_1   s.t.   Y_f = R_f + F(X)   is solved by ADMM whose R_f update is
the projection onto rank-r matrices, computed on the tangent space with two QR decompositions and one 2r x 2r SVD
(eq. 17-26) instead of an SVD of the full matrix. The X update is a soft threshold in the image domain (eq. 29).

The paper does not give lam, mu0, rho, eta or the initialisation: the values below are the usual RPCA ones
(lam = 1/sqrt(max(K, L)), mu0 = 1.25/sigma_1, rho = 1.5) and a randomized SVD start. Everything else follows the
paper; the strips along azimuth, the MDL cap and the choice of the output are our own additions.
"""
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import median_filter


def _fft2(a, inverse=False):
    return (sfft.ifft2 if inverse else sfft.fft2)(a, norm="ortho", workers=-1)    # unitary: keeps norms and dtype


def mdl_rank(lam, n):
    """Wax-Kailath MDL estimate of the number of signal components. lam: eigenvalues (descending) of a p x p
    sample covariance with n snapshots (the squared singular values of the data matrix)."""
    lam = np.maximum(np.asarray(lam, float), 1e-300)
    p = len(lam)
    k = np.arange(p - 1)                                           # candidate ranks 0 .. p-2
    csum = np.cumsum(lam[::-1])[::-1]                              # sum of lam[k:]
    clog = np.cumsum(np.log(lam)[::-1])[::-1]                      # sum of log(lam[k:])
    m = p - k                                                      # number of noise eigenvalues
    mdl = -n * m * (clog[k] / m - np.log(csum[k] / m)) + 0.5 * k * (2 * p - k) * np.log(n)
    return int(np.argmin(mdl))


def _estimate_rank(Y, rule="edge", margin=5, white=101):
    """Rank of the RFI matrix from the eigenvalues of the 2-D spectrum Y (K, L).
    The paper uses MDL on the raw spectrum. On a real SLC the spectrum is shaped by the processing windows and is
    not white noise, so plain MDL returns about min(K, L). white = width (bins) of the running medians of |Y| along
    each axis that are divided out first (a narrow RFI line is narrower than this and survives; None/0 = off).
    rule "mdl": MDL on the whitened eigenvalues (still overestimates on real scenes). rule "edge" (default): number
    of eigenvalues above margin * (sqrt(K) + sqrt(L))^2 * sigma^2, the Marchenko-Pastur edge of the noise, with
    sigma^2 = median(|Y|^2) / ln 2 of the whitened entries; margin > 1 leaves out the excess eigenvalues that
    scene structure produces without RFI. margin is the knob between leaving RFI (large) and removing scene
    (small): on a BIOMASS HV strip 10 gave rank 14 and 10 dB of RFI-line suppression, 5 gave rank 23 and 22 dB;
    on injected RFI (true rank ~10) 10 gives the lowest error and 3 the highest."""
    if white:
        A = np.abs(Y)
        rows = median_filter(np.median(A, axis=1), white, mode="nearest")
        cols = median_filter(np.median(A, axis=0), white, mode="nearest")
        Y = Y / (np.outer(rows, cols) + 1e-30)
    s2 = np.median(np.abs(Y) ** 2) / np.log(2)                     # |y|^2 of complex Gaussian noise is exponential
    if s2 == 0 or min(Y.shape) < 2:                                # no-data strip (mostly zeros) or a single column
        return 0
    Yd = Y.astype(np.complex128)                                   # double: the noise eigenvalues are 1e-6 below the top
    K, L = Yd.shape
    lam = np.linalg.eigvalsh(Yd.conj().T @ Yd if K >= L else Yd @ Yd.conj().T)[::-1]
    if rule == "mdl":
        return mdl_rank(lam, max(K, L))
    return int((lam > margin * (np.sqrt(K) + np.sqrt(L)) ** 2 * s2).sum())


def _rsvd(A, r, rng, extra=10, n_iter=2):
    """Randomized rank-r SVD of A (K, L): returns U (K, r), s (r,), V (L, r) with A ~ U diag(s) V^H."""
    Om = (rng.standard_normal((A.shape[1], r + extra)) + 1j * rng.standard_normal((A.shape[1], r + extra))).astype(A.dtype)
    Q = np.linalg.qr(A @ Om)[0]
    for _ in range(n_iter):                                        # power iterations sharpen the spectrum
        Q = np.linalg.qr(A.conj().T @ Q)[0]
        Q = np.linalg.qr(A @ Q)[0]
    Ub, s, Vh = np.linalg.svd(Q.conj().T @ A, full_matrices=False)
    return (Q @ Ub)[:, :r], s[:r], Vh[:r].conj().T


def _project(Z, U, V):
    """H_r(P_T(Z)) of eq. (17)-(26): project Z onto the tangent space of the rank-r manifold at (U, V) and
    truncate back to rank r. Returns the new (U, s, V); the rank-r matrix is (U * s) @ V^H."""
    r = U.shape[1]
    ZV, ZhU = Z @ V, Z.conj().T @ U
    M11 = U.conj().T @ ZV                                          # U^H Z V
    Q1, R1 = np.linalg.qr(ZV - U @ M11)                            # (I - U U^H) Z V       (18)
    Q2, R2 = np.linalg.qr(ZhU - V @ M11.conj().T)                  # (I - V V^H) Z^H U     (19)
    M = np.block([[M11, R2.conj().T], [R1, np.zeros_like(M11)]])   # 2r x 2r, eq. (20)
    Um, s, Vmh = np.linalg.svd(M)
    return np.hstack([U, Q1]) @ Um[:, :r], s[:r], np.hstack([V, Q2]) @ Vmh.conj().T[:, :r]   # (21)-(23)


def _soft(C, beta):
    """Complex soft threshold, eq. (30): shrinks |C| by beta and keeps the phase."""
    a = np.abs(C)
    return C * (np.maximum(a - beta, 0) / np.maximum(a, 1e-30))


def ap_decompose(img, rank=None, lam=None, mu0=None, rho=1.5, mu_max=None, eta=1e-4, max_iter=60, rmax=100,
                 rank_rule="edge", margin=5, white=101, seed=0):
    """Split a complex image (K, L) into a low-rank RFI part (in its 2-D spectrum) and the rest (Algorithm 1).
    rank: rank of the RFI matrix, None = estimated from the whitened 2-D spectrum (rank_rule "edge" or "mdl",
    margin, white: see _estimate_rank; capped at rmax; 0 = no RFI found).
    lam: weight of the l1 term (threshold lam/mu), default 1/sqrt(max(K, L)); mu0: initial penalty, default
    1.25/sigma_1; rho: growth of mu per iteration; mu_max: cap of mu (default 1e7*mu0); eta: stop when
    ||Y_f - R_f - F(X)|| / ||Y_f|| < eta.
    Returns (rfi, sparse, info): rfi = image-domain RFI estimate F^-1(R_f), sparse = the sparse image X of the
    model, info = dict(rank, iters, err list)."""
    Y = _fft2(img)
    K, L = Y.shape
    if rank is None:
        rank = min(_estimate_rank(Y, rank_rule, margin, white), rmax)
    rank = int(min(rank, K // 2, L // 2))
    if rank == 0:
        z = np.zeros_like(img)
        return z, z, dict(rank=0, iters=0, err=[])
    rng = np.random.default_rng(seed)
    U, s, V = _rsvd(Y, rank, rng)
    if s[0] == 0:                                                  # all-zero strip with a fixed rank: nothing to remove
        z = np.zeros_like(img)
        return z, z, dict(rank=0, iters=0, err=[])
    lam = float(1 / np.sqrt(max(K, L)) if lam is None else lam)    # python floats: a numpy float64 scalar would
    mu = float(1.25 / s[0] if mu0 is None else mu0)                # promote the complex64 arrays to complex128 on numpy 2
    mu_max = float(mu * 1e7 if mu_max is None else mu_max)
    Xf, Y1 = np.zeros_like(Y), np.zeros_like(Y)                    # F(X) and the Lagrange multiplier
    nY, err = np.linalg.norm(Y), []
    for it in range(max_iter):
        U, s, V = _project(Y - Xf + Y1 / mu, U, V)                 # R_f update, eq. (24)
        R = (U * s) @ V.conj().T
        X = _soft(_fft2(Y - R + Y1 / mu, inverse=True), lam / mu)  # X update, eq. (29)
        Xf = _fft2(X)
        res = Y - R - Xf
        Y1 = Y1 + mu * res                                         # eq. (31)
        mu = min(rho * mu, mu_max)                                 # eq. (32)
        err.append(float(np.linalg.norm(res) / nY))
        if err[-1] < eta:
            break
    return _fft2(R, inverse=True), X, dict(rank=rank, iters=it + 1, err=err)


def ap_filter(slc, strip=4096, rank=None, output="subtract", info=None, **kw):
    """AP RFI suppression on slc (rows, cols) or (n_pol, rows, cols); every pol and every strip of `strip`
    azimuth lines (the last one absorbs a short remainder) is filtered on its own, over the full range.
    output: "subtract" = slc - rfi (keeps everything that is not RFI, the radiometry of the scene is
    preserved) or "sparse" = the sparse image X of the model (also drops the non-sparse part of the scene).
    info: list that receives (pol, y0, y1, info dict) per strip. kw: see ap_decompose.
    Returns (cleaned, mask (rows, cols)), mask = 1 where an RFI component was removed (rank > 0) in any pol:
    a strip is marked for all pols as soon as one of them has rank > 0."""
    squeeze = slc.ndim == 2
    S = slc[None] if squeeze else slc
    n, R, C = S.shape
    out, mask = S.copy(), np.zeros((R, C), np.int8)
    edges = list(range(0, R, strip)) + [R]
    if len(edges) > 2 and edges[-1] - edges[-2] < strip // 2:
        edges.pop(-2)                                              # short remainder: join the previous strip
    for k in range(n):
        for y0, y1 in zip(edges[:-1], edges[1:]):
            rfi, sparse, inf = ap_decompose(S[k, y0:y1], rank=rank, **kw)
            if inf["rank"]:
                out[k, y0:y1] = S[k, y0:y1] - rfi if output == "subtract" else sparse
                mask[y0:y1] = 1
            if info is not None:
                info.append((k, y0, y1, inf))
    return (out[0] if squeeze else out), mask


if __name__ == "__main__":      # self-check: injected narrowband RFI must be removed, the scene preserved
    rng = np.random.default_rng(0)
    K, L = 1024, 384
    X = (rng.standard_normal((K, L)) + 1j * rng.standard_normal((K, L))) / np.sqrt(2)
    X[rng.integers(0, K, 30), rng.integers(0, L, 30)] += 40                       # strong point scatterers
    t = np.arange(L)
    A = [(1 + 0.5 * np.cos(2 * np.pi * (n + 1) * np.arange(K) / K))[:, None] * np.exp(2j * np.pi * rng.random()) for n in range(3)]
    Rfi = sum(a * np.exp(2j * np.pi * f * t)[None] for a, f in zip(A, (0.11, -0.23, 0.31)))
    Rfi *= np.sqrt(10 ** (10 / 10) * (np.abs(X) ** 2).mean() / (np.abs(Rfi) ** 2).mean())   # SIR = -10 dB
    Y = (X + Rfi).astype(np.complex64)
    err = lambda a: np.linalg.norm(a - X) / np.linalg.norm(X)
    clean, mask = ap_filter(Y, strip=K, rank=3)
    print(f"rel. error: polluted {err(Y):.2f} -> AP rank 3 {err(clean):.3f}")
    assert err(clean) < 0.1 * err(Y) and mask.all()
    clean_m, _ = ap_filter(Y, strip=K)                                            # rank by MDL
    print(f"auto rank -> error {err(clean_m):.3f}")
    assert err(clean_m) < 0.2 * err(Y)
    inf = []
    free, free_mask = ap_filter(X.astype(np.complex64), strip=K, info=inf)        # no RFI: little should change
    print("no-RFI case: auto rank", inf[0][3]["rank"], "| change", f"{np.linalg.norm(free - X) / np.linalg.norm(X):.3f}")
    assert np.linalg.norm(free - X) / np.linalg.norm(X) < 0.2
    print("ok")
