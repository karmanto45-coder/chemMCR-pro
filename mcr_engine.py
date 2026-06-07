import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

# ── Preprocessing ─────────────────────────────────────────────

def preprocess(spectra_matrix, wavenumber,
               do_norm=True, do_smooth=False, do_baseline=False):
    proc = spectra_matrix.copy().astype(float)
    if do_smooth:
        for i in range(proc.shape[1]):
            proc[:, i] = savgol_filter(proc[:, i], 11, 3)
    if do_baseline:
        for i in range(proc.shape[1]):
            proc[:, i] -= proc[:, i].min()
    if do_norm:
        for i in range(proc.shape[1]):
            area = np.trapz(np.abs(proc[:, i]), wavenumber)
            if area > 0:
                proc[:, i] /= area
    return proc

# ── PCA component detection ───────────────────────────────────

def detect_components(D, max_k=10):
    n = min(max_k, min(D.shape) - 1)
    pca = PCA(n_components=max(n, 2))
    pca.fit(D)
    ev  = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(ev)
    auto_k = int(np.searchsorted(cum, 95)) + 1
    auto_k = max(2, min(auto_k, n))
    return ev, cum, auto_k

# ── MCR-ALS ───────────────────────────────────────────────────

def run_mcr_als(D, n_components, max_iter=200, tol=1e-6,
                closure=False, unimodal=False):
    """
    MCR-ALS with non-negativity constraints.
    D : (n_samples × n_wavelengths)
    Returns C, S, lof_history, r2, converged
    """
    D = np.array(D, dtype=float)
    m, n = D.shape

    # Initialise via PCA
    pca = PCA(n_components=n_components)
    C = np.abs(pca.fit_transform(D))
    C = np.maximum(C, 1e-10)

    lof_history = []
    S = None
    converged = False

    for iteration in range(max_iter):
        # S = (CᵀC)⁻¹ CᵀD
        S = np.linalg.lstsq(C, D, rcond=None)[0]
        S = np.maximum(S, 0)

        # Closure constraint (rows of S sum to 1)
        if closure:
            row_sums = S.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            S /= row_sums

        # C = DSᵀ(SSᵀ)⁻¹
        C = np.linalg.lstsq(S.T, D.T, rcond=None)[0].T
        C = np.maximum(C, 0)

        D_hat = C @ S
        residual = D - D_hat
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum(D ** 2)
        lof = np.sqrt(ss_res / ss_tot) * 100 if ss_tot > 0 else 0.0
        lof_history.append(lof)

        if iteration > 2:
            delta = abs(lof_history[-2] - lof_history[-1])
            if delta < tol:
                converged = True
                break

    D_hat = C @ S
    ss_res = np.sum((D - D_hat) ** 2)
    ss_tot = np.sum((D - np.mean(D)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return C, S, lof_history, r2, converged

# ── Spectral matching ─────────────────────────────────────────

def interpolate_spectrum(wn_ref, sp_ref, wn_target):
    wn_ref  = np.array(wn_ref,  dtype=float)
    sp_ref  = np.array(sp_ref,  dtype=float)
    wn_target = np.array(wn_target, dtype=float)
    # np.interp requires ascending xp
    if wn_ref[0] > wn_ref[-1]:
        wn_ref = wn_ref[::-1]
        sp_ref = sp_ref[::-1]
    ascending = wn_target[0] < wn_target[-1]
    wt = wn_target if ascending else wn_target[::-1]
    result = np.interp(wt, wn_ref, sp_ref)
    return result if ascending else result[::-1]

def apply_window(wn, spec, mode, wmin=None, wmax=None):
    wn   = np.array(wn)
    spec = np.array(spec)
    if mode == "fingerprint":
        mask = (wn >= 400) & (wn <= 1800)
    elif mode == "custom":
        mask = (wn >= wmin) & (wn <= wmax)
    else:   # full
        mask = np.ones(len(wn), dtype=bool)
    return wn[mask], spec[mask]

def cosine_sim(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def hqi_score(a, b):
    c = cosine_sim(a, b)
    return round(c ** 2 * 100, 3)

def batch_match(query_spec, query_wn, library_entries,
                window_mode, wmin, wmax, top_n=10):
    """
    Match one query spectrum against all library entries.
    Returns top_n results sorted by cosine descending.
    """
    wn_q, sp_q = apply_window(query_wn, query_spec, window_mode, wmin, wmax)
    if len(wn_q) < 5:
        return []

    results = []
    for entry in library_entries:
        wn_r = entry["wavenumber"]
        sp_r = entry["spectrum"]
        # Skip if no overlap
        if wn_r.max() < wn_q.min() or wn_r.min() > wn_q.max():
            continue
        wn_r2, sp_r2 = apply_window(wn_r, sp_r, window_mode, wmin, wmax)
        if len(wn_r2) < 5:
            continue
        sp_interp = interpolate_spectrum(wn_r2, sp_r2, wn_q)
        cos = round(cosine_sim(sp_q, sp_interp), 4)
        hqi = round(hqi_score(sp_q, sp_interp), 2)
        results.append({
            "id":       entry["id"],
            "name":     entry["name"],
            "category": entry["category"],
            "cosine":   cos,
            "hqi":      hqi,
        })

    results.sort(key=lambda x: x["cosine"], reverse=True)
    return results[:top_n]

def consensus_label(cos, hqi, thresh_cos=0.95, thresh_hqi=90.25):
    cos_strong = cos  >= thresh_cos
    hqi_strong = hqi  >= thresh_hqi
    cos_med    = cos  >= (thresh_cos - 0.05)
    hqi_med    = hqi  >= (thresh_hqi - 9.25)

    if cos_strong and hqi_strong:
        return "strong",   "✅ Match kuat / Strong match"
    if cos_med and hqi_med:
        if cos_strong != hqi_strong:
            return "conflict", "⚠️ Konflik ranking / Rank conflict"
        return "medium", "🟡 Match sedang / Medium match"
    if cos_strong != hqi_strong:
        return "conflict", "⚠️ Konflik ranking / Rank conflict"
    return "weak", "❌ Tidak match / No match"
