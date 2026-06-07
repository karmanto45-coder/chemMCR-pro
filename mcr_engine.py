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
            area = np.trapezoid(np.abs(proc[:, i]), wavenumber) if hasattr(np, 'trapezoid') else np.trapz(np.abs(proc[:, i]), wavenumber)
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

def _sort_ascending(wn, sp):
    """Ensure wavenumber is ascending (required for interpolation)."""
    wn = np.array(wn, dtype=float)
    sp = np.array(sp, dtype=float)
    if wn[0] > wn[-1]:
        wn = wn[::-1]
        sp = sp[::-1]
    return wn, sp

def interpolate_spectrum(wn_ref, sp_ref, wn_target):
    """Simple linear interpolation — used for overlay plot only."""
    wn_ref, sp_ref = _sort_ascending(wn_ref, sp_ref)
    wn_target = np.array(wn_target, dtype=float)
    ascending = wn_target[0] < wn_target[-1]
    wt = wn_target if ascending else wn_target[::-1]
    result = np.interp(wt, wn_ref, sp_ref)
    return result if ascending else result[::-1]

def build_common_grid(wn_a, wn_b, grid_interval="auto"):
    """
    Build a common wavenumber grid from the overlap of two spectra.

    Parameters
    ----------
    wn_a, wn_b     : array-like wavenumber arrays (any direction)
    grid_interval  : 'auto' → use finest interval, or float (cm⁻¹)

    Returns
    -------
    common_grid : np.ndarray  (ascending)
    info        : dict with overlap details and warnings
    """
    wn_a = np.sort(np.array(wn_a, dtype=float))
    wn_b = np.sort(np.array(wn_b, dtype=float))

    # Overlap region
    ov_min = max(wn_a.min(), wn_b.min())
    ov_max = min(wn_a.max(), wn_b.max())
    overlap = ov_max - ov_min

    info = {
        "wn_a_range": (float(wn_a.min()), float(wn_a.max())),
        "wn_b_range": (float(wn_b.min()), float(wn_b.max())),
        "overlap_min": float(ov_min),
        "overlap_max": float(ov_max),
        "overlap_width": float(overlap),
        "warning": None,
        "error": None,
    }

    if overlap <= 0:
        info["error"] = "No overlap between spectra — matching not possible."
        return None, info

    if overlap < 200:
        info["warning"] = f"Overlap only {overlap:.1f} cm⁻¹ — matching may be unreliable."

    # Determine interval
    if grid_interval == "auto":
        interval_a = float(np.median(np.diff(wn_a))) if len(wn_a) > 1 else 1.0
        interval_b = float(np.median(np.diff(wn_b))) if len(wn_b) > 1 else 1.0
        interval   = min(interval_a, interval_b)   # finest grid
        interval   = max(interval, 0.1)             # safety floor
    else:
        interval = float(grid_interval)

    info["interval_a"]  = float(np.median(np.diff(wn_a))) if len(wn_a) > 1 else 1.0
    info["interval_b"]  = float(np.median(np.diff(wn_b))) if len(wn_b) > 1 else 1.0
    info["grid_interval"] = float(interval)

    common_grid = np.arange(ov_min, ov_max + interval * 0.1, interval)
    info["n_common_points"] = len(common_grid)
    return common_grid, info

def resample_to_grid(wn_src, sp_src, common_grid, method="cubic"):
    """
    Resample spectrum onto common_grid using cubic spline or linear.

    Parameters
    ----------
    wn_src, sp_src : source wavenumber and spectrum arrays
    common_grid    : target wavenumber grid (ascending)
    method         : 'cubic' or 'linear'
    """
    from scipy.interpolate import interp1d
    wn_src, sp_src = _sort_ascending(wn_src, sp_src)
    kind = "cubic" if (method == "cubic" and len(wn_src) >= 4) else "linear"
    f = interp1d(wn_src, sp_src, kind=kind,
                 bounds_error=False, fill_value=0.0)
    return f(common_grid)

def apply_window(wn, spec, mode, wmin=None, wmax=None):
    wn   = np.array(wn,   dtype=float)
    spec = np.array(spec, dtype=float)
    if mode == "fingerprint":
        mask = (wn >= 400) & (wn <= 1800)
    elif mode == "custom":
        mask = (wn >= wmin) & (wn <= wmax)
    else:
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
                window_mode, wmin, wmax, top_n=10,
                grid_interval="auto", interp_method="cubic"):
    # Normalise grid_interval
    if grid_interval != "auto":
        try:
            grid_interval = float(grid_interval)
        except (TypeError, ValueError):
            grid_interval = "auto"
    """
    Match one query spectrum against all library entries using
    common-grid interpolation (cubic spline by default).

    Each pair (query, library) gets its own common grid built
    from their overlap — handles different intervals and ranges
    automatically.

    Returns top_n results sorted by cosine descending, each with
    grid alignment metadata.
    """
    # Apply window to query
    wn_q, sp_q = apply_window(query_wn, query_spec, window_mode, wmin, wmax)
    if len(wn_q) < 5:
        return []

    results = []
    for entry in library_entries:
        wn_r = np.array(entry["wavenumber"], dtype=float)
        sp_r = np.array(entry["spectrum"],   dtype=float)

        # Apply window to library entry
        wn_r2, sp_r2 = apply_window(wn_r, sp_r, window_mode, wmin, wmax)
        if len(wn_r2) < 5:
            continue

        # Build common grid from overlap
        common_grid, grid_info = build_common_grid(wn_q, wn_r2, grid_interval)

        if common_grid is None:
            # No overlap — skip
            continue

        if len(common_grid) < 5:
            continue

        # Resample both spectra onto common grid
        sp_q_common = resample_to_grid(wn_q,  sp_q,  common_grid, interp_method)
        sp_r_common = resample_to_grid(wn_r2, sp_r2, common_grid, interp_method)

        cos = round(cosine_sim(sp_q_common, sp_r_common), 4)
        hqi = round(hqi_score(sp_q_common, sp_r_common), 2)

        results.append({
            "id":              entry["id"],
            "name":            entry["name"],
            "category":        entry["category"],
            "cosine":          cos,
            "hqi":             hqi,
            # Grid alignment metadata
            "overlap_min":     grid_info["overlap_min"],
            "overlap_max":     grid_info["overlap_max"],
            "overlap_width":   grid_info["overlap_width"],
            "grid_interval":   grid_info["grid_interval"],
            "n_common_points": grid_info["n_common_points"],
            "interval_query":  grid_info["interval_a"],
            "interval_lib":    grid_info["interval_b"],
            "grid_warning":    grid_info.get("warning"),
            "interp_method":   interp_method,
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
