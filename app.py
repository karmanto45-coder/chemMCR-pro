import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime

from auth import render_login, is_logged_in, is_admin, logout
from database import (init_db, add_spectrum, delete_spectrum,
                      update_spectrum_meta, get_all_meta,
                      get_spectrum_by_id, get_all_spectra_for_matching,
                      count_spectra, get_categories, import_from_json)
from mcr_engine import (preprocess, detect_components, run_mcr_als,
                        batch_match, consensus_label, interpolate_spectrum,
                        apply_window)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SpectraID Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}

/* Hide GitHub link, footer, deploy button */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
[data-testid="stToolbar"] {visibility:hidden;}
a[href*="github"] {display:none !important;}
.stDeployButton {display:none !important;}
.app-header{background:linear-gradient(135deg,#0f1117,#161b27);
  border:1px solid #2a3142;border-radius:12px;padding:1.2rem 1.8rem;margin-bottom:1.2rem;}
.app-title{font-family:'DM Mono',monospace;font-size:1.5rem;font-weight:500;
  color:#e2e8f0;margin:0;letter-spacing:-0.5px;}
.app-sub{color:#64748b;font-size:0.82rem;margin:3px 0 0;}
.badge{display:inline-block;font-size:0.68rem;padding:2px 8px;border-radius:4px;
  font-family:'DM Mono',monospace;margin-left:8px;}
.badge-admin{background:#1e0a3c;color:#c084fc;}
.badge-user{background:#0a1e2a;color:#7dd3fc;}
.metric-card{background:#161b27;border:1px solid #2a3142;border-radius:10px;
  padding:0.9rem 1.1rem;text-align:center;}
.metric-value{font-family:'DM Mono',monospace;font-size:1.5rem;
  font-weight:500;color:#7dd3fc;}
.metric-label{font-size:0.72rem;color:#64748b;margin-top:2px;
  text-transform:uppercase;letter-spacing:0.05em;}
.sec-hdr{font-family:'DM Mono',monospace;font-size:0.68rem;color:#475569;
  text-transform:uppercase;letter-spacing:0.1em;margin:1.2rem 0 0.6rem;
  padding-bottom:5px;border-bottom:1px solid #1e293b;}
.match-card{border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.45rem;border-left:3px solid;}
.m-strong{background:#0d2018;border-color:#22c55e;}
.m-medium{background:#1a1a08;border-color:#eab308;}
.m-conflict{background:#12100d;border-color:#f97316;}
.m-weak{background:#1a0a08;border-color:#ef4444;}
.m-name{font-weight:500;color:#e2e8f0;font-size:0.92rem;}
.m-scores{font-family:'DM Mono',monospace;font-size:0.78rem;color:#94a3b8;margin-top:3px;}
.m-badge{display:inline-block;font-size:0.68rem;padding:2px 8px;
  border-radius:4px;font-weight:500;float:right;}
.window-chip{display:inline-block;background:#1e293b;border:1px solid #334155;
  border-radius:6px;padding:3px 10px;font-family:'DM Mono',monospace;
  font-size:0.76rem;color:#7dd3fc;margin-right:6px;}
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────
init_db()

# ── Auth gate ─────────────────────────────────────────────────
if not is_logged_in():
    render_login()
    st.stop()

# ── Language helper ───────────────────────────────────────────
lang = st.session_state.get("lang", "id")
def t(id_text, en_text):
    return en_text if lang == "en" else id_text

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    role  = st.session_state.get("role", "user")
    uname = st.session_state.get("display_name", "User")
    badge = "badge-admin" if role == "admin" else "badge-user"
    blabel = "Admin" if role == "admin" else "User"
    st.markdown(f"""
    <div style="padding:0.5rem 0 1rem;">
      <p style="font-family:'DM Mono',monospace;font-size:1rem;
         color:#e2e8f0;margin:0;">{uname}
        <span class="badge {badge}">{blabel}</span>
      </p>
      <p style="font-size:0.75rem;color:#475569;margin:2px 0 0;">
        {st.session_state.get('username','')}
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<p class="sec-hdr">{t("Statistik","Stats")}</p>', unsafe_allow_html=True)
    n_lib = count_spectra()
    st.markdown(f'<div class="metric-card"><div class="metric-value">{n_lib:,}</div>'
                f'<div class="metric-label">{t("Spektra library","Library spectra")}</div></div>',
                unsafe_allow_html=True)

    st.markdown("")
    lang_choice = st.selectbox("🌐 Language",
        ["🇮🇩 Bahasa Indonesia", "🇬🇧 English"],
        index=0 if lang == "id" else 1)
    st.session_state["lang"] = "en" if "English" in lang_choice else "id"

    st.markdown("---")
    if st.button(t("Keluar","Logout"), use_container_width=True):
        logout()
        st.rerun()

# ── Header ────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <p class="app-title">SpectraID Pro
    <span class="badge badge-admin" style="font-size:0.65rem;">v2.0</span>
  </p>
  <p class="app-sub">
    {t("Multivariate Curve Resolution · Identifikasi Spektra ATR-FTIR",
       "Multivariate Curve Resolution · ATR-FTIR Spectral Identification")}
  </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab_labels = (
    [t("📂 Input Data","📂 Input Data"),
     t("🔬 Analisis MCR","🔬 MCR Analysis"),
     t("🔍 Identifikasi","🔍 Identification"),
     t("📚 Library","📚 Library"),
     t("⚙️ Admin","⚙️ Admin"),
     t("📊 Laporan","📊 Report")]
    if is_admin() else
    [t("📂 Input Data","📂 Input Data"),
     t("🔬 Analisis MCR","🔬 MCR Analysis"),
     t("🔍 Identifikasi","🔍 Identification"),
     t("📊 Laporan","📊 Report")]
)

tabs = st.tabs(tab_labels)
tab_input = tabs[0]
tab_mcr   = tabs[1]
tab_match = tabs[2]
tab_lib   = tabs[3] if is_admin() else None
tab_admin = tabs[4] if is_admin() else None
tab_rep   = tabs[5] if is_admin() else tabs[3]

# ════════════════════════════════════════════════════════════════
# TAB 1 — INPUT DATA
# ════════════════════════════════════════════════════════════════
with tab_input:
    st.markdown(f'<p class="sec-hdr">{t("Upload data spektra","Upload spectral data")}</p>',
                unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            t("Upload file (Excel / CSV / TXT)","Upload file (Excel / CSV / TXT)"),
            type=["xlsx","xls","csv","txt","jdx","dx"]
        )
    with col_info:
        st.info(t(
            "**Format kolom:**\nKolom 1 = wavenumber (cm⁻¹)\nKolom 2+ = spektra sampel\nMinimum 4 spektra",
            "**Column format:**\nCol 1 = wavenumber (cm⁻¹)\nCol 2+ = sample spectra\nMinimum 4 spectra"
        ))

    if uploaded:
        try:
            name = uploaded.name.lower()
            if name.endswith((".xlsx",".xls")):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded, sep=None, engine="python", comment="#")

            wn_col   = df.columns[0]
            spec_cols = df.columns[1:]
            wavenumber = df[wn_col].values.astype(float)
            raw_matrix = df[spec_cols].values.astype(float)

            n_spec   = len(spec_cols)
            n_pts    = len(wavenumber)

            c1,c2,c3,c4 = st.columns(4)
            for col, val, lbl in zip(
                [c1,c2,c3,c4],
                [n_spec, n_pts, wavenumber.min(), wavenumber.max()],
                [t("Jumlah spektra","Spectra count"),
                 t("Titik data","Data points"),
                 t("Wavenum. min","Wavenum. min"),
                 t("Wavenum. max","Wavenum. max")]
            ):
                col.markdown(
                    f'<div class="metric-card"><div class="metric-value">{val:.0f}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown(f'<p class="sec-hdr">{t("Pra-pemrosesan","Preprocessing")}</p>',
                        unsafe_allow_html=True)
            p1,p2,p3 = st.columns(3)
            do_norm     = p1.checkbox(t("Normalisasi","Normalize"), value=True)
            do_smooth   = p2.checkbox(t("Smoothing (SG)","Smoothing (SG)"), value=False)
            do_baseline = p3.checkbox(t("Koreksi baseline","Baseline correction"), value=False)

            proc = preprocess(raw_matrix, wavenumber, do_norm, do_smooth, do_baseline)

            st.session_state["wavenumber"]  = wavenumber
            st.session_state["spectra"]     = proc
            st.session_state["spec_names"]  = list(spec_cols.astype(str))

            # Plot
            st.markdown(f'<p class="sec-hdr">{t("Visualisasi spektra","Spectral visualization")}</p>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, col in enumerate(spec_cols):
                fig.add_trace(go.Scatter(
                    x=wavenumber, y=proc[:,i],
                    name=str(col), mode="lines",
                    line=dict(width=1.2, color=colors[i % len(colors)])
                ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                           title=t("Wavenumber (cm⁻¹)","Wavenumber (cm⁻¹)")),
                yaxis=dict(gridcolor="#1e293b", title="Absorbance"),
                legend=dict(bgcolor="#161b27"), height=370,
                margin=dict(l=20,r=20,t=20,b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            if n_spec < 4:
                st.warning(t(f"⚠️ Hanya {n_spec} spektra. Minimum rekomendasi: 4 spektra.",
                             f"⚠️ Only {n_spec} spectra. Recommended minimum: 4."))
            elif n_spec < 10:
                st.warning(t("⚠️ Data cukup untuk analisis, tapi disarankan 10+ spektra.",
                             "⚠️ Sufficient for analysis, but 10+ spectra recommended."))
            else:
                st.success(t(f"✅ {n_spec} spektra siap dianalisis.",
                             f"✅ {n_spec} spectra ready for analysis."))
        except Exception as e:
            st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════════
# TAB 2 — MCR ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab_mcr:
    if "spectra" not in st.session_state:
        st.info(t("Upload data spektra di tab Input Data terlebih dahulu.",
                  "Please upload spectral data in the Input Data tab first."))
    else:
        wn = st.session_state["wavenumber"]
        D  = st.session_state["spectra"].T

        st.markdown(f'<p class="sec-hdr">{t("Deteksi komponen (PCA)","Component detection (PCA)")}</p>',
                    unsafe_allow_html=True)

        ev, cum, auto_k = detect_components(D)
        fig_pca = make_subplots(rows=1, cols=2,
            subplot_titles=(
                t("Variansi tiap komponen (%)","Variance per component (%)"),
                t("Variansi kumulatif (%)","Cumulative variance (%)")
            ))
        fig_pca.add_trace(go.Bar(x=list(range(1,len(ev)+1)), y=ev,
            marker_color="#7dd3fc", name="Var%"), row=1, col=1)
        fig_pca.add_trace(go.Scatter(x=list(range(1,len(cum)+1)), y=cum,
            mode="lines+markers", line=dict(color="#f97316"), name="Cum%"), row=1, col=2)
        fig_pca.add_hline(y=95, line_dash="dash", line_color="#475569",
            annotation_text="95%", row=1, col=2)
        fig_pca.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117", height=260, showlegend=False,
            margin=dict(l=20,r=20,t=40,b=20))
        fig_pca.update_xaxes(gridcolor="#1e293b")
        fig_pca.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption(t(f"Saran otomatis PCA: **{auto_k} komponen** (≥95% variansi)",
                     f"PCA suggestion: **{auto_k} components** (≥95% variance)"))

        st.markdown(f'<p class="sec-hdr">{t("Parameter MCR-ALS","MCR-ALS parameters")}</p>',
                    unsafe_allow_html=True)
        a1,a2,a3,a4 = st.columns(4)
        n_comp   = a1.number_input(t("Jumlah komponen","Components"), 2, 10, auto_k)
        max_iter = a2.number_input(t("Iterasi max","Max iterations"), 50, 1000, 200, step=50)
        tol      = a3.selectbox(t("Toleransi","Tolerance"),
                       [1e-4,1e-5,1e-6,1e-7], index=2,
                       format_func=lambda x: f"{x:.0e}")
        closure  = a4.checkbox(t("Closure constraint","Closure constraint"), value=False)

        if st.button(f"▶  {t('Jalankan MCR-ALS','Run MCR-ALS')}",
                     use_container_width=True):
            with st.spinner(t("Menjalankan MCR-ALS...","Running MCR-ALS...")):
                C, S, lof_hist, r2, conv = run_mcr_als(
                    D, int(n_comp), int(max_iter), float(tol), closure
                )
                st.session_state.update({
                    "mcr_C": C, "mcr_S": S, "mcr_lof": lof_hist,
                    "mcr_r2": r2, "mcr_ncomp": int(n_comp),
                    "mcr_converged": conv
                })
            conv_msg = t("Konvergen","Converged") if conv else t("Belum konvergen","Not converged")
            st.success(f"✅ {conv_msg} — {len(lof_hist)} {t('iterasi','iterations')} "
                       f"| LOF: {lof_hist[-1]:.4f}% | R²: {r2:.5f}")

        if "mcr_S" in st.session_state:
            S_res = st.session_state["mcr_S"]
            C_res = st.session_state["mcr_C"]
            lof_h = st.session_state["mcr_lof"]
            r2    = st.session_state["mcr_r2"]
            nc    = st.session_state["mcr_ncomp"]

            m1,m2,m3,m4 = st.columns(4)
            for col, val, lbl in zip(
                [m1,m2,m3,m4],
                [f"{lof_h[-1]:.3f}%", f"{r2:.4f}", len(lof_h), nc],
                ["LOF", "R²", t("Iterasi","Iterations"), t("Komponen","Components")]
            ):
                col.markdown(
                    f'<div class="metric-card"><div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

            # LOF warning
            if lof_h[-1] > 10:
                st.warning(t("⚠️ LOF > 10% — coba tambah jumlah komponen atau perbaiki preprocessing.",
                             "⚠️ LOF > 10% — try increasing components or improve preprocessing."))
            elif lof_h[-1] > 5:
                st.info(t("ℹ️ LOF 5–10% — hasil dapat diterima, pertimbangkan komponen lebih.",
                          "ℹ️ LOF 5–10% — acceptable, consider more components."))

            colors = px.colors.qualitative.Pastel

            st.markdown(f'<p class="sec-hdr">{t("Spektra murni hasil MCR","MCR pure spectra")}</p>',
                        unsafe_allow_html=True)
            fig_s = go.Figure()
            for i in range(nc):
                fig_s.add_trace(go.Scatter(
                    x=wn, y=S_res[i], name=f"{t('Komponen','Component')} {i+1}",
                    mode="lines", line=dict(width=1.8, color=colors[i%len(colors)])
                ))
            fig_s.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                           title="Wavenumber (cm⁻¹)"),
                yaxis=dict(gridcolor="#1e293b", title=t("Intensitas","Intensity")),
                legend=dict(bgcolor="#161b27"), height=340,
                margin=dict(l=20,r=20,t=20,b=40)
            )
            st.plotly_chart(fig_s, use_container_width=True)

            st.markdown(f'<p class="sec-hdr">{t("Profil konsentrasi","Concentration profiles")}</p>',
                        unsafe_allow_html=True)
            snames = st.session_state.get("spec_names", [f"S{i+1}" for i in range(C_res.shape[0])])
            fig_c = go.Figure()
            for i in range(nc):
                fig_c.add_trace(go.Bar(
                    name=f"{t('Komponen','Component')} {i+1}",
                    x=snames, y=C_res[:,i],
                    marker_color=colors[i%len(colors)]
                ))
            fig_c.update_layout(
                barmode="stack", template="plotly_dark",
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(gridcolor="#1e293b", title=t("Sampel","Sample")),
                yaxis=dict(gridcolor="#1e293b", title=t("Kontribusi relatif","Relative contribution")),
                legend=dict(bgcolor="#161b27"), height=300,
                margin=dict(l=20,r=20,t=20,b=40)
            )
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown(f'<p class="sec-hdr">{t("Konvergensi LOF","LOF convergence")}</p>',
                        unsafe_allow_html=True)
            fig_lof = go.Figure(go.Scatter(y=lof_h, mode="lines",
                line=dict(color="#f97316", width=1.5)))
            fig_lof.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(gridcolor="#1e293b", title=t("Iterasi","Iteration")),
                yaxis=dict(gridcolor="#1e293b", title="LOF (%)"),
                height=200, margin=dict(l=20,r=20,t=10,b=40)
            )
            st.plotly_chart(fig_lof, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — SPECTRAL IDENTIFICATION
# ════════════════════════════════════════════════════════════════
with tab_match:
    if "mcr_S" not in st.session_state:
        st.info(t("Jalankan MCR-ALS terlebih dahulu.",
                  "Please run MCR-ALS first."))
    else:
        n_lib = count_spectra()
        if n_lib == 0:
            st.warning(t("Library masih kosong. Admin perlu menambahkan spektra acuan.",
                         "Library is empty. Admin needs to add reference spectra."))
        else:
            wn  = st.session_state["wavenumber"]
            S   = st.session_state["mcr_S"]
            nc  = st.session_state["mcr_ncomp"]

            st.markdown(f'<p class="sec-hdr">{t("Pengaturan window & threshold","Window & threshold settings")}</p>',
                        unsafe_allow_html=True)
            w1,w2,w3 = st.columns([2,1,1])
            window_mode = w1.selectbox(
                t("Rentang analisis","Analysis range"),
                [t("Fingerprint (400–1800 cm⁻¹)","Fingerprint (400–1800 cm⁻¹)"),
                 t("Full range","Full range"),
                 t("Custom range","Custom range")]
            )
            wmin_input = w2.number_input("Min (cm⁻¹)", value=400, step=50,
                disabled="Custom" not in window_mode)
            wmax_input = w3.number_input("Max (cm⁻¹)", value=4000, step=50,
                disabled="Custom" not in window_mode)

            t1,t2,t3 = st.columns(3)
            top_n      = t1.number_input(t("Top-N kandidat","Top-N candidates"), 3, 20, 10)
            thresh_cos = t2.slider(t("Threshold cosine","Threshold cosine"), 0.70, 1.00, 0.95, 0.01)
            thresh_hqi = t3.slider(t("Threshold HQI (%)","Threshold HQI (%)"), 50.0, 100.0, 90.25, 0.25)

            # Grid alignment settings
            st.markdown(f'<p class="sec-hdr">{t("Pengaturan penyesuaian grid","Grid alignment settings")}</p>',
                        unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            interp_method = g1.selectbox(
                t("Metode interpolasi","Interpolation method"),
                ["cubic", "linear"],
                help=t("Cubic: lebih akurat untuk puncak tajam. Linear: lebih cepat.",
                       "Cubic: more accurate for sharp peaks. Linear: faster.")
            )
            grid_mode = g2.selectbox(
                t("Interval grid bersama","Common grid interval"),
                [t("Otomatis (interval terkecil)","Auto (finest interval)"),
                 t("Manual","Manual")],
            )
            if t("Manual","Manual") in grid_mode:
                grid_interval = st.number_input(
                    t("Interval grid (cm⁻¹)","Grid interval (cm⁻¹)"),
                    min_value=0.1, max_value=10.0, value=1.0, step=0.1
                )
            else:
                grid_interval = "auto"

            # Window mode key
            if "Fingerprint" in window_mode:
                wmode_key = "fingerprint"
                wmin_show, wmax_show = 400, 1800
            elif "Custom" in window_mode:
                wmode_key = "custom"
                wmin_show, wmax_show = wmin_input, wmax_input
            else:
                wmode_key = "full"
                wmin_show = float(np.array(wn).min())
                wmax_show = float(np.array(wn).max())

            wn_arr = np.array(wn)
            n_pts_win = int(np.sum((wn_arr >= wmin_show) & (wn_arr <= wmax_show)))
            wn_interval = float(np.median(np.diff(np.sort(wn_arr)))) if len(wn_arr) > 1 else 1.0
            st.markdown(
                f'<span class="window-chip">{wmin_show:.0f}–{wmax_show:.0f} cm⁻¹</span>'
                f'<span class="window-chip">Δ {wn_interval:.4f} cm⁻¹</span>'
                f'<span style="font-size:0.8rem;color:#475569">{n_pts_win} '
                f'{t("titik · interval MCR","pts · MCR interval")}</span>',
                unsafe_allow_html=True
            )

            # Run identification
            if st.button(t("🔍 Jalankan identifikasi","🔍 Run identification"),
                         use_container_width=True):
                with st.spinner(t(f"Mencocokkan vs {n_lib:,} spektra library...",
                                  f"Matching vs {n_lib:,} library spectra...")):
                    library_entries = get_all_spectra_for_matching()
                    all_results = []
                    for i in range(nc):
                        res = batch_match(
                            S[i], wn, library_entries,
                            wmode_key, wmin_show, wmax_show,
                            int(top_n), grid_interval, interp_method
                        )
                        all_results.append(res)
                    st.session_state["match_results"] = all_results
                st.success(t(f"✅ Identifikasi selesai — {nc} komponen dicocokkan vs {n_lib:,} referensi.",
                             f"✅ Identification complete — {nc} components matched vs {n_lib:,} references."))

            if "match_results" in st.session_state:
                all_results = st.session_state["match_results"]
                colors = px.colors.qualitative.Pastel

                for i, results in enumerate(all_results):
                    with st.expander(f"{t('Komponen','Component')} {i+1}", expanded=(i==0)):
                        if not results:
                            st.warning(t("Tidak ada hasil. Periksa rentang wavenumber.",
                                         "No results. Check wavenumber range."))
                            continue

                        # Overlay plot — top match
                        top = results[0]
                        lib_entry = get_spectrum_by_id(top["id"])
                        if lib_entry:
                            fig_ov = go.Figure()
                            fig_ov.add_trace(go.Scatter(
                                x=wn, y=S[i],
                                name=f"{t('Komponen','Component')} {i+1} (MCR)",
                                line=dict(color=colors[i%len(colors)], width=1.8)
                            ))
                            sp_interp = interpolate_spectrum(
                                lib_entry["wavenumber"], lib_entry["spectrum"], wn
                            )
                            if S[i].max() > 0 and sp_interp.max() > 0:
                                sp_disp = sp_interp / sp_interp.max() * S[i].max()
                            else:
                                sp_disp = sp_interp
                            fig_ov.add_trace(go.Scatter(
                                x=wn, y=sp_disp,
                                name=f"{top['name']} ({t('referensi','reference')})",
                                line=dict(color="#f97316", width=1.5, dash="dot")
                            ))
                            fig_ov.add_vrect(
                                x0=wmin_show, x1=wmax_show,
                                fillcolor="#7dd3fc", opacity=0.04,
                                annotation_text="window",
                                annotation_position="top left"
                            )
                            # Zoom x-axis to selected window range + 5% padding
                            pad = (wmax_show - wmin_show) * 0.05
                            x_lo = wmin_show - pad   # lower wavenumber
                            x_hi = wmax_show + pad   # higher wavenumber
                            fig_ov.update_layout(
                                template="plotly_dark", paper_bgcolor="#0f1117",
                                plot_bgcolor="#0f1117",
                                xaxis=dict(
                                    range=[x_hi, x_lo],  # reversed (high→low FTIR convention)
                                    gridcolor="#1e293b",
                                    title="Wavenumber (cm⁻¹)"
                                ),
                                yaxis=dict(gridcolor="#1e293b",
                                           title=t("Intensitas (norm.)","Intensity (norm.)")),
                                legend=dict(bgcolor="#161b27"),
                                height=260, margin=dict(l=20,r=20,t=20,b=40)
                            )
                            st.plotly_chart(fig_ov, use_container_width=True)

                        # Ranking cards
                        # Show grid info for top result
                        if results:
                            top_r = results[0]
                            grid_warn = top_r.get("grid_warning")
                            ov_w = top_r.get("overlap_width", 0)
                            n_pts = top_r.get("n_common_points", 0)
                            gi   = top_r.get("grid_interval", 0)
                            i_q  = top_r.get("interval_query", 0)
                            i_l  = top_r.get("interval_lib", 0)
                            im   = top_r.get("interp_method","cubic")
                            grid_html = (
                                f'<div style="background:#0f1829;border:1px solid #1e3a5f;'
                                f'border-radius:8px;padding:8px 14px;margin-bottom:10px;'
                                f'font-family:monospace;font-size:0.76rem;color:#7dd3fc;">'
                                f'<b>{t("Info penyesuaian grid","Grid alignment info")}:</b> '
                                f'Overlap {ov_w:.1f} cm⁻¹ · '
                                f'Grid bersama {gi:.4f} cm⁻¹ · {n_pts} titik · '
                                f'Interpolasi: {im} · '
                                f'ΔMCR {i_q:.4f} cm⁻¹ · Δlib {i_l:.4f} cm⁻¹'
                                f'</div>'
                            )
                            if grid_warn:
                                grid_html += (
                                    f'<div style="background:#1a0f00;border:1px solid #f97316;'
                                    f'border-radius:8px;padding:6px 12px;margin-bottom:10px;'
                                    f'font-size:0.76rem;color:#f97316;">⚠️ {grid_warn}</div>'
                                )
                            st.markdown(grid_html, unsafe_allow_html=True)

                        for rank, r in enumerate(results, 1):
                            clabel, cmsg = consensus_label(r["cosine"], r["hqi"],
                                                           thresh_cos, thresh_hqi)
                            card_cls = {"strong":"m-strong","medium":"m-medium",
                                        "conflict":"m-conflict","weak":"m-weak"}.get(clabel,"m-weak")
                            cos_ok = "✓" if r["cosine"] >= thresh_cos else "✗"
                            hqi_ok = "✓" if r["hqi"] >= thresh_hqi else "✗"

                            # Detect rank conflict
                            conflict_note = ""
                            if rank > 1:
                                prev = results[rank-2]
                                if (r["cosine"] > prev["cosine"]) != (r["hqi"] > prev["hqi"]):
                                    conflict_note = f" · ⚠ {t('konflik ranking','rank conflict')}"

                            st.markdown(f"""
                            <div class="match-card {card_cls}">
                              <span class="m-badge" style="background:#1e293b;color:#94a3b8;
                                font-size:0.68rem;padding:2px 8px;border-radius:4px;">
                                {cmsg}
                              </span>
                              <span class="m-name">#{rank} &nbsp; {r['name']}</span>
                              <div class="m-scores">
                                Cosine: <b>{r['cosine']:.4f}</b> {cos_ok} &nbsp;|&nbsp;
                                HQI: <b>{r['hqi']:.2f}%</b> {hqi_ok} &nbsp;|&nbsp;
                                {t('Kategori','Category')}: {r['category']}{conflict_note}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — LIBRARY (ADMIN ONLY)
# ════════════════════════════════════════════════════════════════
if is_admin() and tab_lib:
    with tab_lib:
        st.markdown(f'<p class="sec-hdr">{t("Tambah spektra acuan","Add reference spectrum")}</p>',
                    unsafe_allow_html=True)

        with st.form("add_ref"):
            r1,r2 = st.columns(2)
            ref_name    = r1.text_input(t("Nama senyawa","Compound name"))
            ref_cat     = r2.text_input(t("Kategori","Category"))
            r3,r4 = st.columns(2)
            ref_subcat  = r3.text_input(t("Sub-kategori","Subcategory"))
            ref_cas     = r4.text_input("CAS Number")
            ref_file    = st.file_uploader(
                t("File spektra (Excel/CSV — 2 kolom: wavenumber, absorbance)",
                  "Spectrum file (Excel/CSV — 2 cols: wavenumber, absorbance)"),
                type=["xlsx","xls","csv","txt"]
            )
            ref_notes   = st.text_area(t("Catatan","Notes"), height=68)
            submitted   = st.form_submit_button(t("Tambahkan ke Library","Add to Library"))

            if submitted:
                if not ref_name:
                    st.error(t("Nama senyawa wajib diisi.","Compound name is required."))
                elif ref_file is None:
                    st.error(t("File spektra wajib diupload.","Spectrum file is required."))
                else:
                    try:
                        fn = ref_file.name.lower()
                        if fn.endswith((".xlsx",".xls")):
                            df_r = pd.read_excel(ref_file)
                        else:
                            df_r = pd.read_csv(ref_file, sep=None, engine="python")
                        wn_r = df_r.iloc[:,0].values.tolist()
                        sp_r = df_r.iloc[:,1].values.tolist()
                        new_id = add_spectrum(
                            ref_name, ref_cat, ref_subcat, ref_cas,
                            ref_notes, wn_r, sp_r,
                            added_by=st.session_state.get("username","admin")
                        )
                        st.success(t(f"✅ '{ref_name}' berhasil ditambahkan (ID: {new_id}).",
                                     f"✅ '{ref_name}' added successfully (ID: {new_id})."))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Import batch dari JSON
        st.markdown(f'<p class="sec-hdr">{t("Import batch dari JSON","Batch import from JSON")}</p>',
                    unsafe_allow_html=True)
        imp_file = st.file_uploader(t("Upload file library (.json)","Upload library file (.json)"),
                                    type=["json"], key="imp_json")
        if imp_file:
            import json as _json
            data = _json.load(imp_file)
            st.write(t(f"Ditemukan {len(data)} entri di file.",
                       f"Found {len(data)} entries in file."))
            if st.button(t("Import sekarang","Import now")):
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(_json.dumps(data).encode())
                    tmp_path = tmp.name
                added = import_from_json(tmp_path, st.session_state.get("username","admin"))
                os.unlink(tmp_path)
                st.success(t(f"✅ {added} spektra berhasil diimport.",
                             f"✅ {added} spectra imported successfully."))
                st.rerun()

        # Daftar library
        st.markdown(f'<p class="sec-hdr">{t("Daftar library","Library list")} — {count_spectra():,} {t("entri","entries")}</p>',
                    unsafe_allow_html=True)

        cats = ["— " + t("Semua kategori","All categories") + " —"] + get_categories()
        filter_cat = st.selectbox(t("Filter kategori","Filter category"), cats)

        all_meta = get_all_meta()
        if "Semua" not in filter_cat and "All" not in filter_cat:
            all_meta = [e for e in all_meta if e["category"] == filter_cat]

        for entry in all_meta[:100]:  # Show max 100 at once
            c_info, c_del = st.columns([6, 1])
            with c_info:
                st.markdown(f"""
                <div style="background:#161b27;border:1px solid #2a3142;border-radius:8px;
                  padding:0.6rem 1rem;margin-bottom:5px;">
                  <span style="font-weight:500;color:#e2e8f0;">{entry['name']}</span>
                  <span style="font-size:0.75rem;color:#475569;margin-left:8px;">{entry['category']}</span>
                  <span style="font-size:0.72rem;color:#334155;float:right;">ID:{entry['id']} · {entry['added_at']}</span>
                  <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">
                    {entry['n_points']} pts · {entry['wavenumber_min']:.0f}–{entry['wavenumber_max']:.0f} cm⁻¹
                    {(' · CAS: '+entry['cas_number']) if entry['cas_number'] else ''}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with c_del:
                if st.button("✕", key=f"del_{entry['id']}",
                             help=t("Hapus entri ini","Delete this entry")):
                    delete_spectrum(entry["id"])
                    st.rerun()

        if len(all_meta) > 100:
            st.caption(t(f"Menampilkan 100 dari {len(all_meta)} entri. Gunakan filter kategori.",
                         f"Showing 100 of {len(all_meta)} entries. Use category filter."))

# ════════════════════════════════════════════════════════════════
# TAB 5 — ADMIN PANEL
# ════════════════════════════════════════════════════════════════
if is_admin() and tab_admin:
    with tab_admin:
        st.markdown(f'<p class="sec-hdr">{t("Manajemen pengguna","User management")}</p>',
                    unsafe_allow_html=True)

        from auth import load_users, save_users, hash_password as hp

        users = load_users()

        # Add user
        with st.form("add_user"):
            u1,u2 = st.columns(2)
            new_uname = u1.text_input(t("Username baru","New username"))
            new_name  = u2.text_input(t("Nama lengkap","Full name"))
            u3,u4 = st.columns(2)
            new_pw    = u3.text_input(t("Password","Password"), type="password")
            new_role  = u4.selectbox(t("Role","Role"), ["user", "admin"])
            if st.form_submit_button(t("Tambah pengguna","Add user")):
                if new_uname and new_pw and new_name:
                    if new_uname in users:
                        st.error(t("Username sudah ada.","Username already exists."))
                    else:
                        users[new_uname] = {
                            "password": hp(new_pw),
                            "role": new_role,
                            "name": new_name
                        }
                        save_users(users)
                        st.success(t(f"✅ User '{new_uname}' berhasil ditambahkan.",
                                     f"✅ User '{new_uname}' added successfully."))
                        st.rerun()

        # User list
        st.markdown(f'<p class="sec-hdr">{t("Daftar pengguna","User list")}</p>',
                    unsafe_allow_html=True)
        for uname, udata in users.items():
            uc1, uc2 = st.columns([5,1])
            with uc1:
                badge_cls = "badge-admin" if udata["role"] == "admin" else "badge-user"
                st.markdown(f"""
                <div style="background:#161b27;border:1px solid #2a3142;border-radius:8px;
                  padding:0.6rem 1rem;margin-bottom:5px;">
                  <span style="font-weight:500;color:#e2e8f0;">{udata['name']}</span>
                  <span class="badge {badge_cls}">{udata['role']}</span>
                  <span style="font-size:0.75rem;color:#475569;margin-left:8px;">@{uname}</span>
                </div>
                """, unsafe_allow_html=True)
            with uc2:
                cur = st.session_state.get("username","")
                if uname != cur:
                    if st.button("✕", key=f"delusr_{uname}"):
                        del users[uname]
                        save_users(users)
                        st.rerun()

        # Change password
        st.markdown(f'<p class="sec-hdr">{t("Ganti password","Change password")}</p>',
                    unsafe_allow_html=True)
        with st.form("change_pw"):
            cp1,cp2 = st.columns(2)
            cp_user   = cp1.selectbox(t("Pengguna","User"), list(users.keys()))
            cp_newpw  = cp2.text_input(t("Password baru","New password"), type="password")
            if st.form_submit_button(t("Ganti password","Change password")):
                if cp_newpw:
                    users[cp_user]["password"] = hp(cp_newpw)
                    save_users(users)
                    st.success(t(f"✅ Password '{cp_user}' berhasil diganti.",
                                 f"✅ Password for '{cp_user}' changed successfully."))

# ════════════════════════════════════════════════════════════════
# TAB LAPORAN / REPORT
# ════════════════════════════════════════════════════════════════
with tab_rep:
    if "mcr_S" not in st.session_state:
        st.info(t("Jalankan MCR-ALS terlebih dahulu.",
                  "Please run MCR-ALS first."))
    else:
        S   = st.session_state["mcr_S"]
        C   = st.session_state["mcr_C"]
        wn  = st.session_state["wavenumber"]
        lof = st.session_state["mcr_lof"]
        r2  = st.session_state["mcr_r2"]
        nc  = st.session_state["mcr_ncomp"]
        snames = st.session_state.get("spec_names", [f"S{i+1}" for i in range(C.shape[0])])

        st.markdown(f'<p class="sec-hdr">{t("Export data","Export data")}</p>',
                    unsafe_allow_html=True)

        # Build Excel report
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Sheet 1 — Pure spectra
            df_S = pd.DataFrame(
                S.T, index=wn,
                columns=[f"{t('Komponen','Component')}_{i+1}" for i in range(nc)]
            )
            df_S.index.name = "Wavenumber (cm-1)"
            df_S.to_excel(writer, sheet_name=t("Spektra Murni","Pure Spectra"))

            # Sheet 2 — Concentration
            df_C = pd.DataFrame(
                C, index=snames,
                columns=[f"{t('Komponen','Component')}_{i+1}" for i in range(nc)]
            )
            df_C.index.name = t("Sampel","Sample")
            df_C.to_excel(writer, sheet_name=t("Konsentrasi","Concentration"))

            # Sheet 3 — LOF
            df_lof = pd.DataFrame({
                t("Iterasi","Iteration"): range(1, len(lof)+1),
                "LOF (%)": lof
            })
            df_lof.to_excel(writer, sheet_name="LOF", index=False)

            # Sheet 4 — Matching results
            if "match_results" in st.session_state:
                rows = []
                for i, results in enumerate(st.session_state["match_results"]):
                    for rank, r in enumerate(results, 1):
                        rows.append({
                            t("Komponen","Component"): i+1,
                            t("Rank","Rank"): rank,
                            t("Nama","Name"): r["name"],
                            t("Kategori","Category"): r["category"],
                            "Cosine": r["cosine"],
                            "HQI (%)": r["hqi"],
                            t("Status","Status"): consensus_label(r["cosine"], r["hqi"])[1]
                        })
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name=t("Hasil Matching","Matching Results"), index=False
                )

            # Sheet 5 — Summary
            summary = {
                t("Parameter","Parameter"): [
                    t("Jumlah komponen","Number of components"),
                    t("Iterasi","Iterations"),
                    "LOF akhir / Final LOF (%)",
                    "R²",
                    t("Tanggal analisis","Analysis date"),
                    t("Operator","Operator")
                ],
                t("Nilai","Value"): [
                    nc, len(lof), f"{lof[-1]:.4f}", f"{r2:.6f}",
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    st.session_state.get("display_name","—")
                ]
            }
            pd.DataFrame(summary).to_excel(
                writer, sheet_name=t("Ringkasan","Summary"), index=False
            )

        output.seek(0)
        fname = f"SpectraID_Pro_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

        e1,e2 = st.columns(2)
        with e1:
            st.download_button(
                t("⬇ Download laporan Excel (semua sheet)","⬇ Download Excel report (all sheets)"),
                data=output,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with e2:
            # CSV pure spectra
            df_S_csv = pd.DataFrame(S.T, index=wn,
                columns=[f"Component_{i+1}" for i in range(nc)])
            df_S_csv.index.name = "Wavenumber"
            st.download_button(
                t("⬇ Spektra murni (CSV)","⬇ Pure spectra (CSV)"),
                df_S_csv.to_csv(),
                f"pure_spectra_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
