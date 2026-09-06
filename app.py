"""
Streamlit portal: run the audit on your own data.

The dashboard shows results from one fixed run. This lets anyone upload
their own CSV, choose a target and a protected attribute, and get a
stability matrix back — which is what makes the tool reusable rather
than a script that only works on one dataset.

    streamlit run app.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.audit import run_audit
from src.data_loader import encode_features, encode_target
from src.preprocessing import split
from src.reformat import (
    AGE_VARIANTS,
    GENDER_VARIANTS,
    reformat_categorical,
    reformat_continuous,
)

st.set_page_config(
    page_title="SHAP Stability Audit",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- styling ------------------------------------------------------------

st.markdown("""
<style>
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .block-container { padding-top: 2.4rem; max-width: 1180px; }
  #MainMenu, footer { visibility: hidden; }

  .hero {
    background: linear-gradient(135deg, #12232e 0%, #1f4152 100%);
    color: #fff; padding: 30px 34px; border-radius: 12px;
    margin-bottom: 26px;
  }
  .hero h1 {
    margin: 0 0 8px; font-size: 27px; font-weight: 600;
    letter-spacing: -.02em; color: #fff;
  }
  .hero p {
    margin: 0; font-size: 15px; line-height: 1.55;
    opacity: .82; max-width: 680px;
  }

  div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e2e8ee; border-radius: 10px;
    padding: 16px 18px;
  }
  div[data-testid="stMetricLabel"] p {
    font-size: 11.5px !important; text-transform: uppercase;
    letter-spacing: .07em; color: #5c6b7a !important;
  }
  div[data-testid="stMetricValue"] {
    font-size: 26px !important; font-weight: 600;
    letter-spacing: -.02em;
  }

  section[data-testid="stSidebar"] { background: #f7f9fb; }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

  .stButton > button {
    border-radius: 8px; font-weight: 600; border: none;
    background: #1f6f8b; color: #fff; padding: .55rem 1.4rem;
  }
  .stButton > button:hover { background: #185870; color: #fff; }

  .verdict-fail {
    background: #fdf3e7; border-left: 4px solid #b8500f;
    padding: 16px 20px; border-radius: 6px; font-size: 14.5px;
    line-height: 1.6; margin: 6px 0 22px;
  }
  .verdict-pass {
    background: #e6f2ea; border-left: 4px solid #1d6b45;
    padding: 16px 20px; border-radius: 6px; font-size: 14.5px;
    line-height: 1.6; margin: 6px 0 22px;
  }
  .section-label {
    font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
    color: #5c6b7a; font-weight: 650; margin: 26px 0 10px;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>SHAP Stability Audit</h1>
  <p>Not <em>is this model biased?</em> — but can the SHAP explanation you
     would use to answer that question survive a routine change to how a
     protected attribute is encoded?</p>
</div>
""", unsafe_allow_html=True)

# --- sidebar: data and configuration ------------------------------------

with st.sidebar:
    st.markdown("### Data")
    uploaded = st.file_uploader("Tabular CSV", type=["csv"])

if uploaded is None:
    st.info(
        "**Upload a CSV in the sidebar to begin.** It needs a binary target "
        "column and at least one protected attribute to audit. Text, images "
        "and other non-tabular files are not supported."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded, skipinitialspace=True)
except Exception as e:
    st.error(f"Could not read that file as a table. {e}")
    st.stop()

if df.shape[1] < 2:
    st.error(
        f"Only {df.shape[1]} column(s) found. This tool needs tabular data "
        "with at least a target and one feature."
    )
    st.stop()

with st.sidebar:
    st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} columns")

    st.markdown("### Columns")
    target_col = st.selectbox(
        "Target (binary)",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
        help="The outcome the model predicts.",
    )
    protected_col = st.selectbox(
        "Protected attribute",
        options=[c for c in df.columns if c != target_col],
        help="The column whose SHAP explanation you want to test.",
    )

n_target_classes = df[target_col].nunique()
if n_target_classes != 2:
    st.error(
        f"**'{target_col}' has {n_target_classes} unique values.** "
        "This tool expects a binary target — pick a different column in "
        "the sidebar."
    )
    st.stop()

n_unique = df[protected_col].nunique()
suggested = "continuous" if n_unique > 10 else "categorical"

with st.sidebar:
    st.markdown("### Variants")
    st.caption(f"'{protected_col}' has {n_unique} unique values")
    variant_type = st.radio(
        "Treat as",
        options=["continuous", "categorical"],
        index=0 if suggested == "continuous" else 1,
        horizontal=True,
    )

    if variant_type == "continuous":
        all_variants, reformat_fn = AGE_VARIANTS, reformat_continuous
    else:
        all_variants, reformat_fn = GENDER_VARIANTS, reformat_categorical

    variants = st.multiselect(
        "Reformattings to test", options=all_variants, default=all_variants
    )

    st.markdown("### Model")
    max_depth = st.number_input("Max tree depth (0 = unlimited)", 0, 50, 0)
    n_estimators = st.number_input("Trees", 50, 500, 300, step=50)

    run_clicked = st.button("Run audit", use_container_width=True)

# --- preview ------------------------------------------------------------

with st.expander(f"Preview — {df.shape[0]:,} rows"):
    st.dataframe(df.head(12), use_container_width=True)

if not variants:
    st.warning("Select at least one variant in the sidebar.")
    st.stop()

if run_clicked:
    st.session_state.run_requested = True

if not st.session_state.get("run_requested"):
    st.info("Configure the audit in the sidebar, then press **Run audit**.")
    st.stop()

# --- run (cached so widgets do not retrigger training) ------------------

rf_kwargs = {"n_estimators": int(n_estimators)}
if max_depth > 0:
    rf_kwargs["max_depth"] = int(max_depth)

cache_key = (uploaded.name, target_col, protected_col,
             tuple(variants), int(max_depth), int(n_estimators))

if st.session_state.get("cache_key") != cache_key:
    with st.spinner(f"Training {len(variants) + 1} models and computing SHAP…"):
        y = encode_target(df[target_col])
        X, report = encode_features(df.drop(columns=[target_col]), verbose=False)

        if protected_col not in X.columns:
            st.error(
                f"**'{protected_col}' was dropped during encoding.** It is "
                "either constant or has too many unique values to be a "
                "usable feature."
            )
            st.stop()

        X_train, X_test, y_train, y_test = split(X, y)

        results, baseline, detail = run_audit(
            X_train, X_test, y_train, y_test,
            protected_col=protected_col,
            variants=variants,
            reformat_fn=reformat_fn,
            make_plots=False,
            **rf_kwargs,
        )

    st.session_state.cache_key = cache_key
    st.session_state.results = results
    st.session_state.detail = detail
    st.session_state.report = report

results = st.session_state.results.copy()
detail = st.session_state.detail
report = st.session_state.report

# Threshold sits outside the cache key on purpose: moving it re-labels
# verdicts instantly instead of retraining six models.
threshold = st.slider(
    "Stability threshold (Spearman rho)",
    0.80, 1.00, float(config.STABILITY_THRESHOLD), 0.01,
    help="A convention, not a derived value. Move it and watch the "
         "verdicts change — the underlying numbers do not.",
)
results["stable"] = results["spearman_rho"] >= threshold

# --- headline -----------------------------------------------------------

n_stable = int(results["stable"].sum())
biggest_drop = results.loc[results["rank_shift"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Variants tested", len(results))
c2.metric(f"Passed rho ≥ {threshold:.2f}", n_stable)
c3.metric("Lowest rho", f"{results['spearman_rho'].min():.4f}")
c4.metric("Largest magnitude loss",
          f"{results['magnitude_change_pct'].min():.1f}%")

if n_stable == len(results) and biggest_drop["rank_shift"] > 0:
    st.markdown(f"""
    <div class="verdict-fail">
      <strong>Every variant passed the global threshold</strong> — but under
      <code>{biggest_drop['variant']}</code>, {protected_col} still fell from
      rank {int(biggest_drop['protected_baseline_rank'])} to
      {int(biggest_drop['protected_variant_rank'])}, with its attributed
      importance changing by
      {biggest_drop['magnitude_change_pct']:+.1f}%.
      This is the failure an attribute-level check exists to catch: the
      aggregate metric averages it away.
    </div>
    """, unsafe_allow_html=True)
elif n_stable < len(results):
    st.markdown(f"""
    <div class="verdict-pass">
      <strong>{len(results) - n_stable} of {len(results)} variants flagged
      unstable</strong> at rho ≥ {threshold:.2f}. Lower the threshold and
      they pass — the numbers do not change, only the verdict does.
    </div>
    """, unsafe_allow_html=True)

# --- results table ------------------------------------------------------

st.markdown('<div class="section-label">Results matrix</div>',
            unsafe_allow_html=True)

display = results[[
    "variant", "spearman_rho", "top_k_overlap",
    "protected_baseline_rank", "protected_variant_rank",
    "magnitude_change_pct", "stable",
]].rename(columns={
    "variant": "Variant",
    "spearman_rho": "Spearman rho",
    "top_k_overlap": "Top-3 overlap",
    "protected_baseline_rank": "Rank before",
    "protected_variant_rank": "Rank after",
    "magnitude_change_pct": "Magnitude %",
    "stable": "Stable",
})

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Spearman rho": st.column_config.ProgressColumn(
            "Spearman rho", min_value=0.8, max_value=1.0, format="%.4f"),
        "Magnitude %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Stable": st.column_config.CheckboxColumn(),
    },
)

# --- before / after -----------------------------------------------------

st.markdown('<div class="section-label">Before / after comparison</div>',
            unsafe_allow_html=True)

chosen = st.selectbox("Variant", options=variants, label_visibility="collapsed")
row = results[results["variant"] == chosen].iloc[0]

left, right = st.columns([3, 1])

with right:
    st.metric("Rank",
              f"{int(row['protected_baseline_rank'])} → "
              f"{int(row['protected_variant_rank'])}")
    st.metric("Magnitude", f"{row['magnitude_change_pct']:+.1f}%")
    st.metric("Spearman rho", f"{row['spearman_rho']:.4f}")
    log_scale = st.checkbox("Log scale")

with left:
    base = pd.Series(detail["baseline"])
    var = pd.Series(detail[chosen]).reindex(base.index).fillna(0)
    order = base.sort_values().index

    height = max(3.2, 0.26 * len(order))
    fig, ax = plt.subplots(figsize=(9, height))

    ypos = range(len(order))
    offset = 0.4
    ax.barh([y + offset / 2 for y in ypos], base[order],
            height=0.38,
            color=["#c9938e" if f == protected_col else "#c3ced8"
                   for f in order],
            label="Baseline")
    ax.barh([y - offset / 2 for y in ypos], var[order],
            height=0.38,
            color=["#a8322a" if f == protected_col else "#1f6f8b"
                   for f in order],
            label=f"After: {chosen}")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    if log_scale:
        ax.set_xscale("log")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eef2f6")
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)

st.caption(
    f"**{protected_col}** is highlighted in red. The global correlation "
    "above can pass comfortably while this one feature collapses."
)

# --- provenance ---------------------------------------------------------

if report["constant"] or report["dropped"] or report["encoded"]:
    with st.expander("What the loader did to your data"):
        for col in report["constant"]:
            st.write(f"- dropped **{col}** — constant, one value throughout")
        for col, n in report["dropped"]:
            st.write(f"- dropped **{col}** — {n} unique values, "
                     "looks like an identifier")
        for col, n in report["encoded"]:
            st.write(f"- encoded **{col}** — {n} categories")

st.download_button(
    "Download results as CSV",
    results.to_csv(index=False).encode(),
    "audit_results.csv",
    "text/csv",
)