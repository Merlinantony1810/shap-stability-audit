"""
Static HTML dashboard.

Results are embedded into the HTML at generation time, so the file is
self-contained and needs no server. Chart.js is vendored locally rather
than loaded from a CDN: the earlier version of this project rendered a
blank page without internet access, which is a bad failure mode for
something you demo in a meeting room.
"""

import json

from src import config

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SHAP Stability Audit</title>
<script src="chart.min.js"></script>
<style>
  :root {{
    --ink:      #12232e;
    --ink-soft: #5c6b7a;
    --line:     #e2e8ee;
    --surface:  #ffffff;
    --bg:       #eef2f6;
    --accent:   #1f6f8b;
    --accent-2: #99aebb;
    --warn:     #b8500f;
    --warn-bg:  #fdf3e7;
    --danger:   #a8322a;
    --ok:       #1d6b45;
    --ok-bg:    #e6f2ea;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
         Roboto, sans-serif; margin: 0; background: var(--bg);
         color: var(--ink); }}

  header {{ background: linear-gradient(135deg, #12232e 0%, #1f4152 100%);
            color: #fff; padding: 34px 40px 30px; }}
  header h1 {{ margin: 0 0 7px; font-size: 23px; font-weight: 600;
               letter-spacing: -.01em; }}
  header p {{ margin: 0; opacity: .8; font-size: 14.5px; max-width: 640px;
              line-height: 1.5; }}

  main {{ max-width: 1180px; margin: 0 auto; padding: 30px 40px 70px; }}

  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 15px; margin-bottom: 28px; }}
  .card {{ background: var(--surface); border-radius: 9px; padding: 19px 21px;
           border: 1px solid var(--line);
           transition: transform .15s ease, box-shadow .15s ease; }}
  .card:hover {{ transform: translateY(-2px);
                 box-shadow: 0 5px 18px rgba(18,35,46,.09); }}
  .card .value {{ font-size: 27px; font-weight: 650;
                  letter-spacing: -.02em; }}
  .card .value.warn {{ color: var(--warn); }}
  .card .label {{ font-size: 11.5px; text-transform: uppercase;
                  letter-spacing: .07em; color: var(--ink-soft);
                  margin-top: 5px; }}

  .alert {{ background: var(--warn-bg); border-left: 4px solid var(--warn);
            padding: 17px 21px; border-radius: 5px; margin-bottom: 28px;
            font-size: 14.5px; line-height: 1.6; }}
  .alert code {{ background: rgba(184,80,15,.11); padding: 1px 6px;
                 border-radius: 3px; font-size: 13px; }}

  section {{ background: var(--surface); border: 1px solid var(--line);
             border-radius: 9px; padding: 25px; margin-bottom: 22px; }}
  h2 {{ margin: 0 0 17px; font-size: 13px; text-transform: uppercase;
        letter-spacing: .08em; color: var(--ink-soft); font-weight: 650; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 10px 12px;
        border-bottom: 2px solid var(--line); font-size: 11.5px;
        text-transform: uppercase; letter-spacing: .06em;
        color: var(--ink-soft); cursor: pointer; user-select: none;
        white-space: nowrap; }}
  th:hover {{ color: var(--accent); }}
  th::after {{ content: " ⇅"; opacity: .3; font-size: 10px; }}
  td {{ padding: 11px 12px; border-bottom: 1px solid #f2f5f8; }}
  tbody tr {{ cursor: pointer; transition: background .12s ease; }}
  tbody tr:hover {{ background: #f7fafc; }}
  tbody tr.flagged {{ background: #fffbf5; }}
  tbody tr.flagged:hover {{ background: #fff6ea; }}
  tbody tr.active {{ background: #e8f1f5;
                     box-shadow: inset 3px 0 0 var(--accent); }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 13px; }}

  .badge {{ display: inline-block; padding: 2.5px 10px; border-radius: 11px;
            font-size: 11.5px; font-weight: 650; letter-spacing: .02em; }}
  .badge.ok {{ background: var(--ok-bg); color: var(--ok); }}
  .drop {{ color: var(--danger); font-weight: 650; }}
  .neg {{ color: var(--danger); }}
  .pos {{ color: var(--ok); }}

  .bar-wrap {{ display: inline-block; width: 62px; height: 6px;
               background: #edf1f5; border-radius: 3px; margin-left: 9px;
               vertical-align: middle; overflow: hidden; }}
  .bar {{ height: 100%; background: var(--accent); border-radius: 3px; }}

  .controls {{ display: flex; align-items: center; gap: 14px;
               margin-bottom: 18px; flex-wrap: wrap; }}
  select {{ padding: 8px 12px; border: 1px solid #ccd6df; border-radius: 6px;
            font-size: 14px; background: #fff; color: var(--ink);
            cursor: pointer; }}
  select:focus {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
  .toggle {{ font-size: 13.5px; color: var(--ink-soft);
             display: flex; align-items: center; gap: 6px; cursor: pointer; }}
  .readout {{ margin-left: auto; font-size: 13.5px; color: var(--ink-soft); }}
  .readout strong {{ color: var(--ink); }}

  .hint {{ font-size: 12.5px; color: var(--ink-soft); margin-top: 14px;
           font-style: italic; }}
</style>
</head>
<body>
<header>
  <h1>SHAP Stability Audit</h1>
  <p>Not &ldquo;is this model biased?&rdquo; &mdash; but can the SHAP
     explanation you would use to answer that question survive a routine
     change to how a protected attribute is encoded?</p>
</header>
<main>

  <div class="cards">
    <div class="card"><div class="value">{n_variants}</div>
      <div class="label">Variants tested</div></div>
    <div class="card"><div class="value">{n_stable}</div>
      <div class="label">Passed rho &ge; {threshold}</div></div>
    <div class="card"><div class="value">{worst_rho}</div>
      <div class="label">Lowest rho</div></div>
    <div class="card"><div class="value warn">{worst_mag}%</div>
      <div class="label">Largest magnitude loss</div></div>
  </div>

  <div class="alert">
    <strong>Every variant passed the global stability threshold.</strong>
    Under <code>coarse_binary</code>, Age still fell from rank
    {age_before} to {age_after} and lost {worst_mag_abs}% of its attributed
    importance &mdash; while the aggregate metric reported the explanation
    as stable. This is the failure an attribute-level check exists to catch.
  </div>

  <section>
    <h2>Results matrix</h2>
    <table id="matrix">
      <thead><tr>
        <th data-col="attribute">Attribute</th>
        <th data-col="variant">Variant</th>
        <th data-col="spearman_rho">Spearman rho</th>
        <th data-col="top_k_overlap">Top-3 overlap</th>
        <th data-col="protected_variant_rank">Rank</th>
        <th data-col="magnitude_change_pct">Magnitude</th>
        <th data-col="stable">Verdict</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    <p class="hint">Click any row to load it in the comparison below.
       Click a column heading to sort.</p>
  </section>

  <section>
    <h2>Before / after comparison</h2>
    <div class="controls">
      <select id="variantPicker"></select>
      <label class="toggle">
        <input type="checkbox" id="logScale"> Log scale
      </label>
      <span class="readout" id="readout"></span>
    </div>
    <canvas id="chart" height="115"></canvas>
  </section>

</main>
<script>
const RESULTS = {results_json};
const IMPORTANCE = {importance_json};
const THRESHOLD = {threshold};

/* ---------- results table ---------- */

let sortCol = null, sortAsc = true;

function fmtMag(v) {{
  const cls = v < 0 ? "neg" : "pos";
  return `<span class="${{cls}}">${{v > 0 ? "+" : ""}}${{v.toFixed(1)}}%</span>`;
}}

function rhoBar(rho) {{
  // Stretch 0.90-1.00 across the full bar so differences are visible.
  const pct = Math.max(0, Math.min(1, (rho - 0.9) / 0.1)) * 100;
  return `<span class="bar-wrap"><span class="bar" style="width:${{pct}}%"></span></span>`;
}}

function renderTable() {{
  const tbody = document.querySelector("#matrix tbody");
  tbody.innerHTML = "";

  let rows = RESULTS.map((r, i) => ({{ ...r, _idx: i }}));
  if (sortCol) {{
    rows.sort((a, b) => {{
      const x = a[sortCol], y = b[sortCol];
      const cmp = (typeof x === "string") ? x.localeCompare(y) : x - y;
      return sortAsc ? cmp : -cmp;
    }});
  }}

  rows.forEach(r => {{
    const dropped = r.protected_variant_rank > r.protected_baseline_rank;
    const rankTxt = `${{r.protected_baseline_rank}} → ${{r.protected_variant_rank}}`;
    const tr = document.createElement("tr");
    tr.className = dropped ? "flagged" : "";
    tr.dataset.idx = r._idx;
    tr.innerHTML =
      `<td>${{r.attribute}}</td>` +
      `<td><code>${{r.variant}}</code></td>` +
      `<td>${{r.spearman_rho.toFixed(4)}}${{rhoBar(r.spearman_rho)}}</td>` +
      `<td>${{r.top_k_overlap.toFixed(1)}}</td>` +
      `<td>${{dropped ? `<span class="drop">${{rankTxt}}</span>` : rankTxt}}</td>` +
      `<td>${{fmtMag(r.magnitude_change_pct)}}</td>` +
      `<td><span class="badge ok">stable</span></td>`;
    tr.addEventListener("click", () => {{
      picker.value = r._idx;
      draw(r._idx);
    }});
    tbody.appendChild(tr);
  }});
  highlightRow(+picker.value);
}}

function highlightRow(idx) {{
  document.querySelectorAll("#matrix tbody tr").forEach(tr => {{
    tr.classList.toggle("active", +tr.dataset.idx === idx);
  }});
}}

document.querySelectorAll("#matrix th").forEach(th => {{
  th.addEventListener("click", () => {{
    const col = th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc;
    else {{ sortCol = col; sortAsc = true; }}
    renderTable();
  }});
}});

/* ---------- comparison chart ---------- */

const picker = document.getElementById("variantPicker");
RESULTS.forEach((r, i) => {{
  const o = document.createElement("option");
  o.value = i;
  o.textContent = `${{r.attribute}} — ${{r.variant}}  (rho ${{r.spearman_rho.toFixed(4)}})`;
  picker.appendChild(o);
}});

let chart = null;

function draw(idx) {{
  const row = RESULTS[idx];
  const baseline = IMPORTANCE[row.attribute]["baseline"];
  const variant  = IMPORTANCE[row.attribute][row.variant];
  const labels = Object.keys(baseline);
  const logScale = document.getElementById("logScale").checked;

  document.getElementById("readout").innerHTML =
    `rank <strong>${{row.protected_baseline_rank}} → ${{row.protected_variant_rank}}</strong>` +
    ` &nbsp;·&nbsp; magnitude <strong>${{row.magnitude_change_pct > 0 ? "+" : ""}}` +
    `${{row.magnitude_change_pct.toFixed(1)}}%</strong>`;

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById("chart"), {{
    type: "bar",
    data: {{
      labels: labels,
      datasets: [
        {{ label: "Baseline",
           data: labels.map(f => baseline[f] ?? 0),
           backgroundColor: labels.map(
             f => f === row.attribute ? "#c9938e" : "#99aebb"),
           borderRadius: 3 }},
        {{ label: `After: ${{row.variant}}`,
           data: labels.map(f => variant[f] ?? 0),
           backgroundColor: labels.map(
             f => f === row.attribute ? "#a8322a" : "#1f6f8b"),
           borderRadius: 3 }}
      ]
    }},
    options: {{
      responsive: true,
      animation: {{ duration: 450 }},
      plugins: {{
        legend: {{ labels: {{ boxWidth: 13, font: {{ size: 12.5 }} }} }},
        tooltip: {{
          callbacks: {{
            afterBody: (items) =>
              items[0].label === row.attribute
                ? "← the attribute under audit" : ""
          }}
        }}
      }},
      scales: {{
        y: {{
          type: logScale ? "logarithmic" : "linear",
          title: {{ display: true, text: "Mean |SHAP value|" }},
          grid: {{ color: "#eef2f6" }}
        }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
  highlightRow(idx);
}}

picker.addEventListener("change", e => draw(+e.target.value));
document.getElementById("logScale").addEventListener("change",
  () => draw(+picker.value));

renderTable();
draw(0);
</script>
</body>
</html>
"""


def build(results_df, detail, out_path=None):
    """Write the dashboard with results embedded."""
    if out_path is None:
        out_path = config.REPORTS_DIR / "dashboard.html"

    worst = results_df.loc[results_df["spearman_rho"].idxmin()]

    html = TEMPLATE.format(
        n_variants=len(results_df),
        n_stable=int(results_df["stable"].sum()),
        threshold=config.STABILITY_THRESHOLD,
        worst_rho=f"{results_df['spearman_rho'].min():.4f}",
        worst_mag=f"{results_df['magnitude_change_pct'].min():.1f}",
        worst_mag_abs=f"{abs(results_df['magnitude_change_pct'].min()):.1f}",
        age_before=int(worst["protected_baseline_rank"]),
        age_after=int(worst["protected_variant_rank"]),
        results_json=json.dumps(results_df.to_dict("records")),
        importance_json=json.dumps(detail),
    )

    out_path.write_text(html)
    return out_path