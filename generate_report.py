"""
将翻译评估结果导出为一个自包含的 HTML 报告（无需联网、无外部依赖）。
含：自动指标、显著性检验、人工文化 gold 标注分析、方言词分层、最差案例。

输入: translation_keyword_scores.csv, translation_segment_scores.csv,
      translation_results_200.csv (含 preference gold)
输出: translation_eval_report.html, 并更新 translation_scores.txt 人工评测节
运行: python generate_report.py
"""

import html
import numpy as np
import pandas as pd

KW_CSV = "translation_keyword_scores.csv"
SEG_CSV = "translation_segment_scores.csv"
RES_CSV = "translation_results_200.csv"
OUT_HTML = "translation_eval_report.html"
SCORES_TXT = "translation_scores.txt"
MIN_N = 5
KAPPA = 0.71  # 两人独立标注后讨论定稿的 Cohen's κ

GLM_COL, DS_COL = "GLM-4-Plus", "DeepSeek-V4-Flash"
DS_SHORT = "DS"  # 图表短标签
GLM_C = "#3b82f6"
DS_C = "#16a34a"
TG_C = "#16a34a"   # tie-good
TB_C = "#dc2626"   # tie-bad


def esc(x) -> str:
    return html.escape(str(x))


def bar_chart(rows, glm_key, ds_key, scale, suffix=""):
    """生成一组横向分组条形图的 HTML。"""
    out = []
    for r in rows:
        g, d = float(r[glm_key]), float(r[ds_key])
        out.append(f"""
        <div class="bar-row">
          <div class="bar-label">{esc(r['keyword'])} <span class="muted">n={int(r['n'])}</span></div>
          <div class="bar-track">
            <div class="bar" style="width:{g/scale*100:.1f}%;background:{GLM_C}"></div>
            <span class="bar-val">{g:.1f}{suffix}</span>
          </div>
          <div class="bar-track">
            <div class="bar" style="width:{d/scale*100:.1f}%;background:{DS_C}"></div>
            <span class="bar-val">{d:.1f}{suffix}</span>
          </div>
        </div>""")
    return "".join(out)


def compute_human_eval(results: pd.DataFrame, seg: pd.DataFrame) -> dict:
    """合并 gold (preference) 与逐句自动分数，计算人工评测统计。"""
    m = results.merge(
        seg[[
            "sichuanese", f"{GLM_COL}_COMET", f"{DS_COL}_COMET",
            f"{GLM_COL}_chrF", f"{DS_COL}_chrF",
            f"{GLM_COL}_BLEU", f"{DS_COL}_BLEU",
        ]],
        on="sichuanese",
        how="left",
    )
    lab = m["preference"].astype(str).str.strip()
    n = len(m)
    m["avg_comet"] = (m[f"{GLM_COL}_COMET"] + m[f"{DS_COL}_COMET"]) / 2
    m["comet_winner"] = np.where(
        m[f"{DS_COL}_COMET"] > m[f"{GLM_COL}_COMET"], "d",
        np.where(m[f"{DS_COL}_COMET"] < m[f"{GLM_COL}_COMET"], "g", "tie_comet"),
    )

    counts = {k: int((lab == k).sum()) for k in ["tie-good", "g", "d", "tie-bad", "tie"]}
    tie_bad_n = counts["tie-bad"] + counts["tie"]

    glm_pass = int(lab.isin(["tie-good", "g"]).sum())
    ds_pass = int(lab.isin(["tie-good", "d"]).sum())

    metrics = [
        f"{GLM_COL}_COMET", f"{DS_COL}_COMET",
        f"{GLM_COL}_chrF", f"{DS_COL}_chrF",
        f"{GLM_COL}_BLEU", f"{DS_COL}_BLEU",
    ]
    group_means = {}
    for g in ["tie-good", "g", "d", "tie-bad"]:
        sub = m[lab == g]
        group_means[g] = {met: float(sub[met].mean()) if len(sub) else 0.0 for met in metrics}
        group_means[g]["n"] = len(sub)

    tg = m[lab == "tie-good"]
    tb = m[lab.isin(["tie-bad", "tie"])]
    comp = m[lab.isin(["d", "g"])]
    agree = int((comp["preference"] == comp["comet_winner"]).sum())
    comp_n = len(comp)

    # tie-bad 率按关键词
    from collections import Counter
    kw_tb, kw_n = Counter(), Counter()
    for _, r in m.iterrows():
        kws = [k for k in str(r["matched_keywords"]).split("|") if k]
        is_bad = lab.loc[r.name] in ("tie-bad", "tie")
        for kw in kws:
            kw_n[kw] += 1
            if is_bad:
                kw_tb[kw] += 1
    kw_rows = sorted(
        [(kw, kw_n[kw], kw_tb[kw], kw_tb[kw] / kw_n[kw] * 100) for kw in kw_n if kw_n[kw] >= MIN_N],
        key=lambda x: -x[3],
    )

    # COMET 分布直方图 (avg COMET, bin=5)
    bins = list(range(50, 90, 5))
    hist = {}
    for g in ["tie-good", "tie-bad"]:
        vals = m.loc[lab == g, "avg_comet"].tolist() if g == "tie-good" else tb["avg_comet"].tolist()
        hist[g] = [sum(b <= v < b + 5 for v in vals) for b in bins]

    return {
        "m": m, "n": n, "counts": counts, "tie_bad_n": tie_bad_n,
        "glm_pass": glm_pass, "ds_pass": ds_pass,
        "group_means": group_means, "tg": tg, "tb": tb,
        "agree": agree, "comp_n": comp_n,
        "tb_comet_ds_wins": float((tb[f"{DS_COL}_COMET"] > tb[f"{GLM_COL}_COMET"]).mean()),
        "global_ds_comet": float(m[f"{DS_COL}_COMET"].mean()),
        "kw_rows": kw_rows, "bins": bins, "hist": hist,
    }


def comet_histogram_svg(he: dict) -> str:
    """tie-good vs tie-bad 的 avg COMET 分布（分组柱状图 SVG）。"""
    bins, hist = he["bins"], he["hist"]
    max_c = max(max(hist["tie-good"]), max(hist["tie-bad"]), 1)
    w, h, pad, bw = 640, 220, 48, 28
    bars = []
    for i, b in enumerate(bins):
        x = pad + i * (bw + 14)
        for j, (g, color) in enumerate([("tie-good", TG_C), ("tie-bad", TB_C)]):
            c = hist[g][i]
            bh = int(c / max_c * (h - pad - 30))
            bx = x + j * (bw // 2 + 2)
            bars.append(
                f'<rect x="{bx}" y="{h-pad-bh}" width="{bw//2}" height="{bh}" fill="{color}" rx="2"/>'
            )
            if c:
                bars.append(f'<text x="{bx+bw//4}" y="{h-pad-bh-4}" text-anchor="middle" font-size="10" fill="#444">{c}</text>')
        bars.append(f'<text x="{x+bw//2}" y="{h-pad+16}" text-anchor="middle" font-size="11" fill="#666">{b}-{b+5}</text>')
    return f'''<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" xmlns="http://www.w3.org/2000/svg">
  <text x="{pad}" y="18" font-size="13" fill="#333">逐句平均 COMET 分布（(GLM+DS)/2）</text>
  <line x1="{pad}" y1="{h-pad}" x2="{w-20}" y2="{h-pad}" stroke="#ccc"/>
  {"".join(bars)}
  <rect x="{w-200}" y="8" width="10" height="10" fill="{TG_C}"/><text x="{w-185}" y="17" font-size="11" fill="#666">tie-good (n={he["group_means"]["tie-good"]["n"]})</text>
  <rect x="{w-200}" y="24" width="10" height="10" fill="{TB_C}"/><text x="{w-185}" y="33" font-size="11" fill="#666">tie-bad (n={he["tie_bad_n"]})</text>
</svg>'''


def human_eval_txt(he: dict) -> str:
    c, n = he["counts"], he["n"]
    gm = he["group_means"]
    lines = [
        "",
        "[人工文化充分性评测 HUMAN CULTURAL ADEQUACY]",
        f"  标注: 2 人独立标注后讨论定稿 · Cohen's κ = {KAPPA:.2f} (substantial agreement)",
        f"  gold 列: translation_results_200.csv → preference",
        f"  d/g = 该模型成功传递文化含义; tie-good = 两者均达标; tie-bad = 两者均不达标",
        "",
        "  标签分布:",
        f"    tie-good (两者达标)     {c['tie-good']:>3}/{n}  ({c['tie-good']/n*100:.1f}%)",
        f"    g   (仅 GLM 达标)       {c['g']:>3}/{n}  ({c['g']/n*100:.1f}%)",
        f"    d   (仅 DeepSeek 达标)  {c['d']:>3}/{n}  ({c['d']/n*100:.1f}%)",
        f"    tie-bad (两者不达标)    {he['tie_bad_n']:>3}/{n}  ({he['tie_bad_n']/n*100:.1f}%)",
    ]
    if c.get("tie", 0):
        lines.append(f"    (另有未归类 tie: {c['tie']} 句，已计入 tie-bad 统计)")
    lines += [
        "",
        "  文化充分率:",
        f"    GLM-4-Plus:    {he['glm_pass']/n*100:.1f}%",
        f"    {DS_COL}:   {he['ds_pass']/n*100:.1f}%",
        "",
        "  [核心对比: tie-good vs tie-bad 自动分数均值]",
        f"  {'指标':<14}{'tie-good':>10}{'tie-bad':>10}{'差值':>10}",
        f"  {'-'*14}{'-'*10}{'-'*10}{'-'*10}",
    ]
    pairs = [
        ("GLM COMET", f"{GLM_COL}_COMET"),
        ("DS COMET", f"{DS_COL}_COMET"),
        ("GLM chrF", f"{GLM_COL}_chrF"),
        ("DS chrF", f"{DS_COL}_chrF"),
    ]
    for label, met in pairs:
        tg_v = gm["tie-good"][met]
        tb_v = he["tb"][met].mean()
        lines.append(f"  {label:<14}{tg_v:>10.2f}{tb_v:>10.2f}{tg_v-tb_v:>+10.2f}")
    lines += [
        "",
        "  [人工 vs 自动 一致率]",
        f"    人工 d/g vs COMET 胜者一致率: {he['agree']}/{he['comp_n']} ({he['agree']/he['comp_n']*100:.1f}%)",
        f"    tie-bad 句中 COMET 仍判 DS 更优: {he['tb_comet_ds_wins']*100:.1f}%",
        f"    tie-bad 句 DS COMET 均值: {he['tb'][f'{DS_COL}_COMET'].mean():.2f} (全局 {he['global_ds_comet']:.2f})",
        "",
        f"  [tie-bad 率按方言词 (n>={MIN_N})]",
        f"  {'关键词':<14}{'n':>4}{'tie-bad':>8}{'比率':>8}",
        f"  {'-'*14}{'-'*4}{'-'*8}{'-'*8}",
    ]
    for kw, cnt, tb, pct in he["kw_rows"]:
        lines.append(f"  {kw:<14}{cnt:>4}{tb:>8}{pct:>7.1f}%")
    lines += [
        "",
        "  结论:",
        "    · 仅 14% 句子两模型均成功传递文化义 (tie-good); ~28% 双失败 (tie-bad)。",
        "    · 自动指标能区分 tie-good/tie-bad (COMET 差 ~12)，但 tie-bad 绝对分仍 ~67，",
        "      仅比全局低 ~4 分; 人工–COMET 一致率 ~46%, 不足以反映 culture washing 严重程度。",
    ]
    return "\n".join(lines)


def human_eval_html(he: dict, hist_svg: str) -> str:
    c, n, gm = he["counts"], he["n"], he["group_means"]
    tg, tb = gm["tie-good"], he["tb"]

    def row(label, tg_m, tb_m):
        diff = tg_m - tb_m
        return f"<tr><td>{label}</td><td class='num'>{tg_m:.2f}</td><td class='num'>{tb_m:.2f}</td><td class='num'>{diff:+.2f}</td></tr>"

    dist_rows = "".join(
        f"<tr><td>{lab}</td><td class='num'>{c[lab]}</td><td class='num'>{c[lab]/n*100:.1f}%</td></tr>"
        for lab in ["tie-good", "g", "d", "tie-bad"]
    )
    def kw_row(kw, cnt, tb, pct):
        cls = ' class="low"' if pct >= 35 else ""
        return (
            f"<tr{cls}><td>{esc(kw)}</td><td class='num'>{cnt}</td>"
            f"<td class='num'>{tb}</td><td class='num'>{pct:.1f}%</td></tr>"
        )

    kw_rows = "".join(kw_row(kw, cnt, tb, pct) for kw, cnt, tb, pct in he["kw_rows"])
    group_rows = ""
    for g, title in [("tie-good", "tie-good"), ("g", "g (仅 GLM)"), ("d", "d (仅 DS)"), ("tie-bad", "tie-bad")]:
        gm_g = gm[g]
        group_rows += (
            f"<tr><td>{title}</td><td class='num'>{int(gm_g['n'])}</td>"
            f"<td class='num'>{gm_g[f'{GLM_COL}_COMET']:.2f}</td><td class='num'>{gm_g[f'{DS_COL}_COMET']:.2f}</td>"
            f"<td class='num'>{gm_g[f'{GLM_COL}_chrF']:.2f}</td><td class='num'>{gm_g[f'{DS_COL}_chrF']:.2f}</td></tr>"
        )

    return f"""
  <h2>人工文化充分性评测（gold）</h2>
  <p class="muted" style="font-size:12px">2 人独立标注后讨论定稿 · Cohen's κ = {KAPPA:.2f} · 参考译文为保留文化义的人工翻译</p>

  <div class="stats" style="grid-template-columns:repeat(5,1fr)">
    <div class="stat good"><div class="val">{c['tie-good']/n*100:.0f}%</div><div class="lab">tie-good 双达标</div></div>
    <div class="stat warn"><div class="val">{he['tie_bad_n']/n*100:.0f}%</div><div class="lab">tie-bad 双失败</div></div>
    <div class="stat"><div class="val">{he['glm_pass']/n*100:.0f}%</div><div class="lab">GLM 文化充分率</div></div>
    <div class="stat"><div class="val">{he['ds_pass']/n*100:.0f}%</div><div class="lab">DeepSeek 文化充分率</div></div>
    <div class="stat warn"><div class="val">{he['agree']/he['comp_n']*100:.0f}%</div><div class="lab">人工 vs COMET 一致率</div></div>
  </div>

  <h3 style="font-size:15px;margin:20px 0 8px">表 1 · gold 标签分布</h3>
  <table>
    <thead><tr><th>标签</th><th class="num">n</th><th class="num">占比</th></tr></thead>
    <tbody>{dist_rows}</tbody>
  </table>

  <h3 style="font-size:15px;margin:20px 0 8px">表 2 · 各 gold 组自动分数均值</h3>
  <table>
    <thead><tr><th>gold</th><th class="num">n</th><th class="num">GLM COMET</th><th class="num">DS COMET</th><th class="num">GLM chrF</th><th class="num">DS chrF</th></tr></thead>
    <tbody>{group_rows}</tbody>
  </table>

  <h3 style="font-size:15px;margin:20px 0 8px">表 3 · tie-good vs tie-bad（论点 B 核心）</h3>
  <table>
    <thead><tr><th>指标</th><th class="num">tie-good</th><th class="num">tie-bad</th><th class="num">差值</th></tr></thead>
    <tbody>
      {row("GLM COMET", tg[f"{GLM_COL}_COMET"], tb[f"{GLM_COL}_COMET"].mean())}
      {row("DS COMET", tg[f"{DS_COL}_COMET"], tb[f"{DS_COL}_COMET"].mean())}
      {row("GLM chrF", tg[f"{GLM_COL}_chrF"], tb[f"{GLM_COL}_chrF"].mean())}
      {row("DS chrF", tg[f"{DS_COL}_chrF"], tb[f"{DS_COL}_chrF"].mean())}
    </tbody>
  </table>

  <h3 style="font-size:15px;margin:20px 0 8px">图 1 · tie-good vs tie-bad 的 COMET 分布</h3>
  {hist_svg}
  <p class="muted" style="font-size:12px;margin-top:6px">tie-good 均值 {tg[f'{GLM_COL}_COMET']:.1f}/{tg[f'{DS_COL}_COMET']:.1f} · tie-bad 均值 {tb[f'{GLM_COL}_COMET'].mean():.1f}/{tb[f'{DS_COL}_COMET'].mean():.1f} · 但 tie-bad 仍接近全局 COMET (~{he['global_ds_comet']:.1f})</p>

  <h3 style="font-size:15px;margin:20px 0 8px">表 4 · 人工 vs 自动指标</h3>
  <table>
    <thead><tr><th>对比项</th><th class="num">结果</th></tr></thead>
    <tbody>
      <tr><td>人工 d/g vs COMET 胜者一致率（仅 d/g 句）</td><td class="num">{he['agree']}/{he['comp_n']} ({he['agree']/he['comp_n']*100:.1f}%)</td></tr>
      <tr><td>tie-bad 句中 COMET 仍判 DeepSeek 更优</td><td class="num">{he['tb_comet_ds_wins']*100:.1f}%</td></tr>
      <tr><td>tie-bad 句 DS COMET 均值 vs 全局</td><td class="num">{tb[f'{DS_COL}_COMET'].mean():.2f} vs {he['global_ds_comet']:.2f}</td></tr>
    </tbody>
  </table>

  <h3 style="font-size:15px;margin:20px 0 8px">表 5 · tie-bad 率按方言词（n≥{MIN_N}）</h3>
  <table>
    <thead><tr><th>方言词</th><th class="num">n</th><th class="num">tie-bad</th><th class="num">比率</th></tr></thead>
    <tbody>{kw_rows}</tbody>
  </table>

  <div class="callout" style="margin-top:14px"><b>人工评测结论：</b>仅 {c['tie-good']/n*100:.0f}% 句子两模型均成功传递文化义；
  {he['tie_bad_n']/n*100:.0f}% 双失败。自动 COMET 能区分 tie-good/tie-bad（差 ~12 分），但 tie-bad 绝对分仍 ~67，
  人工–COMET 一致率仅 {he['agree']/he['comp_n']*100:.0f}%，<b>标准化评分系统性低估 culture washing 的严重性</b>。</div>
"""


def patch_scores_txt(human_block: str) -> None:
    with open(SCORES_TXT, encoding="utf-8") as f:
        txt = f.read()
    marker = "\n[人工文化充分性评测"
    if marker in txt:
        txt = txt.split(marker)[0].rstrip()
    if txt.endswith("=" * 64):
        txt = txt[:-64].rstrip()
    txt = txt + human_block + "\n\n" + "=" * 64 + "\n"
    with open(SCORES_TXT, "w", encoding="utf-8") as f:
        f.write(txt)


def main() -> None:
    kw = pd.read_csv(KW_CSV, encoding="utf-8-sig")
    kw_recs = kw.to_dict("records")

    robust = [r for r in kw_recs if r["n"] >= MIN_N]
    comet_rows = sorted(
        robust, key=lambda r: (r[f"{GLM_COL}_COMET"] + r[f"{DS_COL}_COMET"]) / 2
    )
    chrf_rows = sorted(
        robust, key=lambda r: (r[f"{GLM_COL}_chrF"] + r[f"{DS_COL}_chrF"]) / 2
    )

    # 全量表（按 n 降序）
    table_rows = sorted(kw_recs, key=lambda r: -r["n"])
    trows = []
    for r in table_rows:
        avg_c = (r[f"{GLM_COL}_COMET"] + r[f"{DS_COL}_COMET"]) / 2
        cls = ' class="low"' if avg_c < 67 else ""
        trows.append(
            f"<tr{cls}><td>{esc(r['keyword'])}</td><td class='num'>{int(r['n'])}</td>"
            f"<td class='num'>{r[f'{GLM_COL}_chrF']:.1f}</td>"
            f"<td class='num'>{r[f'{DS_COL}_chrF']:.1f}</td>"
            f"<td class='num'>{r[f'{GLM_COL}_COMET']:.1f}</td>"
            f"<td class='num'>{r[f'{DS_COL}_COMET']:.1f}</td></tr>"
        )

    # 最差 15 句
    seg = pd.read_csv(SEG_CSV, encoding="utf-8-sig")
    results = pd.read_csv(RES_CSV, encoding="utf-8-sig")

    he = compute_human_eval(results, seg)
    hist_svg = comet_histogram_svg(he)
    human_txt = human_eval_txt(he)
    patch_scores_txt(human_txt)
    human_html = human_eval_html(he, hist_svg)
    print("已更新 translation_scores.txt（人工评测节）")

    seg["avg_comet"] = (seg[f"{GLM_COL}_COMET"] + seg[f"{DS_COL}_COMET"]) / 2
    worst = seg.sort_values("avg_comet").head(15)

    refusal_glm = int(seg[f"{GLM_COL}_refusal"].sum())
    refusal_ds = int(seg[f"{DS_COL}_refusal"].sum())
    n_total = len(seg)

    cases = []
    for rank, (_, r) in enumerate(worst.iterrows(), 1):
        kws = "".join(
            f"<span class='pill'>{esc(k)}</span>"
            for k in str(r["matched_keywords"]).split("|") if k
        )
        def cell(val, refusal):
            if refusal or str(val).strip() == "ERROR":
                return "<em class='muted'>（拒译 REFUSAL）</em>"
            return esc(val)
        cases.append(f"""
        <div class="case">
          <div class="case-head">
            <span class="rank">#{rank}</span>
            <span class="pills">{kws}</span>
            <span class="scores">COMET&nbsp; GLM {r[f'{GLM_COL}_COMET']:.1f} · DS {r[f'{DS_COL}_COMET']:.1f}</span>
          </div>
          <div class="case-grid">
            <div class="k">四川话</div><div class="v">{esc(r['sichuanese'])}</div>
            <div class="k">参考译文</div><div class="v ref">{esc(r['english'])}</div>
            <div class="k">GLM-4-Plus</div><div class="v">{cell(r[f'{GLM_COL}_pred'], r[f'{GLM_COL}_refusal'])}</div>
            <div class="k">{DS_COL}</div><div class="v">{cell(r[f'{DS_COL}_pred'], r[f'{DS_COL}_refusal'])}</div>
          </div>
        </div>""")

    legend = (
        f"<span class='lg'><i style='background:{GLM_C}'></i>GLM-4-Plus</span>"
        f"<span class='lg'><i style='background:{DS_C}'></i>{DS_COL}</span>"
    )

    doc = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>四川话→英文 翻译质量：方言词分层评估</title>
<style>
  :root {{ --fg:#1a1a1a; --fg2:#5b5b5b; --fg3:#8a8a8a; --line:#e3e3e3; --bg:#fff; --surf:#fafafa; --low:#fdecec; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:32px 24px 64px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:32px 0 12px; }}
  .sub {{ color:var(--fg2); margin:0 0 8px; }}
  .muted {{ color:var(--fg3); }}
  .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:20px 0; }}
  .stat {{ border:1px solid var(--line); border-radius:8px; padding:14px 16px; }}
  .stat .val {{ font-size:22px; font-weight:600; }}
  .stat .lab {{ color:var(--fg2); font-size:13px; margin-top:2px; }}
  .stat.good .val {{ color:{DS_C}; }}
  .stat.warn .val {{ color:#b45309; }}
  .callout {{ border:1px solid #f0d8a8; background:#fdf6e9; border-radius:8px; padding:14px 16px; margin:14px 0; }}
  .callout.n {{ border-color:var(--line); background:var(--surf); }}
  .callout b {{ color:#a15c00; }}
  .legend {{ display:flex; gap:18px; margin:4px 0 14px; font-size:13px; color:var(--fg2); }}
  .lg {{ display:flex; align-items:center; gap:6px; }} .lg i {{ width:12px;height:12px;border-radius:3px;display:inline-block; }}
  .bar-row {{ display:grid; grid-template-columns:150px 1fr 1fr; gap:8px; align-items:center; padding:5px 0; }}
  .bar-label {{ font-size:13px; }}
  .bar-track {{ position:relative; background:var(--surf); border-radius:4px; height:22px; }}
  .bar {{ height:100%; border-radius:4px; }}
  .bar-val {{ position:absolute; right:6px; top:0; line-height:22px; font-size:12px; color:var(--fg); }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }}
  th {{ color:var(--fg2); font-weight:600; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr.low {{ background:var(--low); }}
  .case {{ border:1px solid var(--line); border-radius:8px; margin:12px 0; overflow:hidden; }}
  .case-head {{ display:flex; align-items:center; gap:10px; padding:8px 14px; background:var(--surf); border-bottom:1px solid var(--line); flex-wrap:wrap; }}
  .rank {{ font-weight:600; }}
  .pills {{ display:flex; gap:6px; flex-wrap:wrap; }}
  .pill {{ font-size:12px; border:1px solid #bcd; color:#2563a6; border-radius:10px; padding:1px 8px; background:#eef5fc; }}
  .scores {{ margin-left:auto; font-size:12px; color:var(--fg2); font-variant-numeric:tabular-nums; }}
  .case-grid {{ display:grid; grid-template-columns:96px 1fr; gap:8px 12px; padding:14px; }}
  .case-grid .k {{ color:var(--fg3); font-size:13px; }}
  .case-grid .v.ref {{ font-weight:500; }}
  .foot {{ color:var(--fg3); font-size:12px; margin-top:28px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }}
  @page {{ margin:14mm 12mm; }}
  @media print {{
    body {{ font-size:12px; }}
    .wrap {{ max-width:none; padding:0; }}
    .case, .bar-row, tr, .callout, .stat {{ break-inside:avoid; }}
    h2 {{ break-after:avoid; }}
    .stats {{ gap:8px; }}
  }}
</style>
</head>
<body><div class="wrap">
  <h1>四川话 → 英文 翻译质量：方言词分层评估</h1>
  <p class="sub">按 <b>matched_keywords</b> 分层，对比 GLM-4-Plus 与 DeepSeek-V4-Flash 在每个四川话方言词上的翻译质量，用于定位「文化清洗」洼地。</p>

  <div class="stats">
    <div class="stat"><div class="val">{n_total}</div><div class="lab">方言种子句数</div></div>
    <div class="stat"><div class="val">70.76 / 70.87</div><div class="lab">系统 COMET (GLM / DS)</div></div>
    <div class="stat good"><div class="val">73.35</div><div class="lab">逐句 Oracle 上限 (COMET)</div></div>
    <div class="stat warn"><div class="val">{refusal_glm} / {refusal_ds}</div><div class="lab">拒译数 (GLM / DS)</div></div>
  </div>

  <div class="callout">
    <b>质量洼地 = 文化清洗信号。</b> 实义方言词 安逸 / 老子 / 老表 / 婆娘 / 龟儿子 / 幺 的 COMET 明显低于全局均值（~70.8）：
    模型常把「安逸」直译为 comfortable、把「老子(=我)」误解为 father/Laozi，丢失方言语用色彩。相比之下，语气词 哈/嘛/哦 的 COMET 在 71~73，质量损失主要集中在实义方言词上。
  </div>

  <h2>各方言词 COMET 对比（语义级，越高越好）</h2>
  <div class="legend">{legend}</div>
  {bar_chart(comet_rows, f"{GLM_COL}_COMET", f"{DS_COL}_COMET", 80)}
  <p class="muted" style="font-size:12px;margin-top:8px">来源：WSC-Train 200 句 · COMET = Unbabel/wmt22-comet-da ×100 · 仅含 n≥{MIN_N} 的关键词 · 按两模型 COMET 均值升序</p>

  <h2>各方言词 chrF 对比（字符级 F 值，越高越好）</h2>
  <div class="legend">{legend}</div>
  {bar_chart(chrf_rows, f"{GLM_COL}_chrF", f"{DS_COL}_chrF", 55)}
  <p class="muted" style="font-size:12px;margin-top:8px">来源：WSC-Train 200 句 · chrF = sacreBLEU chrF2 · 仅含 n≥{MIN_N} 的关键词 · 按两模型 chrF 均值升序</p>

  <h2>显著性检验（配对 bootstrap, B=2000）</h2>
  <p class="muted" style="font-size:12px;margin-bottom:8px">差值 = DeepSeek-V4-Flash − GLM-4-Plus（正 = DeepSeek 更好）。α=0.05。</p>
  <table>
    <thead><tr><th>指标</th><th class="num">观测差</th><th class="num">95% CI</th><th class="num">p (双侧)</th><th>显著</th></tr></thead>
    <tbody>
      <tr><td>COMET</td><td class="num">+0.11</td><td class="num">[−0.82, +1.12]</td><td class="num">0.836</td><td>否</td></tr>
      <tr><td>BLEU</td><td class="num">+0.20</td><td class="num">[−1.47, +1.84]</td><td class="num">0.861</td><td>否</td></tr>
      <tr class="low"><td>chrF</td><td class="num">+1.98</td><td class="num">[+0.62, +3.40]</td><td class="num">0.006</td><td>是（DeepSeek 更优）</td></tr>
    </tbody>
  </table>
  <div class="callout n" style="margin-top:12px"><b>结论：</b>COMET / BLEU 上两模型<b>无显著差异</b>，仅 chrF 上 DeepSeek-V4-Flash 有<b>小幅但显著</b>的优势。
  但每类仅 1 个模型（闭源 GLM-4-Plus / DeepSeek-V4-Flash）且均为中文强模型，此结果<b>不足以推广为「开源 vs 闭源整体差距极小」</b>，仅代表这两个具体模型。</div>

  {human_html}

  <h2>关键词全量分层表</h2>
  <p class="muted" style="font-size:12px;margin-bottom:8px">红色行 = 两模型 COMET 均值 &lt; 67（翻译质量洼地）。按命中句数 n 降序。</p>
  <table>
    <thead><tr><th>方言词</th><th class="num">n</th><th class="num">GLM chrF</th><th class="num">DS chrF</th><th class="num">GLM COMET</th><th class="num">DS COMET</th></tr></thead>
    <tbody>{''.join(trows)}</tbody>
  </table>

  <h2>逐句最差案例（COMET 最低 15 句）</h2>
  <div class="callout n"><b>典型误差类型：</b>(1) 实义方言词被直译/误解（安逸→comfortable、老子(=我)→脏话/人名）；(2) 游戏/麻将黑话丢失（幺儿、九通、童子）；(3) ASR 噪声叠加（源文本身连读乱码，放大翻译困难）。</div>
  {''.join(cases)}

  <div class="foot"><span>完整逐句分数：translation_segment_scores.csv</span><span>系统级 BLEU/chrF/COMET：translation_scores.txt</span></div>
</div></body></html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"已生成 -> {OUT_HTML}  ({len(doc):,} bytes)")


if __name__ == "__main__":
    main()
