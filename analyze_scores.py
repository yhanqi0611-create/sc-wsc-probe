"""
翻译质量评估：对每个模型的译文计算 BLEU / chrF / COMET，并汇总 BestScore。
结果写入 translation_scores.txt。

输入: translation_results_200.csv
    - sichuanese    : 源文（四川话）
    - english       : 参考译文 (gold reference)
    - pred_glm4plus : GLM-4-Plus 译文
    - pred_deepseek : DeepSeek-V4-Flash 译文

指标:
    - BLEU  : sacreBLEU 语料级 BLEU
    - chrF  : sacreBLEU chrF2（字符级 F 值，对形态/拼写更鲁棒）
    - COMET : Unbabel/wmt22-comet-da（基于神经网络的语义级指标，需 源+译+参考）
    - BestScore:
        (a) 每个指标上得分最高的模型（谁赢）
        (b) 逐句取各模型最高分再平均 -> "oracle 上限"（理论上的最佳集成）

另外汇报每个模型的「拒译率」（模型输出元评论而非译文，如 "I'm sorry, ...").

依赖:
    pip install pandas sacrebleu unbabel-comet
运行:
    python analyze_scores.py
"""

import re
import pandas as pd
import sacrebleu

INPUT_CSV = "translation_results_200.csv"
OUTPUT_TXT = "translation_scores.txt"
SEG_CSV = "translation_segment_scores.csv"   # 逐句分数
KW_CSV = "translation_keyword_scores.csv"    # 按关键词分层分数
KW_COL = "matched_keywords"
MIN_KW_FOR_TXT = 5  # txt 报告里只展示样本数 >= 该值的关键词（CSV 含全部）
SRC_COL = "sichuanese"
REF_COL = "english"
MODELS = {
    "GLM-4-Plus": "pred_glm4plus",
    "DeepSeek-V4-Flash": "pred_deepseek",
}
COMET_MODEL = "Unbabel/wmt22-comet-da"

# 拒译 / 非译文的启发式特征（小写匹配）
REFUSAL_PATTERNS = [
    r"\bi'?m sorry\b", r"\bi am sorry\b", r"\bi cannot\b", r"\bi can'?t\b",
    r"\bas an ai\b", r"\bwithout proper context\b", r"\bnonsensical\b",
    r"\bunable to (translate|provide)\b", r"\bcannot provide\b",
    r"\bplease (share|provide)\b", r"\bdoes not (appear|seem) to\b",
    r"\bno meaningful\b", r"\bnot .*coherent\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(text: str) -> bool:
    t = str(text).strip()
    return bool(t) and t != "ERROR" and bool(REFUSAL_RE.search(t))


def main() -> None:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    for col in [SRC_COL, REF_COL, *MODELS.values()]:
        df[col] = df[col].fillna("").astype(str)

    srcs = df[SRC_COL].tolist()
    refs = df[REF_COL].tolist()
    n = len(df)

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("四川话 -> 英文  翻译质量评估报告")
    lines.append(f"样本数: {n}    参考译文: {REF_COL}")
    lines.append("=" * 64)

    # ---------- 1. 拒译率 ----------
    lines.append("\n[拒译 / 非译文率 REFUSAL RATE]")
    refusal_flags: dict[str, list[bool]] = {}
    for name, col in MODELS.items():
        flags = df[col].map(is_refusal).tolist()
        refusal_flags[name] = flags
        cnt = sum(flags)
        lines.append(f"  {name:<14} {cnt:>3}/{n}  ({cnt / n * 100:.1f}%)")
    lines.append("  注: 拒译译文为流畅英文，会虚高 BLEU/chrF；下方分数为「原样计分」，")
    lines.append("      另给出「拒译计 0 分」版本，论文可二选一并注明。")

    # ---------- 2. BLEU / chrF ----------
    # 句级分数（用于 oracle 上限）
    seg_scores: dict[str, dict[str, list[float]]] = {m: {} for m in MODELS}
    sys_scores: dict[str, dict[str, float]] = {m: {} for m in MODELS}

    for name, col in MODELS.items():
        hyps = df[col].tolist()
        sys_scores[name]["BLEU"] = sacrebleu.corpus_bleu(hyps, [refs]).score
        sys_scores[name]["chrF"] = sacrebleu.corpus_chrf(hyps, [refs]).score
        seg_scores[name]["BLEU"] = [
            sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)
        ]
        seg_scores[name]["chrF"] = [
            sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps, refs)
        ]

    # ---------- 3. COMET ----------
    comet_ok = True
    try:
        from comet import download_model, load_from_checkpoint

        print(f"加载 COMET 模型 {COMET_MODEL} ...（首次会自动下载）")
        comet_model = load_from_checkpoint(download_model(COMET_MODEL))
        for name, col in MODELS.items():
            data = [
                {"src": s, "mt": m, "ref": r}
                for s, m, r in zip(srcs, df[col].tolist(), refs)
            ]
            print(f"COMET 评分 {name} ...")
            out = comet_model.predict(data, batch_size=16, gpus=0, progress_bar=True)
            seg_scores[name]["COMET"] = [x * 100 for x in out["scores"]]
            sys_scores[name]["COMET"] = out["system_score"] * 100
    except Exception as exc:  # noqa: BLE001
        comet_ok = False
        print(f"[WARN] COMET 不可用，已跳过: {exc}")
        lines.append(f"\n[WARN] COMET 计算失败，已跳过: {exc}")

    metrics = ["BLEU", "chrF"] + (["COMET"] if comet_ok else [])

    # ---------- 4. 系统级分数表 ----------
    lines.append("\n[系统级分数 SYSTEM-LEVEL SCORES]  (越高越好)")
    header = f"  {'Model':<14}" + "".join(f"{m:>10}" for m in metrics)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name in MODELS:
        row = f"  {name:<14}" + "".join(
            f"{sys_scores[name][m]:>10.2f}" for m in metrics
        )
        lines.append(row)

    # 拒译计 0 分版本
    lines.append("\n[系统级分数 - 拒译计0分版本]")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, col in MODELS.items():
        hyps0 = [
            "" if is_refusal(h) else h for h in df[col].tolist()
        ]
        b0 = sacrebleu.corpus_bleu(hyps0, [refs]).score
        c0 = sacrebleu.corpus_chrf(hyps0, [refs]).score
        vals = {"BLEU": b0, "chrF": c0}
        if comet_ok:
            # COMET 逐句：拒译句置 0
            vals["COMET"] = sum(
                0.0 if is_refusal(h) else s
                for h, s in zip(df[col].tolist(), seg_scores[name]["COMET"])
            ) / n
        row = f"  {name:<14}" + "".join(f"{vals[m]:>10.2f}" for m in metrics)
        lines.append(row)

    # ---------- 5. BestScore ----------
    lines.append("\n[BestScore]")
    lines.append("  (a) 每个指标的获胜模型:")
    for m in metrics:
        best_name = max(MODELS, key=lambda nm: sys_scores[nm][m])
        lines.append(
            f"      {m:<6} -> {best_name}  "
            f"({sys_scores[best_name][m]:.2f})"
        )

    lines.append("\n  (b) 逐句 oracle 上限（每句取最高分模型后平均）:")
    names = list(MODELS)
    for m in metrics:
        per_seg_max = [
            max(seg_scores[nm][m][i] for nm in names) for i in range(n)
        ]
        oracle = sum(per_seg_max) / n
        # 各模型在该指标上「逐句最优」的占比
        win_count = {nm: 0 for nm in names}
        for i in range(n):
            best = max(names, key=lambda nm: seg_scores[nm][m][i])
            win_count[best] += 1
        win_str = ", ".join(
            f"{nm}: {win_count[nm]}" for nm in names
        )
        lines.append(f"      {m:<6} oracle = {oracle:6.2f}   逐句最优计数[{win_str}]")

    # ---------- 6. 逐句分数 CSV ----------
    seg_df = pd.DataFrame({
        SRC_COL: srcs,
        REF_COL: refs,
        KW_COL: df[KW_COL] if KW_COL in df.columns else "",
    })
    if "n_content" in df.columns:
        seg_df["n_content"] = df["n_content"]
    for name, col in MODELS.items():
        seg_df[f"{name}_pred"] = df[col]
        seg_df[f"{name}_refusal"] = refusal_flags[name]
        for m in metrics:
            seg_df[f"{name}_{m}"] = [round(s, 4) for s in seg_scores[name][m]]
    seg_df.to_csv(SEG_CSV, index=False, encoding="utf-8-sig")

    # ---------- 7. 按关键词分层 (chrF / COMET) ----------
    strat_metrics = [m for m in ("chrF", "COMET") if m in metrics]
    kw_lists = (
        df[KW_COL].fillna("").astype(str)
        .map(lambda s: [x for x in s.split("|") if x])
        if KW_COL in df.columns else pd.Series([[]] * n)
    )
    all_kws = sorted({kw for labs in kw_lists for kw in labs})

    kw_rows = []
    for kw in all_kws:
        idx = [i for i, labs in enumerate(kw_lists) if kw in labs]
        if not idx:
            continue
        row = {"keyword": kw, "n": len(idx)}
        for name in MODELS:
            for m in strat_metrics:
                row[f"{name}_{m}"] = round(
                    sum(seg_scores[name][m][i] for i in idx) / len(idx), 2
                )
        kw_rows.append(row)

    kw_df = pd.DataFrame(kw_rows).sort_values("n", ascending=False).reset_index(drop=True)
    kw_df.to_csv(KW_CSV, index=False, encoding="utf-8-sig")

    lines.append(f"\n[按关键词分层分数 (样本数 n>={MIN_KW_FOR_TXT})]  指标: {' / '.join(strat_metrics)}")
    # 表头用短码：GLM=GLM-4-Plus, DS=DeepSeek-V4-Flash
    short = {"GLM-4-Plus": "GLM", "DeepSeek-V4-Flash": "DS"}
    cols = [(f"{name}_{m}", f"{short.get(name, name)}_{m}")
            for name in MODELS for m in strat_metrics]
    head = f"  {'关键词':<12}{'n':>4}" + "".join(f"{label:>11}" for _, label in cols)
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for _, r in kw_df[kw_df["n"] >= MIN_KW_FOR_TXT].iterrows():
        line = f"  {r['keyword']:<12}{int(r['n']):>4}" + "".join(
            f"{r[key]:>11.2f}" for key, _ in cols
        )
        lines.append(line)
    lines.append(f"  (完整关键词表见 {KW_CSV}；逐句分数见 {SEG_CSV})")

    lines.append("\n" + "=" * 64)

    report = "\n".join(lines)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\n已保存 -> {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
