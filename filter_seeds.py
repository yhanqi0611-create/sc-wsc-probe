"""
筛选带有强烈语用特征 / 社交称谓 / 语气词的四川话句子，用于研究方言生成中的
“文化清洗”（cultural washing）现象。

本版本改进
----------
A. 文本清洗（clean_text）:
     - 去除句中混入的 ASCII 英文残片（如 "Transation"）。
     - 去除语塞/填充词「呃 / 诶 / 嗯 / 唉」。
B. 关键词分级 + 降低语气助词权重:
     - 语气助词（哈/哦/嘛/噻/得嘛）权重调低，避免抽样被高频虚词主导。
     - 实义词（称谓、方言形容词、骂詈语等）权重为 1。
     - 用「加权随机抽样」选 200 条，使样本更偏向含实义方言词的句子。
C. 高误报单字词上下文约束 + 零命中词写法变体（沿用上一版）。

注：中文无空格分词，\b 词边界对汉字不生效，故普通词用转义子串匹配，
    高误报词用定宽 lookaround 约束。

依赖: pip install pandas
"""

import re
import pandas as pd

INPUT_CSV = "raw_sichuanese_data.csv"
OUTPUT_CSV = "filtered_seeds_200.csv"
TEXT_COLUMN = "sichuanese"
SAMPLE_SIZE = 200
RANDOM_SEED = 42
MAX_LEN = 80  # 句长上限（字符数），过滤超长解说句，便于逐句人工检查

# 抽样权重：实义词高，语气助词低
CONTENT_WEIGHT = 1.0
PARTICLE_WEIGHT = 0.15

# 关键词级权重覆盖（优先级最高）：
#   - 啥子 占比过高 -> 大幅降权，减少「只命中啥子」的句子
#   - 瓜娃子/龟儿子/贼娃子 较稀有但价值高 -> 大幅升权
KEYWORD_WEIGHTS = {
    "啥子": 0.2,
    "瓜娃子": 6.0,
    "龟儿子": 6.0,
    "贼娃子": 6.0,
}

# ---- 文本清洗规则 ----
ASCII_RE = re.compile(r"[A-Za-z]+")      # 句中混入的英文残片
FILLER_RE = re.compile(r"[呃诶嗯唉]")     # 语塞 / 填充词

# 普通关键词（实义词为主）：转义子串匹配
PLAIN_KEYWORDS = [
    "硬是", "极其", "确实", "脑壳昏", "扯把子", "老子", "老太婆", "老太爷",
    "兄弟伙", "婆娘", "老汉", "咋个", "啥子", "莫得", "安逸", "巴适",
    "神戳戳", "坝坝", "老表", "兜兜", "片片", "龟儿子", "瓜娃子", "耙耳朵",
    "孃孃", "先人板板", "嘎嘎", "雄起", "恼火", "鼓到", "背时", "砍脑壳的",
    "贼娃子", "霸道",
]

# 高误报单字词：定宽 lookaround 约束
CONSTRAINED_PATTERNS = {
    "哈(语气词)": r"(?<!哈)哈(?!哈|尔|密|巴|士|喇|雷|利|罗|佛|达|欠|瓦|萨|姆|根|喽|腰|蟆)",
    "哦(语气词)": r"(?<!哦)哦(?!哦)",
    "嘛(语气词)": r"(?<!喇)嘛",
    "幺(称谓/方言)": r"(?<=老)幺|幺(?=妹|儿|女|爸|姑|舅|叔|娃|妈|店|不到|不倒|鸡)",
    "歪(方言:凶/厉害)": r"歪(?=得|货|起|惨)|(?<=好|很|真|太|恁)歪",
    "苟(方言)": r"(?<!不)苟(?!且|同|活|延|安|合|全|言)",
    "蛮(方言:很)": r"(?<!野|刁|撒)蛮(?!横|力|子|夷|缠|干|荒|劲)",
}

# 零命中关键词的写法变体
VARIANT_PATTERNS = {
    "吃九斗碗(含九大碗/九斗碗)": r"九斗碗|九大碗",
    "茂起(含冒起)": r"茂起|(?<!感)冒起",
}

# 语气助词标签集合（抽样时降权）
PARTICLE_LABELS = {
    "哈(语气词)", "哦(语气词)", "嘛(语气词)", "噻", "得嘛",
}

# 单独加入的语气助词（不在上面三组里的）
EXTRA_PARTICLES = {"噻": r"噻", "得嘛": r"得嘛"}


def clean_text(text: str) -> str:
    text = ASCII_RE.sub("", text)
    text = FILLER_RE.sub("", text)
    return text.strip()


def build_patterns() -> dict[str, re.Pattern]:
    patterns: dict[str, re.Pattern] = {}
    seen = set()
    for kw in PLAIN_KEYWORDS:
        if kw in seen:
            continue
        seen.add(kw)
        patterns[kw] = re.compile(re.escape(kw))
    for label, rx in EXTRA_PARTICLES.items():
        patterns[label] = re.compile(rx)
    for label, rx in CONSTRAINED_PATTERNS.items():
        patterns[label] = re.compile(rx)
    for label, rx in VARIANT_PATTERNS.items():
        patterns[label] = re.compile(rx)
    return patterns


def sample_weight(labels: list[str]) -> float:
    """句子抽样权重：关键词级覆盖 > 语气助词降权 > 实义词默认权重。"""
    total = 0.0
    for lab in labels:
        if lab in KEYWORD_WEIGHTS:
            total += KEYWORD_WEIGHTS[lab]
        elif lab in PARTICLE_LABELS:
            total += PARTICLE_WEIGHT
        else:
            total += CONTENT_WEIGHT
    return total


def main() -> None:
    patterns = build_patterns()

    print(f"[1/5] 读取 {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    print(f"      共 {len(df):,} 行。")

    print(f"[2/5] 清洗文本（去除 ASCII 英文残片 + 语塞词 呃/诶/嗯/唉）...")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].map(clean_text)
    df = df[df[TEXT_COLUMN].str.len() > 0]

    print(f"[3/5] 关键词匹配（{len(patterns)} 个模式）...")

    def find_hits(text: str) -> list[str]:
        return [label for label, pat in patterns.items() if pat.search(text)]

    df["matched_keywords"] = df[TEXT_COLUMN].apply(find_hits)
    mask = df["matched_keywords"].map(len) > 0
    filtered = df[mask].copy()

    # 实义词命中数（不含语气助词）
    filtered["n_content"] = filtered["matched_keywords"].map(
        lambda labs: sum(1 for l in labs if l not in PARTICLE_LABELS)
    )
    print(f"      命中句子总数: {len(filtered):,}（占比 {len(filtered)/len(df)*100:.2f}%）")
    print(f"      其中含 >=1 个实义词的句子: {(filtered['n_content']>0).sum():,}")

    print(f"\n[4/5] 各关键词命中统计（按命中句子数降序；P=语气助词）:")
    counts = {
        label: int(df[TEXT_COLUMN].map(lambda t, p=pat: bool(p.search(t))).sum())
        for label, pat in patterns.items()
    }
    counts_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'关键词/模式':<22} {'类别':<4} {'命中句数':>12}")
    print(f"  {'-'*22} {'-'*4} {'-'*12}")
    for label, c in counts_sorted:
        cat = "P" if label in PARTICLE_LABELS else "实义"
        print(f"  {label:<22} {cat:<4} {c:>12,}")

    print(f"\n[5/5] 硬筛选（含 >=1 实义词 且 句长 <= {MAX_LEN} 字）后加权随机抽样并保存 "
          f"(实义词权重={CONTENT_WEIGHT}, 语气助词权重={PARTICLE_WEIGHT}) ...")
    pool = filtered[
        (filtered["n_content"] > 0)
        & (filtered[TEXT_COLUMN].str.len() <= MAX_LEN)
    ].copy()
    print(f"      实义词 & 句长<={MAX_LEN} 句子池: {len(pool):,} 条")
    weights = pool["matched_keywords"].map(sample_weight)
    n = min(SAMPLE_SIZE, len(pool))
    sample = pool.sample(
        n=n, weights=weights, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    sample_out = sample.copy()
    sample_out["matched_keywords"] = sample_out["matched_keywords"].map(
        lambda lst: "|".join(lst)
    )
    sample_out = sample_out[[TEXT_COLUMN, "english", "matched_keywords", "n_content"]]
    sample_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"      已加权抽样 {n} 条 -> {OUTPUT_CSV}")
    print(f"      样本中含实义词的句子: {(sample['n_content']>0).sum()}/{n}")
    print(f"      样本实义词命中数分布:\n{sample['n_content'].value_counts().sort_index().to_string()}")

    # 关键词调权效果检查
    content_labels = sample["matched_keywords"].map(
        lambda labs: [l for l in labs if l not in PARTICLE_LABELS]
    )
    shazi_only = content_labels.map(lambda labs: labs == ["啥子"]).sum()
    print(f"\n      调权效果检查:")
    print(f"        「只命中啥子」(实义词仅啥子) 的句子: {shazi_only}/{n}")
    for kw in ("瓜娃子", "龟儿子", "贼娃子", "啥子"):
        cnt = sample["matched_keywords"].map(lambda labs: kw in labs).sum()
        print(f"        含「{kw}」的句子: {cnt}/{n}")


if __name__ == "__main__":
    main()
