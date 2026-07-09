"""
下载 WenetSpeech-Chuan（四川话语料库）的文本转写，并整理成统一的
Pandas DataFrame，包含两列：
    - sichuanese : 四川话原文（来自语料库的 `text` 字段）
    - english    : 标准英文译文（暂时留空，供后续翻译填充）
最后保存为本地的 raw_sichuanese_data.csv。

数据来源
--------
HuggingFace 数据集: ASLP-lab/WSC-Train
    - 文件 `wsc_metadata.jsonl` 中每行一条 JSON，含四川话转写 `text` 字段。
    - 该 metadata 文件体积小（仅文本，不含音频），因此无需下载多 GB 的音频包。
论文: WenetSpeech-Chuan (arXiv:2509.18004)

依赖
----
    pip install pandas huggingface_hub requests
"""

import json
import pandas as pd

REPO_ID = "ASLP-lab/WSC-Train"
METADATA_FILENAME = "wsc_metadata.jsonl"
OUTPUT_CSV = "raw_sichuanese_data.csv"

# 只保留转写置信度不低于该阈值的句子
MIN_CONFIDENCE = 0.9


def download_metadata() -> str:
    """下载 wsc_metadata.jsonl，返回本地文件路径。

    优先使用 huggingface_hub（带缓存、断点续传）；如不可用则回退到 requests 直接下载。
    """
    try:
        from huggingface_hub import hf_hub_download

        print(f"[1/3] 正在通过 huggingface_hub 下载 {REPO_ID}/{METADATA_FILENAME} ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=METADATA_FILENAME,
            repo_type="dataset",
        )
        print(f"      下载完成: {local_path}")
        return local_path
    except Exception as exc:  # noqa: BLE001  回退到原始 HTTP 下载
        print(f"      huggingface_hub 不可用或失败 ({exc})，改用 requests 直接下载 ...")
        import requests

        url = (
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{METADATA_FILENAME}"
        )
        local_path = METADATA_FILENAME
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"      下载完成: {local_path}")
        return local_path


def load_sichuanese_texts(metadata_path: str) -> list[str]:
    """逐行解析 JSONL，提取四川话转写文本（仅保留 confidence >= MIN_CONFIDENCE）。"""
    print(f"[2/3] 正在解析转写文本（筛选 confidence >= {MIN_CONFIDENCE}）...")
    texts: list[str] = []
    total = 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            confidence = record.get("confidence")
            if confidence is None or confidence < MIN_CONFIDENCE:
                continue
            text = (record.get("text") or "").strip()
            if text:
                texts.append(text)
    print(
        f"      共 {total} 条；保留 {len(texts)} 条 "
        f"(confidence >= {MIN_CONFIDENCE})。"
    )
    return texts


def main() -> None:
    metadata_path = download_metadata()
    sichuanese_texts = load_sichuanese_texts(metadata_path)

    df = pd.DataFrame(
        {
            "sichuanese": sichuanese_texts,
            "english": [""] * len(sichuanese_texts),  # 标准英文译文，暂时留空
        }
    )

    print(f"[3/3] 正在保存到 {OUTPUT_CSV} ...")
    # utf-8-sig 便于 Excel 正确显示中文
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"      已保存 {len(df)} 行 -> {OUTPUT_CSV}")
    print("\n预览前 5 行：")
    print(df.head())


if __name__ == "__main__":
    main()
