"""
四川话 -> 英文 翻译实验：分别调用 GLM-4-Plus 与 DeepSeek-V4-Flash，对比朴素 prompt 下的译文。

运行:
    pip install openai pandas
    export ZHIPUAI_API_KEY=... DEEPSEEK_API_KEY=...
    python translate_experiment.py

说明:
    - 读取 filtered_seeds_200.csv 的 sichuanese 列逐句翻译。
    - 结果写入新列 pred_glm4plus / pred_deepseek，保存为 translation_results_200.csv。
    - API Key 从环境变量读取（见 .env.example）；勿写入代码或提交到 Git。
    - 出错（网络抖动/限流等）该单元格填 "ERROR"。
    - 每 SAVE_EVERY 条实时落盘；若中途崩溃，重跑会自动跳过已完成的单元格。
"""

import os
import pandas as pd
from openai import OpenAI

INPUT_CSV = "filtered_seeds_200.csv"
OUTPUT_CSV = "translation_results_200.csv"
TEXT_COLUMN = "sichuanese"
SAVE_EVERY = 10  # 每翻译多少个单元格落盘一次

SYSTEM_PROMPT = "You are an expert translator."
USER_TEMPLATE = "请将以下四川话翻译为地道的英文：{text}"


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. Export it or copy .env.example to .env and fill in your keys."
        )
    return value


def build_models() -> dict[str, tuple[OpenAI, str]]:
    glm_client = OpenAI(
        api_key=_require_env("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )
    deepseek_client = OpenAI(
        api_key=_require_env("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    return {
        "pred_glm4plus": (glm_client, "glm-4-plus"),
        "pred_deepseek": (deepseek_client, "deepseek-chat"),
    }


def translate(client: OpenAI, model: str, text: str) -> str:
    """调用单个模型翻译一句，失败返回 "ERROR"。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=text)},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001  网络抖动/限流等
        print(f"    [ERROR] {model}: {exc}")
        return "ERROR"


def is_done(value) -> bool:
    """该单元格是否已成功翻译（非空且非 ERROR），用于断点续跑。"""
    return isinstance(value, str) and value.strip() not in ("", "ERROR")


def main() -> None:
    models = build_models()
    # 优先读取已有结果（断点续跑）；否则读原始语料
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
        print(f"检测到已有结果 {OUTPUT_CSV}，将续跑未完成的单元格。")
    else:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    for col in models:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    total = len(df)
    pending = 0

    for i, row in df.iterrows():
        text = row[TEXT_COLUMN]
        if not text:
            continue
        for col, (client, model) in models.items():
            if is_done(df.at[i, col]):
                continue  # 已完成，跳过
            print(f"[{i + 1}/{total}] {model} <- {text[:20]}...")
            df.at[i, col] = translate(client, model, text)
            pending += 1
            if pending % SAVE_EVERY == 0:
                df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
                print(f"    已落盘 -> {OUTPUT_CSV}")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n完成，共保存 {total} 行 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
