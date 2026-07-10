# SC-WSC-Probe: Sichuanese Cultural Adequacy Evaluation

Data and scripts for the paper *The Illusion of Parity: Human Cultural Adequacy Reveals Shared Whitewashing in Dialectal Translation* (INLG 2026 submission).

We release a **200-sentence keyword-stratified probe** from [WenetSpeech-Chuan](https://huggingface.co/datasets/ASLP-lab/WSC-Train), model translations (GLM-4-Plus, DeepSeek-V4-Flash), human **cultural adequacy** gold labels, and automatic metric scores.

## Repository contents

| Path | Description |
|------|-------------|
| `filtered_seeds_200.csv` | 200-sentence probe (`sichuanese`, `matched_keywords`, `n_content`) |
| `translation_results_200.csv` | Source, human references, model outputs, `preference` gold |
| `translation_segment_scores.csv` | Per-sentence BLEU / chrF / COMET |
| `translation_keyword_scores.csv` | Keyword-stratified scores |
| `reverse_misalignment_cases.csv` | Five COMET–human divergence cases |
| `translation_scores.txt` | Full system-level report |
| `docs/annotation_guidelines.md` | Human cultural adequacy protocol (excerpt) |
| `generate_figures.py` | Generate COMET parity / label-distribution figures |

## Human labels (`preference`)

| Label | Meaning |
|-------|---------|
| `tie-good` | Both models culturally adequate |
| `g` | Only GLM-4-Plus adequate |
| `d` | Only DeepSeek-V4-Flash adequate |
| `tie-bad` | Neither adequate (consensus whitewashing) |

## Models

- **GLM-4-Plus** — Zhipu API (`glm-4-plus`)
- **DeepSeek-V4-Flash** — DeepSeek Chat API (`deepseek-chat`; provider dashboard reports V4-Flash)

Zero-shot prompt: system *"You are an expert translator."*; user *"请将以下四川话翻译为地道的英文：{text}"*.

## Quick start (reproduce scores from released data)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# COMET requires Python 3.9 (separate venv recommended)
python3.9 -m venv .venv-comet
source .venv-comet/bin/activate
pip install -r requirements-comet.txt

# Back in main venv, run scoring (reads translation_results_200.csv)
python analyze_scores.py
python generate_report.py   # optional HTML report
```

Outputs: `translation_scores.txt`, `translation_segment_scores.csv`, `translation_keyword_scores.csv`.

## Full pipeline (from scratch)

```bash
pip install -r requirements.txt

# 1. Download WSC-Train metadata (~2.6M sentences, confidence >= 0.9)
python download_sichuanese_data.py
# -> raw_sichuanese_data.csv (not in repo; ~163MB)

# 2. Build 200-sentence keyword probe (seed=42)
python filter_seeds.py
# -> filtered_seeds_200.csv

# 3. Translate (requires API keys; skip if using released translation_results_200.csv)
cp .env.example .env   # fill keys locally; never commit .env
export $(grep -v '^#' .env | xargs)
python translate_experiment.py
# -> translation_results_200.csv

# 4. Score (COMET venv)
source .venv-comet/bin/activate
python analyze_scores.py
```

## API keys

Set environment variables (see `.env.example`):

- `ZHIPUAI_API_KEY` — GLM-4-Plus
- `DEEPSEEK_API_KEY` — DeepSeek Chat

**Never commit real keys.** If a key was ever committed, rotate it in the provider dashboard.

## Citation

```bibtex
@inproceedings{sc-wsc-probe2026,
  title     = {The Illusion of Parity: Human Cultural Adequacy Reveals Shared Whitewashing in Dialectal Translation},
  author    = {Anonymous},
  booktitle = {Proceedings of INLG},
  year      = {2026},
  note      = {Under review}
}
```

Update author fields after acceptance.

## License

Code: MIT (add `LICENSE` file if desired).  
Data annotations and references: CC BY 4.0 recommended for academic reuse.

## Contact

Open an issue or contact the authors (update after de-anonymization).
