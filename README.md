# AuditAgent: Cross-Regulatory AI-Augmented Financial Fraud Evidence Discovery

AI-augmented financial fraud evidence discovery system operating across US (SEC Complaints + EDGAR reports) and China (CSRC enforcement + CNINFO/SSE reports) markets.

## ⚠️ IMPORTANT: API Key Setup

**All API keys in this repository have been replaced with `YOUR_API_KEY` placeholders.**

Before running any pipeline, you MUST set your own API keys. The original keys are backed up in `api_keys_backup.txt` **(excluded from git — local only)**.

Affected files:
- `run_accounting_cn_eval_all.py` — 6 provider API configs
- `us_pipeline_v2/agents/base_agent.py` — 5 provider API configs
- `us_pipeline_v2/evaluation/llm_judge.py` — MiniMax Judge config
- `LLM-FinRisk/LLM-FinRisk/evaluation/llm_judge.py` — MiniMax Judge config
- `LLM-FinRisk/LLM-FinRisk/evaluation/metrics.py` — 3 provider API configs
- `LLM-FinRisk/LLM-FinRisk/solution/Hierarchical_RAG/SingleReportAnalyzer.py` — 1 provider config
- `LLM-FinRisk/LLM-FinRisk/solution/Hierarchical_RAG/CrossReportAnalyzer.py` — 1 provider config

**Recommended:** Search for `YOUR_API_KEY` across the codebase and replace with your own keys. Alternatively, modify the code to read from environment variables via `os.getenv()`.

## Repository Structure

```
auditagent/
  .gitignore                           # Excludes all data, caches, artifacts
  README.md
  requirements.txt
  .claude.md                           # Claude Code project guide
  run_accounting_cn_eval_all.py        # CN Pipeline entry point

  us_pipeline_v2/                      # US Pipeline V2
    run.py                             #   Entry point
    pipeline.py                        #   Orchestrator
    config.py                          #   PipelineConfig
    ablation.py                        #   Ablation presets
    prior_subjects.py                  #   Prior accounting subjects
    plot.py                            #   Visualization
    agents/                            #   LLM agents
    retrieval/                         #   BM25 + Dense + Section retrievers
    prompts/                           #   Prompt templates
    evaluation/                        #   4-dim LLM Judge + metrics
    verification/                      #   Retrieval verifier
    results/                           #   Output directory

  LLM-FinRisk/LLM-FinRisk/             # CN Pipeline core (shared modules)
    solution/Hierarchical_RAG/         #   Single/Cross report analyzers
    evaluation/                        #   LLM judge + metrics

  finfraud_processing/data/processed/
    cases/account_classification.json  #   77 accounting categories (metadata)
```

## Pipelines

### Pipeline 1: US Pipeline V2

Prior-guided multi-path retrieval (BM25 + Dense + Section) with multi-expert LLM analysis and 4-dimension LLM Judge evaluation.

```bash
# Basic run
python us_pipeline_v2/run.py --limit 50

# With specific ablation
python us_pipeline_v2/run.py --limit 50 --ablation baseline --output us_pipeline_v2/results/baseline.xlsx

# Re-evaluation only (on existing results)
python us_pipeline_v2/run.py --reeval-only
```

**Ablation modes:** `baseline`, `no_prior`, `no_hybrid`, `no_multi_expert`, `no_cross_doc`, `cn_to_us`, `cn_us_to_us`

**Required external data (NOT included in this repo):**
- Ground truth JSONs: `case_LR-XXXXX.json` files (contact authors for access)
- EDGAR report text: `finfraud_processing/data/edgar_reports_text/`

### Pipeline 2: CN Pipeline

Hierarchical RAG (SingleAnalyzer + CrossAnalyzer + Aggregation) with LLM Judge evaluation on Chinese financial fraud data.

```bash
# Basic run
python run_accounting_cn_eval_all.py --limit 50

# With specific model and prior
python run_accounting_cn_eval_all.py --limit 50 --model deepseek-v4 --prior cn_15
```

**Required external data (NOT included in this repo):**
- CN dataset: `FinFraud-dataset-txt-cross.xlsx`
- Context txt files from pre-parsed financial reports
- Raw financial report txt files

## Setup

```bash
pip install -r requirements.txt
```

## Hardcoded Paths

This codebase contains hardcoded absolute Windows paths (e.g., `d:/mainfiles/AuditAgent/...`). You will need to update the following before running:

**`us_pipeline_v2/run.py`** (lines ~51-53):
- `CASES_BASE` — path to ground truth JSON cases
- `EDGAR_TEXT_BASE` — path to EDGAR report text files

**`us_pipeline_v2/config.py`** (lines ~13-17):
- `edgar_text_base`, `cases_base`, `classification_file`, `output_dir`, `report_dir`

**`run_accounting_cn_eval_all.py`** (lines ~51-54):
- `DATASET_XLSX`, `TXT_PREFIX`, `REPORT_TXT_BASE`, `AUDIT_RESULT_PATH`

**`us_pipeline_v2/prior_subjects.py`** (line ~238):
- Path to `account_classification.json`

## Reference

This repository accompanies the AuditAgent research paper. For questions about data access or the pipelines, please contact the authors.

## License

[MIT License](https://opensource.org/licenses/MIT)
