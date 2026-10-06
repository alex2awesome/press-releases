# press-release-exploration

Data collection, modeling and evaluation code for a study of how journalists cover corporate press releases, and how that compares with the coverage plans a large language model would suggest. We build a corpus of press releases from S&P 500 company newsrooms and the news articles that link to them (via Common Crawl hyperlinks and the Wayback Machine). We treat critical coverage as contrastive summarization: sentence-level NLI between a press release and an article measures how much the article challenges the release. We then prompt LLMs (GPT-3/4, Mixtral, Command-R, Llama-2) for the angles and sources a reporter should pursue and compare them with what journalists actually did. The questions: what planning strategies do journalists follow when covering press releases, and how closely do LLM suggestions match them?

## Related paper

*Do LLMs Plan Like Human Writers? Comparing Journalist Coverage of Press Releases with LLMs* (EMNLP 2024). LaTeX in `latex/ACL2024/`; `latex/bloomberg-internship-wrapup/` is an earlier internal write-up.

## Layout

- `src/` -- corpus construction: Common Crawl and Wayback fetching, press release text, HTML/PDF parsing, merging into sqlite, coreference resolution.
- `src/pipelines/` -- coreference, NLI, stance and source-scoring pipelines with SLURM launchers.
- `src/eval/` -- NLI aggregation into document scores, SummaC, GPT grading of LLM plans.
- `src/gcf_*` -- Cloud Run / Cloud Functions services for HTML/PDF and Wayback fetching.
- `models/` -- news-discourse classifier, hyperlink predictor, factual-consistency scorers (BARTScore, QAFactEval, ANLI), summarization.
- `experiments/` -- prompts and `run_opensource_model.py` for the LLM planning experiments.
- `javascript/` -- d3 demo of press-release-to-article NLI alignment.
- `notebooks/` -- dated notebooks (2023-2025): corpus construction, NLI analysis, LLM querying, figures.
- `press-release-urls-and-article-manual-list.txt` -- hand-picked press release / article pairs with notes.

## How to run

- `pip install -r requirements.txt` (Spark pipelines: `requirements-spark.txt`).
- Corpus: `src/common_crawl_gce_runner.py` -> `get_timestamps_wayback.py` -> `parse_html_and_other_retrievals_from_wayback.py` -> `get_press_release_text.py` -> `combine_data.py`.
- Scoring: `src/resolve_coref.py`, then `src/pipelines/nli_pipeline_hf_datasets.py`.
- LLM plans: `cd experiments && python run_opensource_model.py --model_id mixtral --prompt_type separate_zeroshot`.

## Data

Not tracked in git (`data/`, `*.csv` are gitignored). Locally, `data/open-sourced-articles/` and `data/s_p_500_backlinks/` hold press release text, article text, hyperlink maps, and coref-resolved and NLI-scored versions (tens of GB). The paper describes a public release of press release text, article URLs and derived data. The earliest notebooks query Bloomberg-internal databases.

## Status

Collection and modeling June 2023 - April 2024; notebooks edited through May 2025.
