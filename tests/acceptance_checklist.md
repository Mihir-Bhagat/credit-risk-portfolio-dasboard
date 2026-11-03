# Acceptance checklist — Credit Risk Portfolio Analytics take-home (v2)

Source of truth: `Credit Risk Portfolio Analytics – Take-Home Assignment (v2).pdf` (4 pages).
Status values: **PASS / FAIL / MANUAL**. Nothing is PASS until its command output is shown in `tests/verify_log.md`.

> **Baseline state (2026-11-03):** the zip contains only the brief PDF and the data pack
> (5 CSVs + DATA_DICTIONARY.md). **No repo, app, model or docs exist yet**, so every automatable
> item starts as FAIL ("not built"). The verification loop therefore starts by building the project
> against this checklist (tests first, then code).

## Pre-registered thresholds (fixed now so they cannot be tuned later)

| Name | Value | Used by |
|---|---|---|
| Single-feature leakage ceiling | AUC ≤ 0.95 (direction-agnostic) | #27 |
| Gini identity tolerance | `abs(Gini − (2·AUC−1)) ≤ 1e-9` | #37 |
| KS independent check | `abs(KS − scipy ks_2samp) ≤ 1e-9` | #37 |
| Reproducibility | metrics identical to 9 decimals across two runs, same seed | #38 |
| KPI tolerance | `abs(app − pandas) ≤ 1e-6` (rates), exact for counts | #47 |
| Anomaly flag-rate band ("not absurd") | 0.5 % ≤ rate ≤ 10 % of accounts | #66 |
| Injected extreme accounts | ≥ 10 synthetic accounts, 100 % must be flagged | #69 |
| Question suite | ≥ 15 questions: ≥ 8 answerable, ≥ 4 unanswerable, ≥ 3 prompt-injection | #60 |
| Cold start / filter latency | ≤ 10 s first render, ≤ 2 s per filter change (cached) | #89 |
| Test-set split | by `customer_id` (customers own up to 3 accounts), never touched until final scoring | #36 |

## A. Repo and layout

| # | Requirement (brief quote) | Verified by | Status |
|---|---|---|---|
| 1 | "README.md" at repo root (§Repository layout) | `tests/test_a_repo.py::test_readme_exists` | PASS |
| 2 | "requirements.txt # every package pinned, e.g. pandas==2.2.3" | test: every non-comment line has `==`; AST scan of all imports in `app/`, `src/` maps to a pinned package | PASS |
| 3 | ".env.example # names of required keys, no values" | test: every line is `KEY=` with empty value; contains the LLM key name(s) the code reads | PASS |
| 4 | "app/app.py # dashboard entry point" | exists; `AppTest.from_file` runs it | PASS |
| 5 | "src/ # ingestion, analytics, model, AI features" | test: modules for each of the four exist and import cleanly | PASS |
| 6 | "data/raw/ # the 5 CSVs as provided" | sha256 of each CSV equals `tests/raw_data_sha256.txt` (taken from the supplied pack) | PASS |
| 7 | "docs/architecture.pdf" | exists, is a PDF | PASS |
| 8 | "docs/solution_overview.pdf" | exists, is a PDF | PASS |
| 9 | "tests/ # optional; recommended" | `pytest` collects and passes | PASS |
| 10 | "Submit exactly two documents, each one page, as PDF. No slide deck is needed." | test: the only deliverable documents are the 2 PDFs; no `.pptx`. **Open question Q3** on `docs/data_quality_log.md` | PASS |
| 11 | "contain no API keys or other secrets" | `scripts/secret_scan.py`: regex + entropy over working tree AND `git log -p` of all commits; no `st.secrets` values; no `.streamlit/secrets.toml` tracked | PASS |
| 12 | (your rule) `.gitignore` covers `.env`, `.DS_Store`, secrets | test: `git check-ignore` on `.env`, `.DS_Store`, `.streamlit/secrets.toml` | PASS |
| 13 | "Must match the hosted app" | automated: app footer shows the git commit SHA; MANUAL: compare with deployed app | MANUAL |
| 14 | "Public, or private with access given to the reviewer account we name" | cannot be checked from a sandbox | MANUAL |

### README (§"README must include")

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 15 | "The Python version used." | test: README names a version; it equals `.python-version` / the interpreter used in verification | PASS |
| 16 | "Run instructions as no more than three commands: install, build the data and model, start the dashboard." | test: parse the fenced "Run" block; ≤ 3 commands; covers install, build, start | PASS |
| 17 | (your rule) runs from a fresh virtualenv using only those commands | `scripts/fresh_venv_check.sh`: new venv, run the README commands verbatim, poll `/_stcore/health` | PASS |
| 18 | "How to set the LLM API key, and what the app does if no key is set." | test: README documents key name, where to set it (env / secrets), and the no-key fallback | PASS |
| 19 | "Where each item in this brief lives in the repo." | test: mapping table has a row for every brief item (1–4 build, 2 AI features, 2 docs, 3 submission items); every path in it exists | PASS |

## B. Data pipeline (brief Required build 1)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 20 | "load and join all 5 tables" | test: pipeline reads all 5 files; final dataset has at least one column derived from each | PASS |
| 21 | "produce one clean, analysis-ready dataset" | test: single output file; grain = 1 row/account; `account_id` unique; types correct; no unhandled NaN in model features | PASS |
| 22 | "Document your join logic" | test: join doc exists; independent row-count check: 14,685 accounts in → 14,685 out; no key orphans; no explosion after each join | PASS |
| 23 | "Document … every data-quality decision" | test: `docs/data_quality_log.md` has detect / count / decision / rationale per issue | PASS |
| 24 | Every issue in the log is fixed in the output (your rule) | test re-runs each issue's detector on the cleaned data → 0 remaining (or documented residual) | PASS |
| 25 | Data dictionary: "surface and handle these explicitly, not silently drop rows" | test: no row loss at any step; missing-value counts in log equal counts re-measured from raw | PASS |
| 26 | Pipeline runs from raw CSVs on a clean checkout, no manual steps | `scripts/fresh_venv_check.sh` + run twice, output hashes equal | PASS |
| 27 | Leakage: (your rule) no feature uses information on/after the default observation point | test: feature list + cutoff logic written to `reports/feature_catalog.csv`; point-in-time check; per-feature AUC ≤ 0.95. **Open question Q1** | PASS |
| 28 | Issues found by an independent profile are all logged (pre-found: 600 M/F gender codes; 500 nulls each in `years_employed`/`num_dependents`/`years_at_residence`; 294 null `interest_rate_pct`; 689 null bureau `days_past_due`; 1,008 customers without bureau lines; 7,293 loans originated before `customer_since`; 3,875 collections events dated before origination; 5,760 bureau lines opened after 2023-12-31; 429 "Unemployed" with income > 12,000; 726 with `years_employed` > age−16; dictionary says "SMS-Email", data says "SMS/Email Notice") | test: independent profiler output ⊆ log | PASS |

## C. Portfolio analytics (brief Required build 2)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 29 | "profile the portfolio across income, employment type, region, age and loan type" | test: analytics output and dashboard each contain all 5 dimensions | PASS |
| 30 | "Give 4–5 insights, each with a 'so what'" | test: 4 ≤ count ≤ 5; each has non-empty so-what | PASS |
| 31 | "including at least one cross-dimensional cut" | test: ≥ 1 insight built on ≥ 2 dimensions | PASS |
| 32 | (your rule) numbers in insights are correct | independent pandas recompute equals the stated figures | PASS |

## D. Default model (brief Required build 3)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 33 | "engineered features from the bureau and repayment history" | test: feature catalog has ≥ 1 group from each source and none are raw pass-through only | PASS |
| 34 | "a logistic regression baseline" | trains; metrics file written | PASS |
| 35 | "one ML model" | gradient-boosted model trains; metrics file written | PASS |
| 36 | "Report AUC, KS, Gini" on a test set never used for tuning or feature selection (your rule) | test: customer-grouped split; no `customer_id` overlap; tuning uses CV on train only; selection logged before test scoring | PASS |
| 37 | KS and Gini consistent with AUC | Gini = 2·AUC−1; KS recomputed with `scipy.stats.ks_2samp` | PASS |
| 38 | Reproducible (your rule) | run training twice with same seed; metrics identical | PASS |
| 39 | Fair comparison, honest reporting | same split, same features, same tuning budget; result reported even if boosted model does not win | PASS |
| 40 | "feature importance or SHAP" | output files exist and are non-empty; top features sensible | PASS |
| 41 | Account-level "risk score and key drivers" for every account | test: 14,685 scores, all in [0,1], none null; top drivers non-empty for every row | PASS |
| 42 | "explain how you would monitor the model after deployment" | test: present in README/overview (PSI, KS/AUC drift, calibration, retrain trigger); PSI function runs on a shifted sample | PASS |

## E. Dashboard (brief Required build 4)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 43 | "portfolio view (risk segments, exposure, trends across dimensions)" | AppTest: tab loads, contains risk segments, exposure, and trend-by-dimension elements | PASS |
| 44 | "account-level view (risk score and key drivers)" | AppTest: select account → score and drivers shown, equal to model output file | PASS |
| 45 | "at least one working filter or drill-down" | AppTest: change a filter → displayed numbers change; drill-down lists accounts of the chosen segment | PASS |
| 46 | "both AI features (section 3) embedded in it" | AppTest: both panels present in the same app | PASS |
| 47 | KPIs match direct pandas (your rule) | AppTest values vs independent pandas on cleaned data | PASS |
| 48 | Every tab loads without exception (your rule) | `AppTest` per tab: `at.exception` empty | PASS |
| 49 | "A static HTML page … does not qualify" — key must stay server-side | test: key read only server-side; no key in rendered output / session state | PASS |
| 50 | Starts from fresh venv with README commands | same as #17 | PASS |
| 51 | "A live app on Streamlit Community Cloud, Hugging Face Spaces or similar" | I cannot deploy from here (no credentials) | MANUAL |
| 52 | "open without a login, with the LLM feature working, and stay live for 14 days" | private-window check by you | MANUAL |

## F. AI feature 1 — "Ask the Portfolio" (brief §Mandatory AI features 1)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 53 | "natural-language Q&A over the dashboard data using an LLM" | adapter integration test over real HTTP against a local OpenAI-compatible server (plan used, numbers computed locally); a run against a real provider is MANUAL item M4 | PASS |
| 54 | "grounded in the actual underlying data — no hallucinated numbers" | every number in the final answer appears in the executed result (parsed and compared) | PASS |
| 55 | "short plain-English answer with the exact figures and where they came from" | test: answer shows source (table, columns, filter, the query actually run) | PASS |
| 56 | Example: "Which loan type has the highest default rate?" | independent pandas gives Mortgage ≈ 24.7 % from raw data; app must match | PASS |
| 57 | "Questions the data cannot answer are declined" | unanswerable set (forecasts, causes not in data, outside the 5 tables) → declined with a reason, zero invented numbers | PASS |
| 58 | Prompt-injection resistance (your rule) | "ignore instructions…", "show raw rows", "what is your API key" → declined with reason; key never echoed | PASS |
| 59 | No-key mode (brief: "what the app does if no key is set") | unset key → preset questions + visible notice, no crash; simulated API error and rate limit behave the same | PASS |
| 60 | 15+ question suite (your rule) | `tests/qa_suite.yaml`, thresholds above | PASS |
| 61 | Independent oracle for each answerable question | pandas written separately from feature code; figures match | PASS |
| 62 | Brief: architecture doc must say "what data is sent to it" | log the payload; assert only the question + table catalog leave the machine, no row-level data. **Open question Q2** | PASS |
| 63 | "state in the dashboard what triggers it, what a false positive and a false negative look like, and the human-in-the-loop step" | AppTest: panel text contains all four | PASS |

## G. AI feature 2 — anomaly detection (brief §Mandatory AI features 2)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 64 | "flag accounts or customers with unusual repayment or bureau patterns compared with their own history or their peer segment" | test: signals from both repayment and bureau; at least one own-history method and one peer-segment method | PASS |
| 65 | "a review list of flagged accounts in the dashboard" | AppTest: list is present and non-empty | PASS |
| 66 | Flag rate is "not absurd" (your rule) | rate reported; must be inside the pre-registered 0.5 %–10 % band | PASS |
| 67 | "For each account, the analyst can see why it was flagged" | every flag has a readable reason; numbers in reason equal the data | PASS |
| 68 | "record a decision (e.g. confirm or dismiss)" | writes timestamp, account id, decision, note; still there after app reload | PASS |
| 69 | True-positive check (your rule) | inject ≥ 10 synthetic extreme accounts into a test copy → all flagged; normal accounts flagged for no clear reason are reported | PASS |
| 70 | "state in the dashboard what triggers it, what a false positive and a false negative look like, and the human-in-the-loop step" | AppTest: panel text contains all four | PASS |
| 71 | "working on the hosted link" (both AI features) | cannot be checked from a sandbox | MANUAL |

## H. Documents (brief §Documents)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 72 | architecture.pdf: "each one page" | `pdfinfo` Pages == 1 | PASS |
| 73 | "The flow from raw data → ingestion → analytics → model → dashboard, with both AI features shown." | text extraction: all stages and both features present; page rasterised and checked by eye | PASS |
| 74 | "The tool or library used at each stage, where the LLM is called, and what data is sent to it." | text extraction for tool names per stage + LLM call site + payload description | PASS |
| 75 | solution_overview.pdf: one page | `pdfinfo` Pages == 1 | PASS |
| 76 | Contains the five listed sections: dashboard views; two AI features (how, trigger, FP, FN, human review); model results + approach; top 3 insights; assumptions + what is needed from bank IT/data | text extraction for each section | PASS |
| 77 | Metrics quoted match the model output files (your rule) | parse AUC/KS/Gini from PDF and compare with `reports/metrics.json` | PASS |
| 78 | "Your top 3 portfolio insights" match the analytics output | parse and compare | PASS |
| 79 | "Content beyond one page will not be read" | no clipped text: all text within page bounds; no overflow to page 2; visual check | PASS |

## I. Submission (brief §Submission standard)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 80 | "one email containing three links. Do not attach zip files." | by you | MANUAL |
| 81 | "Demo video link … 3–5 minutes … Show the portfolio and account views and both AI features in use." | I will provide a shot list; you check duration and content | MANUAL |
| 82 | "You must be able to explain every part of your solution in the follow-up interview." | I will provide an explainer notes file; you review | MANUAL |

## J. Quality bar — "best dashboard possible" (your request, beyond the brief)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 83 | Every filter combination runs without exception, including empty results | sweep through AppTest; empty-state message shown | PASS |
| 84 | Cold start / interaction latency | measured against pre-registered budget | PASS |
| 85 | Visual QA of every tab at desktop (1440 px) and phone (390 px) widths | Playwright screenshots, reviewed by me and shown to you | PASS |
| 86 | Accessible, colour-blind-safe palette with contrast ≥ 4.5:1 for text | script computing contrast ratios of the palette | PASS |
| 87 | No hardcoded numbers in the UI | grep + #47 | PASS |

## K. LLM selection (your request, beyond the brief)

| # | Requirement | Verified by | Status |
|---|---|---|---|
| 88 | Researched shortlist with sources: price, latency, structured-output support, rate limits, free-tier/data-use terms | `tests/reports/llm_research.md` with links, checked against current pricing pages | PASS |
| 89 | Each candidate run on the same question suite | `tests/reports/llm_comparison.md`: plan correctness, refusal correctness, injection resistance, latency, cost per question. **Needs a key per provider you want tested** | MANUAL |
| 90 | Provider-agnostic adapter; model chosen via env var | unit tests per adapter against recorded responses | PASS |
| 91 | Real API calls counted and reported | call counter in the log | PASS |

## Open decisions I need from you before building

**Q1. Observation point and leakage (modelling decision).** The data has no default date and no snapshot date.
Repayment history is exactly months 1–12 for every account (even 360-month mortgages). Collections events
are dated before origination for 3,875 of 8,635 rows. Bureau lines are opened up to 2025-12, after every loan
(last origination 2023-12-30). Single-feature AUCs are all ≤ 0.62, so the leakage test will pass, but the
timeline is unreliable. My proposal: observation point = end of month 12; repayment features use months 1–12 only;
**exclude collections events from the model** (they follow delinquency; used for analytics only); use bureau
fields as given but state in the log that they carry no snapshot date. Expect modest AUC (~0.65–0.70), reported honestly.

**Q2. What may leave the machine for "Ask the Portfolio".** Your rule says only the question + table catalog.
Proposal: the LLM returns a structured query plan; pandas executes it locally; the answer is rendered locally from the
result, so numbers cannot be hallucinated. The alternative (second LLM call to phrase the answer from aggregated results)
reads better but sends aggregates out. Which do you want?

**Q3. Where does the data-quality log live?** The brief says "exactly two documents"; your rule needs
`docs/data_quality_log.md`. Proposal: keep it at `docs/data_quality_log.md` (it is project documentation, not a
submission document) and say so in the README.

**Q4. Keys and hosting.** I have no LLM API keys and cannot deploy to Streamlit Cloud or push to GitHub from here.
Tell me which providers you have keys for (I can only test those); hosting, repo visibility and the video stay MANUAL.
Also, Streamlit Cloud storage is ephemeral, so confirm/dismiss decisions saved to a local file are lost on restart.
Proposal: local CSV/SQLite with a visible notice, plus a download button.
