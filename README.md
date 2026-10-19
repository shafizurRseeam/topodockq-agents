# TopoDockQ Agents

An LLM-based multi-agent framework for protein–peptide complex quality assessment, built around the
published **TopoDockQ** method.

AlphaFold can generate dozens of candidate 3D structures for one peptide bound to one protein, and its own
confidence score often ranks a wrong structure first. TopoDockQ ([Dai, Wang & Zhang, *Communications
Chemistry* 2025](https://doi.org/10.1038/s42004-025-01727-4)) predicts how good each candidate is (p-DockQ,
from 0 to 1) from topological features of the protein–peptide interface, so the best candidate can be chosen.

This project runs the published TopoDockQ pipeline unchanged and puts a team of Claude agents in charge of it.
You describe what you want in plain language; the agents find and validate the input structures, run the
scientific tools in batches, check every hand-off, handle failures, and report the best candidate for each
complex. Every decision is recorded, and every result is re-checked by an independent run of the same
pipeline without any LLM.

The scientific method, its code, the trained model and the data are the work of the TopoDockQ authors (see
[Credits](#credits-and-licenses)). This repository contributes the agentic orchestration around them.

## How it works

```
your request + structures
        │
        ▼
   MainAgent ── plans the job, delegates whole batches, checks each stage, stops or retries, reports
        │
        ├─▶ DataAgent   finds the structures, validates them, extracts each protein–peptide interface (10 Å)
        ├─▶ TopoAgent   computes 2,754 topological features per interface
        └─▶ MLAgent     verifies the published model, predicts p-DockQ, ranks candidates per complex
        │
        ▼
ranking (top-1 per complex), predictions, report, full trace  ──▶  independent deterministic re-check
```

The agents decide *what* to run; all numbers come from deterministic tools that wrap the authors' code:
their interface-extraction script, their feature generator (or a Python 3.13 port verified to match it), and
the published network with its original preprocessing. Agents pass file paths and metadata to each other,
never feature arrays. Structures are excluded only by fixed, documented validation rules.

## Repository structure

```
├── topodockq_agents/           agent layer
│   ├── agents/                 MainAgent, DataAgent, TopoAgent, MLAgent (prompts and tools)
│   ├── tools/                  the functions the agents call, bound to the run state
│   ├── pipeline.py             command-line entry point: `topodockq`
│   ├── verify.py               independent deterministic re-check of every agent run
│   ├── state.py                artifact registry and run trace
│   ├── llm.py, config.py       Claude API wrapper (Anthropic SDK tool runner), settings
├── tdq_adapters/               deterministic tools around the authors' code (no LLM)
│   ├── inputs.py               turns a folder or manifest into a validated list of structures
│   ├── structures.py           fixed validation rules
│   ├── data_prep.py            interface extraction with the authors' script
│   ├── topo_features.py        Python 3.13 port of the authors' feature generator
│   ├── feature_gen.py          runs either feature engine and converts with the authors' script
│   ├── preprocessing.py        the published column selection and scaler, integrity-checked
│   ├── pipeline.py             the whole chain in one deterministic call
│   ├── evaluation.py           the paper's top-1 selection metrics
│   ├── bootstrap.py            `topodockq-setup`: downloads the model and data, verifies everything
│   ├── step1.py … step4.py     reproduction checks against the published results
│   └── resources/              expected hashes of the authors' code
├── authors_code/TopoDockQ/     the authors' repository (MIT License), unmodified, at commit 5696f82
├── artifacts/
│   ├── preprocessing/          column selection and scaler of the published model
│   └── audits/                 dataset audit of the published interface files
├── tests/                      unit and integration tests
├── pyproject.toml, uv.lock     dependencies, pinned by uv
└── .env.example                template for your API key
```

Created on your machine and never committed: `data/` (model and structures, downloaded from Zenodo by the
setup step), `.venv/`, `.venv-prody/`, `.venv-feature38/` (Python environments) and `runs/` (outputs).

## Requirements

- [uv](https://docs.astral.sh/uv/), which installs the right Python version and every dependency for you.
- An internet connection for the setup step.
- Disk space: about 1.6 GB for the Python environments and the model, plus 1.3 GB for the example
  structures. The optional reproduction data needs about 6.5 GB more.
- Your own Anthropic API key for the agents (see step 5). The setup, the tests and dry runs work without one.
- Developed and tested on Windows 11. Linux and macOS should work; see [Troubleshooting](#troubleshooting).

## Installation

**1. Install uv** (skip if `uv --version` already works).

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Clone the repository.**

```bash
git clone https://github.com/shafizurRseeam/topodockq-agents.git
cd topodockq-agents
```

**3. Install the Python dependencies.**

```bash
uv sync
```

This creates `.venv/` with Python 3.13 and installs the exact versions in `uv.lock`, including a CPU build of
PyTorch.

**4. Download the model and the example structures.**

```bash
uv run topodockq-setup --examples
```

`uv sync` installs Python packages only, and the model and data are too large for git, so this one command
fetches them from their original source and prepares everything else:

- checks that the authors' code shipped in `authors_code/TopoDockQ` is complete and unmodified (SHA-256 of
  every file);
- downloads the trained model (64 MB) and the example AlphaFold structures (222 MB) from
  [Zenodo record 15469415](https://zenodo.org/records/15469415), checked against Zenodo's size and MD5, and
  checks every unpacked file;
- creates two small side environments that the authors' tools need (Python 3.9 with ProDy 2.4.1, and
  Python 3.8 with gudhi 3.8.0, as in the authors' environment files).

It finishes with a self-test: it regenerates the authors' published 4k38 example and must obtain their
published score, 0.15463054. Re-running the command re-verifies what is already in place and downloads only
what is missing. Leave out `--examples` if you only want to score your own structures.

**5. Add your Anthropic API key.**

The agents call Claude through the Anthropic API, which needs a key of your own. The repository does not
include one.

1. Sign in at [console.anthropic.com](https://console.anthropic.com), add credit under **Billing**, and create
   a key under **API Keys**.
2. Create your `.env` from the template:
   ```bash
   cp .env.example .env              # Windows PowerShell: Copy-Item .env.example .env
   ```
3. Open `.env` and replace the placeholder:
   ```
   TOPODOCKQ_ANTHROPIC_API_KEY=sk-ant-...your key...
   TOPODOCKQ_MODEL=claude-sonnet-5-5
   ```
4. Check it (this makes no billed request):
   ```bash
   uv run topodockq --check
   ```

`.env` is git-ignored; never commit it. The project reads the key only from `.env` and deliberately ignores a
system-wide `ANTHROPIC_API_KEY`.

## Usage

**Try it without the API** (a dry run calls the tools in a fixed order, no agents, no cost):

```bash
uv run topodockq --input data/external/PFPD_AF2_models/1AWR --dry-run
```

**Run the agents:**

```bash
uv run topodockq --input data/external/PFPD_AF2_models \
    --task "For complexes 1AWR, 1ER8 and 2FMF, rank AlphaFold models 0 to 9 and tell me the best model per complex."
```

The request is plain language: you can name complexes or models, ask for the authors' original feature code,
or ask for everything in the input. Useful options:

| Option | Effect |
|---|---|
| `--expect-structures N` | the run fails its check unless exactly N structures were selected (guards against a misread selection) |
| `--strict` | a complex whose top-1 candidate has unconfirmed, implausible chain roles is marked `needs_confirmation` instead of `accepted` |
| `--mode sequential` | run DataAgent, TopoAgent and MLAgent in fixed order without the MainAgent |
| `--no-verify` | skip the deterministic re-check |

**Your own structures.** Protein as chain A, peptide as chain B, in any of these forms:

- a folder with one sub-folder per complex holding AlphaFold models `ranked_<N>.pdb` (or `ranked_<N>_sp.pdb`);
- a single such complex folder;
- a folder of `.pdb` files;
- a CSV manifest with columns `pdb_id`, `af_model_id`, `input_pdb`, and optionally `protein_chain`,
  `peptide_chain`.

Chain IDs alone do not prove which chain is the protein. If the peptide chain is longer than the protein
chain, the structure gets the warning `W_CHAIN_ROLE_SUSPICIOUS` and is still scored. The roles count as
confirmed, and the warning is not raised, when a manifest lists `protein_chain` and `peptide_chain`, or when
your request states them. A structure whose protein is not chain A is excluded (`E_CHAIN_IDS`): the authors'
feature code reads the protein from chain A and the peptide from chain B.

**Output.** Each run writes `runs/<run id>/`:

| File | Contents |
|---|---|
| `report.md` | run status, stages, input selection per complex, top-ranked candidate per complex, exclusions, verification result, the MainAgent's report, token usage |
| `artifacts/ranking.csv` | top-1 candidate per complex: p-DockQ, predicted high quality (> 0.82), validation warnings, ranking status |
| `artifacts/predictions.csv` | p-DockQ for every structure, or the rule code that excluded it, plus its warnings |
| `trace.jsonl` | every stage, delegation, tool call (arguments, outcome, timing) and model message |
| `verification.json` | comparison with an independent deterministic re-run of the same structures, and the input-selection audit |

**How a run is checked.** Each stage's outcome is decided by Python checks on its output files, not by the
agent's own account: `done`; `partial` (valid outputs, but some structures failed at that stage and are
excluded with a rule code); `incomplete` or `failed` (the MainAgent may retry once). After the agents finish,
the same structures are re-run without any LLM and compared: same structures (duplicates included), same
statuses, exclusion rules and warnings, p-DockQ within 1e-6, one ranking row per complex and the same top-1.
The selection itself is audited too: the structure list must match a fresh scan of the input with the
recorded selection, and with `--expect-structures` its size must match yours. A run whose checks fail, or that
crashes, still writes its report and trace, with run status `failed`.

Example: the request above produced this ranking (p-DockQ is a predicted DockQ; above 0.82 counts as
predicted high quality):

| complex | top-1 candidate | p-DockQ | predicted high quality |
|---|---|---|---|
| 1awr | 1awr_ranked_0 | 0.4634 | no |
| 1er8 | 1er8_ranked_6 | 0.7397 | no |
| 2fmf | 2fmf_ranked_2 | 0.5382 | no |

With Claude Sonnet 5.5 that run took under two minutes and used about 25,000 input and 3,000 output tokens,
a few cents at current API prices. The independent re-check matched it exactly.

**From Python, without agents:**

```python
from tdq_adapters import inputs, pipeline

structures, _ = inputs.discover("data/external/PFPD_AF2_models", complexes=["1AWR"], models=[0, 1, 2])
result = pipeline.score_structures(structures, "runs/my-run")
print(result["ranking"])
```

## Reproducing the published results (optional)

The `tdq_adapters` step runners check this implementation against the paper and the authors' files. They
need the larger data downloads (about 4.7 GB to download, 6.5 GB on disk):

```bash
uv run topodockq-setup --reproduction
uv run python -m tdq_adapters.step1   # published model on the authors' example: p-DockQ 0.15463054
uv run python -m tdq_adapters.step2   # Figure 2, SinglePPD-Test_70%: FPR and precision vs AlphaFold confidence
uv run python -m tdq_adapters.step3   # interface extraction on 27 PFPD complexes x 50 models vs the authors' files
uv run python -m tdq_adapters.step4   # features vs the published feature table (both engines)
uv run python -m tdq_adapters.e2e     # full structure -> ranking, three independent paths
```

Each writes a JSON report to `runs/` and checks that no downloaded file was modified.

## Tests

```bash
uv run pytest
```

The integration tests use the example structures and are skipped if `--examples` was not downloaded.

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `TOPODOCKQ_ANTHROPIC_API_KEY` | none | your API key; required for agent runs |
| `TOPODOCKQ_MODEL` | `claude-opus-5-5` | model for every agent; the template sets `claude-sonnet-5-5` |
| `TOPODOCKQ_{MAIN,DATA,TOPO,ML}_MODEL` | `TOPODOCKQ_MODEL` | model for one agent |
| `TOPODOCKQ_{MAIN,DATA,TOPO,ML}_EFFORT` | `high` (Main), `medium` | reasoning effort: low, medium, high, xhigh, max |
| `TOPODOCKQ_INPUT` | none | default `--input` |
| `TOPODOCKQ_WORKERS` | 8 | structures processed in parallel |
| `TOPODOCKQ_VERIFY` | 1 | re-run the deterministic pipeline after each agent run and compare |
| `TOPODOCKQ_STRICT` | 0 | same as `--strict` |
| `TOPODOCKQ_RUNS_DIR` | `./runs` | output folder |

Command-line options: `uv run topodockq --help`.

## Troubleshooting

- **"No project API key configured"**: `.env` is missing or still has the placeholder; see step 5.
- **The setup cannot create `.venv-prody` or `.venv-feature38`** (most likely on macOS with Apple silicon,
  where some old packages have no prebuilt wheels): the agents still work for everything except the
  authors' original feature code if only `.venv-feature38` fails; interface extraction needs `.venv-prody`.
  The authors' `environment.yaml` and `src/prody_environment.yml` (in `authors_code/TopoDockQ/`) describe a
  conda alternative.
- **Windows "path too long" errors during `uv sync`**: PyTorch has deep file paths. Enable long paths in
  Windows, or clone into a short folder such as `C:\src\topodockq-agents`.
- **"authors_code/TopoDockQ is incomplete or modified"**: a file in the shipped authors' code was changed
  or deleted; restore it with `git checkout -- authors_code/TopoDockQ`.
- **GPU**: not needed; the network is small and runs on CPU.
- **Re-downloading**: `uv run topodockq-setup --force` downloads the Zenodo files again and re-verifies them.

## Credits and licenses

- **Method and paper:** Dai X., Wang R., Zhang Y. Topological deep learning for enhancing peptide-protein
  complex prediction. *Communications Chemistry* 8, 347 (2025).
  [doi:10.1038/s42004-025-01727-4](https://doi.org/10.1038/s42004-025-01727-4)
- **Code:** [XDaiNYU/TopoDockQ](https://github.com/XDaiNYU/TopoDockQ), Copyright (c) 2025 Xuhang Dai,
  Rui Wang, Yingkai Zhang, MIT License. It is included unmodified in `authors_code/TopoDockQ` (commit
  5696f82) with its `LICENSE` file. Its compiled feature generator is the same as the one published in
  [wangru25/TopoDockQ-Feature](https://github.com/wangru25/TopoDockQ-Feature) (byte-identical files).
  `tdq_adapters/topo_features.py` is a Python 3.13 port derived from that MIT-licensed code.
- **Data and trained model:** [Zenodo record 15469415](https://zenodo.org/records/15469415), CC BY 4.0,
  downloaded by `topodockq-setup`. `artifacts/preprocessing/` contains the column selection and scaler
  statistics derived from that data.
- **Agents:** Claude by Anthropic, through the Anthropic Python SDK.
- **This repository:** MIT License, see [LICENSE](LICENSE). The authors' code in `authors_code/TopoDockQ`
  keeps its own MIT license and copyright notice.

If you use this work, please cite the TopoDockQ paper above.
