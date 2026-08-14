# SAILOR Longitudinal Research

Primary Cursor implementation of the locked SAILOR v2.0 research protocol.

## Phase 1

Phase 1 is a CPU-only, read-only audit of the legacy EBRAINS data. It never
downloads data and never writes beneath `/content/drive/MyDrive/sailor_v1`.

In Colab:

```bash
pip install -r requirements.txt
python scripts/run_stage1_audit.py
```

After Phase 1 is complete, preview the Phase 2 extraction plan:

```bash
python scripts/run_stage2_preprocessing.py --section 10
```

Selective extraction requires a separate explicit approval flag:

```bash
python scripts/run_stage2_preprocessing.py --section 10 --execute --approve-extraction
```

Phase 3 evaluates persistence, MRI-only, and MRI+Δt baselines. Preview the plan:

```bash
python scripts/run_stage3_baselines.py --section 14
python scripts/run_stage3_baselines.py --section 15
```

Approved execution writes patient-level results under `07_BASELINE_RESULTS/p3.0/`:

```bash
python scripts/run_stage3_baselines.py --section 14 --execute
python scripts/run_stage3_baselines.py --section 15 --execute
```

C0/C1 require PyTorch in the runtime. Persistence C−1 is CPU-only. Do not write
into `SAILOR_READY_v2.0`; that folder is a read-only distribution copy.

Build the separate teammate distribution package with a read-only dry run:

```bash
python scripts/build_sailor_ready.py
```

After reviewing the copy plan:

```bash
python scripts/build_sailor_ready.py --execute --approve-copy
python scripts/verify_sailor_ready.py /content/drive/MyDrive/SAILOR_READY_v2.0
```

This creates a sibling copy only. It never modifies the authoritative Phase-2
project root.

The production data root is locked to:

```text
/content/drive/MyDrive/SAILOR_Longitudinal_Research_Cursor
```

Local paths belong in the ignored `configs/local_paths.py`; see
`configs/local_paths.example.py`.
