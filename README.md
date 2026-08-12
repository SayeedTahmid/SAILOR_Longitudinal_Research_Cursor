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

The production data root is locked to:

```text
/content/drive/MyDrive/SAILOR_Longitudinal_Research_Cursor
```

Local paths belong in the ignored `configs/local_paths.py`; see
`configs/local_paths.example.py`.
