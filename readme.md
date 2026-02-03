# 3DICE: Interpretable 3D Cross-Modal Learning for Drug–Target Interaction

## Quick Start

### 1) Environment
Create the Conda environment (macOS, Linux, or Windows with CPU/GPU):

```bash
conda env create -f environment.yml
conda activate SRP
```

Model dependencies (from [environment.yml](environment.yml)):
- Core: PyTorch (`torch`), NumPy, pandas
- Protein embeddings: `fair-esm`, `biotite`, `gemmi`
- Drug embeddings: `unimol-tools`, `rdkit`
- GNN/graph utilities: `torch-geometric`, `torch-scatter`
- Config + metrics: `yacs`, `scikit-learn`
- Plotting + utilities: `matplotlib`, `tqdm`


### 2) Data Layout
3DICE expects three CSV splits and embedding folders (defaults in [config/cfg.py](config/cfg.py)):

- CSVs: [lists/db_train.csv](lists/db_train.csv), [lists/db_val.csv](lists/db_val.csv), [lists/db_test.csv](lists/db_test.csv)
	- Required columns: `uniprot_id`, `drug_id`, `interaction` (0/1)
	- Optional column: `SMILES` (used by drug embedding script)

- Protein embeddings: [embeddings/](embeddings/)
	- Per-Protein folder: `embeddings/{uniprot_id}/...`
	- Latest `*.pt` file in each folder is loaded as the token sequence (e.g., per-residue 512-D vectors).
	- **Protein embeddings should be computed with Python 3.10 (see ESM‑IF1 section)**

- Drug embeddings: [drug/embeddings_atomic/](drug/embeddings_atomic/)
	- Per-Drug file: `{drug_id}.pt` or `{drug_id}_unimol.pt` (contains `atomic_reprs` → reshaped to `(L_atoms, 512)`).

You can change paths in [config/cfg.py](config/cfg.py):

- `DATA.TRAIN_CSV_PATH`, `DATA.VAL_CSV_PATH`, `DATA.TEST_CSV_PATH`
- `DATA.PROTEIN_DIR`, `DATA.DRUG_DIR`


### 3) Generate Drug Embeddings (UniMol)
If your CSVs include `SMILES`, generate UniMol atomic embeddings:

Edit input/output in [drug/02_embed_drugs.py](drug/02_embed_drugs.py) if needed (defaults to `lists/db_train.csv` and `drug/embeddings_atomic/`), then run:

```bash
python drug/02_embed_drugs.py
```

This writes `drug/embeddings_atomic/{drug_id}.pt` with atomic representations.


### 4) Train
Run the training loop in [solver.py](solver.py):

```bash
python solver.py
```

Outputs
- Checkpoints: [saved/](saved/) and best models in [best_models/](best_models/) (created on first run).
- Metrics log: [logs/training_log_YYYYMMDD_HHMMSS.csv](logs/) with loss, accuracy, MCC, AUC, AUPRC, confusion counts, epoch time.

Adjust training config in [config/cfg.py](config/cfg.py):
- `SOLVER.BATCH_SIZE`, `SOLVER.EPOCHS`, `SOLVER.LR`, `SOLVER.WEIGHT_DECAY`, `SOLVER.DROPOUT`
- `SOLVER.LOSS_FN`: `"cross_entropy"`
- Model dims: `DRUG.*`, `PROTEIN.*`, `MLP.*`

### 5) Evaluate
Set your checkpoint path in [eval.py](eval.py) (`CKPT_PATH` at the top), then:

```bash
python eval.py
```

## Interpretability

Use [attentiontest.py](attentiontest.py) to inspect residue–atom attention for a specific pair.

Example (choose your own IDs and checkpoint):

```bash
python attentiontest.py \
	--ckpt saved/model_XXXXXXXX_epoch_YY.pt \
	--csv lists/db_train.csv \
	--drug_id DB00001 \
	--uniprot_id P00533 \
	--top_k 20 \
	--plot_style poster \
	--save_plots
```

Flags (selected)
- `--joint_mode {geom,arith,harm,prod,min}`: combine bidirectional attentions into joint scores.
- `--atom_labels {none,topk,all}` and `--label_with_weight`: control py3Dmol labeling in 3D views.
- `--save_plots`, `--save_dir`, `--save_formats`: persist figures for papers/posters.

Tip: For macOS poster-quality exports, the script sets larger fonts and DPI when `--plot_style poster`.

## ESM‑IF1 Embeddings (Python 3.10)

ESM‑IF1 (inverse folding) embeddings are structure‑conditioned per‑residue features (512‑D) computed from PDB/mmCIF backbones. We create a separate Python 3.10 env for maximum compatibility with `fair-esm` and geometry parsers.

Environment (separate env)
```bash
conda create -n 3DICE-ESMIF1 python=3.10 -y
conda activate 3DICE-ESMIF1

# Install PyTorch matching your platform (CUDA/MPS/CPU)
pip install torch torchvision torchaudio

# ESM, Biotite (structure parsing), and helpers
pip install fair-esm biotite gemmi tqdm numpy pandas
```

Fetch structures
- Use [protein/01_fetch_structures_v2.py](protein/01_fetch_structures_v2.py) to download mmCIF/PDB per UniProt ID:

```bash
python protein/01_fetch_structures_v2.py \
  --input_csv lists/db_train.csv \
  --uniprot_col uniprot_id \
  --out_dir structures \
  --format mmcif \
  --min_coverage 0.2 \
  --alphafold_fallback 1
```

Embeddings can be computed using `protein/02_embed_protein_3D.py`
