# TourMT-Hi

Tourism-domain machine translation for English↔Hindi, with a low-resource extension to Maithili.
We measure how much open MT models (NLLB-200, IndicTrans2) degrade on tourism text compared with
general-domain text (FLORES-200), and test whether LoRA fine-tuning reduces the gap.

> Work in progress (Dec 2026 – Mar 2027).

## Layout

```
configs/    one YAML file per experiment
src/        scripts (translate.py, evaluate.py, ...)
data/       tourism test set (released) and fine-tuning scripts
outputs/    system translations (one folder per run)
results/    metric tables (CSV)
notebooks/  Colab launchers and analysis
tests/      quick tests that run without a GPU
```

## Running an experiment

Every result comes from one command:

```bash
pip install -r requirements.txt
python src/translate.py --config configs/nllb600m_flores-devtest_en-hi.yaml
```

Add `--limit 20` for a quick test. Outputs go to `outputs/<run_name>/` together with the exact
config and package versions used.

FLORES+ (`openlanguagedata/flores_plus`) and IndicTrans2 models are gated on Hugging Face:
accept their terms and log in (`huggingface-cli login`) first.

## Tests

```bash
python -m pytest tests/
```

## Licences

Wikivoyage-derived test data: CC BY-SA 4.0. NLLB-200: CC BY-NC 4.0 (non-commercial).
