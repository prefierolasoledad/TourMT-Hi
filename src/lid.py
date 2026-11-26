"""Language identification with GlotLID (fastText), used for the Maithili off-target check.

Hindi and Maithili are closely related and both use Devanagari, so a model asked for Maithili may
answer in Hindi ("off-target"). GlotLID labels each output with a language; we count the share that
is not labelled Maithili. check_lid.py first measures how well GlotLID separates Hindi from Maithili
on FLORES references, so we know how much to trust these labels.
"""
from __future__ import annotations

GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model_v4.bin"                                  # 783 MB
GLOTLID_REVISION = "e9592e0a9bc3b6f2a946de94faaa188d1dd6f522"  # pinned commit

# Our short language codes -> GlotLID labels
LID_LABELS = {"hi": "__label__hin_Deva", "mai": "__label__mai_Deva", "en": "__label__eng_Latn"}

_MODEL = None


def load_glotlid(model_path: str | None = None):
    """Load GlotLID once (downloads the pinned file from Hugging Face unless model_path is given)."""
    global _MODEL
    if _MODEL is None:
        import fasttext

        if model_path is None:
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(GLOTLID_REPO, GLOTLID_FILE, revision=GLOTLID_REVISION)
        _MODEL = fasttext.load_model(model_path)
    return _MODEL


def predict_labels(texts: list[str], model_path: str | None = None) -> list[str]:
    """Top-1 GlotLID label for each text, e.g. '__label__mai_Deva'."""
    model = load_glotlid(model_path)
    # fastText needs single-line input
    labels, _ = model.predict([t.replace("\n", " ") for t in texts], k=1)
    return [lab[0] if lab else "__label__unknown" for lab in labels]


def label_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def off_target(labels: list[str], expected_lang: str) -> dict:
    """Share of outputs not labelled as the expected language, and the share labelled Hindi."""
    expected = LID_LABELS[expected_lang]
    n = len(labels)
    return {
        "off_target_rate": round(sum(lab != expected for lab in labels) / n, 4),
        "hindi_rate": round(sum(lab == LID_LABELS["hi"] for lab in labels) / n, 4),
        "label_counts": label_counts(labels),
    }
