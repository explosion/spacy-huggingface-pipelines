<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-transformers-pipeline: Use pretrained transformer models for text and token classification

This package provides [spaCy](https://github.com/explosion/spaCy) components to
use pretrained
[transformers pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
for inference only.

[![PyPi](https://img.shields.io/pypi/v/spacy-transformers-pipeline.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/spacy-transformers-pipeline)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-transformers-pipeline/all.svg?style=flat-square&logo=github)](https://github.com/explosion/spacy-transformers-pipeline/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

## Features

- Apply pretrained transformers models like
  [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) and
  [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).

## 🚀 Installation

Installing the package from pip will automatically install all dependencies,
including PyTorch and spaCy.

```bash
pip install -U pip setuptools wheel
pip install spacy-transformers-pipeline
```

For GPU installation, follow the
[spaCy installation quickstart with GPU](https://spacy.io/usage/), e.g.

```bash
pip install -U spacy[cuda-autodetect]
```

If you are having trouble installing PyTorch, follow the
[instructions](https://pytorch.org/get-started/locally/) on the official website
for your specific operating system and requirements.

## 📖 Documentation

This module provides spaCy wrappers for around the inference-only transformers
[`TokenClassificationPipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TokenClassificationPipeline)
and
[`TextClassificationPipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextClassificationPipeline)
pipelines.

The models are downloaded on initialization from the Hugging Face Hub if they're
not already in your local cache, or alternatively they can be loaded from a
local path.

Note that the transformer model data is not saved with the pipeline when you
call `nlp.to_disk`, so if you are loading pipelines in an environment with
limited internet access, make sure the model is available in your
[transformers cache directory](https://huggingface.co/docs/transformers/main/en/installation#cache-setup)
and enable offline mode if needed.

### Token classification

Config settings for `ext_tok_cls_trf`:

```ini
[components.ext_tok_cls_trf]
factory = "ext_tok_cls_trf"
aggregation_strategy = "average"  # "simple", "first", "average", "max"
alignment_mode = "strict"         # "strict", "contract", "expand"
annotate = "ents"                 # "ents", "pos", "spans", "tag"
annotate_spans_key = null         # Doc.spans key for annotate = "spans"
model = "dslim/bert-base-NER"     # Model name or path
revision = "main"                 # Model revision
scorer = null                     # Optional scorer, e.g. {"@scorers": "spacy.ner_scorer.v1"}
stride = 16                       # Use stride > 0 to process long texts in
                                  # overlapping windows of the model max length.
                                  # The value is the length of the window
                                  # overlap in transformer tokenizer tokens,
                                  # NOT the length of the stride.
```

You can save the output as `token.tag_`, `token.pos_` (only for UPOS tags),
`doc.ents` or `doc.spans`.

1. Save named entity annotation as `Doc.ents`:

```python
import spacy
nlp = spacy.blank("en")
nlp.add_pipe("ext_tok_cls_trf", config={"model": "dslim/bert-base-NER"})
doc = nlp("My name is Sarah and I live in London")
print(doc.ents)
# (Sarah, London)
```

2. Save named entity annotation as `Doc.spans[spans_key]`:

```python
import sapcy
nlp = spacy.blank("en")
nlp.add_pipe(
    "ext_tok_cls_trf",
    config={
        "model": "dslim/bert-base-NER",
        "annotate": "spans",
        "annotate_spans_key": "bert-base-ner",
    },
)
doc = nlp("My name is Sarah and I live in London")
print(doc.spans["bert-base-ner"])
# [Sarah, London]
```

3. Save fine-grained tags as `Token.tag`:

```python
import spacy
nlp = spacy.blank("en")
nlp.add_pipe(
    "ext_tok_cls_trf",
    config={
        "model": "QCRI/bert-base-multilingual-cased-pos-english",
        "annotate": "tag",
    },
)
doc = nlp("My name is Sarah and I live in London")
print([t.tag_ for t in doc])
# ['PRP$', 'NN', 'VBZ', 'NNP', 'CC', 'PRP', 'VBP', 'IN', 'NNP']
```

4. Save coarse-grained tags as `Token.pos`:

```python
import spacy
nlp = spacy.blank("en")
nlp.add_pipe(
    "ext_tok_cls_trf",
    config={"model": "vblagoje/bert-english-uncased-finetuned-pos", "annotate": "pos"},
)
doc = nlp("My name is Sarah and I live in London")
print([t.pos_ for t in doc])
# ['PRON', 'NOUN', 'AUX', 'PROPN', 'CCONJ', 'PRON', 'VERB', 'ADP', 'PROPN']
```

### Text classification

Config settings for `ext_txt_cls_trf`:

```ini
[components.ext_txt_cls_trf]
factory = "ext_txt_cls_trf"
model = "distilbert-base-uncased-finetuned-sst-2-english"  # Model name or path
revision = "main"                                          # Model revision
scorer = null                                              # Optional scorer.
```

The input texts are truncated according to the transformers model max length.

```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe(
    "ext_txt_cls_trf",
    config={"model": "distilbert-base-uncased-finetuned-sst-2-english"},
)
doc = nlp("This is great!")
print(doc.cats)
# {'POSITIVE': 0.9998694658279419, 'NEGATIVE': 0.00013048505934420973}
```

### Batching and GPU

Both token and text classification support batching with `nlp.pipe`:

```python
for doc in nlp.pipe(texts, batch_size=256):
    my_func(doc)
```

Switch to GPU:

```python
import spacy
spacy.require_gpu()

for doc in nlp.pipe(texts):
    my_func(doc)
```
