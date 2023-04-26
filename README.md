<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-huggingface-pipelines: Use pretrained transformer models for text and token classification

This package provides [spaCy](https://github.com/explosion/spaCy) components to
use pretrained
[Hugging Face Transformers pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
for inference only.

[![PyPi](https://img.shields.io/pypi/v/spacy-huggingface-pipelines.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/spacy-huggingface-pipelines)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-huggingface-pipelines/all.svg?style=flat-square&logo=github)](https://github.com/explosion/spacy-huggingface-pipelines/releases)

## Features

- Apply pretrained transformers models like
  [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) and
  [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).

## 🚀 Installation

Installing the package from pip will automatically install all dependencies,
including PyTorch and spaCy.

```bash
pip install -U pip setuptools wheel
pip install spacy-huggingface-pipelines
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

This module provides spaCy wrappers for the inference-only transformers
[`TokenClassificationPipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TokenClassificationPipeline)
and
[`TextClassificationPipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextClassificationPipeline)
pipelines.

The models are downloaded on initialization from the
[Hugging Face Hub](https://huggingface.co/models) if they're not already in your
local cache, or alternatively they can be loaded from a local path.

Note that the transformer model data **is not saved with the pipeline** when you
call `nlp.to_disk`, so if you are loading pipelines in an environment with
limited internet access, make sure the model is available in your
[transformers cache directory](https://huggingface.co/docs/transformers/main/en/installation#cache-setup)
and enable offline mode if needed.

### Token classification

Config settings for `hf_token_pipe`:

```ini
[components.hf_token_pipe]
factory = "hf_token_pipe"
model = "dslim/bert-base-NER"     # Model name or path
revision = "main"                 # Model revision
aggregation_strategy = "average"  # "simple", "first", "average", "max"
stride = 16                       # If stride >= 0, process long texts in
                                  # overlapping windows of the model max
                                  # length. The value is the length of the
                                  # window overlap in transformer tokenizer
                                  # tokens, NOT the length of the stride.
kwargs = {}                       # Any additional arguments for
                                  # TokenClassificationPipeline
alignment_mode = "strict"         # "strict", "contract", "expand"
annotate = "ents"                 # "ents", "pos", "spans", "tag"
annotate_spans_key = null         # Doc.spans key for annotate = "spans"
scorer = null                     # Optional scorer
```

#### `TokenClassificationPipeline` settings

- `model`: The model name or path.
- `revision`: The model revision. For production use, a specific git commit is
  recommended instead of the default `main`.
- `stride`: For `stride >= 0`, the text is processed in overlapping windows
  where the `stride` setting specifies the number of overlapping tokens between
  windows (NOT the stride length). If `stride` is `None`, then the text may be
  truncated. `stride` is only supported for fast tokenizers.
- `aggregation_strategy`: The aggregation strategy determines the word-level
  tags for cases where subwords within one word do not receive the same
  predicted tag. See:
  https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline.aggregation_strategy
- `kwargs`: Any additional arguments to
  [`TokenClassificationPipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline).

#### spaCy settings

- `alignment_mode` determines how transformer predictions are aligned to spaCy
  token boundaries as described for
  [`Doc.char_span`](https://spacy.io/api/doc#char_span).
- `annotate` and `annotate_spans_key` configure how the annotation is saved to
  the spaCy doc. You can save the output as `token.tag_`, `token.pos_` (only for
  UPOS tags), `doc.ents` or `doc.spans`.

#### Examples

1. Save named entity annotation as `Doc.ents`:

```python
import spacy
nlp = spacy.blank("en")
nlp.add_pipe("hf_token_pipe", config={"model": "dslim/bert-base-NER"})
doc = nlp("My name is Sarah and I live in London")
print(doc.ents)
# (Sarah, London)
```

2. Save named entity annotation as `Doc.spans[spans_key]`:

```python
import spacy
nlp = spacy.blank("en")
nlp.add_pipe(
    "hf_token_pipe",
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
    "hf_token_pipe",
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
    "hf_token_pipe",
    config={"model": "vblagoje/bert-english-uncased-finetuned-pos", "annotate": "pos"},
)
doc = nlp("My name is Sarah and I live in London")
print([t.pos_ for t in doc])
# ['PRON', 'NOUN', 'AUX', 'PROPN', 'CCONJ', 'PRON', 'VERB', 'ADP', 'PROPN']
```

### Text classification

Config settings for `hf_text_pipe`:

```ini
[components.hf_text_pipe]
factory = "hf_text_pipe"
model = "distilbert-base-uncased-finetuned-sst-2-english"  # Model name or path
revision = "main"                 # Model revision
kwargs = {}                       # Any additional arguments for
                                  # TextClassificationPipeline
scorer = null                     # Optional scorer
```

The input texts are truncated according to the transformers model max length.

#### `TextClassificationPipeline` settings

- `model`: The model name or path.
- `revision`: The model revision. For production use, a specific git commit is
  recommended instead of the default `main`.
- `kwargs`: Any additional arguments to
  [`TextClassificationPipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline).

#### Example

```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe(
    "hf_text_pipe",
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
    do_something(doc)
```

If the component runs into an error processing a batch (e.g. on an empty text),
`nlp.pipe` will back off to processing each text individually. If it runs into
an error on an individual text, a warning is shown and the doc is returned
without additional annotation.

Switch to GPU:

```python
import spacy
spacy.require_gpu()

for doc in nlp.pipe(texts):
    do_something(doc)
```

## Bug reports and issues

Please report bugs in the
[spaCy issue tracker](https://github.com/explosion/spaCy/issues) or open a new
thread on the [discussion board](https://github.com/explosion/spaCy/discussions)
for other issues.
