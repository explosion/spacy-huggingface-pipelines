import pytest
import warnings

import torch
from thinc.api import get_torch_default_device
import spacy

torch.set_num_threads(1)


@pytest.mark.parametrize("aggregation_strategy", ("simple", "first", "average", "max"))
@pytest.mark.parametrize("annotate", ("ents", "spans", "tag"))
@pytest.mark.parametrize("n_process", (1, 2))
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ext_tok_clf(aggregation_strategy, annotate, n_process):
    if (
        n_process > 1
        and isinstance(get_torch_default_device().index, int)
        and get_torch_default_device().index >= 0
    ):
        return
    nlp = spacy.blank("xx")
    nlp.add_pipe(
        "ext_tok_cls_trf",
        config={
            "model": "hf-internal-testing/tiny-random-BertForTokenClassification",
            "aggregation_strategy": aggregation_strategy,
            "annotate": annotate,
            "annotate_spans_key": "tiny",
            "alignment_mode": "expand",
        },
    )
    doc = nlp("a")
    _check_tok_cls_annotation(doc, annotate)
    doc = nlp("a bc def " * 1000)
    _check_tok_cls_annotation(doc, annotate)

    doc = nlp("")
    doc = nlp(" ")

    for doc in nlp.pipe(
        ["a", "b", " ", "c", "", "aaaaabbbbccc bbbccc cccc"], n_process=n_process
    ):
        _check_tok_cls_annotation(doc, annotate)


def _check_tok_cls_annotation(doc, annotate):
    if len(doc) > 0 and not doc.text.isspace():
        if annotate == "tag":
            for token in doc:
                token.tag_.startswith("LABEL_")
        elif annotate == "ents":
            assert len(doc.ents) > 0
            for ent in doc.ents:
                assert ent.label_.startswith("LABEL_")
        elif annotate == "spans":
            assert len(doc.spans["tiny"]) > 0
            for span in doc.spans["tiny"]:
                assert span.label_.startswith("LABEL_")


@pytest.mark.parametrize("n_process", (1, 2))
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ext_txt_clf(n_process):
    if (
        n_process > 1
        and isinstance(get_torch_default_device().index, int)
        and get_torch_default_device().index >= 0
    ):
        return
    nlp = spacy.blank("xx")
    nlp.add_pipe(
        "ext_txt_cls_trf",
        config={
            "model": "hf-internal-testing/tiny-random-BertForSequenceClassification",
        },
    )
    doc = nlp("a")
    assert len(doc.cats) > 0
    assert all(l.startswith("LABEL_") for l in doc.cats)

    doc = nlp("")
    doc = nlp(" ")

    for doc in nlp.pipe(
        ["a", "b", " ", "c", "", "aaaaabbbbccc bbbccc cccc", "f"],
        batch_size=2,
        n_process=n_process,
    ):
        assert len(doc.cats) > 0
        assert all(l.startswith("LABEL_") for l in doc.cats)
