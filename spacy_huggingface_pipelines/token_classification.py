from typing import Callable, Iterable, Iterator, List, Literal, Optional
import warnings

from thinc.api import get_torch_default_device
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc, Span, SpanGroup
from spacy import util

from transformers import pipeline


@Language.factory(
    "hf_token_pipe",
    assigns=[],
    default_config={
        "model": "",
        "revision": "main",
        "stride": 16,
        "aggregation_strategy": "average",
        "annotate": "ents",
        "annotate_spans_key": None,
        "alignment_mode": "strict",
        "scorer": None,
        "kwargs": {},
    },
    default_score_weights={},
)
def make_hf_token_pipe(
    nlp: Language,
    name: str,
    model: str,
    revision: str,
    # note that the tokenizer stride is the size of the overlap, not the size of
    # the stride
    stride: Optional[int],
    # this is intentionally omitting "none" from the aggregation strategies
    aggregation_strategy: Literal["simple", "first", "average", "max"],
    annotate: Literal["ents", "pos", "spans", "tag"],
    annotate_spans_key: Optional[str],
    alignment_mode: Literal["strict", "contract", "expand"],
    scorer: Optional[Callable],
    kwargs: dict,
):
    try:
        device = get_torch_default_device().index
        if device is None:
            device = -1
    except Exception:
        device = -1
    if model == "":
        raise ValueError(
            "No model provided. Specify the model in your config, e.g.:\n\n"
            'nlp.add_pipe("hf_token_pipe", config={"model": "dslim/bert-base-NER"})'
        )
    hf_pipeline = pipeline(
        task="token-classification",
        model=model,
        revision=revision,
        aggregation_strategy=aggregation_strategy,
        device=device,
        stride=stride,
        **kwargs,
    )
    return HFTokenPipe(
        name=name,
        hf_pipeline=hf_pipeline,
        annotate=annotate,
        annotate_spans_key=annotate_spans_key,
        alignment_mode=alignment_mode,
        scorer=scorer,
    )


class HFTokenPipe(Pipe):
    def __init__(
        self,
        name: str,
        hf_pipeline: pipeline,
        *,
        annotate: Literal["ents", "pos", "spans", "tag"] = "ents",
        annotate_spans_key: Optional[str] = None,
        alignment_mode: str = "strict",
        scorer: Optional[Callable] = None,
    ):
        self.name = name
        self.hf_pipeline = hf_pipeline
        self.annotate = annotate
        if self.annotate == "spans":
            if isinstance(annotate_spans_key, str):
                self.annotate_spans_key = annotate_spans_key
            else:
                raise ValueError(
                    "'annotate_spans_key' setting required to set spans annotations for hf_token_pipe"
                )
        self.alignment_mode = alignment_mode
        self.scorer = scorer

    def __call__(self, doc: Doc) -> Doc:
        return next(self.pipe([doc]))

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        for docs in util.minibatch(stream, size=batch_size):
            outputs = self._get_annotations(docs)
            for doc, output in zip(docs, outputs):
                output_spans = SpanGroup(doc, attrs={"scores": []})
                prev_ann_end = 0
                for ann in output:
                    if ann["start"] >= prev_ann_end:
                        output_span = doc.char_span(
                            ann["start"],
                            ann["end"],
                            label=ann["entity_group"],
                            alignment_mode=self.alignment_mode,
                        )
                        if (
                            output_span is not None
                            and output_span.start_char >= prev_ann_end
                        ):
                            output_spans.append(output_span)
                            output_spans.attrs["scores"].append(ann["score"])
                            prev_ann_end = ann["end"]
                        else:
                            text_excerpt = (
                                doc.text
                                if len(doc.text) < 100
                                else doc.text[:100] + "..."
                            )
                            warnings.warn(
                                f"Skipping annotation, {ann} is overlapping or can't be aligned for doc '{text_excerpt}'"
                            )
                self._set_annotation_from_spans(doc, output_spans)
                yield doc

    def _get_annotations(self, docs: List[Doc]) -> List[List[dict]]:
        with warnings.catch_warnings():
            # the PipelineChunkIterator does not report its length correctly,
            # leading to many spurious warnings from torch
            warnings.filterwarnings(
                "ignore", message="Length of IterableDataset", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore",
                message="You seem to be using the pipelines sequentially on GPU",
                category=UserWarning,
            )
            if len(docs) > 1:
                try:
                    return self.hf_pipeline([doc.text for doc in docs])
                except Exception:
                    warnings.warn(
                        "Unable to process texts as batch, backing off to processing texts individually"
                    )
            outputs = []
            for doc in docs:
                try:
                    outputs.append(self.hf_pipeline(doc.text))
                except Exception:
                    text_excerpt = (
                        doc.text if len(doc.text) < 100 else doc.text[:100] + "..."
                    )
                    warnings.warn(
                        f"Unable to process, skipping annotation for doc '{text_excerpt}'"
                    )
                    outputs.append([])
            return outputs

    def _set_annotation_from_spans(self, doc: Doc, spans: SpanGroup) -> Doc:
        if self.annotate == "ents":
            doc.set_ents(list(spans))
        elif self.annotate == "spans":
            doc.spans[self.annotate_spans_key] = spans
        elif self.annotate == "tag":
            for span in spans:
                for token in span:
                    token.tag_ = span.label_
        elif self.annotate == "pos":
            for span in spans:
                for token in span:
                    token.pos_ = span.label_
        return doc

    # dummy serialization methods
    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes_data, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
