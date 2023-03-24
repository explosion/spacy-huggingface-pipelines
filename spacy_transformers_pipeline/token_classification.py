from typing import Callable, Iterable, Iterator, List, Literal, Optional
import warnings

from thinc.api import get_torch_default_device
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc, Span
from spacy import util

from transformers import pipeline


@Language.factory(
    "ext_tok_cls_trf",
    assigns=[],
    default_config={
        "model": "",
        "revision": "main",
        "aggregation_strategy": "average",
        "annotate": "ents",
        "annotate_spans_key": None,
        "get_spans": {
            "@span_getters": "spacy-transformers.strided_spans.v1",
            "window": 128,
            "stride": 96,
        },
        "alignment_mode": "strict",
        "scorer": None,
        "kwargs": {},
    },
    default_score_weights={},
)
def make_tok_cls_trf(
    nlp: Language,
    name: str,
    model: str,
    revision: str,
    # this is intentionally omitting "none" from the aggregation strategies
    aggregation_strategy: Literal["simple", "first", "average", "max"],
    get_spans: Callable,
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
            'nlp.add_pipe("ext_tok_cls_trf", config={"model": "dslim/bert-base-NER"})'
        )
    tf_pipeline = pipeline(
        task="token-classification",
        model=model,
        revision=revision,
        aggregation_strategy=aggregation_strategy,
        device=device,
        **kwargs,
    )
    return ExternalTokenClassificationTransformer(
        name=name,
        tf_pipeline=tf_pipeline,
        get_spans=get_spans,
        annotate=annotate,
        annotate_spans_key=annotate_spans_key,
        alignment_mode=alignment_mode,
        scorer=scorer,
    )


class ExternalTokenClassificationTransformer(Pipe):
    def __init__(
        self,
        name: str,
        tf_pipeline: pipeline,
        *,
        get_spans: Callable,
        annotate: Literal["ents", "pos", "spans", "tag"] = "ents",
        annotate_spans_key: Optional[str] = None,
        alignment_mode: str = "strict",
        scorer: Optional[Callable] = None,
    ):
        self.name = name
        self.tf_pipeline = tf_pipeline
        self.get_spans = get_spans
        self.annotate = annotate
        if self.annotate == "spans":
            if isinstance(annotate_spans_key, str):
                self.annotate_spans_key = annotate_spans_key
            else:
                raise ValueError(
                    "'annotate_spans_key' setting required to set spans annotations for ext_tok_cls_trf"
                )
        self.alignment_mode = alignment_mode
        self.scorer = scorer

    def __call__(self, doc: Doc) -> Doc:
        return next(self.pipe([doc]))

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        for docs in util.minibatch(stream, size=batch_size):
            input_spans = [span for spans in self.get_spans(docs) for span in spans]
            outputs = self._get_annotations_from_spans(input_spans)
            doc = docs[0]
            prev_ann_end = -1
            output_spans: List[Span] = []
            for input_span, output in zip(input_spans, outputs):
                if doc != input_span.doc:
                    doc = self._set_annotation_from_spans(doc, output_spans)
                    yield doc
                    doc = input_span.doc
                    prev_ann_end = -1
                    output_spans = []
                for ann in output:
                    if ann["start"] + input_span.start_char >= prev_ann_end:
                        output_span = doc.char_span(
                            ann["start"] + input_span.start_char,
                            ann["end"] + input_span.start_char,
                            label=ann["entity_group"],
                            alignment_mode=self.alignment_mode,
                        )
                        if (
                            output_span is not None
                            and output_span.start_char >= prev_ann_end
                        ):
                            output_spans.append(output_span)
                            prev_ann_end = ann["end"] + input_span.start_char
                        else:
                            warnings.warn(
                                f"Skipping annotation, {ann} is overlapping or can't be aligned for span {repr(input_span.text)}"
                            )
            doc = self._set_annotation_from_spans(doc, output_spans)
            yield doc

    def _get_annotations_from_spans(self, spans: List[Span]) -> List[List[dict]]:
        # TODO: warn when truncating? (I'm not sure you can detect this
        # easily through the current pipeline API)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return self.tf_pipeline([span.text for span in spans])
        except Exception:
            # TODO: better UX
            pass
        outputs = []
        for span in spans:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    outputs.append(self.tf_pipeline(span.text))
            except Exception:
                # TODO: better UX
                warnings.warn(f"Unable to process, skipping span {repr(span.text)}")
                outputs.append([])
        return outputs

    def _set_annotation_from_spans(self, doc: Doc, spans: List[Span]) -> Doc:
        if self.annotate == "ents":
            doc.set_ents(spans)
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
