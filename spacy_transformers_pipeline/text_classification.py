from typing import Callable, Iterable, Iterator, List, Optional
import warnings

from thinc.api import get_torch_default_device
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy import util

from transformers import pipeline


@Language.factory(
    "trf_text_pipe",
    assigns=[],
    default_config={
        "model": "",
        "revision": "main",
        "scorer": None,
        "kwargs": {},
    },
    default_score_weights={},
)
def make_trf_text_pipe(
    nlp: Language,
    name: str,
    model: str,
    revision: str,
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
            'nlp.add_pipe("trf_text_pipe", config={"model": "distilbert-base-uncased-finetuned-sst-2-english"})'
        )
    tf_pipeline = pipeline(
        task="text-classification",
        model=model,
        revision=revision,
        device=device,
        truncation=True,
        **kwargs,
    )
    return TrfTextPipe(
        name=name,
        tf_pipeline=tf_pipeline,
        scorer=scorer,
    )


class TrfTextPipe(Pipe):
    def __init__(
        self, name: str, tf_pipeline: pipeline, *, scorer: Optional[Callable] = None
    ):
        self.name = name
        self.tf_pipeline = tf_pipeline
        self.scorer = scorer

    def __call__(self, doc: Doc) -> Doc:
        """Set transformers pipeline output on a spaCy Doc.

        doc (Doc): The doc to process.
        RETURNS (Doc): The spaCy Doc object.
        """
        return next(self.pipe([doc]))

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        for docs in util.minibatch(stream, size=batch_size):
            outputs = self._get_annotations(docs, batch_size=batch_size)
            for doc, output in zip(docs, outputs):
                doc.cats.update({a["label"]: a["score"] for a in output})
                yield doc

    def _get_annotations(self, docs: List[Doc], batch_size) -> List[List[dict]]:
        # TODO: warn when truncating? (I'm not sure you can detect this
        # easily through the current pipeline API)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return self.tf_pipeline(
                    [doc.text for doc in docs], batch_size=batch_size, top_k=None
                )
        except Exception:
            # TODO: better UX
            pass
        outputs = []
        for doc in docs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    outputs.append(self.tf_pipeline(doc.text, top_k=None))
            except Exception:
                # TODO: better UX
                warnings.warn(f"Unable to process, skipping doc {repr(doc)}")
                outputs.append([])
        return outputs

    # dummy serialization methods
    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes_data, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
