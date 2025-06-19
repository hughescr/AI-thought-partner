"""Simple named-entity extraction helper using spaCy.

The original version executed the spaCy pipeline at import time and read from
``stdin``.  The logic has been refactored into small functions to make testing
straightforward. ``main`` retains the original behaviour when run as a script.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Iterable, Sequence

import spacy

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)

ENTITY_LABELS: list[str] = [
    "PERSON",
    "ORG",
    "PRODUCT",
    "LOC",
    "WORK_OF_ART",
    "GPE",
    "FAC",
    "EVENT",
]


def map_spacy_label_to_type(label: str) -> str:
    """Map spaCy entity labels to coarser types."""
    if label == "PERSON":
        return "Person"
    if label == "ORG":
        return "Organization"
    if label in {"LOC", "GPE", "FAC"}:
        return "Location"
    if label == "PRODUCT":
        return "Object"
    if label == "WORK_OF_ART":
        return "Work of art"
    if label == "EVENT":
        return "Event"
    return "Concept"


def extract_entities(doc: object, labels: Sequence[str] = ENTITY_LABELS) -> list[dict]:
    """Return text/label pairs for matching entities in ``doc``."""
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in getattr(doc, "ents", [])
        if ent.label_ in labels
    ]


def build_entity_objects(entities: Iterable[dict]) -> list[dict]:
    """Convert raw entities to our structured representation."""
    objects = [
        {
            "name": e["text"].strip(),
            "type": map_spacy_label_to_type(e["label"]),
            "description": e["label"],
            "aliases": [],
        }
        for e in entities
    ]
    # deduplicate
    normalised = [
        {k: tuple(v) if isinstance(v, list) else v for k, v in obj.items()}
        for obj in objects
    ]
    deduped = [dict(t) for t in {tuple(d.items()) for d in normalised}]
    for obj in deduped:
        if isinstance(obj.get("aliases"), tuple):
            obj["aliases"] = list(obj["aliases"])
    return deduped


def process_text(text: str, nlp=None) -> list[dict]:
    """Run ``text`` through the spaCy pipeline and return entity objects."""
    if nlp is None:
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_trf")
    doc = nlp(text)
    ents = extract_entities(doc)
    return build_entity_objects(ents)


def main(text: str | None = None, nlp=None) -> list[dict]:
    """Entry point used by the command line."""
    if text is None:
        text = sys.stdin.read()
    entities = process_text(text, nlp)
    print(json.dumps(entities, ensure_ascii=False))
    return entities


if __name__ == "__main__":  # pragma: no cover - invoke CLI when run directly
    main()
