import json
import sys
from pathlib import Path
from types import SimpleNamespace
import types
import io

# ruff: noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a minimal stub for the ``spacy`` module so ``ner`` can be imported
spacy_stub = types.SimpleNamespace(prefer_gpu=lambda: None, load=lambda _: None)
sys.modules.setdefault("spacy", spacy_stub)

import ner


class StubEnt:
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label_ = label


class StubNLP:
    def __init__(self, ents):
        self.doc = SimpleNamespace(ents=ents)

    def __call__(self, text: str):
        self.called_with = text
        return self.doc


def test_map_spacy_label_to_type_all_branches():
    assert ner.map_spacy_label_to_type("PERSON") == "Person"
    assert ner.map_spacy_label_to_type("ORG") == "Organization"
    for lbl in ["LOC", "GPE", "FAC"]:
        assert ner.map_spacy_label_to_type(lbl) == "Location"
    assert ner.map_spacy_label_to_type("PRODUCT") == "Object"
    assert ner.map_spacy_label_to_type("WORK_OF_ART") == "Work of art"
    assert ner.map_spacy_label_to_type("EVENT") == "Event"
    assert ner.map_spacy_label_to_type("OTHER") == "Concept"


def test_extract_entities_filters_by_label():
    doc = SimpleNamespace(ents=[StubEnt("Alice", "PERSON"), StubEnt("ACME", "ORG"), StubEnt("foo", "NORP")])
    result = ner.extract_entities(doc)
    assert result == [{"text": "Alice", "label": "PERSON"}, {"text": "ACME", "label": "ORG"}]


def test_build_entity_objects_and_deduplication():
    ents = [{"text": "Alice", "label": "PERSON"}, {"text": "Alice", "label": "PERSON"}]
    objs = ner.build_entity_objects(ents)
    assert objs == [{"name": "Alice", "type": "Person", "description": "PERSON", "aliases": []}]


def test_process_text_uses_nlp_and_returns_entities():
    stub_nlp = StubNLP([StubEnt("Alice", "PERSON")])
    output = ner.process_text("hello", nlp=stub_nlp)
    assert stub_nlp.called_with == "hello"
    assert output == [{"name": "Alice", "type": "Person", "description": "PERSON", "aliases": []}]


def test_main_prints_json_and_returns_data(capsys):
    stub_nlp = StubNLP([StubEnt("Alice", "PERSON")])
    result = ner.main("hello", nlp=stub_nlp)
    captured = capsys.readouterr().out
    assert json.loads(captured) == result


def test_process_text_loads_spacy_when_nlp_none(monkeypatch):
    loaded = {}

    def fake_load(name):
        loaded["name"] = name
        return StubNLP([StubEnt("Bob", "PERSON")])

    monkeypatch.setattr(ner.spacy, "load", fake_load)
    monkeypatch.setattr(ner.spacy, "prefer_gpu", lambda: loaded.setdefault("pref", True))

    result = ner.process_text("hi")
    assert loaded["name"] == "en_core_web_trf"
    assert loaded["pref"] is True
    assert result == [{"name": "Bob", "type": "Person", "description": "PERSON", "aliases": []}]


def test_main_reads_from_stdin(monkeypatch, capsys):
    stub_nlp = StubNLP([StubEnt("Alice", "PERSON")])
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello"))
    result = ner.main(nlp=stub_nlp)
    captured = capsys.readouterr().out
    assert json.loads(captured) == result
