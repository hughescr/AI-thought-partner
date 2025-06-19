import json
import sys
import warnings

import spacy

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)

# Pass input text from stdin
text = sys.stdin.read()

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
doc = nlp(text)
types = ["PERSON", "ORG", "PRODUCT", "LOC", "WORK_OF_ART", "GPE", "FAC", "EVENT"]
# Get a list of all the entities in the doc where the label is in types
ents = [{'text': e.text, 'label': e.label_} for e in doc.ents if e.label_ in types]
# You may want to output each entity as a structured JSON object. For example:
# {
#   "name": "Alice",
#   "type": "PERSON",
#   "description": null,
#   "aliases": []
# }
# For now, let's just return simple objects. You can enrich them as needed later.

# Map spaCy labels directly to known entity types
def map_spacy_label_to_type(label):
    # Map spaCy label to one of ['Person', 'Location', 'Organization', ...] etc.
    if label == "PERSON":
        return "Person"
    elif label == "ORG":
        return "Organization"
    elif label in ["LOC", "GPE", "FAC"]:
        return "Location"
    elif label == "PRODUCT":
        return "Object"
    elif label == "WORK_OF_ART":
        return "Work of art"
    elif label == "EVENT":
        return "Event"
    else:
        return "Concept"  # fallback

entity_objects = []
for e in ents:
    entity_objects.append(
        {
            "name": e['text'].strip(),
            "type": map_spacy_label_to_type(
                e['label']
            ),
            "description": e['label'],
            "aliases": [],
        }
    )
# Filter for uniqueness
entity_objects = [
    dict(
        (key, tuple(value) if isinstance(value, list) else value)
        for key, value in d.items()
    )
    for d in entity_objects
]
entity_objects = [dict(t) for t in {tuple(d.items()) for d in entity_objects}]

# Print the JSON array
print(json.dumps(entity_objects, ensure_ascii=False))
