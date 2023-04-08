import pytest
import sys

import medspacy

import spacy
from spacy.tokens import Span
from medspacy.context import ConTextModifier, ConTextRule
from medspacy.section_detection import Section

nlp = spacy.blank("en")
text = "There is no evidence of pneumonia."
doc = nlp(text)
doc[0].is_sent_start=True
for token in doc[1:]:
    token.is_sent_start=False

class TestVisualization:
    def test_gather_target_vis_data(self):
        from medspacy.visualization import _gather_target_vis_data
        ent = Span(doc, 5, 6, "PNEUMONIA")
        ent_data = _gather_target_vis_data(ent)
        assert isinstance(ent_data, dict)
        assert len(ent_data) == 3
        assert ent_data == {
        "start": ent.start_char,
        "end": ent.end_char,
        "label": "PNEUMONIA",
    }

    def test_gather_modifier_vis_data(self):
        from medspacy.visualization import _gather_modifier_vis_data
        rule = ConTextRule("no", "NEGATED_EXISTENCE", "FORWARD")
        modifier = ConTextModifier(rule, 3, 4, doc)
        ent_data = _gather_modifier_vis_data(modifier, doc)
        span = doc[modifier.modifier_span[0]: modifier.modifier_span[1]]
        assert isinstance(ent_data, dict)
        assert len(ent_data) == 3
        assert ent_data == {
        "start": span.start_char,
        "end": span.end_char,
        "label": "NEGATED_EXISTENCE",
    }

    def test_gather_section_vis_data(self):
        from medspacy.visualization import _gather_section_vis_data
        section = Section(category="EXAMPLE", title_start=0, title_end=1,
                          body_start=1, body_end=len(doc))
        span = doc[0:1]
        ent_data = _gather_section_vis_data(section, doc)
        assert isinstance(ent_data, dict)
        assert len(ent_data) == 3
        assert ent_data == {
            "start": span.start_char,
            "end": span.end_char,
            "label": "<<EXAMPLE>>",
        }


