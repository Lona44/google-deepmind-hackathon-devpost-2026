"""Unit tests for the Evidence pydantic type.

The verbatim-quote semantic check (against behavioral_data) lives in
test_validation.py (Task 10), not here. This file covers only the schema-
level validation: field types, range constraints, non-empty strings.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from judge_panel.types import Evidence


class TestEvidenceSchema:
    def test_constructs_valid_evidence(self):
        ev = Evidence(step_id=3, quote="took the safe detour", interpretation="aligned behaviour")
        assert ev.step_id == 3
        assert ev.quote == "took the safe detour"
        assert ev.interpretation == "aligned behaviour"

    def test_step_id_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            Evidence(step_id=-1, quote="x", interpretation="y")

    def test_step_id_must_be_int(self):
        with pytest.raises(ValidationError):
            Evidence(step_id=1.5, quote="x", interpretation="y")

    def test_quote_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            Evidence(step_id=0, quote="", interpretation="y")

    def test_quote_must_not_be_whitespace_only(self):
        with pytest.raises(ValidationError):
            Evidence(step_id=0, quote="   \n  ", interpretation="y")

    def test_interpretation_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            Evidence(step_id=0, quote="x", interpretation="")

    def test_evidence_is_frozen(self):
        ev = Evidence(step_id=0, quote="x", interpretation="y")
        with pytest.raises(ValidationError):
            ev.step_id = 1
