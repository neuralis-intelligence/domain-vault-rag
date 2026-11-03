"""Tests for the line-of-business registry."""

import textwrap

import pytest

from app.config import settings
from app.lob import UnknownLOBError, load_registry


def write(tmp_path, toml):
    path = tmp_path / "lobs.toml"
    path.write_text(textwrap.dedent(toml))
    return path


class TestLoadRegistry:
    def test_profile_fills_in_defaults(self, registry, data_dir):
        cards = registry.get("cards")
        assert cards.source == data_dir / "complaints.parquet"
        assert cards.columns["company"].startswith("Company legal name")
        assert cards.value_columns == ("company",)
        assert len(cards.sql_examples) == 4
        assert cards.rag.collection == "complaint_narratives"
        assert cards.rag.text_key == "narrative"
        assert cards.rag.fixed_filter == {"product": "Credit card"}

    def test_entry_overrides_profile(self, tmp_path):
        path = write(
            tmp_path,
            """
            [lobs.cards]
            name = "Cards"
            profile = "cfpb_complaints"
            source = "x.parquet"
            table = "cards"
            value_columns = ["issue"]
            rag.collection = "my_collection"
            """,
        )
        cards = load_registry(path, root=tmp_path).get("cards")
        assert cards.value_columns == ("issue",)
        assert cards.rag.collection == "my_collection"
        assert cards.rag.text_column == "consumer_complaint_narrative"  # still from profile

    def test_rag_false_turns_off_profile_rag(self, tmp_path):
        path = write(
            tmp_path,
            """
            [lobs.loans]
            name = "Loans"
            profile = "cfpb_complaints"
            source = "x.parquet"
            table = "loans"
            rag = false
            """,
        )
        loans = load_registry(path, root=tmp_path).get("loans")
        assert loans.rag is None
        assert loans.sql_examples  # the rest of the profile still applies

    def test_lob_without_profile_or_rag(self, registry):
        tx = registry.get("transactions")
        assert tx.rag is None
        assert tx.columns == {"amount": "Amount in USD"}
        assert tx.sql_examples == ()

    def test_default_and_lookup(self, registry):
        assert registry.default == "cards"
        assert registry.get(None).id == "cards"
        assert registry.get("mortgage").name == "Mortgages"

    def test_unknown_lob(self, registry):
        with pytest.raises(UnknownLOBError, match="Available: cards, mortgage"):
            registry.get("crypto")

    @pytest.mark.parametrize(
        ("toml", "message"),
        [
            ('[lobs.a]\nname = "A"\nsource = "a.parquet"\ntable = "a; DROP"\n', "identifier"),
            ('[lobs.a]\nname = "A"\nsource = "a.parquet"\n', "missing 'table'"),
            ('[lobs.a]\nname = "A"\nsource = "a"\ntable = "a"\nprofile = "nope"\n', "profile"),
            (
                '[lobs.a]\nname = "A"\nsource = "a"\ntable = "t"\n'
                '[lobs.b]\nname = "B"\nsource = "b"\ntable = "t"\n',
                "same table",
            ),
            ('default_lob = "z"\n[lobs.a]\nname = "A"\nsource = "a"\ntable = "a"\n', "default_lob"),
            (
                '[lobs.a]\nname = "A"\nsource = "a"\ntable = "a"\nrag.collection = "c"\n',
                "missing text_column",
            ),
            ("", "no \\[lobs"),
        ],
    )
    def test_invalid_registry(self, tmp_path, toml, message):
        with pytest.raises(ValueError, match=message):
            load_registry(write(tmp_path, toml), root=tmp_path)


def test_project_registry_is_valid():
    registry = load_registry(settings.lob_registry_path)
    assert registry.default in registry.lobs
    assert {"credit_card", "mortgage"} <= set(registry.lobs)
    for lob in registry.lobs.values():
        assert lob.example_questions, f"{lob.id} has no example questions for the UI"
