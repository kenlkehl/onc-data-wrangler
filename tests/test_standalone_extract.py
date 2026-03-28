"""Tests for standalone extraction CLI cohort-style behavior."""

from argparse import Namespace
import json

import pandas as pd

from onc_data_wrangler import cli
from onc_data_wrangler.extraction.chunked import chunk_text_by_chars, ChunkedExtractor


def _make_args(tmp_path, input_name: str, output_name: str = "out.json"):
    return Namespace(
        input=str(tmp_path / input_name),
        output=str(tmp_path / output_name),
        ontology=["naaccr"],
        cancer_type="generic",
        vllm_url="http://localhost:8000/v1",
        model="test-model",
        provider="openai",
        api_key="none",
        text_column="text",
        patient_id_column="patient_id",
        date_column="date",
        note_type_column="note_type",
        items_per_call=50,
        format="json",
        max_tokens=1234,
        chunk_tokens=40,
        overlap_tokens=5,
        patient_workers=3,
    )


def test_prepare_standalone_notes_df_filters_and_orders_rows():
    df = pd.DataFrame(
        [
            {"patient_id": "b", "date": "2024-01-02", "text": "this note is long enough"},
            {"patient_id": "a", "date": "2024-01-03", "text": "keep me as well"},
            {"patient_id": "a", "date": "2024-01-01", "text": "short"},
            {"patient_id": "a", "date": "2024-01-02", "text": None},
        ]
    )

    notes_df, patient_order = cli._prepare_standalone_notes_df(
        df,
        patient_id_col="patient_id",
        text_col="text",
        date_col="date",
    )

    assert patient_order == ["a", "b"]
    assert notes_df["patient_id"].tolist() == ["a", "b"]
    assert notes_df["text"].tolist() == ["keep me as well", "this note is long enough"]


def test_chunk_text_by_chars_respects_note_boundaries():
    text = "A" * 170 + "\n--- note ---\n" + "B" * 170

    chunks = chunk_text_by_chars(
        text,
        chunk_size_chars=180,
        overlap_chars=0,
        boundary_window_chars=40,
    )

    assert len(chunks) >= 2
    assert chunks[0] == "A" * 170
    assert chunks[1].startswith("\n--- note ---\n")


def test_chunked_extractor_passes_max_tokens_to_extract_iterative():
    calls = {}

    class FakeExtractor:
        def extract_iterative(self, chunks, cancer_type=None, max_tokens=None, max_retries=None):
            calls["chunks"] = chunks
            calls["max_tokens"] = max_tokens
            calls["max_retries"] = max_retries
            return [{"naaccr": {"field": "value"}}]

    chunked = ChunkedExtractor(
        extractor=FakeExtractor(),
        tokenizer=None,
        chunk_size=20,
        overlap=0,
        max_retries=7,
        patient_workers=1,
        max_tokens=4321,
    )

    result = chunked.extract_patient("p1", "A" * 200)

    assert result["patient_id"] == "p1"
    assert result["extractions"] == [{"naaccr": {"field": "value"}}]
    assert calls["max_tokens"] == 4321
    assert calls["max_retries"] == 7
    assert len(calls["chunks"]) >= 2


def test_run_standalone_extract_uses_chunked_cohort_path(tmp_path, monkeypatch):
    input_path = tmp_path / "notes.csv"
    pd.DataFrame(
        [
            {"patient_id": "p2", "date": "2024-01-02", "text": "second patient note", "note_type": "path"},
            {"patient_id": "p1", "date": "2024-01-03", "text": "later note for p1", "note_type": "visit"},
            {"patient_id": "p1", "date": "2024-01-01", "text": "earlier note for p1", "note_type": "path"},
        ]
    ).to_csv(input_path, index=False)

    args = _make_args(tmp_path, "notes.csv")

    monkeypatch.setattr(cli, "_create_standalone_llm", lambda args: object())
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda model_name: object())

    captured = {}
    final_extractions = {
        "p1": [{"naaccr": {"field_a": "value_a"}}, {"_extraction_results": {}}],
        "p2": [{"naaccr": {"field_b": "value_b"}}, {"_extraction_results": {}}],
    }

    class FakeChunkedExtractor:
        def __init__(self, extractor, tokenizer, chunk_size, overlap, max_retries, patient_workers, max_tokens):
            captured["init"] = {
                "extractor": extractor,
                "tokenizer": tokenizer,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "max_retries": max_retries,
                "patient_workers": patient_workers,
                "max_tokens": max_tokens,
            }

        def extract_cohort(self, notes_df, output_dir, patient_id_column, text_column, date_column, type_column, resume):
            captured["extract_cohort"] = {
                "notes_df": notes_df.copy(),
                "output_dir": output_dir,
                "patient_id_column": patient_id_column,
                "text_column": text_column,
                "date_column": date_column,
                "type_column": type_column,
                "resume": resume,
            }
            return pd.DataFrame()

    class FakeCheckpointManager:
        def __init__(self, output_dir):
            captured["checkpoint_output_dir"] = output_dir

        def load_final_extractions(self):
            return final_extractions

    monkeypatch.setattr(
        "onc_data_wrangler.extraction.extractor.create_extractor",
        lambda llm_client, ontology_ids, cancer_type, items_per_call: "fake-extractor",
    )
    monkeypatch.setattr("onc_data_wrangler.extraction.chunked.ChunkedExtractor", FakeChunkedExtractor)
    monkeypatch.setattr("onc_data_wrangler.extraction.chunked.CheckpointManager", FakeCheckpointManager)

    cli._run_standalone_extract(args)

    assert captured["init"]["chunk_size"] == 40
    assert captured["init"]["overlap"] == 5
    assert captured["init"]["patient_workers"] == 3
    assert captured["init"]["max_tokens"] == 1234
    assert captured["extract_cohort"]["resume"] is False
    assert captured["extract_cohort"]["type_column"] == "note_type"
    assert captured["extract_cohort"]["notes_df"]["patient_id"].tolist() == ["p1", "p1", "p2"]
    assert captured["extract_cohort"]["notes_df"]["text"].tolist() == [
        "earlier note for p1",
        "later note for p1",
        "second patient note",
    ]

    with open(args.output) as f:
        output = json.load(f)

    assert list(output.keys()) == ["p1", "p2"]
    assert output["p1"][0]["naaccr"]["field_a"] == "value_a"
    assert output["p2"][0]["naaccr"]["field_b"] == "value_b"
