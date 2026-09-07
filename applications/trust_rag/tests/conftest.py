import json
from dataclasses import replace

import pytest

from applications.trust_rag.config import Settings
from applications.trust_rag.corpus import build_corpus


@pytest.fixture
def corpus(tmp_path):
    record = {
        "evidence_id": "ev1",
        "doc_id": "doc1",
        "source_title": "银行监管报表",
        "quote": "总资产为125亿元。",
        "location": {"type": "table_cell", "sheet": "表一", "cell": "C5"},
        "metadata": {
            "file_name": "report.xlsx",
            "metric": "总资产",
            "cell_value": "125",
            "unit": "亿元",
            "qa_type": "MUST_NOT_LEAK",
            "correct_option": "MUST_NOT_LEAK",
        },
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    settings = replace(Settings(), data_dir=str(tmp_path))
    build_corpus(source, tmp_path / "corpus.sqlite")
    return settings
