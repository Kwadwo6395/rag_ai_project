import json
from pathlib import Path
from rag.logging_utils import QueryLogger


def test_query_logger_writes_ordered_stages(tmp_path):
    log = QueryLogger(log_dir=tmp_path)
    log.stage("query_received", {"query": "hello"})
    log.stage("retrieval", {"top_k": 3})
    path = log.flush()
    data = json.loads(Path(path).read_text())
    assert data["query_id"]
    assert [s["stage"] for s in data["stages"]] == ["query_received", "retrieval"]
    assert data["stages"][0]["payload"]["query"] == "hello"
