# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/logging_utils.py
import json
import time
import uuid
from pathlib import Path


class QueryLogger:
    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.query_id = uuid.uuid4().hex[:12]
        self.started_at = time.time()
        self.stages: list[dict] = []

    def stage(self, name: str, payload: dict) -> None:
        self.stages.append({
            "stage": name,
            "t": time.time() - self.started_at,
            "payload": payload,
        })

    def flush(self) -> str:
        path = self.log_dir / f"query_{int(self.started_at)}_{self.query_id}.json"
        path.write_text(json.dumps({
            "query_id": self.query_id,
            "started_at": self.started_at,
            "stages": self.stages,
        }, indent=2, default=str))
        return str(path)
