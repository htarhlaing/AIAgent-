from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from threading import RLock

from AI_agent.utils.config_handler import agent_config
from AI_agent.utils.path_tool import get_abs_path


@dataclass(frozen=True)
class UsageRecord:
    user_id: str
    month: str
    feature: str
    efficiency: str
    consumption: str
    comparison: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class RecordsService:
    """Loads quoted/multiline CSV records once and exposes typed lookups."""

    def __init__(self, csv_path: str | None = None) -> None:
        self.csv_path = csv_path or get_abs_path(agent_config["external_data_path"])
        self._records: dict[tuple[str, str], UsageRecord] | None = None
        self._lock = RLock()

    def _load(self) -> dict[tuple[str, str], UsageRecord]:
        with self._lock:
            if self._records is not None:
                return self._records

            records: dict[tuple[str, str], UsageRecord] = {}
            with open(self.csv_path, "r", encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    record = UsageRecord(
                        user_id=row["用户ID"].strip(),
                        month=row["时间"].strip(),
                        feature=row["特征"].strip(),
                        efficiency=row["清洁效率"].strip(),
                        consumption=row["耗材"].strip(),
                        comparison=row["对比"].strip(),
                    )
                    records[(record.user_id, record.month)] = record
            self._records = records
            return records

    def get_record(self, user_id: str, month: str) -> UsageRecord | None:
        return self._load().get((user_id.strip(), month.strip()))

    def available_months(self, user_id: str) -> list[str]:
        return sorted(month for uid, month in self._load() if uid == user_id.strip())


records_service = RecordsService()
