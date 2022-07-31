from dataclasses import dataclass, field
from typing import List


@dataclass
class WandbConfig:
    entity: str
    prj_name: str
    notes: str = ""
    tags: List[str] = None
    enabled: bool = True