from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Term:
    """용어 데이터 모델"""
    id: str
    name: str
    description: str
    abbreviation: str
    domain: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return f"Term(name='{self.name}', domain='{self.domain}')"
    
    def __repr__(self) -> str:
        return self.__str__()