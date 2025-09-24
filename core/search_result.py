from dataclasses import dataclass
from typing import Optional, List
from core.term import Term

@dataclass
class SearchResult:
    """검색 결과 모델"""
    term: Term
    semantic_score: float
    colbert_score: float
    final_score: float
    rank: int
    matched_tokens: Optional[List[str]] = None
    
    def __str__(self) -> str:
        return f"SearchResult(term='{self.term.name}', score={self.final_score:.3f}, rank={self.rank})"
    
    def __repr__(self) -> str:
        return self.__str__()