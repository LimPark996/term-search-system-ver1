import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SimpleQueryProcessor:
    """규칙 기반 간단한 쿼리 전처리기 (GPT 없이 동작)"""
    
    def __init__(self):
        # 구어체 → 표준어 변환 룩업 테이블
        self.replacements = {
            # 질문 패턴
            '어떻게 해': '방법',
            '뭐가 필요해': '항목',
            '어떤 거야': '종류',
            '뭔지 알려줘': '정의',
            
            # 구어체 표현
            '하는 거': '',
            '인 거': '',
            '은 거': '',
            '를 거': '',
            '이랑': '과',
            '주고받는': '교환',
            '저장할 때': '저장',
            '관리하는': '관리',
            
            # 줄임말
            'DB': '데이터베이스',
            'API': '인터페이스',
            'UI': '사용자인터페이스',
            'UX': '사용자경험',
            
            # 불필요한 표현 제거
            '좀': '',
            '그냥': '',
            '막': '',
            '진짜': '',
            '정말': '',
            '많이': '',
            '완전': ''
        }
    
    def process(self, query: str, user_id: Optional[str] = None) -> str:
        """규칙 기반 쿼리 정제"""
        try:
            # 1. 기본 정제
            cleaned = self._basic_clean(query)
            
            # 2. 구어체 변환
            refined = self._apply_replacements(cleaned)
            
            # 3. 최종 정리
            final = self._final_cleanup(refined)
            
            logger.info(f"간단 정제: '{query}' → '{final}'")
            return final if final.strip() else query
            
        except Exception as e:
            logger.error(f"쿼리 정제 실패: {e}")
            return query
    
    def _basic_clean(self, text: str) -> str:
        """기본적인 텍스트 정제"""
        # 특수문자 제거 (한글, 영문, 숫자만 유지)
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 정리
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def _apply_replacements(self, text: str) -> str:
        """룩업 테이블 기반 치환"""
        result = text
        
        for old, new in self.replacements.items():
            result = result.replace(old, new)
        
        return result
    
    def _final_cleanup(self, text: str) -> str:
        """최종 정리"""
        # 연속된 공백 재정리
        cleaned = ' '.join(text.split())
        
        # 너무 짧으면 원본 반환
        if len(cleaned.strip()) < 2:
            return text
        
        return cleaned
    
    def add_replacement(self, old_pattern: str, new_pattern: str):
        """새로운 치환 규칙 추가"""
        self.replacements[old_pattern] = new_pattern
        logger.info(f"치환 규칙 추가: '{old_pattern}' → '{new_pattern}'")