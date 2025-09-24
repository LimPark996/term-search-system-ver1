import re
import logging
from typing import List
from konlpy.tag import Okt

logger = logging.getLogger(__name__)

class TextUtils:
    """텍스트 처리 유틸리티"""
    
    def __init__(self):
        try:
            self.okt = Okt()
            self.morpheme_available = True
        except Exception as e:
            logger.warning(f"형태소 분석기 초기화 실패: {e}")
            self.okt = None
            self.morpheme_available = False
    
    def tokenize(self, text: str) -> List[str]:
        """한국어 텍스트 토큰화"""
        if not text or not text.strip():
            return []
        
        # 1. 정규식 기반 기본 토큰화
        basic_tokens = self._basic_tokenize(text)
        
        # 2. 형태소 분석 (가능한 경우)
        morpheme_tokens = []
        if self.morpheme_available and self.okt:
            morpheme_tokens = self._morpheme_tokenize(text)
        
        # 3. 결합 및 중복 제거
        all_tokens = list(set(basic_tokens + morpheme_tokens))
        
        # 4. 필터링 (길이 2 이상, 의미있는 토큰만)
        filtered_tokens = [
            token for token in all_tokens 
            if len(token) >= 2 and self._is_meaningful_token(token)
        ]
        
        return filtered_tokens
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """정규식 기반 기본 토큰화"""
        # 한글, 영문, 숫자 조합만 추출
        tokens = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        
        # 길이 2 이상인 토큰만 선택
        return [token for token in tokens if len(token) >= 2]
    
    def _morpheme_tokenize(self, text: str) -> List[str]:
        """형태소 분석 기반 토큰화"""
        try:
            # 품사 태깅
            pos_result = self.okt.pos(text)
            
            # 의미있는 품사만 선택
            meaningful_pos = ['Noun', 'Verb', 'Adjective', 'Alpha', 'Number']
            tokens = [
                word for word, pos in pos_result 
                if pos in meaningful_pos and len(word) >= 2
            ]
            
            return tokens
            
        except Exception as e:
            logger.debug(f"형태소 분석 실패: {e}")
            return []
    
    def _is_meaningful_token(self, token: str) -> bool:
        """의미있는 토큰인지 판단"""
        # 너무 반복적인 문자는 제외
        if len(set(token)) < 2 and len(token) > 3:
            return False
        
        # 숫자만으로 구성된 토큰은 제외 (단, 영문+숫자 조합은 포함)
        if token.isdigit():
            return False
        
        # 불용어 제외
        stopwords = {
            '에서', '에게', '으로', '로서', '처럼', '같이', '하지만', 
            '그러나', '하지만', '그래서', '그런데', '그리고'
        }
        
        if token in stopwords:
            return False
        
        return True
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not text:
            return ""
        
        # 1. HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 특수문자 정리 (일부 유지)
        text = re.sub(r'[^\w\s가-힣.-]', ' ', text)
        
        # 3. 연속된 공백 정리
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """키워드 추출"""
        tokens = self.tokenize(text)
        
        # 토큰 빈도 계산
        token_freq = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        # 빈도순 정렬
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 키워드 반환
        keywords = [token for token, freq in sorted_tokens[:max_keywords]]
        
        return keywords
    
    def similarity_score(self, text1: str, text2: str) -> float:
        """간단한 텍스트 유사도 계산 (Jaccard)"""
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 and not tokens2:
            return 1.0
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def normalize_query(self, query: str) -> str:
        """쿼리 정규화"""
        # 기본 정제
        normalized = self.clean_text(query)
        
        # 소문자 변환 (영문)
        normalized = re.sub(r'[A-Z]', lambda m: m.group().lower(), normalized)
        
        # 연속된 같은 문자 축약 (예: "좋아아아" -> "좋아")
        normalized = re.sub(r'(.)\1{2,}', r'\1\1', normalized)
        
        return normalized