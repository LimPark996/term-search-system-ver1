import logging
from typing import Optional
from openai import OpenAI
from core.custom_exceptions import RateLimitExceeded, QueryProcessingError
from utils.rate_limiter import MemoryRateLimiter
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException

logger = logging.getLogger(__name__)

class GPTQueryProcessor:
    """GPT 기반 쿼리 전처리기 (Rate Limiting + Circuit Breaker)"""
    
    def __init__(self, 
                 api_key: str, 
                 rate_limiter: Optional[MemoryRateLimiter] = None,
                 max_requests_per_minute: int = 10):
        self.client = OpenAI(api_key=api_key)
        self.rate_limiter = rate_limiter or MemoryRateLimiter()
        self.max_requests = max_requests_per_minute
        
        # Circuit Breaker 설정 (GPT API 장애 대응)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,    # 5번 실패하면 차단
            recovery_timeout=60,    # 60초 후 복구 시도
            expected_exception=Exception
        )
    
    def process(self, query: str, user_id: Optional[str] = None) -> str:
        """쿼리 전처리 (Rate Limiting + Circuit Breaker 적용)"""
        
        # 1. Rate Limiting 확인
        if user_id and not self.is_rate_limited(user_id):
            raise RateLimitExceeded(f"사용자 {user_id}의 요청 제한을 초과했습니다")
        
        # 2. Circuit Breaker를 통한 GPT 호출
        try:
            return self.circuit_breaker.call(self._gpt_process, query)
        except CircuitBreakerOpenException:
            # GPT 서비스 장애 시 간단한 fallback 사용
            logger.warning("GPT 서비스 장애로 인해 간단한 전처리 사용")
            return self._simple_fallback(query)
    
    def is_rate_limited(self, user_id: str) -> bool:
        """Rate limit 상태 확인 (허용되면 True, 제한되면 False)"""
        return self.rate_limiter.is_allowed(user_id, self.max_requests)
    
    def _gpt_process(self, query: str) -> str:
        """실제 GPT API 호출"""
        prompt = f"""다음 자연어 질의에서 핵심 키워드를 추출하고 명확한 검색어로 변환해주세요.

입력 질의: "{query}"

조건:
- 구어체, 은어, 줄임말을 표준 용어로 변환
- 핵심 개념을 명확히 식별
- 불필요한 조사, 감정 표현 제거
- 검색에 적합한 키워드 조합으로 구성
- 20-30자 내외로 간결하게 작성

예시:
"고객 번호 관리하는 거 어떻게 해?" → "고객번호 관리 방법"
"직원 정보 저장할 때 뭐가 필요해?" → "직원정보 저장 항목"
"웹사이트에서 다른 시스템이랑 데이터 주고받는 거" → "웹사이트 시스템간 데이터 교환"

변환된 검색어만 출력하세요:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            
            if response and response.choices:
                refined_query = response.choices[0].message.content.strip()
                logger.info(f"GPT 쿼리 정제: '{query}' → '{refined_query}'")
                return refined_query
            else:
                raise QueryProcessingError("GPT 응답이 없습니다")
                
        except Exception as e:
            logger.error(f"GPT API 호출 실패: {e}")
            raise QueryProcessingError(f"GPT 처리 실패: {str(e)}")
    
    def _simple_fallback(self, query: str) -> str:
        """GPT 실패 시 간단한 텍스트 정제"""
        import re
        
        # 기본적인 텍스트 정제
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', query)  # 특수문자 제거
        cleaned = ' '.join(cleaned.split())       # 공백 정리
        
        # 구어체 → 표준어 간단 변환
        replacements = {
            '어떻게 해': '방법',
            '뭐가 필요해': '항목',
            '하는 거': '',
            '이랑': '과',
            '주고받는': '교환'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        cleaned = ' '.join(cleaned.split())  # 다시 공백 정리
        
        logger.info(f"간단 정제: '{query}' → '{cleaned}'")
        return cleaned if cleaned else query