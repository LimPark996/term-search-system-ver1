import logging
import numpy as np
from openai import OpenAI
from services.file_cache import CacheManager
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException

logger = logging.getLogger(__name__)

class EmbeddingService:
    """OpenAI 임베딩 생성 서비스 (캐시 + Circuit Breaker)"""
    
    def __init__(self, api_key: str, cache_manager: CacheManager):
        self.client = OpenAI(api_key=api_key)
        self.cache = cache_manager
        
        # Circuit Breaker 설정
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,    # 3번 실패하면 차단
            recovery_timeout=30,    # 30초 후 복구 시도
            expected_exception=Exception
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """임베딩 조회 (캐시 우선, 없으면 API 호출)"""
        
        # 1. 캐시 확인 (캐시된 임베딩을 가져온다)
        cached_embedding = self.cache.get_embedding(text)
        if cached_embedding is not None:
            logger.debug(f"캐시 HIT: {text[:30]}...")
            return cached_embedding
        logger.debug(f"캐시 MISS: {text[:30]}...")
        
        # 캐시된 임베딩이 만약 존재하지 않는다면
        # 2. Circuit Breaker를 통한 API 호출 (_generate_embedding 메서드를 호출한다)
        try:
            embedding = self.circuit_breaker.call(self._generate_embedding, text)
            
            # 3. 캐시에 저장 - text와 embedding과 ttl을 지정하여 저장한다.
            self.cache.set_embedding(text, embedding, ttl=604800)  # ttl = 7일
            
            return embedding
            
        except CircuitBreakerOpenException:
            # API 장애 시 제로 벡터 반환
            logger.warning(f"OpenAI API 장애로 제로 벡터 반환: {text[:30]}...")
            return np.zeros(1536, dtype=np.float32)

    def _generate_embedding(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩 생성"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # L2 정규화
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            logger.debug(f"임베딩 생성 완료: {text[:30]}...")
            return embedding # 텍스트에 대한 임베딩을 생성한다.
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def get_circuit_breaker_state(self):
        """Circuit Breaker 상태 조회"""
        return self.circuit_breaker.get_state()
    
    def reset_circuit_breaker(self):
        """Circuit Breaker 강제 리셋"""
        self.circuit_breaker.force_close()
        logger.info("임베딩 서비스 Circuit Breaker 리셋됨")