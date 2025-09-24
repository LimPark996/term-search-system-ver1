import pickle
import hashlib
import time
import logging
from typing import Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheManager:
    """파일 기반 캐시 매니저"""
    
    def __init__(self, cache_dir: str = "cache", ttl: int = 604800):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = ttl  # 기본 7일
        
        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_dir = self.cache_dir / "embeddings"
        self.embedding_dir.mkdir(exist_ok=True)
        
        logger.info(f"파일 캐시 매니저 초기화: {self.cache_dir}")

    def _hash_text(self, text: str) -> str:
        """텍스트 해시 생성"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, text: str) -> Path:
        """캐시 파일 경로 생성"""
        hash_key = self._hash_text(text)
        return self.embedding_dir / f"{hash_key}.pkl"
    
    def _is_cache_valid(self, file_path: Path, ttl: int) -> bool:
        """캐시 파일이 유효한지 확인"""
        if not file_path.exists():
            return False
        
        # 파일 수정 시간 확인
        file_mtime = file_path.stat().st_mtime
        current_time = time.time()
        
        return (current_time - file_mtime) < ttl
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """임베딩 조회"""
        file_path = self._get_cache_file_path(text)
        
        if not self._is_cache_valid(file_path, self.default_ttl):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                embedding = pickle.load(f)
            logger.debug(f"캐시 HIT: {text[:30]}...")
            return embedding
        except (FileNotFoundError, pickle.PickleError, EOFError) as e:
            logger.debug(f"캐시 파일 읽기 실패: {e}")
            # 손상된 캐시 파일 삭제
            if file_path.exists():
                file_path.unlink()
            return None
    
    def set_embedding(self, text: str, embedding: np.ndarray, ttl: int = None):
        """임베딩 저장"""
        file_path = self._get_cache_file_path(text)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(embedding, f)
            logger.debug(f"캐시 저장: {text[:30]}...")
        except Exception as e:
            logger.error(f"캐시 파일 저장 실패: {e}")
    
    def clear_cache(self):
        """캐시 전체 삭제"""
        try:
            for file_path in self.embedding_dir.glob("*.pkl"):
                file_path.unlink()
            logger.info("캐시 전체 삭제 완료")
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {e}")
    
    def cleanup_expired_cache(self):
        """만료된 캐시 파일들 정리"""
        try:
            current_time = time.time()
            deleted_count = 0
            
            for file_path in self.embedding_dir.glob("*.pkl"):
                file_mtime = file_path.stat().st_mtime
                if (current_time - file_mtime) >= self.default_ttl:
                    file_path.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"만료된 캐시 파일 {deleted_count}개 삭제")
        except Exception as e:
            logger.error(f"캐시 정리 실패: {e}")
        
    def close(self):
        """캐시 매니저 종료 시 정리"""
        try:
            self.cleanup_expired_cache()
            logger.info("파일 캐시 매니저 종료")
        except Exception as e:
            logger.error(f"캐시 매니저 종료 중 오류: {e}")