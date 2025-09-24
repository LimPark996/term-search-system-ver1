import pickle
import hashlib
import time
import logging
from typing import Dict, Optional, Any, List
import numpy as np
from pathlib import Path
import faiss

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

        # 인덱스 캐시 디렉토리 생성
        self.index_dir = self.cache_dir / "indexes"
        self.index_dir.mkdir(exist_ok=True)
        
        logger.info(f"파일 캐시 매니저 초기화: {self.cache_dir}")

    def _hash_text(self, text: str) -> str:
        """텍스트 해시 생성"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _generate_terms_hash(self, terms: List) -> str:
            """용어 목록의 해시 생성 - 데이터 변경 감지용"""
            # 용어 ID와 내용을 결합하여 해시 생성
            terms_content = []
            for term in terms:
                content = f"{term.id}:{term.name}:{term.description}:{term.abbreviation}:{term.domain}"
                terms_content.append(content)
            
            combined = "|".join(sorted(terms_content))
            return hashlib.sha256(combined.encode('utf-8')).hexdigest() # 용어 목록 전체 덩어리의 해시값

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
    
    # FAISS 인덱스 캐시 메서드들
    def get_search_index(self, terms_hash: str) -> Optional[Dict[str, Any]]:
        """검색 인덱스 캐시 조회"""
        index_file = self.index_dir / f"{terms_hash}.index" # FAISS 인덱스 파일 조회
        metadata_file = self.index_dir / f"{terms_hash}_meta.pkl" # 메타데이터 파일 조회
        
        if not (index_file.exists() and metadata_file.exists()): # 둘 중 하나라도 없으면 캐시 미스
            logger.debug(f"인덱스 캐시 MISS: {terms_hash}")
            return None
        
        # TTL 확인
        if not (self._is_cache_valid(index_file, self.default_ttl) and # 둘 중 하나라도 만료되면?
                self._is_cache_valid(metadata_file, self.default_ttl)):
            logger.debug(f"인덱스 캐시 만료: {terms_hash}")
            self._remove_index_cache(terms_hash) # 만료된 캐시 파일 삭제
            return None
        
        try:
            # FAISS 인덱스 로드 (의미적 유사도?가 높은 순서에서 낮은 순서로 인덱스 정렬)
            semantic_index = faiss.read_index(str(index_file))
            
            # 메타데이터 로드
            with open(metadata_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            logger.info(f"인덱스 캐시 HIT: {terms_hash}")
            return {
                'semantic_index': semantic_index,
                'token_metadata': cache_data['token_metadata'],
                'semantic_embeddings': cache_data['semantic_embeddings'],
                'semantic_texts': cache_data['semantic_texts']
            }
            
        except Exception as e:
            logger.error(f"인덱스 캐시 로드 실패: {e}")
            self._remove_index_cache(terms_hash) # 손상된 캐시 파일 삭제
            return None

    def set_search_index(self, terms_hash: str, 
                    semantic_index, 
                    token_metadata: List[Dict[str, Any]],
                    semantic_embeddings: List[np.ndarray],
                    semantic_texts: List[str]):
        """검색 인덱스 캐시 저장"""
        index_file = self.index_dir / f"{terms_hash}.index"
        metadata_file = self.index_dir / f"{terms_hash}_meta.pkl"
        
        try:
            # FAISS 인덱스 저장
            faiss.write_index(semantic_index, str(index_file))
            
            # 메타데이터 저장
            cache_data = {
                'token_metadata': token_metadata,
                'semantic_embeddings': semantic_embeddings,
                'semantic_texts': semantic_texts,
                'cached_at': time.time(),
                'terms_hash': terms_hash
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"인덱스 캐시 저장 완료: {terms_hash}")
            
        except Exception as e:
            logger.error(f"인덱스 캐시 저장 실패: {e}")
            # 실패 시 부분적으로 생성된 파일들 정리
            for file_path in [index_file, metadata_file]:
                if file_path.exists():
                    file_path.unlink()
        
    def _remove_index_cache(self, terms_hash: str):
        """특정 인덱스 캐시 파일들 삭제"""
        files_to_remove = [
            self.index_dir / f"{terms_hash}.index",
            self.index_dir / f"{terms_hash}_meta.pkl"
        ]
        
        for file_path in files_to_remove:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"인덱스 캐시 파일 삭제 실패: {file_path}: {e}")
            
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
                if (current_time - file_mtime) >= self.default_ttl: # ttl 초과 -> 만료(expired)이다.
                    file_path.unlink() # 파일 삭제
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