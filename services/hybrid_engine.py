import logging
from typing import List
import numpy as np
import faiss
from core.term import Term
from core.search_result import SearchResult
from services.embedding_service import EmbeddingService
from services.file_cache import CacheManager
from utils.text_utils import TextUtils

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """FAISS + ColBERT 하이브리드 검색 엔진"""
    
    def __init__(self, 
                 terms: List[Term],
                 embedding_service: EmbeddingService,
                 cache_manager: CacheManager,
                 semantic_weight: float = 0.6,
                 colbert_weight: float = 0.4,
                 search_threshold: float = 0.3):
        
        self.terms = terms
        self.embedding_service = embedding_service
        self.cache_manager = cache_manager
        self.text_utils = TextUtils()
        
        # 검색 파라미터
        self.semantic_weight = semantic_weight
        self.colbert_weight = colbert_weight
        self.search_threshold = search_threshold
        
        # 인덱스 및 메타데이터
        self.semantic_index = None
        self.token_metadata = []
        self.semantic_embeddings = []
        self.semantic_texts = []
        self._ready = False

        # 용어 데이터 해시 생성
        self.terms_hash = self.cache_manager._generate_terms_hash(terms)
        logger.info(f"용어 데이터 해시: {self.terms_hash}")
    
    def initialize(self) -> bool:
        """검색 엔진 초기화"""
        try:
            logger.info("하이브리드 검색 엔진 초기화 시작...")

            # 캐시에서 인덱스 로드 시도
            if self._load_from_cache():
                logger.info("캐시에서 인덱스 로드 성공")
                self._ready = True
                return True
            
            # 캐시 미스
            logger.info("캐시 미스 - 새 인덱스 구축 시작")
            if self._build_fresh_index():
                # 구축 완료 후 캐시에 저장
                self._save_to_cache()
                self._ready = True
                logger.info(f"검색 엔진 초기화 완료: {len(self.terms)}개 용어")
                return True

            return False
            
        except Exception as e:
            logger.error(f"검색 엔진 초기화 실패: {e}")
            self._ready = False
            return False
        
    def _load_from_cache(self) -> bool:
        """캐시에서 인덱스 로드"""
        try:
            cached_data = self.cache_manager.get_search_index(self.terms_hash) # 전체 용어 사전에 대한 캐시 한번에 조회
            if cached_data is None:
                return False
            
            # 캐시된 데이터 복원
            self.semantic_index = cached_data['semantic_index']
            self.token_metadata = cached_data['token_metadata']
            self.semantic_embeddings = cached_data['semantic_embeddings']
            self.semantic_texts = cached_data['semantic_texts']
            
            logger.info(f"캐시에서 인덱스 복원: {len(self.token_metadata)}개 메타데이터")
            return True
            
        except Exception as e:
            logger.error(f"캐시 로드 실패: {e}")
            return False

    def _save_to_cache(self):
        """현재 인덱스를 캐시에 저장"""
        try:
            self.cache_manager.set_search_index(
                terms_hash=self.terms_hash,
                semantic_index=self.semantic_index,
                token_metadata=self.token_metadata,
                semantic_embeddings=self.semantic_embeddings,
                semantic_texts=self.semantic_texts
            )
            logger.info("인덱스 캐시 저장 완료")
            
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")

    def is_ready(self) -> bool:
        """검색 엔진 준비 상태"""
        return self._ready
    
    def search(self, query: str, k: int = 20) -> List[SearchResult]:
        """하이브리드 검색 수행"""
        if not self._ready:
            raise RuntimeError("검색 엔진이 초기화되지 않았습니다")
        
        return self._perform_search(query, k)
    
    def _perform_search(self, query: str, k: int) -> List[SearchResult]:
        """실제 검색 로직"""
        # 1. 사용자 질의에 대한 임베딩 생성
        query_semantic_emb = self.embedding_service.get_embedding(query)
        query_tokens = self.text_utils.tokenize(query)
        
        # 2. 사용자 질의에 대한 토큰별 임베딩 병렬 생성
        query_token_embs = []
        if query_tokens:
            query_token_embs = [
                self.embedding_service.get_embedding(token) 
                for token in query_tokens
            ]
        
        # 3. FAISS로 의미적 유사도 기반 후보 추출
        top_candidates = min(100, len(self.token_metadata))
        scores, indices = self.semantic_index.search(
            query_semantic_emb.reshape(1, -1).astype('float32'), 
            top_candidates
        )
        
        # 4. 후보"들"에 대해 ColBERT 점수 계산
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.token_metadata):
                continue
            
            term_data = self.token_metadata[idx] # {용어:정의} 한 덩어리에 대해서 진행! 전체 문서 아님!!!
            term = self.terms[term_data['term_index']]
            
            # FAISS 의미적 점수
            semantic_score = float(scores[0][i])
            
            # ColBERT 점수 계산
            colbert_score = self._calculate_colbert_score(
                query_token_embs, 
                term_data['token_embeddings']
            )
            
            # 하이브리드 점수 -> 하나의 {용어:정의}와 쿼리 간의 최종 점수
            final_score = (self.semantic_weight * semantic_score + 
                          self.colbert_weight * colbert_score)
            
            # 임계값 필터링 -> 최종 점수가 검색 임계값 이상인 경우에만 후보로 추가
            if final_score >= self.search_threshold:
                matched_tokens = [
                    token for token in query_tokens 
                    if token in term_data['tokens']
                ]
                
                result = SearchResult(
                    term=term,
                    semantic_score=semantic_score,
                    colbert_score=colbert_score,
                    final_score=final_score,
                    rank=0,
                    matched_tokens=matched_tokens
                )
                candidates.append(result)
        
        # 5. 최종 점수로 정렬 및 순위 부여
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        for rank, candidate in enumerate(candidates[:k], 1):
            candidate.rank = rank
        
        logger.info(f"검색 완료: {query} -> {len(candidates[:k])}개 결과")
        return candidates[:k]
    
    def _build_fresh_index(self) -> bool:
        """새 인덱스 구축"""
        try:
            logger.info("의미적 임베딩 생성 중...")
            
            # 1. 의미적 임베딩 생성
            self.semantic_texts = [f"{term.name}: {term.description}" for term in self.terms] # 전체 용어에 대해서 "용어명:정의" 문자열 형태의 리스트로 생성
            self.semantic_embeddings = []
            
            # 배치 처리로 진행상황 표시
            batch_size = 50
            total_batches = (len(self.semantic_texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(self.semantic_texts))
                
                batch_texts = self.semantic_texts[start_idx:end_idx] # 배치 단위로 문자열들을 batch_texts에 담기
                batch_embeddings = [
                    self.embedding_service.get_embedding(text) 
                    for text in batch_texts # 배치 안에서 각 문자열에 대해 임베딩 생성
                ]
                self.semantic_embeddings.extend(batch_embeddings) # sementic_embeddings에 임베딩들 추가
                
                logger.info(f"의미적 임베딩 진행: {end_idx}/{len(self.semantic_texts)}")
            
            # 2. FAISS 인덱스 생성
            if self.semantic_embeddings:
                embeddings_array = np.array(self.semantic_embeddings, dtype=np.float32)
                dimension = embeddings_array.shape[1]
                
                self.semantic_index = faiss.IndexFlatIP(dimension)
                self.semantic_index.add(embeddings_array)
                logger.info(f"FAISS 인덱스 생성: {embeddings_array.shape}")
            
            # 3. 토큰 메타데이터 생성
            logger.info("토큰 메타데이터 생성 중...")
            self._build_token_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"인덱스 구축 실패: {e}")
            return False
                
    def _build_token_metadata(self):
        """토큰 메타데이터 구축 (기존 로직 유지)"""
        # 모든 토큰 수집
        all_tokens = []
        token_mapping = []
        
        for i, text in enumerate(self.semantic_texts):
            tokens = self.text_utils.tokenize(text) # 각 "용어명:정의" 문자열을 토큰화 진행함
            
            if tokens: # 하나의 문자열에 대한 토큰화 진행 결과가 있다면,
                start_idx = len(all_tokens) # 해당 문자열의 인덱스 시작 위치, 끝 위치 저장
                all_tokens.extend(tokens)
                end_idx = len(all_tokens)
                
                token_mapping.append({ # 하나의 문자열에 대한 토큰화 결과를 token_mapping에 저장
                    'term_index': i,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'tokens': tokens
                })
            else:
                token_mapping.append({ # 하나의 문자열에 대한 토큰화 결과가 없다면, 빈 리스트로 저장
                    'term_index': i,
                    'start_idx': -1,
                    'end_idx': -1,
                    'tokens': []
                })
        
        # 모든 토큰의 임베딩 생성 (100개씩 배치 처리)
        logger.info(f"토큰 임베딩 생성 중: {len(all_tokens)}개")
        all_token_embeddings = []
        if all_tokens:
            batch_size = 100
            total_batches = (len(all_tokens) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(all_tokens))
                
                batch_tokens = all_tokens[start_idx:end_idx]
                batch_embeddings = [
                    self.embedding_service.get_embedding(token) 
                    for token in batch_tokens # 100개씩의 배치 중 각 토큰에 대해 임베딩 생성
                ]
                all_token_embeddings.extend(batch_embeddings) # all_token_embeddings에 임베딩들 추가
                
                logger.info(f"토큰 임베딩 진행: {end_idx}/{len(all_tokens)}")
        
        # 용어별 토큰 임베딩 분배
        self.token_metadata = []
        for mapping_info in token_mapping:
            term_index = mapping_info['term_index']
            start_idx = mapping_info['start_idx']
            end_idx = mapping_info['end_idx']
            
            if start_idx >= 0:
                token_embeddings = all_token_embeddings[start_idx:end_idx]
            else:
                token_embeddings = []
            
            self.token_metadata.append({ # 앞에서 만든 all_token_embeddings에서 각 용어별로 토큰 임베딩 분배 (기준: 하나의 문자열)
                'term_index': term_index,
                'tokens': mapping_info['tokens'],
                'token_embeddings': token_embeddings,
                'semantic_embedding': self.semantic_embeddings[term_index]
            })
        
        logger.info(f"토큰 메타데이터 생성 완료: {len(self.token_metadata)}개")
        
    def _calculate_colbert_score(self, query_embeds: List[np.ndarray], doc_embeds: List[np.ndarray]) -> float:
        """ColBERT 스타일 점수 계산"""
        if not query_embeds or not doc_embeds:
            return 0.0
        
        total_score = 0
        for query_embed in query_embeds:
            max_sim = max(
                self._cosine_similarity(query_embed, doc_embed) 
                for doc_embed in doc_embeds
            )
            total_score += max_sim
        
        return total_score / len(query_embeds)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0