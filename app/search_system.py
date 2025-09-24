import logging
from typing import Dict, Any, List
from core.search_result import SearchResult
from core.custom_exceptions import (
    RateLimitExceeded, 
    QueryProcessingError, 
    SearchEngineError
)

logger = logging.getLogger(__name__)

class TermSearchSystem:
    """3단계 용어 검색 시스템 (쿼리 전처리 + 하이브리드 검색 + 결과 포맷팅)"""
    
    def __init__(self, 
                 query_processor,  # 타입 힌트에서 인터페이스만 제거
                 search_engine,    # 타입 힌트에서 인터페이스만 제거
                 cache_manager):   # 타입 힌트에서 인터페이스만 제거
        
        self.query_processor = query_processor
        self.search_engine = search_engine
        self.cache_manager = cache_manager
        
        logger.info("용어 검색 시스템 초기화 완료")
    
    def search(self, natural_query: str, user_id: str = None, k: int = 20) -> Dict[str, Any]:
        """3단계 검색 파이프라인 실행"""
        
        logger.info(f"검색 파이프라인 시작: '{natural_query}' (사용자: {user_id})")

        try:
            # Phase 1: 쿼리 전처리
            logger.debug("Phase 1: 쿼리 전처리 시작")
            refined_query = self.query_processor.process(natural_query, user_id)
            logger.info(f"쿼리 정제: '{natural_query}' → '{refined_query}'")
            
            # Phase 2 & 3: 하이브리드 검색
            logger.debug("Phase 2-3: 하이브리드 검색 시작")
            
            if not self.search_engine.is_ready():
                raise SearchEngineError("검색 엔진이 준비되지 않았습니다")
            
            candidates = self.search_engine.search(refined_query, k)
            
            if not candidates:
                return {
                    'success': False,
                    'message': '검색 조건에 맞는 용어를 찾을 수 없습니다.',
                    'original_query': natural_query,
                    'refined_query': refined_query,
                    'search_results': []
                }
            
            # 성공 응답
            return {
                'success': True,
                'original_query': natural_query,
                'refined_query': refined_query,
                'candidates_count': len(candidates),
                'search_results': candidates,
                'processing_info': {
                    'query_processor': type(self.query_processor).__name__,
                    'search_engine': type(self.search_engine).__name__,
                    'cache_manager': type(self.cache_manager).__name__
                }
            }
            
        except RateLimitExceeded as e:
            logger.warning(f"Rate limit 초과: {e}")
            return {
                'success': False,
                'error_type': 'rate_limit',
                'message': str(e),
                'original_query': natural_query
            }
            
        except QueryProcessingError as e:
            logger.error(f"쿼리 처리 실패: {e}")
            return {
                'success': False,
                'error_type': 'query_processing',
                'message': f'쿼리 처리 중 오류가 발생했습니다: {str(e)}',
                'original_query': natural_query
            }
            
        except SearchEngineError as e:
            logger.error(f"검색 엔진 오류: {e}")
            return {
                'success': False,
                'error_type': 'search_engine',
                'message': f'검색 중 오류가 발생했습니다: {str(e)}',
                'original_query': natural_query
            }
            
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}", exc_info=True)
            return {
                'success': False,
                'error_type': 'unknown',
                'message': f'시스템 오류가 발생했습니다: {str(e)}',
                'original_query': natural_query
            }
        
    def clear_cache(self) -> Dict[str, Any]:
        """캐시 전체 삭제"""
        try:
            self.cache_manager.clear_cache()
            logger.info("캐시 전체 삭제 완료")
            return {'success': True, 'message': '캐시가 삭제되었습니다'}
            
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {e}")
            return {'success': False, 'message': f'캐시 삭제 실패: {str(e)}'}
    
    def search_similar_terms(self, term_name: str, k: int = 10) -> List[SearchResult]:
        """특정 용어와 유사한 용어들 검색"""
        try:
            # 용어명으로 직접 검색
            results = self.search_engine.search(term_name, k)
            
            # 자기 자신 제외
            filtered_results = [
                result for result in results 
                if result.term.name.lower() != term_name.lower()
            ]
            
            return filtered_results[:k-1]  # 원하는 개수만큼 반환
            
        except Exception as e:
            logger.error(f"유사 용어 검색 실패: {e}")
            return []
    
    def format_results_for_display(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """검색 결과를 표시용 형태로 변환"""
        formatted = []
        
        for result in results:
            formatted.append({
                'rank': result.rank,
                'term_name': result.term.name,
                'term_description': result.term.description,
                'term_abbreviation': result.term.abbreviation,
                'term_domain': result.term.domain,
                'scores': {
                    'final_score': round(result.final_score, 3),
                    'semantic_score': round(result.semantic_score, 3),
                    'colbert_score': round(result.colbert_score, 3)
                },
                'matched_tokens': result.matched_tokens or []
            })
        
        return formatted
    
    def close(self):
        """시스템 종료 시 리소스 정리"""
        try:
            if hasattr(self.cache_manager, 'close'):
                self.cache_manager.close()
            
            logger.info("용어 추천 시스템 종료")
            
        except Exception as e:
            logger.error(f"시스템 종료 중 오류: {e}")