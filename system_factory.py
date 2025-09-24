import logging
from core.custom_exceptions import DataLoadError

# 직접 클래스 import (인터페이스 제거)
from services.gpt_processor import GPTQueryProcessor
from services.simple_processor import SimpleQueryProcessor
from services.hybrid_engine import HybridSearchEngine
from services.embedding_service import EmbeddingService
from services.file_cache import CacheManager  
from services.sheets_loader import SheetsDataLoader, ExcelDataLoader

from utils.rate_limiter import MemoryRateLimiter

logger = logging.getLogger(__name__)

class SystemFactory:
    """시스템 컴포넌트 팩토리 - 의존성 주입과 객체 조립"""
    
    @staticmethod
    def create_system(config: dict) -> 'TermSearchSystem':
        """시스템 생성"""
        logger.info("시스템 생성 중...")
        
        # 1. 캐시 매니저 생성
        cache_manager = SystemFactory._create_cache_manager(config)
        
        # 2. 데이터 로더 생성
        data_loader = SystemFactory._create_data_loader(config)
        
        # 3. 용어 데이터 로드
        terms = data_loader.load_terms()
        if not terms:
            raise DataLoadError("용어 데이터를 로드할 수 없습니다")
        
        logger.info(f"용어 데이터 로드 완료: {len(terms)}개")
        
        # 4. 임베딩 서비스 생성
        embedding_service = EmbeddingService(
            api_key=config['openai_api_key'],
            cache_manager=cache_manager
        )
        
        # 5. 검색 엔진 생성
        search_engine = HybridSearchEngine(
            terms=terms,
            embedding_service=embedding_service,
            cache_manager=cache_manager,
            semantic_weight=config.get('semantic_weight', 0.6),
            colbert_weight=config.get('colbert_weight', 0.4),
            search_threshold=config.get('search_threshold', 0.3)
        )
        
        # 6. 쿼리 프로세서 생성 (GPT 우선, 실패시 Simple)
        query_processor = SystemFactory._create_query_processor(config, cache_manager)
        
        # 7. 검색 엔진 초기화
        search_engine.initialize()
        if not search_engine.initialize():
            raise RuntimeError("검색 엔진 초기화에 실패했습니다")
        
        # 8. 메인 시스템 조립
        from app.search_system import TermSearchSystem
        system = TermSearchSystem(
            query_processor=query_processor,
            search_engine=search_engine,
            cache_manager=cache_manager
        )
        
        logger.info(f"시스템 생성 완료: {len(terms)}개 용어")
        return system
    
    @staticmethod
    def _create_cache_manager(config: dict) -> CacheManager:
        """파일 캐시 매니저 생성"""
        cache_dir = config.get('cache_dir', 'cache')
        ttl = config.get('cache_ttl', 604800)  # 7일
        
        cache_manager = CacheManager(cache_dir=cache_dir, ttl=ttl)
        
        # 시작 시 만료된 캐시 정리
        cache_manager.cleanup_expired_cache()
        
        return cache_manager
    
    @staticmethod
    def _create_data_loader(config: dict):
        """데이터 로더 생성"""
        
        # Google Sheets 설정 확인
        sheets_config = config.get('google_sheets', {})
        spreadsheet_id = sheets_config.get('spreadsheet_id')
        
        if spreadsheet_id:
            column_mapping = sheets_config.get('column_mapping', {})
            
            loader = SheetsDataLoader(
                spreadsheet_id=spreadsheet_id,
                column_mapping=column_mapping,
                credentials_file=sheets_config.get('credentials_file', 'credentials.json'),
                token_file=sheets_config.get('token_file', 'token.json')
            )
            
            if loader.is_available():
                logger.info("Google Sheets 데이터 로더 생성")
                return loader
            else:
                logger.warning("Google Sheets 접근 불가, Excel 파일로 전환")
        
        # Excel 파일로 fallback
        excel_file = config.get('excel_file', 'terms.xlsx')
        column_mapping = config.get('column_mapping', {})
        
        loader = ExcelDataLoader(excel_file, column_mapping)
        
        if loader.is_available():
            logger.info(f"Excel 데이터 로더 생성: {excel_file}")
            return loader
        else:
            raise RuntimeError(f"데이터 소스를 찾을 수 없습니다: Sheets({spreadsheet_id}), Excel({excel_file})")
    
    @staticmethod
    def _create_query_processor(config: dict, cache_manager: CacheManager):
        """쿼리 프로세서 생성"""
        
        # GPT 프로세서 시도
        openai_api_key = config.get('openai_api_key')
        
        if openai_api_key:
            try:
                rate_limiter = MemoryRateLimiter()
                
                gpt_processor = GPTQueryProcessor(
                    api_key=openai_api_key,
                    rate_limiter=rate_limiter,
                    max_requests_per_minute=config.get('max_requests_per_minute', 10) # 1분당 최대 API 요청 수: 10회
                )
                
                logger.info("GPT 쿼리 프로세서 생성 성공")
                return gpt_processor
                
            except Exception as e:
                logger.warning(f"GPT 프로세서 생성 실패, Simple 프로세서로 전환: {e}")
        
        # Simple 프로세서로 fallback
        logger.info("Simple 쿼리 프로세서 생성")
        return SimpleQueryProcessor()