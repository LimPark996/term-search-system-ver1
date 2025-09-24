import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """단순화된 애플리케이션 설정 (운영/개발 환경 분리 제거)"""
    
    # 기본 설정
    PORT = int(os.getenv('PORT', 5000))
    
    # API 키
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Google Sheets 설정 (선택사항)
    SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', '')
    
    # 파일 캐시 설정
    CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
    CACHE_TTL = int(os.getenv('CACHE_TTL', 604800))  # 7일
    
    # 검색 파라미터
    SEMANTIC_WEIGHT = float(os.getenv('SEMANTIC_WEIGHT', 0.6))
    COLBERT_WEIGHT = float(os.getenv('COLBERT_WEIGHT', 0.4))
    SEARCH_THRESHOLD = float(os.getenv('SEARCH_THRESHOLD', 0.3))
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', 10))
    
    # 파일 경로
    EXCEL_FILE = os.getenv('EXCEL_FILE', 'terms.xlsx')
    
    # 컬럼 매핑
    COLUMN_MAPPING = {
        'term_abbr': os.getenv('COLUMN_TERM_ABBR', '공통표준용어영문약어명'),
        'term_name': os.getenv('COLUMN_TERM_NAME', '공통표준용어명'),
        'term_desc': os.getenv('COLUMN_TERM_DESC', '공통표준용어설명'),
        'domain': os.getenv('COLUMN_DOMAIN', '공통표준도메인명'),
        'word_name': os.getenv('COLUMN_WORD_NAME', '공통표준단어명'),
        'word_abbr': os.getenv('COLUMN_WORD_ABBR', '공통표준단어영문약어명')
    }
    
    @classmethod
    def get_system_config(cls) -> dict:
        """시스템 팩토리에서 사용할 설정 딕셔너리 반환"""
        return {
            # API 키
            'openai_api_key': cls.OPENAI_API_KEY,
            
            # Google Sheets 설정
            'google_sheets': {
                'spreadsheet_id': cls.SPREADSHEET_ID,
                'column_mapping': cls.COLUMN_MAPPING,
                'credentials_file': 'credentials.json',
                'token_file': 'token.json'
            },
            
            # 파일 캐시 설정
            'cache_dir': cls.CACHE_DIR,
            'cache_ttl': cls.CACHE_TTL,

            # 검색 파라미터
            'semantic_weight': cls.SEMANTIC_WEIGHT,
            'colbert_weight': cls.COLBERT_WEIGHT,
            'search_threshold': cls.SEARCH_THRESHOLD,
            
            # Rate Limiting
            'max_requests_per_minute': cls.MAX_REQUESTS_PER_MINUTE,
            
            # 파일 경로
            'excel_file': cls.EXCEL_FILE,
            'column_mapping': cls.COLUMN_MAPPING
        }
    
    @classmethod
    def validate_config(cls) -> list:
        """간단한 설정 검증"""
        warnings = []
        
        if not cls.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY가 설정되지 않았습니다")
        
        # Excel 파일 또는 Google Sheets 중 하나는 있어야 함
        excel_exists = os.path.exists(cls.EXCEL_FILE)
        sheets_configured = bool(cls.SPREADSHEET_ID)
        
        if not excel_exists and not sheets_configured:
            warnings.append(f"데이터 소스가 없습니다: Excel({cls.EXCEL_FILE}) 또는 Google Sheets 설정 필요")
        
        # 캐시 디렉토리 생성 가능 여부 확인
        try:
            os.makedirs(cls.CACHE_DIR, exist_ok=True)
        except PermissionError:
            warnings.append(f"캐시 디렉토리 생성 권한이 없습니다: {cls.CACHE_DIR}")
        
        return warnings

# 설정 검증 (앱 로드 시)
config_warnings = Config.validate_config()
if config_warnings:
    print("⚠️ 설정 경고:")
    for warning in config_warnings:
        print(f"  - {warning}")