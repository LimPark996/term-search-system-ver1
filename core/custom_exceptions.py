"""커스텀 예외 정의"""

class TermSearchException(Exception):
    """용어 검색 시스템 기본 예외"""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

class RateLimitExceeded(TermSearchException):
    """API 호출 제한 초과"""
    def __init__(self, message: str = "API 호출 제한을 초과했습니다"):
        super().__init__(message, "RATE_LIMIT_EXCEEDED")

class DataLoadError(TermSearchException):
    """데이터 로드 실패"""
    def __init__(self, message: str = "데이터 로드에 실패했습니다"):
        super().__init__(message, "DATA_LOAD_ERROR")

class QueryProcessingError(TermSearchException):
    """쿼리 처리 실패"""
    def __init__(self, message: str = "쿼리 처리에 실패했습니다"):
        super().__init__(message, "QUERY_PROCESSING_ERROR")

class SearchEngineError(TermSearchException):
    """검색 엔진 오류"""
    def __init__(self, message: str = "검색 엔진에서 오류가 발생했습니다"):
        super().__init__(message, "SEARCH_ENGINE_ERROR")

class CacheError(TermSearchException):
    """캐시 오류"""
    def __init__(self, message: str = "캐시에서 오류가 발생했습니다"):
        super().__init__(message, "CACHE_ERROR")

class EmbeddingServiceError(TermSearchException):
    """임베딩 서비스 오류"""
    def __init__(self, message: str = "임베딩 서비스에서 오류가 발생했습니다"):
        super().__init__(message, "EMBEDDING_SERVICE_ERROR")

class ConfigurationError(TermSearchException):
    """설정 오류"""
    def __init__(self, message: str = "시스템 설정에 오류가 있습니다"):
        super().__init__(message, "CONFIGURATION_ERROR")

class CircuitBreakerOpenException(TermSearchException):
    """Circuit Breaker가 열린 상태에서 호출 시 발생하는 예외"""
    def __init__(self, message: str = "Circuit Breaker가 열린 상태입니다"):
        super().__init__(message, "CIRCUIT_BREAKER_OPEN")