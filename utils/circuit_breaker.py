import time  # 시간 관련 함수 사용 (복구 타이밍 체크용)
import logging  # 로그 출력용
from typing import Callable, Any, Optional  # 타입 힌트용
from enum import Enum  # 상태를 나타내는 열거형 생성용
from core.custom_exceptions import TermSearchException  # 커스텀 예외 클래스

logger = logging.getLogger(__name__)  # 현재 모듈용 로거 생성

class CircuitBreakerState(Enum):
    """Circuit Breaker의 3가지 상태를 정의"""
    CLOSED = "closed"     # 정상 상태 - API 호출 허용
    OPEN = "open"         # 장애 상태 - API 호출 차단 (실패가 너무 많아서)
    HALF_OPEN = "half_open"  # 복구 시도 상태 - 1번만 호출해서 테스트

class CircuitBreakerOpenException(TermSearchException):
    """Circuit Breaker가 열린 상태에서 호출 시 발생하는 예외"""
    # 이 예외가 발생하면 "지금은 API 호출을 차단 중입니다"라는 의미
    pass

class CircuitBreaker:
    """외부 API 호출 보호를 위한 Circuit Breaker"""
    
    def __init__(self, 
                 failure_threshold: int = 5,      # 몇 번 실패하면 차단할지
                 recovery_timeout: int = 60,      # 몇 초 후에 복구 시도할지
                 expected_exception: type = Exception):  # 어떤 예외를 실패로 볼지
        
        # === 설정값 저장 ===
        self.failure_threshold = failure_threshold  # 5번 실패하면 차단
        self.recovery_timeout = recovery_timeout    # 60초 후 복구 시도
        self.expected_exception = expected_exception  # 이 예외가 나면 실패로 카운트
        
        # === 현재 상태 초기화 ===
        self.state = CircuitBreakerState.CLOSED  # 처음엔 정상 상태로 시작
        self.failure_count = 0  # 실패 횟수 0으로 초기화
        self.last_failure_time: Optional[float] = None  # 마지막 실패 시간 (아직 없음)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Circuit Breaker를 통한 안전한 함수 호출"""
        
        # === 현재 상태 확인 및 처리 ===
        if self.state == CircuitBreakerState.OPEN:  # 차단 상태라면
            if self._should_attempt_reset():  # 충분한 시간이 지났는지 확인
                self.state = CircuitBreakerState.HALF_OPEN  # 복구 시도 상태로 변경
                logger.info("Circuit Breaker: HALF_OPEN 상태로 전환")
            else:  # 아직 복구 시간이 안 됐다면
                # 예외를 던져서 "지금은 호출 불가"라고 알림
                raise CircuitBreakerOpenException("Circuit Breaker가 열린 상태입니다")
        
        try:
            # === 실제 함수 호출 시도 ===
            result = func(*args, **kwargs)  # 전달받은 함수를 실행
            
            # === 호출 성공 시 처리 ===
            if self.state == CircuitBreakerState.HALF_OPEN:  # 복구 시도 중이었다면
                self._reset()  # 완전 복구 (모든 카운터 초기화)
                logger.info("Circuit Breaker: 복구 성공, CLOSED 상태로 전환")
            
            return result  # 성공한 결과 반환
            
        except self.expected_exception as e:  # 예상된 예외 (API 실패 등)가 발생하면
            self._record_failure()  # 실패 횟수 증가 및 상태 체크
            logger.warning(f"Circuit Breaker: 실패 기록 ({self.failure_count}/{self.failure_threshold})")
            raise e  # 예외를 다시 던져서 호출자에게 알림
    
    def _should_attempt_reset(self) -> bool:
        """복구 시도 여부 판단"""
        if self.last_failure_time is None:  # 실패한 적이 없다면
            return True  # 복구 시도 가능
        
        # 현재 시간 - 마지막 실패 시간 >= 복구 대기 시간
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _record_failure(self):
        """실패 기록 및 상태 변경"""
        self.failure_count += 1  # 실패 횟수 1 증가
        self.last_failure_time = time.time()  # 현재 시간을 마지막 실패 시간으로 기록
        
        # 실패 횟수가 임계값에 도달했다면
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN  # 차단 상태로 변경
            logger.error(f"Circuit Breaker: OPEN 상태로 전환 (실패 {self.failure_count}회)")
    
    def _reset(self):
        """Circuit Breaker 상태 초기화 (완전 복구)"""
        self.state = CircuitBreakerState.CLOSED  # 정상 상태로 변경
        self.failure_count = 0  # 실패 횟수 초기화
        self.last_failure_time = None  # 실패 시간 초기화
    
    def get_state(self) -> CircuitBreakerState:
        """현재 상태 조회 (외부에서 상태 확인용)"""
        return self.state
    
    def force_open(self):
        """강제로 Circuit Breaker 열기 (테스트나 긴급 상황용)"""
        self.state = CircuitBreakerState.OPEN  # 강제로 차단 상태로 변경
        self.last_failure_time = time.time()  # 현재 시간을 실패 시간으로 설정
        logger.warning("Circuit Breaker: 강제로 OPEN 상태로 설정됨")
    
    def force_close(self):
        """강제로 Circuit Breaker 닫기 (관리자가 수동 복구할 때)"""
        self._reset()  # 모든 상태 초기화
        logger.info("Circuit Breaker: 강제로 CLOSED 상태로 설정됨")

# === Circuit Breaker가 왜 필요한가? ===
#
# 예시 상황:
# 1. 우리 서비스가 외부 API를 호출합니다
# 2. 외부 API가 갑자기 응답하지 않습니다 (서버 다운, 네트워크 장애 등)
# 3. 우리 서비스는 계속 재시도합니다
# 4. 모든 요청이 타임아웃될 때까지 기다립니다 (예: 30초씩)
# 5. 결국 우리 서비스도 느려지고 사용자 경험이 나빠집니다
#
# Circuit Breaker 적용 후:
# 1. 5번 실패하면 더 이상 호출하지 않습니다
# 2. 60초 후 1번만 테스트해봅니다
# 3. 성공하면 정상 호출 재개, 실패하면 다시 60초 대기
# 4. 우리 서비스는 빠르게 에러를 반환하여 사용자 경험 보호

# === 사용 예시 ===
# def unreliable_api_call():
#     # 외부 API 호출 코드
#     pass
#
# circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
# 
# try:
#     result = circuit_breaker.call(unreliable_api_call)
# except CircuitBreakerOpenException:
#     # Circuit Breaker가 열려있음 - 대체 로직 실행
#     result = "서비스 일시 중단 중입니다"