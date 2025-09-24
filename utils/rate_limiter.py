import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MemoryRateLimiter:
    """메모리 기반 Rate Limiter"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, user_id: str, max_requests: int = 10) -> bool:
        """간단한 메모리 기반 rate limiting"""
        current_time = time.time()
        
        # 사용자별 요청 기록 가져오기
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        user_requests = self.requests[user_id]
        
        # 윈도우 밖의 오래된 요청들 제거
        cutoff_time = current_time - self.window_size
        user_requests[:] = [req_time for req_time in user_requests if req_time > cutoff_time]
        
        # 요청 수 확인
        if len(user_requests) < max_requests:
            user_requests.append(current_time)
            return True
        else:
            return False