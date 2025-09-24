import logging
import os
import sys

# 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.search_system import TermSearchSystem
from config import Config
from system_factory import SystemFactory

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_system():
    """시스템 초기화"""
    try:
        logger.info("검색 시스템 초기화 시작...")
        
        # 설정 생성
        config = Config.get_system_config()
        
        # 시스템 생성
        search_system = SystemFactory.create_system(config)
        
        logger.info("검색 시스템 초기화 완료")
        return search_system
        
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def search_demo(search_system):
    """검색 데모"""
    print("\n=== 용어 검색 시스템 ===")
    print("사용법:")
    print("  - 검색어를 입력하고 Enter")
    print("  - 'status' 입력하면 시스템 상태 확인")
    print("  - 'quit' 또는 'exit' 입력하면 종료")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n검색어 입력: ").strip()
            
            if query.lower() in ['quit', 'exit', '종료', 'q']:
                print("프로그램을 종료합니다.")
                break
                
            if not query:
                continue
                
            # 검색 실행
            print(f"\n'{query}' 검색 중...")
            result = search_system.search(query, "user", 10)
            
            if result['success']:
                search_results = result['search_results']
                print(f"\n검색 결과 ({len(search_results)}개):")
                print(f"원본 쿼리: {result['original_query']}")
                print(f"정제된 쿼리: {result['refined_query']}")
                print("-" * 50)
                
                for i, search_result in enumerate(search_results, 1):
                    term = search_result.term
                    print(f"{i}. {term.name}")
                    print(f"   약어: {term.abbreviation}")
                    print(f"   설명: {term.description}")
                    print(f"   도메인: {term.domain}")
                    print(f"   점수: {search_result.final_score:.3f}")
                    if search_result.matched_tokens:
                        print(f"   매칭 토큰: {', '.join(search_result.matched_tokens)}")
                    print()
            else:
                print(f"검색 실패: {result.get('message', '알 수 없는 오류')}")
                if result.get('error_type') == 'rate_limit':
                    print("잠시 후 다시 시도해주세요.")
                    
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            logger.error(f"검색 중 오류: {e}")
            print(f"오류 발생: {e}")

def main():
    """메인 함수"""
    # 시스템 초기화
    search_system = initialize_system()  # 여기서 변수 할당!
    
    if not search_system:
        print("시스템 초기화에 실패했습니다.")
        return
    
    try:
        # 검색 데모 실행
        search_demo(search_system)
    finally:
        # 시스템 정리
        if hasattr(search_system, 'close'):
            search_system.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()