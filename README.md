# 용어 검색 시스템 (Term Search System)

지능형 하이브리드 검색을 통한 용어 사전 검색 시스템

## 시스템 개요

자연어 질의를 통해 용어 사전에서 관련 용어를 검색하는 지능형 시스템입니다. GPT 기반 쿼리 전처리와 FAISS + ColBERT 하이브리드 검색을 결합하여 높은 정확도의 검색 결과를 제공합니다.

### 핵심 특징
- **3단계 검색 파이프라인**: 쿼리 전처리 → 하이브리드 검색 → 결과 포맷팅
- **하이브리드 검색 방식**: 의미적 유사도(60%) + 토큰별 최대 유사도(40%)
- **장애 대응 메커니즘**: Circuit Breaker, Rate Limiting, Fallback 시스템
- **다중 데이터 소스**: Google Sheets, Excel 파일 지원
- **파일 기반 캐싱**: 임베딩 결과 7일간 캐시

## 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ 쿼리 전처리     │ -> │ 하이브리드 검색  │ -> │ 결과 포맷팅     │
│ QueryProcessor  │    │ HybridEngine     │    │ TermSearchSystem│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         |                        |
         v                        v
┌─────────────────┐    ┌──────────────────┐
│ GPT/Simple      │    │ FAISS + ColBERT  │
│ 자연어 정제     │    │ 벡터 + 토큰 검색 │
└─────────────────┘    └──────────────────┘
```

### 주요 컴포넌트

#### 1. 검색 시스템 (TermSearchSystem)
- 전체 검색 파이프라인 조율
- 예외 처리 및 응답 포맷팅
- 시스템 상태 모니터링

#### 2. 쿼리 처리 (QueryProcessor)
**GPTQueryProcessor**
- OpenAI GPT-4o-mini를 사용한 자연어 쿼리 정제
- 구어체 → 표준 용어 변환
- Rate Limiting (분당 10회 제한)
- Circuit Breaker (5회 실패 시 60초 차단)

**SimpleQueryProcessor** (Fallback)
- 정규식 기반 간단한 텍스트 정제
- GPT API 장애 시 자동 전환

#### 3. 하이브리드 검색 엔진 (HybridSearchEngine)
**의미적 검색**
- OpenAI text-embedding-3-small 모델
- FAISS IndexFlatIP (내적 기반)
- 1536차원 벡터 공간

**토큰별 검색 (ColBERT 스타일)**
- 쿼리 토큰 vs 문서 토큰 최대 유사도
- 형태소 분석 + 정규식 토큰화
- 토큰별 임베딩 벡터 비교

**점수 계산**
```
최종점수 = 의미적유사도 × 0.6 + ColBERT점수 × 0.4
```

#### 4. 임베딩 서비스 (EmbeddingService)
- OpenAI API 호출 관리
- 파일 기반 캐시 (7일 TTL)
- Circuit Breaker (3회 실패 시 30초 차단)
- API 장애 시 제로 벡터 반환

#### 5. 캐시 관리 (CacheManager)
- MD5 해시 기반 키 생성
- Pickle을 통한 NumPy 배열 직렬화
- TTL 기반 자동 만료 관리
- 서브디렉토리 구조로 성능 최적화

#### 6. 데이터 로더
**SheetsDataLoader** (Primary)
- Google Sheets API 연동
- OAuth 2.0 인증
- 실시간 데이터 동기화

**ExcelDataLoader** (Fallback)
- pandas를 통한 Excel 파일 읽기
- 로컬 파일 기반 백업

## 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (.env)
```env
# OpenAI API 설정
OPENAI_API_KEY=sk-...

# 파일 경로
EXCEL_FILE=terms.xlsx

# Google Sheets 설정 (선택사항)
SPREADSHEET_ID=...

# 캐시 설정
CACHE_DIR=cache
CACHE_TTL=604800

# 검색 파라미터
SEMANTIC_WEIGHT=0.6
COLBERT_WEIGHT=0.4
SEARCH_THRESHOLD=0.3

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=10

# 컬럼 매핑
COLUMN_TERM_NAME=공통표준용어명
COLUMN_TERM_DESC=공통표준용어설명
COLUMN_TERM_ABBR=공통표준용어영문약어명
COLUMN_DOMAIN=공통표준도메인명
```

### 3. 인증 파일 설정 (Google Sheets 사용 시)
```
credentials.json  # Google API 서비스 계정 키
token.json        # OAuth 토큰 (자동 생성)
```

### 4. 데이터 파일 준비
**Excel 파일 형식**
```
공통표준용어명 | 공통표준용어설명 | 공통표준용어영문약어명 | 공통표준도메인명
고객번호      | 고객을 식별하는... | CUST_NO              | 고객관리
```

## 사용법

### 1. 시스템 실행
```bash
python main.py
```

### 2. 검색 예시
```
검색어 입력: 고객 번호 관리하는 거 어떻게 해?

검색 결과 (3개):
원본 쿼리: 고객 번호 관리하는 거 어떻게 해?
정제된 쿼리: 고객번호 관리 방법

1. 고객번호
   약어: CUST_NO
   설명: 고객을 식별하는 유일한 번호
   도메인: 고객관리
   점수: 0.892

2. 고객관리시스템
   약어: CMS
   설명: 고객 정보를 통합 관리하는 시스템
   도메인: 고객관리
   점수: 0.745
```

## API 구조

### 검색 API
```python
from app.search_system import TermSearchSystem
from system_factory import SystemFactory
from config import Config

# 시스템 초기화
config = Config.get_system_config()
search_system = SystemFactory.create_system(config)

# 검색 실행
result = search_system.search(
    natural_query="고객 번호 관리 방법",
    user_id="user123",
    k=10
)

# 결과 구조
{
    'success': True,
    'original_query': '고객 번호 관리하는 거 어떻게 해?',
    'refined_query': '고객번호 관리 방법',
    'search_results': [SearchResult, ...],
    'candidates_count': 10
}
```

### 검색 결과 구조
```python
@dataclass
class SearchResult:
    term: Term                    # 용어 객체
    semantic_score: float         # 의미적 유사도 점수
    colbert_score: float         # ColBERT 점수
    final_score: float           # 최종 점수
    rank: int                    # 순위
    matched_tokens: List[str]    # 매칭된 토큰들
```

## 장애 대응 메커니즘

### 1. Circuit Breaker
**GPT API (쿼리 전처리)**
- 실패 임계값: 5회
- 복구 대기: 60초
- Fallback: SimpleQueryProcessor

**OpenAI Embedding API**
- 실패 임계값: 3회
- 복구 대기: 30초
- Fallback: 제로 벡터 반환

### 2. Rate Limiting
- 사용자별 분당 10회 제한
- 메모리 기반 슬라이딩 윈도우
- 60초 윈도우 크기

### 3. Fallback 시스템
```
GPT 실패 → Simple Processor
Google Sheets 실패 → Excel 파일
OpenAI API 장애 → 제로 벡터 (서비스 지속)
```

## 성능 최적화

### 1. FAISS 인덱싱
- C++ 기반 최적화된 벡터 검색
- IndexFlatIP (내적 기반)
- 대규모 벡터 컬렉션 지원

### 2. 캐시 전략
- 파일 기반 영구 캐시
- MD5 해시 충돌 최소화
- 서브디렉토리 구조 (파일 시스템 성능)
- TTL 기반 자동 정리

### 3. 토큰화 최적화
- 정규식 + 형태소 분석 결합
- 의미있는 토큰 필터링
- 불용어 제거

## 모니터링 및 로깅

### 로그 레벨
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 주요 로그
- 검색 파이프라인 실행 과정
- API 호출 성공/실패
- Circuit Breaker 상태 변화
- 캐시 히트/미스
- 시스템 초기화 진행상황

## 파일 구조
```
SEARCHTERM_V2/
├── app/
│   └── search_system.py          # 메인 검색 시스템
├── core/
│   ├── term.py                   # 용어 데이터 모델
│   ├── search_result.py          # 검색 결과 모델
│   └── custom_exceptions.py      # 커스텀 예외
├── services/
│   ├── embedding_service.py      # 임베딩 생성 서비스
│   ├── hybrid_engine.py          # 하이브리드 검색 엔진
│   ├── gpt_processor.py          # GPT 쿼리 처리
│   ├── simple_processor.py       # 간단한 쿼리 처리
│   ├── file_cache.py            # 파일 캐시 관리
│   └── sheets_loader.py          # 데이터 로더
├── utils/
│   ├── circuit_breaker.py        # Circuit Breaker 구현
│   ├── rate_limiter.py          # Rate Limiter 구현
│   └── text_utils.py            # 텍스트 유틸리티
├── config.py                     # 설정 관리
├── system_factory.py            # 시스템 팩토리
├── main.py                      # 메인 실행 파일
├── .env                         # 환경 변수
└── requirements.txt             # 의존성
```

## 개발 및 기여

### 개발 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 개발 의존성 설치
pip install -r requirements.txt
```

### 테스트
```bash
python main.py
```

### 설정 검증
시스템 시작 시 자동으로 설정 검증이 수행됩니다:
- OpenAI API 키 확인
- 데이터 소스 존재 확인
- 캐시 디렉토리 권한 확인

## 라이센스

이 프로젝트는 내부 사용을 위한 것입니다.

## 버전 히스토리

### v2.0.0 (리팩토링 후)
- 3단계 파이프라인으로 단순화
- Circuit Breaker 및 Rate Limiting 추가
- Fallback 메커니즘 강화
- AI 재순위화 기능 제거 (비용 및 성능 최적화)

### v1.0.0 (리팩토링 전)
- 4단계 파이프라인 (AI 재순위화 포함)
- 기본적인 하이브리드 검색
- 파일 기반 캐시 시스템 도입