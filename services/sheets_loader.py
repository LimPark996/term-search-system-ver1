import logging
import gspread
import pandas as pd
from typing import List
from core.term import Term
from core.custom_exceptions import DataLoadError

logger = logging.getLogger(__name__)

class SheetsDataLoader:
    """Google Sheets 데이터 로더"""
    
    def __init__(self, 
                 spreadsheet_id: str,
                 column_mapping: dict,
                 credentials_file: str = 'credentials.json',
                 token_file: str = 'token.json'):
        self.spreadsheet_id = spreadsheet_id
        self.column_mapping = column_mapping
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.gc = None
        
        self._setup_connection()
    
    def _setup_connection(self):
        """Google Sheets 연결 설정"""
        try:
            logger.info("Google Sheets 연결 중...")
            self.gc = gspread.oauth(
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ],
                credentials_filename=self.credentials_file,
                authorized_user_filename=self.token_file
            )
            logger.info("Google Sheets 연결 성공")
        except Exception as e:
            logger.error(f"Google Sheets 인증 실패: {e}")
            self.gc = None
    
    def is_available(self) -> bool:
        """Google Sheets 접근 가능 여부"""
        return self.gc is not None and bool(self.spreadsheet_id)
    
    def load_terms(self) -> List[Term]:
        """Google Sheets에서 용어 데이터 로드"""
        if not self.is_available():
            raise DataLoadError("Google Sheets 연결이 설정되지 않았습니다")
        
        try:
            logger.info(f"스프레드시트 로드 중: {self.spreadsheet_id}")
            
            # 스프레드시트 열기
            spreadsheet = self.gc.open_by_key(self.spreadsheet_id)
            worksheet = spreadsheet.worksheets()[0]  # 첫 번째 시트
            
            # 모든 데이터 가져오기
            all_values = worksheet.get_all_values()
            if not all_values:
                logger.warning("스프레드시트가 비어있습니다")
                return []
            
            # DataFrame으로 변환
            headers = all_values[0]
            rows = all_values[1:]
            df = pd.DataFrame(rows, columns=headers)
            
            # 빈 행 제거
            df = df.dropna(how='all')
            
            # Term 객체로 변환
            terms = self._convert_to_terms(df)
            
            logger.info(f"용어 로드 완료: {len(terms)}개")
            return terms
            
        except Exception as e:
            logger.error(f"Google Sheets 로드 실패: {e}")
            raise DataLoadError(f"데이터 로드 실패: {str(e)}")
    
    def _convert_to_terms(self, df: pd.DataFrame) -> List[Term]:
        """DataFrame을 Term 객체로 변환"""
        terms = []
        
        for idx, row in df.iterrows():
            try:
                term = Term(
                    id=str(idx),
                    name=str(row.get(self.column_mapping.get('term_name', ''), '')).strip(),
                    description=str(row.get(self.column_mapping.get('term_desc', ''), '')).strip(),
                    abbreviation=str(row.get(self.column_mapping.get('term_abbr', ''), '')).strip(),
                    domain=str(row.get(self.column_mapping.get('domain', ''), '')).strip(),
                    metadata=row.to_dict()
                )
                
                # 필수 필드 검증
                if term.name and term.description:
                    terms.append(term)
                else:
                    logger.debug(f"행 {idx}: 필수 필드 누락 - 건너뜀")
                    
            except Exception as e:
                logger.warning(f"행 {idx} 변환 실패: {e}")
                continue
        
        logger.info(f"유효한 용어 {len(terms)}개 변환 완료")
        return terms
    
    def get_sheet_info(self) -> dict:
        """시트 정보 조회"""
        if not self.is_available():
            return {'error': 'Google Sheets 연결 불가'}
        
        try:
            spreadsheet = self.gc.open_by_key(self.spreadsheet_id)
            worksheet = spreadsheet.worksheets()[0]
            
            return {
                'spreadsheet_title': spreadsheet.title,
                'worksheet_title': worksheet.title,
                'row_count': worksheet.row_count,
                'col_count': worksheet.col_count,
                'last_updated': spreadsheet.lastUpdateTime
            }
            
        except Exception as e:
            logger.error(f"시트 정보 조회 실패: {e}")
            return {'error': str(e)}

class ExcelDataLoader:
    """Excel 파일 데이터 로더 (Fallback)"""
    
    def __init__(self, excel_file: str, column_mapping: dict):
        self.excel_file = excel_file
        self.column_mapping = column_mapping
    
    def is_available(self) -> bool:
        """Excel 파일 존재 여부"""
        import os
        return os.path.exists(self.excel_file)
    
    def load_terms(self) -> List[Term]:
        """Excel 파일에서 용어 데이터 로드"""
        if not self.is_available():
            raise DataLoadError(f"Excel 파일을 찾을 수 없습니다: {self.excel_file}")
        
        try:
            logger.info(f"Excel 파일 로드 중: {self.excel_file}")
            
            # Excel 파일 읽기
            df = pd.read_excel(self.excel_file)
            df = df.dropna(how='all')
            
            # Term 객체로 변환
            terms = self._convert_to_terms(df)
            
            logger.info(f"Excel 파일에서 용어 로드 완료: {len(terms)}개")
            return terms
            
        except Exception as e:
            logger.error(f"Excel 파일 로드 실패: {e}")
            raise DataLoadError(f"Excel 로드 실패: {str(e)}")
    
    def _convert_to_terms(self, df: pd.DataFrame) -> List[Term]:
        """DataFrame을 Term 객체로 변환"""
        terms = []
        
        for idx, row in df.iterrows():
            try:
                term = Term(
                    id=str(idx),
                    name=str(row.get(self.column_mapping.get('term_name', ''), '')).strip(),
                    description=str(row.get(self.column_mapping.get('term_desc', ''), '')).strip(),
                    abbreviation=str(row.get(self.column_mapping.get('term_abbr', ''), '')).strip(),
                    domain=str(row.get(self.column_mapping.get('domain', ''), '')).strip(),
                    metadata=row.to_dict()
                )
                
                # 필수 필드 검증
                if term.name and term.description:
                    terms.append(term)
                else:
                    logger.debug(f"행 {idx}: 필수 필드 누락 - 건너뜀")
                    
            except Exception as e:
                logger.warning(f"행 {idx} 변환 실패: {e}")
                continue
        
        logger.info(f"Excel에서 유효한 용어 {len(terms)}개 변환 완료")
        return terms