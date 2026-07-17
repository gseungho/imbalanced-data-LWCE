# src/data_handler.py

import os # 파일 경로 관리를 위한 모듈
import numpy as np # 수치 연산을 위한 모듈
import pandas as pd # 데이터 조작 및 분석을 위한 모듈
from sklearn.model_selection import train_test_split # 데이터를 학습용과 테스트용으로 나누기 위한 함수
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 숫자 스케일링 및 범주형 데이터 인코딩을 위한 클래스
from sklearn.compose import ColumnTransformer # 여러 전처리 단계를 하나로 묶기 위한 클래스

# 데이터셋을 불러오고 전처리하는 메인 함수 정의
def load_dataset(dataset_name: str, base_data_path: str, test_size: float = 0.3, random_state: int = 42) -> tuple:
    # `dataset_name` 문자열에 따라 해당 데이터셋을 불러오는 로직을 실행
    if dataset_name == 'aps_failure': # APS 고장 데이터셋 처리
        train_path = os.path.join(base_data_path, 'aps_failure_training_set.csv') # 학습 데이터 파일 경로 생성
        test_path = os.path.join(base_data_path, 'aps_failure_test_set.csv') # 테스트 데이터 파일 경로 생성
        
        train_df = pd.read_csv(train_path, skiprows=20, na_values='na') # 학습 데이터 로드, 상위 20줄 설명은 건너뛰고 'na'를 결측치로 인식
        test_df = pd.read_csv(test_path, skiprows=20, na_values='na') # 테스트 데이터 로드
        
        train_df['class'] = train_df['class'].map({'pos': 1, 'neg': 0}) # 'pos'/'neg' 문자 레이블을 숫자 1/0으로 변환
        test_df['class'] = test_df['class'].map({'pos': 1, 'neg': 0}) # 테스트 데이터에도 동일하게 적용

        X_train = train_df.drop('class', axis=1) # 'class' 열을 제외한 나머지를 피처(X_train)로 사용
        y_train = train_df['class'] # 'class' 열을 레이블(y_train)로 사용
        X_test = test_df.drop('class', axis=1) # 테스트 데이터도 동일하게 피처(X_test) 분리
        y_test = test_df['class'] # 테스트 데이터도 동일하게 레이블(y_test) 분리

        mean_values = X_train.mean() # 학습 데이터의 각 피처별 평균값 계산
        X_train.fillna(mean_values, inplace=True) # 학습 데이터의 결측치를 위에서 계산한 평균값으로 대체
        X_test.fillna(mean_values, inplace=True) # 테스트 데이터의 결측치도 학습 데이터의 평균값으로 대체 (중요)
        
        scaler = StandardScaler() # 데이터 스케일링을 위한 StandardScaler 객체 생성
        X_train = scaler.fit_transform(X_train) # 학습 데이터에 맞춰 스케일러를 학습시키고 변환 적용
        X_test = scaler.transform(X_test) # 학습된 기준으로 테스트 데이터도 변환
        
        return X_train, X_test, y_train, y_test # 이 데이터셋은 자체 분할되어 있으므로 바로 반환
    

    elif dataset_name == 'credit_card_fraud': # 신용카드 사기 탐지 데이터셋 처리
        file_path = os.path.join(base_data_path, 'creditcard.csv') # 파일 경로 생성
        df = pd.read_csv(file_path) # CSV 파일 로드
        df = df.drop('Time', axis=1) # 불필요한 'Time' 열 제거
        scaler = StandardScaler() # 'Amount' 열 스케일링을 위한 객체 생성
        df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1)) # 거래금액(Amount)만 스케일링 적용
        X = df.drop('Class', axis=1) # 'Class' 열을 제외한 나머지를 피처(X)로 사용
        y = df['Class'] # 'Class' 열을 레이블(y)로 사용

    elif dataset_name == 'credit_card_default': # 신용카드 채무 불이행 데이터셋 처리
        file_path = os.path.join(base_data_path, 'default of credit card clients.xls') # 파일 경로 생성
        df = pd.read_excel(file_path, header=1) # 엑셀 파일 로드, 첫 행은 건너뜀
        df = df.drop(columns=['ID']) # 불필요한 'ID' 열 제거
        df.rename(columns={'default payment next month': 'target'}, inplace=True) # 컬럼명을 'target'으로 변경
        for col in df.columns: # 모든 컬럼에 대해
            df[col] = pd.to_numeric(df[col], errors='coerce') # 숫자형으로 변환, 변환 불가 시 결측치(NaN)로 처리
        df.fillna(0, inplace=True) # 결측치를 0으로 채움
        X = df.drop('target', axis=1) # 'target' 열을 제외하고 피처(X)로 사용
        y = df['target'] # 'target' 열을 레이블(y)로 사용
        categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE'] # 범주형 피처 목록
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # 원-핫 인코딩 객체 생성
        preprocessor = ColumnTransformer( # 특정 열에만 변환을 적용하는 객체
            [('one_hot', one_hot_encoder, categorical_features)], # 범주형 피처에 원-핫 인코딩 적용
            remainder='passthrough' # 나머지 열은 그대로 둠
        )
        X = preprocessor.fit_transform(X) # 전처리기 적용

    elif dataset_name == 'telco_churn': # 통신사 고객 이탈 데이터셋 처리
        file_path = os.path.join(base_data_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv') # 파일 경로 생성
        df = pd.read_csv(file_path) # CSV 파일 로드
        df = df.drop('customerID', axis=1) # 불필요한 'customerID' 열 제거
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') # 숫자형으로 변환
        df['TotalCharges'].fillna(0, inplace=True) # 결측치를 0으로 채움
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0) # 'Yes'/'No'를 1/0으로 변환
        X = df.drop('Churn', axis=1) # 'Churn' 열을 제외하고 피처(X)로 사용
        y = df['Churn'] # 'Churn' 열을 레이블(y)로 사용
        binary_cols = [col for col in X.columns if X[col].nunique() == 2 and X[col].dtype == 'object'] # 고유값이 2개인 문자열 열 목록
        multi_cols = [col for col in X.columns if X[col].nunique() > 2 and X[col].dtype == 'object'] # 고유값이 2개 초과인 문자열 열 목록
        for col in binary_cols: # 2개인 열에 대해
            X[col], _ = pd.factorize(X[col]) # 0과 1로 인코딩
        X = pd.get_dummies(X, columns=multi_cols, drop_first=True) # 2개 초과 열은 원-핫 인코딩

    elif dataset_name == 'wdbc': # 유방암 진단 데이터셋 처리
        file_path = os.path.join(base_data_path, 'wdbc.data') # 파일 경로 생성
        df = pd.read_csv(file_path, header=None) # CSV 파일 로드, 헤더 없음
        y = df[1].map({'M': 1, 'B': 0}) # 'M'/'B' 레이블을 1/0으로 변환
        X = df.drop(columns=[0, 1]) # ID와 레이블 열을 제외하고 피처(X)로 사용

    elif dataset_name == 'german_credit': # 독일 신용 데이터셋 처리
        file_path = os.path.join(base_data_path, 'german.data') # 파일 경로 생성
        column_names = [f'attr_{i}' for i in range(20)] + ['target'] # 컬럼명 리스트 생성
        df = pd.read_csv(file_path, header=None, sep=r'\s+', names=column_names) # 공백으로 구분된 파일 로드
        df['target'] = df['target'].map({1: 0, 2: 1}) # 1/2 레이블을 0/1로 변환
        X = df.drop('target', axis=1) # 'target' 열 제외하고 피처(X)로 사용
        y = df['target'] # 'target' 열을 레이블(y)로 사용
        categorical_features = X.select_dtypes(include=['object']).columns # 문자열 타입의 범주형 피처 선택
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True) # 원-핫 인코딩 적용

    elif dataset_name == 'secom': # 반도체 제조 공정 불량 데이터셋 처리
        features_path = os.path.join(base_data_path, 'secom.data') # 피처 데이터 경로
        labels_path = os.path.join(base_data_path, 'secom_labels.data') # 레이블 데이터 경로
        X = pd.read_csv(features_path, header=None, sep=' ') # 공백으로 구분된 피처 파일 로드
        y_df = pd.read_csv(labels_path, header=None, sep=' ') # 공백으로 구분된 레이블 파일 로드
        y = y_df[0] # 첫 번째 열이 레이블
        y = y.map({-1: 0, 1: 1}) # -1/1 레이블을 0/1로 변환
        X.fillna(X.mean(), inplace=True) # 결측치를 평균값으로 대체
    
    elif dataset_name == 'bank_marketing': # 은행 마케팅 데이터셋 처리
        file_path = os.path.join(base_data_path, 'bank-full.csv') # 파일 경로 생성
        df = pd.read_csv(file_path, sep=';') # 세미콜론으로 구분된 파일 로드
        df['y'] = df['y'].map({'yes': 1, 'no': 0}) # 'yes'/'no' 레이블을 1/0으로 변환
        X = df.drop('y', axis=1) # 'y' 열을 제외하고 피처(X)로 사용
        y = df['y'] # 'y' 열을 레이블(y)로 사용
        categorical_features = X.select_dtypes(include=['object']).columns # 범주형 피처 선택
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True) # 원-핫 인코딩 적용
        
    elif dataset_name == 'glass': # 유리 식별 데이터셋 처리
        file_path = os.path.join(base_data_path, 'glass.data') # 파일 경로 생성
        df = pd.read_csv(file_path, header=None) # CSV 파일 로드
        X = df.drop(columns=[0, 10]) # ID와 레이블 열을 제외하고 피처(X)로 사용
        y = df[10] - 1 # 1부터 시작하는 레이블을 0부터 시작하도록 1씩 빼줌
        
    elif dataset_name == 'steel_faults': # 강판 불량 유형 데이터셋 처리
        file_path = os.path.join(base_data_path, 'Faults.NNA') # 파일 경로 생성
        df = pd.read_csv(file_path, header=None, sep=r'\s+') # 공백으로 구분된 파일 로드
        X = df.iloc[:, :-7] # 마지막 7개 열을 제외하고 피처(X)로 사용
        y = np.argmax(df.iloc[:, -7:].values, axis=1) # 마지막 7개 열(원-핫 인코딩된 레이블)에서 가장 큰 값의 인덱스를 찾아 레이블로 변환
        y = pd.Series(y) # NumPy 배열을 Pandas Series로 변환

    elif dataset_name == 'covertype': # 산림 피복 유형 데이터셋 처리
        file_path = os.path.join(base_data_path, 'covtype.data') # 파일 경로 생성
        df = pd.read_csv(file_path, header=None) # CSV 파일 로드
        X = df.drop(columns=[54]) # 마지막 열을 제외하고 피처(X)로 사용
        y = df[54] - 1 # 1부터 시작하는 레이블을 0부터 시작하도록 1씩 빼줌
        
    elif dataset_name == 'yeast': # 효모 단백질 데이터셋 처리
        file_path = os.path.join(base_data_path, 'yeast.data') # 파일 경로 생성
        df = pd.read_csv(file_path, header=None, sep=r'\s+') # 공백으로 구분된 파일 로드
        X = df.drop(columns=[0, 9]) # ID와 레이블 열을 제외하고 피처(X)로 사용
        y, _ = pd.factorize(df[9]) # 문자열 레이블을 0부터 시작하는 정수 인덱스로 변환
        y = pd.Series(y) # NumPy 배열을 Pandas Series로 변환

    elif dataset_name == 'page_blocks': # 웹 페이지 블록 데이터셋 처리
        file_path = os.path.join(base_data_path, 'page-blocks.data') # 파일 경로 생성
        df = pd.read_csv(file_path, header=None, sep=r'\s+') # 공백으로 구분된 파일 로드
        X = df.drop(columns=[10]) # 마지막 열을 제외하고 피처(X)로 사용
        y = df[10] - 1 # 1부터 시작하는 레이블을 0부터 시작하도록 1씩 빼줌
        
    else: # 목록에 없는 데이터셋 이름이 들어온 경우
        raise ValueError(f"알 수 없는 데이터셋 이름입니다: {dataset_name}") # 에러 발생

    # 데이터를 학습용과 테스트용으로 분할 (aps_failure 제외)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # stratify=y: 원본 데이터의 클래스 비율을 유지하며 분할
    )
    
    # 숫자형 피처들의 스케일을 표준화하기 위한 객체 생성
    scaler = StandardScaler()
    
    # 데이터가 Pandas DataFrame 형태인지 확인 (컬럼명, 인덱스 보존을 위해)
    if isinstance(X_train, pd.DataFrame):
        X_train_cols = X_train.columns # 학습 데이터의 컬럼명 저장
        X_train_idx = X_train.index # 학습 데이터의 인덱스 저장
        X_test_idx = X_test.index # 테스트 데이터의 인덱스 저장
        
        # 학습 데이터 기준으로 스케일러 학습 및 변환
        X_train = scaler.fit_transform(X_train)
        # 학습된 스케일러로 테스트 데이터 변환
        X_test = scaler.transform(X_test)
        
        # NumPy 배열로 변환된 데이터를 다시 DataFrame으로 만들어 컬럼명과 인덱스 복원
        X_train = pd.DataFrame(X_train, columns=X_train_cols, index=X_train_idx)
        X_test = pd.DataFrame(X_test, columns=X_train_cols, index=X_test_idx)
    else: # 데이터가 NumPy 배열이면
        # 바로 스케일링 적용
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 최종적으로 전처리된 데이터 4종류(학습용 X,y / 테스트용 X,y)를 반환
    return X_train, X_test, y_train, y_test