# AI Hackathon: Company Success Prediction

기업 데이터를 바탕으로 성공 확률(0~1)을 예측하는 해커톤입니다.
기업의 재무 상태, 성장성, 고객 수, SNS 영향력 등의 다양한 지표를 활용하여 미래의 성공 가능성을 수치로 정량화합니다.

---

## 프로젝트 개요

- 데이터: 기업 재무 및 성과 데이터 (train.csv, test.csv)  
- 목표 변수: 성공확률 (0~1 사이의 연속 값)  
- 모델: XGBoost Regressor  
- 튜닝: Optuna 기반 하이퍼파라미터 최적화 (5-Fold)  
- 최종 학습 및 예측: 10-Fold 교차 검증 앙상블  
- 평가지표: MAE (Mean Absolute Error)

---

## 전체 파이프라인

### 1. 데이터 전처리

- 설립연도 → 연차로 변환  
- 수치형 결측값 → 중앙값 대체  
- 범주형 변수 → Label Encoding  
- 투자단계 → 순서형 수치 매핑 (Seed → IPO)  
- 불리언 값 → 0/1 매핑

---

### 2. 피처 엔지니어링

**주요 원본 Feature**  
연차, 투자단계, 직원 수, 고객 수, 총 투자금, 연매출, SNS 팔로워 수, 기업가치

**파생 Feature**

- 투자_1인당: 총 투자금 ÷ 직원 수  
- 매출_1인당: 연매출 ÷ 직원 수  
- 투자대비매출: 연매출 ÷ 총 투자금  
- 가치대비투자비율: 총 투자금 ÷ 기업가치  
- 고객당매출: 연매출 ÷ 고객 수  
- 연차_루트: √연차  
- 투자_회수율: 기업가치 ÷ 총 투자금  
- SNS_노출도: 팔로워 수 ÷ 고객 수  
- SNS_영향력: 팔로워 수 ÷ 기업가치  
- 직원당_고객수: 고객 수 ÷ 직원 수  
- 연매출_비율: 연매출 ÷ 기업가치  
- 성장도: 연매출 ÷ 연차  

**제거 Feature**

- 국가, 분야: 범주 수 많고 편향 우려  
- 인수여부, 상장여부: 정보량 부족  
- 직원 수: 파생 변수로 대체됨

---

## 모델 학습 및 예측 전략

### 1. Optuna 하이퍼파라미터 튜닝 (5-Fold 기준)

- KFold(n_splits=5) 기반 MAE 최적화  
- 총 300회 trial을 통해 best hyperparameter 도출  
- 최적 파라미터: learning_rate, max_depth, subsample, colsample_bytree, lambda, alpha 등

---

### 2. 최종 모델 학습 (10-Fold 기준)

- Optuna로 도출된 파라미터로 10-Fold KFold 학습  
- 각 Fold마다 XGBoost 모델 학습 및 MAE 계산  
- 평균 MAE로 전체 성능 평가

---

### 3. 테스트셋 예측 및 앙상블

- 학습된 10개의 모델로 test.csv 예측  
- 모델별 예측값 평균 → 최종 제출값 생성  
- sample_submission.csv 포맷으로 저장

---

## 모델 해석

**SHAP 기반 시각화**  
- 모델의 예측 근거를 설명할 수 있도록 shap.summary_plot 사용  

**XGBoost Feature Importance**  
- gain 기준으로 변수 중요도 분석  

---


