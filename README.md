# AI Hackathon: Company Success Prediction

기업 데이터를 바탕으로 성공 확률(0~1)을 예측하는 해커톤입니다.
기업의 다양한 지표를 활용하여 미래의 성공 가능성을 수치로 정량화합니다.

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

- **Origin Features**  
  : 국가, 분야, 설립연도, 투자단계, 직원 수, 고객수(백만명), 총 투자금(억원), 연매출(억원), SNS 팔로워 수(백만명), 기업가치(백억원), 인수여부, 상장여부

- **설립연도 → 연차로 변환**  
  : 현재연도에서 설립연도를 빼서 '연차' 변수 생성

- **수치형 결측값 처리**  
  : 각 수치형 피처의 중앙값(median)으로 결측값 대체

- **범주형 변수 처리**  
  : '국가', '분야'는 Label Encoding 적용

- **투자단계 처리**  
  : Seed, Series A, ..., IPO 순으로 숫자 매핑 (예: Seed=0, Series A=1, ..., IPO=5)

- **불리언 변수 처리**  
  : 인수여부, 상장여부를 True/False → 1/0으로 변환

---

### 2. 피처 엔지니어링

**파생 Feature**

- 

**제거 Feature**

- 

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

---

## 모델 해석

**SHAP 기반 시각화**  
- 모델의 예측 근거를 설명할 수 있도록 shap.summary_plot 사용  

---


