
<h1 align="center">🍄 Mushroom Classification — Machine Learning HW1</h1>
<p align="center"><b>Binary Classification of Edible vs Poisonous Mushrooms</b></p>

---

> ⚠️ <b>WARNING</b>  
> 본 프로젝트는 외형적 feature만을 기반으로 버섯의 독성 여부를 예측하는 머신러닝 모델입니다.  
> <b>절대 실제 섭취 여부 판단에 사용해서는 안 됩니다.</b>

---

# Project Overview

이 프로젝트는 버섯의 형태·색상·표면·계절 등 **20개의 feature**를 활용해  
버섯이 **먹을 수 있는지(edible)** 또는 **독성(poisonous)** 인지를 예측하는  
<b>이진 분류 머신러닝 과제</b>입니다.

- Python built-in 라이브러리만 사용 (scikit-learn 금지)
- 모든 알고리즘 **수식부터 직접 구현**

---

# Dataset: `mushroom.csv`

> 전체 feature 설명은 `mushroom_meta.txt` 참고  

| 항목            | 내용                                                           |
|-----------------|----------------------------------------------------------------|
| 데이터 수       | **61,069개**                                                   |
| 클래스          | `e` = edible, `p` = poisonous                                  |
| Feature 개수    | 총 20개                                                        |
| Feature 타입    | **17개 명목형(nominal)** + **3개 수치형(metrical)**           |
| 데이터 특징     | synthetic dataset (랜덤 기반)                                   |
| 주요 변수       | cap, gill, stem, veil, color, habitat, season 등 20개 속성     |

### Feature 예시 요약  
- `cap-diameter` (m): 모자 지름  
- `cap-shape` (n): bell, conical, convex, flat 등  
- `cap-surface`, `stem-surface`: scaly, fibrous, smooth 등  
- `cap-color`, `gill-color`, `stem-color`: 다양한 색상 코드  
- `habitat`: woods, meadows, urban 등  
- `season`: spring, summer, autumn, winter  

명목형 feature는 모두 **1글자 코드**로 인코딩되어 있어 데이터 처리 시 매핑이 필요합니다.

---

# Implemented Algorithms (From Scratch)

본 프로젝트에는 다음 5가지 알고리즘이 구현되었습니다.

---

## ✔️ Accuracy Summary

| Algorithm        | Accuracy |
|------------------|----------|
| **kNN (k=5)**    | **0.8819** |
| Naive Bayes      | 0.8039 |
| Decision Tree    | 0.5549 |
| Neural Network   | 0.5549 |
| SVM (Linear)     | 0.5417 |

> kNN이 가장 높은 성능을 기록 (synthetic + 범주형 비중 높은 데이터 특성)

---

# Algorithm Descriptions

## 1. Decision Tree (ID3)
- Entropy / Information Gain 기반 분기  
- 명목형 feature에 대해 value별 split  
- Tree Depth 제한 적용  
- 데이터 랜덤성이 높아 일반화 성능은 제한됨

---

## 2️. k-Nearest Neighbors (kNN)
- 거리 기반 분류(Euclidean)  
- 명목형 feature → 일치 여부 0/1 처리  
- k = 5 설정  
- 실험 결과 가장 높은 accuracy 달성  

---

## 3. Naive Bayes
- Feature 독립 가정  
- 명목형 feature → 빈도 기반 확률 + Laplace smoothing  
- 수치형 feature → Gaussian  
- 매우 빠르고 baseline 성능 우수  

---

## 4. Neural Network (2-layer MLP)
- hidden layer 16, activation = sigmoid  
- binary cross-entropy  
- backpropagation 직접 구현  
- synthetic 데이터 특성상 높은 성능은 어려움  

---

## 5. Support Vector Machine (Linear SVM)
- Hinge loss + Gradient Descent  
- Kernel 미사용(직접 구현 난이도 고려)  
- 선형 분리 어려운 데이터 → 낮은 accuracy  

---