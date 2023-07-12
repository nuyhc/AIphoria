# Feature Selection

---

**‣ Feature Selection(기능 선택)이란?**  
기계 학습 모델 구축에 사용할 관련 기능의 하위 집합을 선택하는 프로세스  
  
데이터가 많을수록 결과가 더 좋아진다는 것은 항상 옮은 것이 아니다.  
관련 없는 기능(예측에 도움이 되지 않는 기능)과 중복 기능을 포함하면 학습 프로세스가 과적합이 발생하기 쉽게된다.

  
**Feature Selection**을 이용한다면,  

-   해석하기 쉽도록 모델 단순화
-   훈련 시간 단축 및 계산 비용 절감
-   데이터 수집 비용 절감
-   과적합을 줄여 일반화 향상

---

## ❖Filter Method


**Filter Method**는 Feature을 선택하는데 있어 성능을 우선적으로 선택하게 되는데  
p\_value의 성능 유무를 가지고 판단하거나 각 관측 값들의 변화 관계 등을 이용해서 비슷하거나,  
너무 값이 떨어지게 되면, 그 부분을 하나로 보거나 제외하고 측정한다.  
  
(Feature 간의 관계들을 파악하고, 이에 대해 수치적으로 판단하지만, 각각의 데이터들의 관계에 따라  
선택한 피처가 완벽한 선택 방법이라고 판단 할 수는 없음)  
  
필터 방법은 다음과 같음  

-   모델에 관계없이 변수 선택
-   적은 계산 비용
-   일반적으로 낮은 예측 성능 제공  
    (띄)

**Filter Method**는 빠르게 관련 없는 기능을 제거하거나 과적합한 기능들을 제거하기 용이함  

-   Variance method : 대다수/모든 관측값에 대해 동일한 값을 나타내는 기능 제거
-   Correlation method : 서로 높은 상관 관계가 있는 기능 제거
-   Mutual Information Filter : 상호 정보는 기능의 존재/부재가 Y에 대한 올바른 예측을 만드는 데 기여하는 정보의 양을 측정
-   Chi-Square Filter : 음수가 아닌 각 기능과 클래스 사이의 카이제곱 통계를 계산
-   Univariate ROC-AUC or MSE
  
**\[Demo\]**

```
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
# plt.style.use('seaborn-colorblind')
# %matplotlib inline
from feature_selection import filter_method as ft
```

## Load Dataset

```
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data = pd.DataFrame(np.c_[data['data'], data['target']],
                  columns= np.append(data['feature_names'], ['target']))
```

```
data.head(5)
```

|   | mean radius | mean texture | mean perimeter | mean area | mean smoothness | mean compactness | mean concavity | mean concave points | mean symmetry | mean fractal dimension | ... | worst texture | worst perimeter | worst area | worst smoothness | worst compactness | worst concavity | worst concave points | worst symmetry | worst fractal dimension | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 17.99 | 10.38 | 122.80 | 1001.0 | 0.11840 | 0.27760 | 0.3001 | 0.14710 | 0.2419 | 0.07871 | ... | 17.33 | 184.60 | 2019.0 | 0.1622 | 0.6656 | 0.7119 | 0.2654 | 0.4601 | 0.11890 | 0.0 |
| 1 | 20.57 | 17.77 | 132.90 | 1326.0 | 0.08474 | 0.07864 | 0.0869 | 0.07017 | 0.1812 | 0.05667 | ... | 23.41 | 158.80 | 1956.0 | 0.1238 | 0.1866 | 0.2416 | 0.1860 | 0.2750 | 0.08902 | 0.0 |
| 2 | 19.69 | 21.25 | 130.00 | 1203.0 | 0.10960 | 0.15990 | 0.1974 | 0.12790 | 0.2069 | 0.05999 | ... | 25.53 | 152.50 | 1709.0 | 0.1444 | 0.4245 | 0.4504 | 0.2430 | 0.3613 | 0.08758 | 0.0 |
| 3 | 11.42 | 20.38 | 77.58 | 386.1 | 0.14250 | 0.28390 | 0.2414 | 0.10520 | 0.2597 | 0.09744 | ... | 26.50 | 98.87 | 567.7 | 0.2098 | 0.8663 | 0.6869 | 0.2575 | 0.6638 | 0.17300 | 0.0 |
| 4 | 20.29 | 14.34 | 135.10 | 1297.0 | 0.10030 | 0.13280 | 0.1980 | 0.10430 | 0.1809 | 0.05883 | ... | 16.67 | 152.20 | 1575.0 | 0.1374 | 0.2050 | 0.4000 | 0.1625 | 0.2364 | 0.07678 | 0.0 |

5 rows × 31 columns

```
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=['target'], axis=1), 
                                                    data.target, test_size=0.2,
                                                    random_state=0)
X_train.shape, X_test.shape
```

```
((455, 30), (114, 30))
```

## Variance Method

특성들의 분산(Variance)을 기준으로 중요한 특성을 선택하는 방법  

**분산**은 데이터의 분포에서 데이터 포인트들이 중심으로부터 얼마나 퍼져있는지 나타내는 지표  
특성의 분산이 낮을수록 해당 특성은 데이터 포인트들이 중심에 모여있는 경향이 있고, 데이터의 변동성이  
작다는 것을 의미함  

**Variance Method는 다음과 같은 절차로 작동한다.**

1.  각 특성의 분산을 계산한다. 이를 통해 각 특성의 데이터 포인트들이 얼마나 분산됭 있는지를 측정  
    낮은 분산 값을 가지는 특성은 데이터 포인트들이 중심에 모여있어 예측에 유용하지 않을 가능성이 높음
2.  분산 값을 기준으로 특성들을 순위화한다. 높은 분산 값을 가진 특성들이 중요하다고 간주되며, 낮은 값은  
    예측에 덜 유용한 특성을 나타낸다.
3.  사용자가 지정한 임계값(threshold)을 사용하여 상위 rank feature을 선택한다. 임계값 이상의 분산 값을  
    가진 특성들을 선택하여 최종적으로 중요한 Feature-Subset(특성의 부분집함)을 형성한다.
 
**계산 비용이 낮고, 불필요한 특성 제거를 통해 모델의 복잡성을 줄일 수 있다. 하지만, 일부 특성은 분산이 낮더라도  
예측에 중요한 역활을 할 수 있는 경우가 발생할 수 있다.**

```
# the original dataset has no constant variable
quasi_constant_feature = ft.constant_feature_detect(data=X_train,threshold=0.9)
```

```
0  variables are found to be almost constant
```

```
# lets create a duumy variable that help us do the demonstration
X_train['dummy'] = np.floor(X_train['worst smoothness']*10)
# variable dummy has> 92% of the observations show one value, 1.0
X_train.dummy.value_counts() / np.float(len(X_train))
```

```
1.0    0.923077
0.0    0.068132
2.0    0.008791
Name: dummy, dtype: float64
```

```
quasi_constant_feature = ft.constant_feature_detect(data=X_train,threshold=0.9)
quasi_constant_feature
```

```
1  variables are found to be almost constant





['dummy']
```

```
# drop that variable
X_train.drop(labels=quasi_constant_feature,axis=1,inplace=True)
print(X_train.shape)
```

```
(455, 30)
```

## Correlation method

특성(feature)들 간의 상관관계를 기반으로 중요한 특성을 선택하는 방법  
  
상관관계가 높은 feature들을 식별하고, 이에 대한 처리를 고려한다. 상관관계가 높은  
특성들은 중복된 정보를 제공할 수 있고, 모델의 성능을 감소시킬 수 있다.  
  
하지만, 두 변수간의 상관관계 이외에 다중관계가 형성되어 있는 경우, 중요한 정보를 제외하는 경우가 발생할 수 있다.

```
corr = ft.corr_feature_detect(data=X_train,threshold=0.9)
# print all the correlated feature groups!
for i in corr:
    print(i,'\n')
```

```
          feature1         feature2      corr
0   mean perimeter      mean radius  0.998185
6   mean perimeter        mean area  0.986692
14  mean perimeter  worst perimeter  0.970507
19  mean perimeter     worst radius  0.969520
33  mean perimeter       worst area  0.941920 

           feature1      feature2      corr
12  perimeter error  radius error  0.978323
30  perimeter error    area error  0.944995 

          feature1             feature2      corr
36  mean concavity  mean concave points  0.914627 

        feature1       feature2      corr
38  mean texture  worst texture  0.908182 

                feature1             feature2      corr
40  worst concave points  mean concave points  0.906312 
```

## Mutual Information Filter

특성들(Features)간의 **상호 정보량(Mutual Information)을** 기반으로 중요한 특성을 선택하는 방법  
  
두 변수 간의 종속성 또는 상호 의존성을 측정하는 정보 이론 기반 개념이다.  
두 변수 사이의 상호 의존성을 측정하여, 한 변수의 값을 알면 다른 변수의 값을 예측하기 용이하다.  

```
# select the top 3 features
mi = ft.mutual_info(X=X_train,y=y_train,select_k=3)
print(mi)
```

```
Index(['mean concave points', 'worst perimeter', 'worst area'], dtype='object')
```

```
# select the top 20% features
mi = ft.mutual_info(X=X_train,y=y_train,select_k=0.2)
print(mi)
```

```
Index(['mean perimeter', 'mean concave points', 'worst radius',
       'worst perimeter', 'worst area', 'worst concave points'],
      dtype='object')
```

## Chi-Square Filter

카이제곱 검정(Chi-Square test)을 기반으로 중요한 특성을 선택하는 방법  

카이제곱 검정은 범주형 변수 간의 독립성을 검정하는 통계적인 방법이다. 따라서, Chi-Square Filter은  
범주형 특성과 target변수 간의 독립성을 검정하여 중요한 특성을 선택한다

```
# select the top 3 features
chi = ft.chi_square_test(X=X_train,y=y_train,select_k=3)
print(chi)
```

```
Index(['mean area', 'area error', 'worst area'], dtype='object')
```

```
# select the top 20% features
chi = ft.chi_square_test(X=X_train,y=y_train,select_k=0.2)
print(chi)
```

```
Index(['mean perimeter', 'mean area', 'area error', 'worst radius',
       'worst perimeter', 'worst area'],
      dtype='object')
```

## Univariate ROC-AUC or MSE

Univariate ROC-AUC는 이진 분류 문제에서, MSE는 회귀 문제에서 모델의 성능을 평가하는데 사용  
  
**Univariate ROC-AUC**

[##_Image|kage@bzzU0W/btsngsHr0rc/VVjVrJOw8t4Rb1C0nIho6k/img.png|CDM|1.3|{"originWidth":689,"originHeight":377,"style":"alignCenter","filename":"edited_스크린샷 2023-07-11 오후 5.59.57.png"}_##]

-   Roc-Auc는 분류 모델의 성능을 평가하는 지표 중 하나로, 이진 분류 문제에서 주로 사용한다.
-   Roc 곡선은 모델의 분류 threshold을 변경함에 따라 Ture Positive Rate, Sensitivity에 대한  
    False Positive Rate 의 변화를 나타냄
-   Roc-Auc는 ROC 곡선 아래 영역의 면적을 계산한 값으로, 모델의 분류 성능을 나타낸다.
-   Roc-Auc 값은 0부터 1 사이의 값을 가지며, 1에 가까울 수록 모델의 성능이 우수함을 나타낸다.  


**MSE(Mean Squared Error)**

[##_Image|kage@byfYmi/btsnaL1TQvr/wRpVh40v2niMO9ZGPczwU1/img.png|CDM|1.3|{"originWidth":769,"originHeight":243,"style":"alignCenter","filename":"edited_스크린샷 2023-07-11 오후 6.00.59.png"}_##]

-   MSE는 회귀 모델의 성능을 평가하는 지표로 사용된다.
-   MSE는 예측값과 실제값 간의 제곱 오차를 평균한 값으로, 오차의 제곱을 사용하므로 양수의 값이 된다.
-   MSE 값이 작을 수록 모델의 예측이 실제값과 잘 일치한다는 것을 의미한다.
-   MSE는 오차의 제곱을 사용하기 떄문에 예측값과 실제값의 차이가 큰 경우에 더 큰 패널티를 부여한다.
-   연속형 값 예측 문제에서 주로사용되고, 0에 가까울 수록 모델의 성능이 우수하다.

```
uni_roc_auc = ft.univariate_roc_auc(X_train=X_train,y_train=y_train,
                                   X_test=X_test,y_test=y_test,threshold=0.8)
print(uni_roc_auc)
```

```
worst perimeter            0.917275
worst area                 0.895840
worst radius               0.893458
worst concave points       0.863131
mean concavity             0.856939
mean radius                0.849000
mean area                  0.839314
worst concavity            0.831375
mean perimeter             0.829628
mean concave points        0.826453
area error                 0.812321
worst compactness          0.742299
radius error               0.740235
mean compactness           0.734360
perimeter error            0.680534
worst texture              0.647666
worst fractal dimension    0.640997
concavity error            0.640203
worst symmetry             0.620991
concave points error       0.618133
compactness error          0.607336
mean symmetry              0.591775
mean texture               0.573357
texture error              0.568593
worst smoothness           0.565100
mean smoothness            0.557637
fractal dimension error    0.542077
smoothness error           0.522706
symmetry error             0.493649
mean fractal dimension     0.475548
dtype: float64
11 out of the 30 featues are kept
mean radius             0.849000
mean perimeter          0.829628
mean area               0.839314
mean concavity          0.856939
mean concave points     0.826453
area error              0.812321
worst radius            0.893458
worst perimeter         0.917275
worst area              0.895840
worst concavity         0.831375
worst concave points    0.863131
dtype: float64
```

```
uni_mse = ft.univariate_mse(X_train=X_train,y_train=y_train,
                            X_test=X_test,y_test=y_test,threshold=0.4)
print(uni_mse)
```

```
mean fractal dimension     0.491228
symmetry error             0.480750
fractal dimension error    0.456140
smoothness error           0.449561
texture error              0.412281
worst smoothness           0.403265
mean smoothness            0.399123
mean texture               0.396930
mean symmetry              0.363060
compactness error          0.361842
concave points error       0.357456
worst fractal dimension    0.355263
worst symmetry             0.350877
worst texture              0.333333
concavity error            0.333333
perimeter error            0.300439
mean compactness           0.258772
worst compactness          0.254386
radius error               0.245614
area error                 0.179825
mean perimeter             0.166667
mean concave points        0.166667
worst concavity            0.162281
mean radius                0.146930
mean concavity             0.142544
mean area                  0.140351
worst concave points       0.123782
worst area                 0.103070
worst radius               0.100877
worst perimeter            0.098684
dtype: float64
6 out of the 30 featues are kept
mean fractal dimension     0.491228
texture error              0.412281
smoothness error           0.449561
symmetry error             0.480750
fractal dimension error    0.456140
worst smoothness           0.403265
dtype: float64
```

---

## ❖Wrapper Method

**Wrapper Method**는 검색 전략을 사용하여 가능한 기능 하위 집합의 공간을 검색하고 각 하위 집합을 ML 알고리즘의 성능 품질로 평가한다.  
  
ML의 예측 정확도 측면에서 가장 좋은 성능을 보이는 Subset을 뽑아낼 수 있는 방법이며, ML을 진행하며 Best Feature Subset을 찾아가는  
방법이기 떄문에 많은 시간과 비용이 발생하지만, 최종적으로는 Best Subset을 찾아가는 방법이기 떄문에 매우 바람직한 방법이다.  

-   매우 계산적으므로 비싸다
-   임의로 정의된 중지 기준이 필요하다
-   일반적으로 주어진 ML 알고리즘에 대해 최고 성능의 하위 집합을 제공하지만, 다른 알고리즘에는 그렇지 않을 수 있다.
-   각 하위 집합에서 새 모델 훈련시킨다.  
  
일반적으로 **검색 전략 그룹**에는 Forward Selection, Backward Elimination 및 Exhaustive Search를 포함하는 순차 검색이 있다.  
  
그렇다면, 언제 검색을 중지해야 할까?  
보통 세 가지가 있다.
    
-   성능 향상
-   성능 저하
-   미리 정의된 기능 수에 도달했을 경우

---

## Foward Selection(전진 선택)

변수가 없는 상태로 시작하여 반복할 때 마다 가장 중요한 변수를 추가하여, 더 이상 성능의 향상이 없을 떄까지 변수를 추가한다.  
(미리 설정된 평가 기준에 따라 최고 성능 알고리즘을 생성한다.)

## Backward Elimination(후방 제거)

모든 변수를 가지고 시작하며, 가장 덜 중요한 변수를 하나씩 제거하면서 모델의 성능을 향상시킨다.  
더 이상 성능의 향상이 없을 때 까지 반복한다.  
(미리 설정된 평가 기준에 따라 최고 성능 알고리즘을 생성한다.)

## Exhaustive Feature Selection

**Exhaustive Feature Selection** 모든 가능한 특정 부분 집합(Feature Subset)을 고려하여 최적의 특성 조합을 찾는 방법이다.  
모든 특정 부분 집합(Feature Subset)을 평가하기 떄문에 정확한 결과를 제공하지만,  
최적의 특성 조합을 선택하기 떄문에 계산 비용이 매우 많이든다.

## RandomForest

[##_Image|kage@ch01pz/btsngUckCIV/MjiCaI0RqP0RsfbuOB4amK/img.png|CDM|1.3|{"originWidth":757,"originHeight":508,"style":"alignCenter","filename":"스크린샷 2023-07-11 오후 5.56.51.png"}_##]

**RandomForest**는 앙상블 학습(Ensemble Learning)의 일종으로,  
여러개의 의사결정 트리를 조합하여 예측 모델을 구성하는 방법이다.  
**RandomForest**는 다수의 의사 결정 트리를 생성하고,  
각 트리의 예측 결과를 모아 다수결이나 평균을 통해 최종 예측을 수행한다.  
  
**RandomForest**의 동작 방식

1.  **의사결정 트리의 앙상블**: 랜덤 포레스트는 여러 개의 의사결정 트리를 동시에 생성하여 앙상블합니다.  
    각 트리는 데이터의 일부만 사용하여 학습하므로 과적합을 방지하고 예측 성능을 향상시킵니다.
2.  **부트스트래핑(Bootstraping)**: 각 의사결정 트리의 학습 데이터는 원본 데이터셋에서 무작위로 중복을 허용하여 추출한 샘플로 구성됩니다.  
    이를 부트스트래핑이라고 합니다. 부트스트래핑을 통해 다양한 학습 데이터셋을 생성하여 다양성을 확보합니다.
3.  **랜덤 특성 선택**: 의사결정 트리를 생성할 때, 각 노드에서 최적의 분할을 결정할 때 랜덤하게 선택된 특성들만 고려합니다.  
    이는 트리 간의 상관 관계를 줄이고 다양한 특성들을 활용하여 예측 성능을 향상시킵니다.
4.  **다수결 또는 평균 예측**: 각 의사결정 트리의 예측 결과를 다수결이나 평균을 통해 최종 예측을 수행합니다.  
    분류 문제의 경우 다수결에 의해 예측 클래스가 결정되며, 회귀 문제의 경우 평균을 통해 예측 값이 계산됩니다.

  
**RandomForestClassifier() 참고 함수**  
  
n\_estimators : 모델에서 사용할 트리 갯수(학습시 생성할 트리 갯수)  
criterion : 분할 품질을 측정하는 기능 (default : gini)  
max\_depth : 트리의 최대 깊이  
min\_samples\_split : 내부 노드를 분할하는데 필요한 최소 샘플 수 (default : 2)  
min\_samples\_leaf : 리프 노드에 있어야 할 최소 샘플 수 (default : 1)  
min\_weight\_fraction\_leaf : min\_sample\_leaf와 같지만 가중치가 부여된 샘플 수에서의 비율  
max\_features : 각 노드에서 분할에 사용할 특징의 최대 수  
max\_leaf\_nodes : 리프 노드의 최대수  
min\_impurity\_decrease : 최소 불순도  
min\_impurity\_split : 나무 성장을 멈추기 위한 임계치  
bootstrap : 부트스트랩(중복허용 샘플링) 사용 여부  
oob\_score : 일반화 정확도를 줄이기 위해 밖의 샘플 사용 여부  
n\_jobs :적합성과 예측성을 위해 병렬로 실행할 작업 수  
random\_state : 난수 seed 설정  
verbose : 실행 과정 출력 여부  
warm\_start : 이전 호출의 솔루션을 재사용하여 합계에 더 많은 견적가를 추가  
class\_weight : 클래스 가중치

---

**\[Demo Code\]**

## Forward Selection

```
# step forward feature selection
# select top 10 features based on the optimal roc_auc and RandomForest Classifier

sfs1 = SFS(RandomForestClassifier(n_jobs=-1,n_estimators=5), 
           k_features=10, 
           forward=True, 
           floating=False, 
           verbose=1,
           scoring='roc_auc',
           cv=3)

sfs1 = sfs1.fit(np.array(X_train), y_train)
```

```
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:   11.4s finished
Features: 1/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  29 out of  29 | elapsed:   11.2s finished
Features: 2/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  28 out of  28 | elapsed:   10.7s finished
Features: 3/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  27 out of  27 | elapsed:   10.3s finished
Features: 4/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  26 out of  26 | elapsed:   10.0s finished
Features: 5/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  25 out of  25 | elapsed:    9.6s finished
Features: 6/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  24 out of  24 | elapsed:    9.2s finished
Features: 7/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  23 out of  23 | elapsed:    8.8s finished
Features: 8/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  22 out of  22 | elapsed:    8.4s finished
Features: 9/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  21 out of  21 | elapsed:    8.1s finished
Features: 10/10
```

```
selected_feat1= X_train.columns[list(sfs1.k_feature_idx_)]
selected_feat1
```

```
Index(['mean texture', 'mean perimeter', 'mean concavity',
       'mean fractal dimension', 'area error', 'compactness error',
       'worst perimeter', 'worst area', 'worst smoothness', 'worst symmetry'],
      dtype='object')
```

**☑︎ SFS 알고리즘 랜덤 프레스트 분류기를 사용하였고, 전방 선택 방법을 사용하여 최적의 10개 특성을 선택하고, ROC-AUC를 기준으로  
모델의 분류 성능을 평가했다. 또한, 교차 검증(cv)을 통해 일반화 성능을 평가하고, 상세한 정보(verbose)를 출력했다.**

## Backward Elimination

```
# step backward feature selection
# select top 10 features based on the optimal roc_auc and RandomForest Classifier

sfs2 = SFS(RandomForestClassifier(n_jobs=-1,n_estimators=5), 
           k_features=10, 
           forward=False, 
           floating=False, 
           verbose=1,
           scoring='roc_auc',
           cv=3)

sfs2 = sfs1.fit(np.array(X_train.fillna(0)), y_train)
```

```
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:   11.5s finished
Features: 1/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  29 out of  29 | elapsed:   11.2s finished
Features: 2/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  28 out of  28 | elapsed:   10.7s finished
Features: 3/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  27 out of  27 | elapsed:   10.2s finished
Features: 4/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  26 out of  26 | elapsed:   10.1s finished
Features: 5/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  25 out of  25 | elapsed:    9.6s finished
Features: 6/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  24 out of  24 | elapsed:    9.2s finished
Features: 7/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  23 out of  23 | elapsed:    8.8s finished
Features: 8/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  22 out of  22 | elapsed:    8.5s finished
Features: 9/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  21 out of  21 | elapsed:    8.2s finished
Features: 10/10
```

```
selected_feat2= X_train.columns[list(sfs2.k_feature_idx_)]
selected_feat2
```

```
Index(['mean area', 'mean compactness', 'texture error', 'area error',
       'compactness error', 'concavity error', 'worst texture',
       'worst perimeter', 'worst smoothness', 'worst concavity'],
      dtype='object')
```

## Exhaustive Feature Selection

```
efs1 = EFS(RandomForestClassifier(n_jobs=-1,n_estimators=5, random_state=0), 
           min_features=1,
           max_features=6, 
           scoring='roc_auc',
           print_progress=True,
           cv=2)

# in order to shorter search time for the demonstration
# we only try all possible 1,2,3,4,5,6
# feature combinations from a dataset of 10 features

efs1 = efs1.fit(np.array(X_train[X_train.columns[0:10]].fillna(0)), y_train)
```

```
Features: 847/847
```

```
selected_feat3= X_train.columns[list(efs1.best_idx_)]
selected_feat3
```

```
Index(['mean radius', 'mean texture', 'mean area', 'mean smoothness',
       'mean concavity'],
      dtype='object')
```

---

## ❖Embedded Method(Filter Method + Wrapper Method)

**:Fliter/Wrapper 방식에서 장점들을 결합한 것**  
자체 변수 선택 프로세스를 활용하여 기능 선택과 분류를 동시에 수행 / Feature 선택과 모델 훈련을 동시에 수행하는 방법이다.  
ML 알고리즘 자체에 내장된 특성 선택 기능을 사용하여 중요한 특성을 식별하는 것이다.  
  
계수가 0이 아닌 Feature가 선택되어, 더 낮은 복잡성으로 모델을 훈련하며, 학습 절차를 최적화한다.  
또한, 기능 간의 상호 작용을 고려하고, Wrapper에 비해 모델을 한 번만 훈련하므로 계산 비용이 저렴하다.  


### 정규화(Regularized Method)란?

:정규화는 모델의 자유도를 줄이기 위해 기계 학습 모델의 다양한 매개변수에 패널티를 추가하는 것으로 구성. 따라서 모델이 훈련 데이터의 노이즈에 적합 할 가능성이 적어 과적합될 가능성이 적다. 제약조건(패널티)는 선형회귀 계수(weight)주어, 과적합을 줄일 수 있다.  
  

**Embedded Method** 정규화 방식은 크게 다음과 같다.  


1.  L1 정규화(Lasso)
2.  L2 정규화(Ridge)
3.  L1/L2 (Elastic net)  

Lasso를 사용한 정규화:Lasso(L1)는 일부 계수를 0으로 축소할 수 있는 속성을 가지고 있다.(0에 수렵하는 과정은 식을 참고할 것)  
이러한 특징은 y(종속변수)에 영향을 적게주는 x를 제거해 줄 수 있다.  

패널티를 높이면 제거되는 기능의 수가 늘어나는데 적절한 패널티의 경우 과적합을 방지하고, 효율성을 높이는데 용이하지만  
너무 높게 설정할 경우 중요한 기능이 배제될 수 있다.

### Ridge를 사용한 정규화

:Reidge(L2)는 큰 숫자들을 줄이는데 사용되는데 1보다 큰 숫자들을 1에 수렴시키려는 속성을 가지고 있다.  
이러한 특징은 x(독립변수)의 값들의 크키(scale)을 조정해서 좀 더 현실적인 데이터로 만들어준다.

---

**\[Demo Code\]**

## Lasso

```
# linear models benefit from feature scaling

scaler = RobustScaler()
scaler.fit(X_train)
```

```
RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
       with_scaling=True)
```

### Data Scaling

: Data Scaling은 데이터 전처리 과정 중의 하나이다.  
피처(feature)들마다 데이터값의 범위가 다 제각각이기 때문에 범위 차이가 클 경우 데이터를 갖고 모델을  
학습할 때 0으로 수렴하거나 무한으로 발산할 수 있다.  
  
즉, 이러한 머신러닝을 Data Scaling 방법을 이용하여 모든 피처들의 데이터 분포나 범위를 동일하게 조정할 수 있다.  


### sklearn - Data Scaling 방법 다섯가지

-   StandardScaler() : 모든 피처들을 평균이 0 , 분산이 1인 정규분포를 갖도록 만들어준다.
-   MinMaxscaler() : 모든 피처들이 0과 1 사이의 데이터값을 갖도록 만들어준다.
-   MaxAbsScaler() : 모든 피처들의 절대값이 0과 1사이에 놓이도록 만드러준다.
-   RobustScaler() : 중간값(median)과 사분위값을 사용하여 StandardScaler보다 데이터가 더 넓게 분포되도록 만들어준다.
-   Normalizer() : 각 행(row)마다 정규화를 진행하여, 한 행의 모든 피처들 사이의 유클리드 거리가 1이 되도록 데이터값을 만들어준다.

```
# fit the LR model
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
sel_.fit(scaler.transform(X_train), y_train)
```

```
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)





SelectFromModel(estimator=LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False),
        max_features=None, norm_order=1, prefit=False, threshold=None)
```

```
# make a list with the selected features
selected_feat = X_train.columns[(sel_.get_support())]

print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))
```

```
total features: 30
selected features: 14
features with coefficients shrank to zero: 16
```

```
# we can identify the removed features like this:
removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
removed_feats
```

```
Index(['mean radius', 'mean perimeter', 'mean area', 'mean smoothness',
       'mean compactness', 'mean concavity', 'mean fractal dimension',
       'texture error', 'perimeter error', 'smoothness error',
       'concavity error', 'concave points error', 'symmetry error',
       'worst radius', 'worst perimeter', 'worst compactness'],
      dtype='object')
```

```
# remove the features from the training and testing set

X_train_selected = sel_.transform(X_train.fillna(0))
X_test_selected = sel_.transform(X_test.fillna(0))

X_train_selected.shape, X_test_selected.shape
```

```
((455, 14), (114, 14))
```

## Random Forest Importance

```
model = embedded_method.rf_importance(X_train=X_train,y_train=y_train,
                             max_depth=10,top_n=10)
```

```
Feature ranking:
1. feature no:27 feature name:worst concave points (0.206316)
2. feature no:22 feature name:worst perimeter (0.147163)
3. feature no:7 feature name:mean concave points (0.100672)
4. feature no:20 feature name:worst radius (0.082449)
5. feature no:6 feature name:mean concavity (0.060420)
6. feature no:2 feature name:mean perimeter (0.048284)
7. feature no:23 feature name:worst area (0.046151)
8. feature no:3 feature name:mean area (0.038594)
9. feature no:13 feature name:area error (0.035627)
10. feature no:0 feature name:mean radius (0.030476)
11. feature no:10 feature name:radius error (0.028711)
12. feature no:26 feature name:worst concavity (0.028533)
13. feature no:12 feature name:perimeter error (0.019986)
14. feature no:21 feature name:worst texture (0.018623)
15. feature no:1 feature name:mean texture (0.013840)
16. feature no:25 feature name:worst compactness (0.013195)
17. feature no:29 feature name:worst fractal dimension (0.011840)
18. feature no:24 feature name:worst smoothness (0.008988)
19. feature no:28 feature name:worst symmetry (0.008973)
20. feature no:18 feature name:symmetry error (0.007378)
21. feature no:11 feature name:texture error (0.006736)
22. feature no:15 feature name:compactness error (0.005464)
23. feature no:19 feature name:fractal dimension error (0.005117)
24. feature no:16 feature name:concavity error (0.004957)
25. feature no:8 feature name:mean symmetry (0.004660)
26. feature no:4 feature name:mean smoothness (0.004614)
27. feature no:9 feature name:mean fractal dimension (0.003689)
28. feature no:17 feature name:concave points error (0.002993)
29. feature no:5 feature name:mean compactness (0.002844)
30. feature no:14 feature name:smoothness error (0.002706)
```

[##_Image|kage@bkHfNF/btsnbJiE2WG/22LnUSyXvNq4tCZZg7RL8K/img.png|CDM|1.3|{"originWidth":776,"originHeight":550,"style":"alignCenter","width":400,"height":284,"filename":"스크린샷 2023-07-11 오후 9.40.08.png"}_##]

```
# select features whose importance > threshold
from sklearn.feature_selection import SelectFromModel

# only 5 features have importance > 0.05
feature_selection = SelectFromModel(model, threshold=0.05,prefit=True) 
selected_feat = X_train.columns[(feature_selection.get_support())]
selected_feat
```

```
Index(['mean concavity', 'mean concave points', 'worst radius',
       'worst perimeter', 'worst concave points'],
      dtype='object')
```

```
# only 12 features have importance > 2 times median
feature_selection2 = SelectFromModel(model, threshold='2*median',prefit=True) 
selected_feat2 = X_train.columns[(feature_selection2.get_support())]
selected_feat2
```

```
Index(['mean radius', 'mean perimeter', 'mean area', 'mean concavity',
       'mean concave points', 'radius error', 'area error', 'worst radius',
       'worst perimeter', 'worst area', 'worst concavity',
       'worst concave points'],
      dtype='object')
```

## Gradient Boosted Trees Importance

```
model = embedded_method.gbt_importance(X_train=X_train,y_train=y_train,
                             max_depth=10,top_n=10)
```

```
Feature ranking:
1. feature no:27 feature name:worst concave points (0.694636)
2. feature no:23 feature name:worst area (0.131077)
3. feature no:4 feature name:mean smoothness (0.033800)
4. feature no:8 feature name:mean symmetry (0.018609)
5. feature no:22 feature name:worst perimeter (0.015998)
6. feature no:21 feature name:worst texture (0.013732)
7. feature no:2 feature name:mean perimeter (0.010792)
8. feature no:26 feature name:worst concavity (0.010138)
9. feature no:17 feature name:concave points error (0.008941)
10. feature no:13 feature name:area error (0.008934)
11. feature no:0 feature name:mean radius (0.007928)
12. feature no:12 feature name:perimeter error (0.006268)
13. feature no:18 feature name:symmetry error (0.005472)
14. feature no:3 feature name:mean area (0.005069)
15. feature no:1 feature name:mean texture (0.005034)
16. feature no:10 feature name:radius error (0.004299)
17. feature no:16 feature name:concavity error (0.003595)
18. feature no:6 feature name:mean concavity (0.003354)
19. feature no:19 feature name:fractal dimension error (0.003092)
20. feature no:14 feature name:smoothness error (0.002149)
21. feature no:29 feature name:worst fractal dimension (0.001952)
22. feature no:25 feature name:worst compactness (0.001149)
23. feature no:9 feature name:mean fractal dimension (0.000942)
24. feature no:11 feature name:texture error (0.000917)
25. feature no:15 feature name:compactness error (0.000671)
26. feature no:5 feature name:mean compactness (0.000636)
27. feature no:20 feature name:worst radius (0.000354)
28. feature no:24 feature name:worst smoothness (0.000305)
29. feature no:28 feature name:worst symmetry (0.000145)
30. feature no:7 feature name:mean concave points (0.000013)
```

[##_Image|kage@ybHj5/btsnhSyDvBx/UDIPKjEfSd552wWtBt2YDK/img.png|CDM|1.3|{"originWidth":762,"originHeight":534,"style":"alignCenter","width":400,"height":280,"filename":"스크린샷 2023-07-11 오후 9.40.45.png"}_##]

```
# select features whose importance > threshold

# only 8 features have importance > 0.01
feature_selection = SelectFromModel(model, threshold=0.01,prefit=True) 
selected_feat = X_train.columns[(feature_selection.get_support())]
selected_feat
```

```
Index(['mean perimeter', 'mean smoothness', 'mean symmetry', 'worst texture',
       'worst perimeter', 'worst area', 'worst concavity',
       'worst concave points'],
      dtype='object')
```

---

참고 [GitHub / Yimeng-Zhang (Amazing-Feature-Engineering)]
블로그 작성 [https://traceofones.tistory.com/entry/Feature-Engineering-Feature-Selection]
