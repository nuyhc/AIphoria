    # 1_Demo_Data_Explore

## Data Explore(데이터 탐색)

### 1.1 Variable(변수)

타입 - 범주형, 수치형

범주형

- Nominal : 범주 그룹에서 선택한 값을 가진 변수이며 자연 순서는 없습니다. ex) Gender, car types
- Ordinal : 범주가 의미 있게 정렬될 수 있는 범주형 변수 ex) Grade of an exam

수치형

- Discrete : 값이 유한하거나 셀 수 없이 무한한 변수입니다. ex) Number of children in a family
- Continuous : 무한히 많은 셀 수 없는 값을 가질 수 있는 변수입니다. ex) House prices, time passed

### 1.2 Variable Identification(변수 식별)

각 데이터 유형을 식별하는것
실제로 다양한 이유로 인해 변수 유형이 혼합될 수 있습니다. 데이터 정리의 특정 단계 후에 데이터 유형을 변환해야 할 수도 있습니다.

### 1.3 Univariate Analysis(일변량 분석)

단일 변수에 대한 기술 통계량
분류 - 범주형, 수치형

범주형

- 모양: 히스토그램/주파수 표

수치

- 중심 경향: 평균/ 중위수/ 평균
- 산포 : 최소/ 최대/ 범위/ 분위수/ IQR/ MAD/ 분산/ 표준편차/
- 모양: 왜도/ 히스토그램/ 상자 그림

<방법>
pandas.Dataframe.describe() - 숫자형 데이터에 대한 통계 정보
pandas.Dataframe.dtypes - 각 열에 대한 데이터 타입을 반환하는 속성
Barplot - 막대 그래프를 생성하는 데이터 시각화 방법. 막대 그래프는 범주형 데이터의 빈도, 크기, 비교 등을 시각적으로 표현하는 데 사용된다.
Countplot - 각 카테고리별 빈도수를 막대 그래프로 시각화하는 방법. 주로 범주형 데이터의 빈도수를 시각화할 때 사용된다.
Boxplot - 연속형 변수의 분포와 이상치를 시각화하는 방법. 주로 데이터의 중앙값, 사분위수, 이상치 등을 확인하고 비교할 때 사용된다.
Distplot - 연속형 변수의 분포를 시각화하는 방법. 주로 데이터의 분포 형태를 확인하고, 확률 밀도 함수(PDF)를 표시할 때 사용된다.

### 1.4 Bi-variate Analysis(이항 분석)

둘 이상의 변수 사이의 관계를 나타내는 통계량

Scatter Plot(산점도) - 데카르트 좌표를 사용하여 일반적으로 데이터 집합에 대한 두 변수의 값을 표시하는 그래프 또는 수학적 다이어그램의 한 유형. 주로 두 변수 간의 상관관계, 분포, 이상치 등을 확인할 때 사용된다.
Correlation Plot(상관 그림) - 주로 데이터셋의 변수들 간의 상관관계를 확인하고, 변수들 간의 패턴이나 의존성을 시각적으로 파악할 때 사용됩니다.
Heat Map(열 지도) - 렬에 포함된 개별 값이 색상으로 표현되는 데이터의 그래픽 표현

### 실습

```python
use_cols = [
    'Pclass', 'Sex', 'Age', 'Fare', 'SibSp',
    'Survived'
]

data = pd.read_csv('../data/train.csv', usecols=use_cols)
data.head(3)
```

- usecols를 통해 필요한 열만 선택하여 데이터를 읽어오는 작업을 수행할 수 있다.
- data.head(3)처럼 head 괄호 안에 숫자를 넣어 무조건 상위 5개 자료가 아닌 위에서부터 원하는 자료만큼 뽑아낼 수 있다.

```js
str_var_list, num_var_list, (all_var_list = explore.get_dtypes((data = data)));
```

주어진 데이터프레임에서 변수의 데이터 유형을 분석하는 함수

#### describe

```js
explore.describe(data=data,output_path=r'./output/')
```

#### barplot

```js
explore.discrete_var_barplot(
  (x = "Pclass"),
  (y = "Survived"),
  (data = data),
  (output_path = "./output/")
);
```

#### countplot

```js
explore.discrete_var_countplot(
  (x = "Pclass"),
  (data = data),
  (output_path = "./output/")
);
```

#### boxplot

```js
explore.discrete_var_boxplot(
  (x = "Pclass"),
  (y = "Fare"),
  (data = data),
  (output_path = "./output/")
);
```

#### distplot

```js
explore.continuous_var_distplot(
  (x = data["Fare"]),
  (output_path = "./output/")
);
```

#### scatterplot

```js
explore.scatter_plot(
  (x = data.Fare),
  (y = data.Pclass),
  (data = data),
  (output_path = "./output/")
);
```

#### correlation plot

```js
explore.correlation_plot((data = data), (output_path = "./output/"));
```

#### heat map

```js
flights = sns.load_dataset("flights")
print(flights.head(5))
# explore.heatmap(data=data[['Sex','Survived']])
flights = flights.pivot("month", "year", "passengers")
explore.heatmap(data=flights,output_path='./output/')
```
