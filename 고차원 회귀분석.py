from matplotlib import pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins
from sklearn.linear_model import LinearRegression
import pandas as pd
import patsy
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform


df = load_penguins()
penguins=df.dropna()

model=LinearRegression()

# patsy를 사용하여 수식으로 상호작용으로 항 생성
# 0 + 는 절편을 제거함
formula = 'bill_depth_mm ~ 0 + bill_length_mm * species'
y, x = patsy.dmatrices(formula, df,
                       return_type = 'dataframe')
x.iloc[:, 1:]

model.fit(x, y)

model.coef_
model.intercept_

regline_y=model.predict(x)

import numpy as np
index_1=np.where(penguins['species'] == "Adelie")
index_2=np.where(penguins['species'] == "Gentoo")
index_3=np.where(penguins['species'] == "Chinstrap")

sns.scatterplot(data=df, 
                x="bill_length_mm", 
                y="bill_depth_mm",
                hue="species")
plt.plot(penguins["bill_length_mm"].iloc[index_1], regline_y[index_1], color="black")
plt.plot(penguins["bill_length_mm"].iloc[index_2], regline_y[index_2], color="black")
plt.plot(penguins["bill_length_mm"].iloc[index_3], regline_y[index_3], color="black")
plt.xlabel("부리길이")
plt.ylabel("부리깊이")




# 2차방정식 그래프 그리기
a=2
b=3
c=5

x = np.linspace(-8,8,100)
y = a*x**2 +b*x +c

plt.plot(x, y, color='black')
plt.show()


# 3차 곡선의 방정식

plt.clf()
a=1
b=0
c=-10
d=0
e=10

x = np.linspace(-4,4,1000)
y = a*x**4 + b*x**3 + c*x**2 +d*x +e

plt.plot(x, y, color='black')
plt.show()








#####
norm.rvs(size = 1, loc = 0, scale = 3)
x = uniform.rvs(size = 20, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size = 20, loc = 0, scale = 0.3)
k = np.linspace(-4,4,200)
sin_y = np.sin(k)

plt.plot(k, sin_y, color='black')
plt.scatter(x, y, color='blue')


np.random.seed(42)
x = uniform.rvs(size = 30, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size = 30, loc = 0, scale = 0.3)

df = pd.DataFrame({'x' : x,
                   'y' : y})
train_df = df.loc[:19]
train_df
test_df = df.loc[20:]
test_df

plt.scatter(train_df['x'], train_df['y'], color='blue')

model=LinearRegression()
x = train_df[['x']]
y = train_df['y']
model.fit(x, y)

model.coef_
model.intercept_

reg_line = model.predict(train_df[['x']])
plt.plot(train_df[['x']], reg_line, color = 'red')
plt.scatter(train_df['x'], train_df['y'], color='blue')

# 2차 곡선 회귀
train_df['x2'] = train_df['x']**2

x = train_df[['x', 'x2']]
y = train_df['y']

model.fit(x, y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    'x':k, 'x2':k**2
})
df_k


reg_line = model.predict(df_k)
plt.plot(k, reg_line, color = 'red')
plt.scatter(train_df['x'], train_df['y'], color='blue')

# 3차 곡선 회귀
train_df['x3'] = train_df['x']**3

x = train_df[['x', 'x2', 'x3']]
y = train_df['y']

model.fit(x, y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    'x':k, 'x2':k**2, 'x3':k**3
})
df_k


reg_line = model.predict(df_k)
plt.plot(k, reg_line, color = 'red')
plt.scatter(train_df['x'], train_df['y'], color='blue')

# 9차 곡선 회귀
train_df['x4'] = train_df['x']**4
train_df['x5'] = train_df['x']**5
train_df['x6'] = train_df['x']**6
train_df['x7'] = train_df['x']**7
train_df['x8'] = train_df['x']**8
train_df['x9'] = train_df['x']**9

x = train_df[['x', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']]
y = train_df['y']

model.fit(x, y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    'x':k, 'x2':k**2, 'x3':k**3, 'x4':k**4, 'x5':k**5, 'x6':k**6, 'x7':k**7, 'x8':k**8, 'x9':k**9
})
df_k


reg_line = model.predict(df_k)
plt.plot(k, reg_line, color = 'red')
plt.scatter(train_df['x'], train_df['y'], color='blue')


pred_test_df = test_df.copy()
pred_test_df['x2'] = pred_test_df['x']**2
pred_test_df['x3'] = pred_test_df['x']**3
pred_test_df['x4'] = pred_test_df['x']**4
pred_test_df['x5'] = pred_test_df['x']**5
pred_test_df['x6'] = pred_test_df['x']**6
pred_test_df['x7'] = pred_test_df['x']**7
pred_test_df['x8'] = pred_test_df['x']**8
pred_test_df['x9'] = pred_test_df['x']**9

x = pred_test_df[['x', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']]
y = pred_test_df['y']

y_hat = model.predict(x)

sum((test_df['y'] - y_hat)**2)

reg_line_test = model.predict(test_df)
plt.plot(k, reg_line, color = 'red')
plt.scatter(train_df['x'], train_df['y'], color='blue')










#============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "x" : x , "y" : y
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
y = train_df["y"]

model=LinearRegression()
model.fit(x,y)

test_df = df.loc[20:]
test_df

for i in range(2, 21):
    test_df[f"x{i}"] = test_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = test_df[["x"] + [f"x{i}" for i in range(2, 21)]]

y_hat = model.predict(x)

# 모델 성능
sum((test_df["y"] - y_hat)**2)
#======================================================================














# 선생님 레포 houseprice8 참조
# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!
# 원하는 변수 2개
# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/train.csv")
house_test=pd.read_csv("./data/test.csv")
sub_df=pd.read_csv("./data/sample_submission.csv")

house_train.shape
house_test.shape

df = pd.concat([house_train, house_test], ignore_index = True)

house_train["Neighborhood"]
neighborhood_dummies = pd.get_dummies(
    df["Neighborhood"],
    drop_first=True
    )

x= pd.concat([df[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies], axis=1)
y = df["SalePrice"]

train_x = x.iloc[:1460]
test_x = x.iloc[1460:]
train_y = y[:1460]

# val(모의고사셋) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(1460), size = 438, replace=False)

valid_x = train_x.loc[val_index]
train_x = train_x.drop(val_index)
valid_y = train_y[val_index]
train_y = train_y.drop(val_index)

from sklearn.model_selection import train_test_split
train_x, valid_x = train_test_split(train_x, test_size=0.3, random_state=42)
train_y, valid_y = train_test_split(train_y, test_size=0.3, random_state=42)



model = LinearRegression()
model.fit(train_x, train_y)

y_hat = model.predict(valid_x)
np.sqrt(np.mean((valid_y - y_hat)**2))















## 이상치 탐색
# house_train=house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
house_train["Neighborhood"]
neighborhood_dummies = pd.get_dummies(
    house_train["Neighborhood"],
    drop_first=True
    )
# pd.concat([df_a, df_b], axis=1)
x= pd.concat([house_train[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies], axis=1)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

neighborhood_dummies_test = pd.get_dummies(
    house_test["Neighborhood"],
    drop_first=True
    )
test_x= pd.concat([house_test[["GrLivArea", "GarageArea"]], 
                   neighborhood_dummies_test], axis=1)
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)