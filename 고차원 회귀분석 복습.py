import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# 1차 방정식 만들기
a=2
b=3

x=np.linspace(-4, 4, 100)
y=a*x + b

plt.plot(x,y)

# 2차 방정식 만들기

a=2
b=3
c=4

x=np.linspace(-4, 4, 100)
y=a*x**2 + b*x + c

plt.plot(x,y)

# 3차 방정식 만들기

a=2
b=1
c=4
d=2

x=np.linspace(-4, 4, 100)
y=a*x**3 + b*x**2 + c*x +d

plt.plot(x,y)


# sin함수의 그래프 근처의 점들에 회귀직선을 그어보자

# 그래프확인 예제
k=np.linspace(-5,5,100)
sin_y = np.sin(k)
x=uniform.rvs(size=20, loc=-5, scale=10)
y=np.sin(x) + norm.rvs(size=20, loc=0, scale=0.3)

plt.plot(k, sin_y)
plt.scatter(x,y)

# 임의의 30개 데이터셋
np.random.seed(42)
x=uniform.rvs(size=30, loc=-5, scale=10)
y=np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 30개의 데이터를 train과 val로 나눠보자!
df = pd.DataFrame({
    'x' : x,
    'y' : y
})
train_df = df.iloc[:20]
test_df = df.iloc[20:]

plt.scatter(train_df['x'], train_df['y'])

# train데이터의 회귀직선 그려보기
model = LinearRegression()
x=train_df[['x']]
y=train_df['y']
model.fit(x,y)

reg_line = model.predict(x)
plt.plot(x, reg_line)
plt.scatter(train_df['x'], train_df['y'])

# train데이터의 2차곡선 회귀
train_df['x2'] = train_df['x']**2

x = train_df[['x', 'x2']]
y=train_df['y']
model.fit(x,y)

k = np.linspace(-5, 5, 200)
df_k=pd.DataFrame({
    'x' : k, 'x2' : k**2
})
df_k

reg_line = model.predict(df_k)
plt.plot(k, reg_line, color = 'red')
plt.scatter(train_df['x'], train_df['y'], color='blue')

# train데이터의 3차곡선 회귀
train_df['x3'] = train_df['x']**3
x=train_df[['x', 'x2', 'x3']]
y=train_df['y']

model.fit(x,y)

df_k = pd.DataFrame({
    'x':k, 'x2':k**2, 'x3':k**3
})

reg_line = model.predict(df_k)
plt.plot(k, reg_line)
plt.scatter(train_df['x'], train_df['y'])

# f'{}' 구문으로 더 고차원의 곡선회귀를 그려보자
np.random.seed(45)
x = uniform.rvs(size=30, loc=-5, scale=10)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    'x' : x, 'y' : y
})

train_df = df.iloc[:20]
test_df = df.iloc[20:]

for i in range(2,5):
    train_df[f'x{i}'] = train_df['x'] ** i

x = train_df[['x'] + [f'x{i}' for i in range(2,5)]]
y = train_df['y']

model.fit(x,y)

for i in range(2,5):
    test_df[f'x{i}'] = test_df['x'] ** i
x = test_df[['x'] + [f'x{i}' for i in range(2,5)]]

pred_y = model.predict(x)

plt.plot(x, pred_y)