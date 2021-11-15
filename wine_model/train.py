import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# 환경설정
np.random.seed(20211115)

# 입력변수
alpha = float(sys.argv[1])
l1_ratio = float(sys.argv[2])

# 데이터 불러오기
wine = pd.read_csv('../data/wine-quality.csv')

# 데이터 가공하기
train, test = train_test_split(wine)
train_X, train_y = train.drop(['quality'], axis=1), train[['quality']]
test_X, test_y = test.drop(['quality'], axis=1), test[['quality']]
with mlflow.start_run():
    # 모델학습
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    lr.fit(train_X, train_y)

    # 성능평가
    predict_y = lr.predict(test_X)
    rmse = np.sqrt(mean_squared_error(test_y, predict_y))
    mae = mean_absolute_error(test_y, predict_y)
    r2 = r2_score(test_y, predict_y)

    ## 성능평가 결과 출력
    print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
    print(f"  RMSE: {rmse}")
    print(f"  MAE : {mae}")
    print(f"  R2  : {r2}")

    ## 모델 기록
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('l1_ratio', l1_ratio)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('mae', mae)
    mlflow.sklearn.log_model(lr, 'model')