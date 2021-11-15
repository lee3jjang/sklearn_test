# 터미널에서 ipynb 실행하기
ipython -c "%run wine_model/train.ipynb"

# 모델 학습하기
mlflow run \
    wine_model \
    -P alpha=0.4 \
    --no-conda

# UI 오픈하기
mlflow server \
    --default-artifact-root ./mlruns \
    --backend-store-uri sqlite:///mlflow.db

mlflow ui

# 모델 서빙하기
mlflow models serve \
    -m runs:/2a40b3a4f04447459a8f4e2f2621fff1/model \
    -p 1234 \
    --no-conda

# 입력값 보내기
curl \
    -d '{"columns": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"], "data": [[7.3000e+00, 1.9000e-01, 2.7000e-01, 1.3900e+01, 5.7000e-02, 4.5000e+01, 1.5500e+02, 9.9807e-01, 2.9400e+00, 4.1000e-01, 8.8000e+00]]}' \
    -H 'Content-Type: application/json' \
    -X POST \
    localhost:1234/invocations