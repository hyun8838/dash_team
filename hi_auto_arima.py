# !pip install pmdarima
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pmdarima as pm

class AutoArimaPipeline:
    def __init__(self):
        pipe1 = Pipeline([
            ('step1', SimpleImputer(strategy="mean")),
            ('step2', StandardScaler()),
        ])

        pipe2 = Pipeline([
            ('step1', SimpleImputer(strategy="most_frequent")),
            ('step2', OneHotEncoder()),
        ])

        self.transform = ColumnTransformer([
            ('num', pipe1, make_column_selector(dtype_include=np.number)),
            ('cat', pipe2, make_column_selector(dtype_exclude=np.number)),
        ])

    def fit(self, train_data):
        self.models = []
        for data in train_data:
            auto_arima_model = pm.auto_arima(data['마케팅비용'])
            self.models.append(auto_arima_model)

    def predict(self, test_data):
        predictions = []
        for i, data in enumerate(test_data):
            marketing_pred = []
            pred_upper = []
            pred_lower = []

            for new_ob in data['마케팅비용']:
                # 새로운 데이터를 이용하여 모델 업데이트
                self.models[i].update(new_ob)

                # 예측
                fc, conf = self.models[i].predict(n_periods=1, return_conf_int=True)
                fc = pd.Series(fc)

                marketing_pred.append(fc.values[0])
                pred_upper.append(conf[0][1])
                pred_lower.append(conf[0][0])

            marketing_pred = pd.Series(marketing_pred, index=data.index)
            marketing_pred = pd.DataFrame(marketing_pred).rename(columns={0: 'pred'})

            marketing_test_pred = pd.concat([data, marketing_pred], axis=1)
            predictions.append(marketing_test_pred)

        return predictions

    # 시계열 데이터 전처리 함수
    def split_data_by_cluster(self, data):
        clusters = np.sort(data['Cluster'].unique())
        data['거래날짜'] = pd.to_datetime(data['거래날짜'], format='%Y-%m-%d')
        out = []
        for cluster in clusters:
            cluster_data = data[data['Cluster'] == cluster][['거래날짜', '마케팅비용', 'Cluster']]
            cluster_data = cluster_data.drop_duplicates(subset=['거래날짜']).sort_values('거래날짜')
            out.append(cluster_data)

        out.append(data[['거래날짜', '마케팅비용', 'Cluster']].drop_duplicates(subset=['거래날짜']).sort_values('거래날짜'))
        return out

    def time_train_test_split(self, data, test_size=0.2):
        test_size = 1 - test_size
        data.set_index('거래날짜', inplace=True)

        train = data[:int(test_size * len(data))]
        test = data[int(test_size * len(data)):]
        return train, test

    def create_train_test_by_cluster(self, data, out):
        clusters = np.append(np.sort(data['Cluster'].unique()), np.sort(data['Cluster'].unique())[-1] + 1)
        train = []
        test = []

        for cluster_name, cluster_data in zip(clusters, out):
            cluster_train, cluster_test = self.time_train_test_split(cluster_data)
            train.append(cluster_train)
            test.append(cluster_test)

        return train, test

    def run(self, data, test_size=0.2):
        out = self.split_data_by_cluster(data)
        train, test = self.create_train_test_by_cluster(data, out)

        self.fit(train)
        predictions = self.predict(test)

        for i, prediction in enumerate(predictions):
            print(f"Predictions for Cluster {i}:")
            print(prediction)

# Usage 1:
# auto_arima_pipeline = AutoArimaPipeline()

# auto_arima_pipeline.fit(train)

# predictions = auto_arima_pipeline.predict(test)

# output
# for i, prediction in enumerate(predictions):
#     print(f"Predictions for Cluster {i}:")
#     print(prediction)

# Usage 2:
# auto_arima_pipeline = AutoArimaPipeline()

# auto_arima_pipeline.run(ecommerce_df)
