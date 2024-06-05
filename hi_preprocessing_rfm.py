import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def data_preprocessing(marketing_info, onlinesales_info, customer_info, discount_info):
  # 마케팅비용 = 오프라인비용 + 온라인비용
  marketing_info['마케팅비용'] = marketing_info['오프라인비용'] + marketing_info['온라인비용']

  # 총구매금액 = 수량 * 평균금액
  onlinesales_info['총구매금액'] = onlinesales_info['수량'] * onlinesales_info['평균금액']

  # 쿠폰상태 1: 사용, 0: 나머지
  onlinesales_info['쿠폰상태'] = onlinesales_info['쿠폰상태'].map({'Used': 1, 'Not Used': 0, 'Clicked': 0})

  # 고객지역: Chicago = 1, California = 2, New York = 3, New Jersey = 4, Washington DC = 5
  region_mapping = {'Chicago': 1, 'California': 2, 'New York': 3, 'New Jersey': 4, 'Washington DC': 5}
  customer_info['고객지역'] = customer_info['고객지역'].map(region_mapping)

  # 남자 = 1, 여자 = 0
  customer_info['성별'] = customer_info['성별'].map({'남': 1, '여': 0})

  # 고객정보, 할인정보, 마케팅정보, 온라인판매정보 병합
  data = pd.merge(customer_info, onlinesales_info, on='고객ID')

  marketing_info = marketing_info.rename(columns = {"날짜" : "거래날짜"})
  data = pd.merge(data, marketing_info, on='거래날짜')

  data['월'] = pd.to_datetime(data['거래날짜']).dt.month
  month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
  discount_info['월'] = discount_info['월'].map(month_mapping)
  data = pd.merge(data, discount_info, on=['월', '제품카테고리'])
  data = data.drop(['월', '쿠폰코드'], axis = 1)


  # 필요없는 문자 제거
  data['고객ID'] = data['고객ID'].str.replace("USER_", "")
  data['거래ID'] = data['거래ID'].str.replace("Transaction_", "")
  data['제품ID'] = data['제품ID'].str.replace("Product_", "")

  # integer to string
  data = data.astype({"성별": 'str', "고객지역": 'str', "쿠폰상태": 'str', "할인율": 'str'})

  return data



def marketing_preprocessing(data, marketing_info):
  # 마케팅비용 = 오프라인비용 + 온라인비용
  marketing_info['마케팅비용'] = marketing_info['오프라인비용'] + marketing_info['온라인비용']
  marketing_info = marketing_info.rename(columns = {"날짜" : "거래날짜"})
  data = pd.merge(data, marketing_info, on='거래날짜')

  return data


def rfm_data(data):
  # 거래날짜를 datetime 형태로 변환
  data['거래날짜'] = pd.to_datetime(data['거래날짜'])
  data['총구매금액'] = data['수량'] * data['평균금액']
  data['고객ID'] = data['고객ID'].str.replace("USER_", "")
  data['성별'] = data['성별'].str.replace("남", "1").str.replace("여", "0")
  data['거래ID'] = data['거래ID'].str.replace("Transaction_", "")
  data['제품ID'] = data['제품ID'].str.replace("Product_", "")
  #region_mapping = {'Chicago': 1, 'California': 2, 'New York': 3, 'New Jersey': 4, 'Washington DC': 5}
  #data['고객지역'] = data['고객지역'].map(region_mapping)
  data = data.astype({"성별": 'str', "고객지역": 'str', "거래ID": 'str', "제품ID": 'str'})

  # RFM 데이터 생성
  rfm_data = data.groupby('고객ID').agg({
      '거래날짜': lambda x: (data['거래날짜'].max() - x.max()).days,
      '거래ID': 'nunique',
      '총구매금액': 'sum'
      }).rename(columns={'거래날짜': 'Recency', '거래ID': 'Frequency', '총구매금액': 'MonetaryValue'})
  rfm_data.reset_index(inplace=True)
  rfm_data = pd.merge(rfm_data, data, on='고객ID')


  return rfm_data



class numeric_filtering(BaseEstimator, TransformerMixin):
    def __init__(self, check_const_col=True, check_id_col=True):
        self.check_const_col = check_const_col
        self.check_id_col = check_id_col

    def fit(self, X, y=None):
        if self.check_const_col:
            self.constant_col = [i for i in range(X.shape[1]) if X[:,i].std() == 0]
        else:
            self.constant_col = []

        if self.check_id_col:
            self.id_col = [i for i in range(X.shape[1]) if len(np.unique(np.diff(X[:,i]))) == 1]
        else:
            self.id_col = []

        self.rm_cols = self.constant_col + self.id_col
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self

    def transform(self, X):
        return X[:, self.final_cols]

class categorical_filtering(BaseEstimator, TransformerMixin):
    def __init__(self, check_const_col=True, check_id_col=True, check_cardinality=True):
        self.check_const_col = check_const_col
        self.check_id_col = check_id_col
        self.check_cardinality = check_cardinality

    def fit(self, X, y=None):
        if self.check_const_col:
            self.constant_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) == 1]
        else:
            self.constant_col = []

        if self.check_id_col:
            self.id_col = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) == X.shape[0]]
        else:
            self.id_col = []

        if self.check_cardinality:
            self.cardinality = [i for i in range(X.shape[1]) if len(np.unique(X[:,i])) > 50]
        else:
            self.cardinality = []

        self.rm_cols = self.constant_col + self.id_col + self.cardinality
        self.final_cols = [i for i in range(X.shape[1]) if i not in self.rm_cols]
        return self

    def transform(self, X):
        return X[:, self.final_cols]




class RFMClusteringPipeline:
    def __init__(self):
        self.pipeline = self.create_pipeline()
        self.rfm_scaled = None
        self.optimal_clusters = None
        self.kmeans_model = None

    def create_pipeline(self):
        num_pipeline = Pipeline(steps=[
            ('step1',   SimpleImputer(strategy="mean") ),
            ('step2',   numeric_filtering()  ),
            ('step3',   StandardScaler()  ),
        ])

        cat_pipeline = Pipeline(steps=[
            ('step1',   SimpleImputer(strategy="most_frequent") ),
            ('step2',   categorical_filtering()  ),
            ('step3',   OneHotEncoder()  ),
        ])

        transformer = ColumnTransformer(transformers=[
            ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
            ('cat', cat_pipeline, make_column_selector(dtype_exclude=np.number))
        ])

        return transformer

    def fit_transform(self, X):
        self.pipeline.fit(X[['Recency', 'Frequency', 'MonetaryValue']])
        self.rfm_scaled = self.pipeline.transform(X[['Recency', 'Frequency', 'MonetaryValue']])
        return self.rfm_scaled

    def elbow_method(self):
        sse = {}
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=1)
            kmeans.fit(self.rfm_scaled)
            sse[k] = kmeans.inertia_

        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()), marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("SSE")
        plt.title("Elbow Method for Optimal Clusters")
        plt.show()

        return self.find_elbow_point(sse)

    def find_elbow_point(self, sse):
        keys = list(sse.keys())
        for i in range(len(keys) - 1):
            value1 = sse[keys[i]]
            value2 = sse[keys[i + 1]]
            if abs(value1 - value2) >= 250:
                return keys[i + 1]
        return keys[-1]

    def silhouette_method(self):
        silhouette_scores = {}
        for k in range(3, 8):
            kmeans = KMeans(n_clusters=k, random_state=1)
            kmeans.fit(self.rfm_scaled)
            score = silhouette_score(self.rfm_scaled, kmeans.labels_)
            silhouette_scores[k] = score

        plt.figure()
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Scores for Different Number of Clusters")
        plt.show()

        return max(silhouette_scores, key=silhouette_scores.get)

    def fit_kmeans(self, method='silhouette'):
        if self.rfm_scaled is None:
            raise ValueError("Data has not been fit and transformed. Call fit_transform first.")

        if method == 'silhouette':
            self.optimal_clusters = self.silhouette_method()
        elif method == 'elbow':
            self.optimal_clusters = self.elbow_method()
        else:
            raise ValueError("Invalid method. Choose 'silhouette' or 'elbow'.")

        self.kmeans_model = KMeans(n_clusters=self.optimal_clusters, random_state=1)
        clusters = self.kmeans_model.fit_predict(self.rfm_scaled)
        return clusters


def add_cluster(data, clusters, on = '고객ID'):
  clusters = clusters[['고객ID', 'Cluster']]
  data = pd.merge(data, clusters, on=on)
  return data


# Usage:
# rfm_data = rfm_data(ecommerce_df)
# rfm_pipeline = RFMClusteringPipeline()
# rfm_scaled = rfm_pipeline.fit_transform(rfm_data)
# rfm_data['Cluster'] = rfm_pipeline.fit_kmeans(method='silhouette')
# print(rfm_data.head())

