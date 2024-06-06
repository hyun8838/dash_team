#!/usr/bin/env python
# coding: utf-8

# In[208]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# In[35]:


# train = pd.read_csv("./train.csv")
# val = pd.read_csv("./val.csv")
# test = pd.read_csv("./test.csv")


# # bw preprocessing class

# In[121]:


import pandas as pd

class bw_preprocessing:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df["거래날짜"] = pd.to_datetime(self.df["거래날짜"])
    
    # 쿠폰을 쓰지 않은 구매내력에 대해서는 할인율 = 0 으로 수정
    def update_discount(self):
        self.df.loc[self.df['쿠폰상태'] != 'Used', '할인율'] = 0
    
    # 고객 소비액, 순수매출 변수 추가
    def total_sales(self):
        self.df["고객소비액"] = self.df["수량"] * self.df["평균금액"] * (1 - self.df["할인율"]/100) * (1 + self.df["GST"]) + self.df["배송료"]
        self.df["매출"] = self.df["수량"] * self.df["평균금액"]

    # 첫거래, 마지막거래, 거래날차이 ,재구매여부 변수 추가
    def add_transaction_info(self):
        result = self.df.groupby('고객ID').agg(처음거래날짜=('거래날짜', 'min'),
                                              마지막거래날짜=('거래날짜', 'max')).reset_index()

        result['마지막거래날짜'] = pd.to_datetime(result['마지막거래날짜'])
        result['처음거래날짜'] = pd.to_datetime(result['처음거래날짜'])

        result['거래날차이'] = result['마지막거래날짜'] - result['처음거래날짜']

        result['거래날차이'] = result['거래날차이'].astype(str)
        result['재방문여부'] = result['거래날차이'].apply(lambda x: 0 if x == "0 days" else 1)
        
        self.df = self.df.merge(result, on=["고객ID"],how="left")    
     
    # 함수 적용    
    def apply_my_function(self):
        self.update_discount()
        self.total_sales()
        self.add_transaction_info()
    
    
    # 데이터프레임 만들기
    def return_dataframe(self):
        return self.df

# # 사용 예시
# bw = bw_preprocessing(val)
# bw.apply_my_function()
# bw_df = bw.return_dataframe()
# bw_df


# In[166]:


import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
import math

class RFMProcessor:
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.data["거래날짜"] = pd.to_datetime(self.data["거래날짜"])
        self.rfm = None
    
    def total_sales(self):
        self.data["고객소비액"] = self.data["수량"] * self.data["평균금액"] * (1 - self.data["할인율"]/100) * (1 + self.data["GST"]) + self.data["배송료"]
    
    def calculate_rfm(self):
        # Reference date for Recency calculation is set to '2020-01-01 00:00:00'
        reference_date = self.data["거래날짜"].max() + datetime.timedelta(days=1) 
        print('Reference Date:', reference_date)
        self.data['Recency'] = (reference_date - self.data["거래날짜"]).dt.days 

        # Recency calculation
        rfm = self.data[['고객ID', 'Recency']].groupby("고객ID").min().reset_index()

        # Frequency calculation
        freq = self.data[['고객ID', '거래ID']].drop_duplicates().groupby('고객ID').count().reset_index()
        freq.columns = ['고객ID', 'Frequency']
        rfm = pd.merge(rfm, freq, how='left')

        # Monetary calculation
        monetary = self.data[['고객ID', '고객소비액']].groupby("고객ID").sum().reset_index()
        rfm = rfm.merge(monetary, on="고객ID", how="left").rename(columns={"고객소비액": "Monetary"})

        self.rfm = rfm
    
    def separate_outliers(self, column):
        Q1 = self.rfm[column].quantile(0.25)
        Q3 = self.rfm[column].quantile(0.75)
        IQR = Q3 - Q1

        filter = (self.rfm[column] >= Q1 - 1.5 * IQR) & (self.rfm[column] <= Q3 + 1.5 * IQR)
        
        df_outliers = self.rfm.loc[~filter]  # DataFrame containing only outliers
        self.rfm = self.rfm.loc[filter]  # DataFrame with outliers removed
        
        return self.rfm, df_outliers
    
    def process_data(self):
        self.total_sales()
        self.calculate_rfm()
        rfm_without_outliers_f, rfm_outliers_f = self.separate_outliers('Frequency')
        rfm_without_outliers_m, rfm_outliers_m = self.separate_outliers('Monetary')
        
        # Select rows without outliers
        rfm_without_outliers = self.rfm.loc[
            self.rfm.index.isin(rfm_without_outliers_f.index) & self.rfm.index.isin(rfm_without_outliers_m.index)
        ]
        
        # Select rows with outliers
        rfm_outliers = self.rfm.loc[
            self.rfm.index.isin(rfm_outliers_f.index) | self.rfm.index.isin(rfm_outliers_m.index)
        ]
        
        # Log transformation for R, F, M columns
        rfm_without_outliers['Recency_log'] = rfm_without_outliers['Recency'].apply(math.log)
        rfm_without_outliers['Frequency_log'] = rfm_without_outliers['Frequency'].apply(math.log)
        rfm_without_outliers['Monetary_log'] = rfm_without_outliers['Monetary'].apply(math.log)
        
        # Feature selection for normalization
        features = ['Monetary_log', 'Recency_log','Frequency_log']

        # Normalize columns
        X_subset = rfm_without_outliers[features]
        st = StandardScaler().fit(X_subset)
        X_scaled = st.transform(X_subset)
        rfm_without_outliers_log = pd.DataFrame(X_scaled, columns=X_subset.columns)
        
        return rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled

# # Usage
# processor = RFMProcessor(test)
# rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()

# print("RFM without outliers:")
# print(rfm_without_outliers)

# print("\nOutliers:")
# print(rfm_outliers)

# print("\nNormalized RFM without outliers:")
# print(rfm_without_outliers_log)


# # RFM class

# In[219]:


import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math

class RFMProcessor:
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.data["거래날짜"] = pd.to_datetime(self.data["거래날짜"])
        self.rfm = None
        self.kmeans = None  # 클러스터링 모델
    
    # rfm을 계산하기 위한 컬럼 생성
    def total_sales(self):
        self.data["고객소비액"] = self.data["수량"] * self.data["평균금액"] * (1 - self.data["할인율"]/100) * (1 + self.data["GST"]) + self.data["배송료"]

    # rfm 계산
    def calculate_rfm(self):
        reference_date = self.data["거래날짜"].max() + datetime.timedelta(days=1) 
        self.data['Recency'] = (reference_date - self.data["거래날짜"]).dt.days 

        rfm = self.data[['고객ID', 'Recency']].groupby("고객ID").min().reset_index()
        freq = self.data[['고객ID', '거래ID']].drop_duplicates().groupby('고객ID').count().reset_index()
        freq.columns = ['고객ID', 'Frequency']
        rfm = pd.merge(rfm, freq, how='left')
        monetary = self.data[['고객ID', '고객소비액']].groupby("고객ID").sum().reset_index()
        rfm = rfm.merge(monetary, on="고객ID", how="left").rename(columns={"고객소비액": "Monetary"})
        self.rfm = rfm
    
    # 이상치 고객을 분류하기 위함
    def separate_outliers(self, column):
        Q1 = self.rfm[column].quantile(0.25)
        Q3 = self.rfm[column].quantile(0.75)
        IQR = Q3 - Q1

        filter = (self.rfm[column] >= Q1 - 1.5 * IQR) & (self.rfm[column] <= Q3 + 1.5 * IQR)
        
        df_outliers = self.rfm.loc[~filter]
        self.rfm = self.rfm.loc[filter]
        
        return self.rfm, df_outliers
    
    # rfm 생성 및 이상치 분류 함수 적용
    def process_data(self):
        self.total_sales()
        self.calculate_rfm()
        rfm_without_outliers_f, rfm_outliers_f = self.separate_outliers('Frequency')
        rfm_without_outliers_m, rfm_outliers_m = self.separate_outliers('Monetary')
        
        rfm_without_outliers = self.rfm.loc[
            self.rfm.index.isin(rfm_without_outliers_f.index) & self.rfm.index.isin(rfm_without_outliers_m.index)
        ]
        
        rfm_outliers = self.rfm.loc[
            self.rfm.index.isin(rfm_outliers_f.index) | self.rfm.index.isin(rfm_outliers_m.index)
        ]
        
        rfm_without_outliers['Recency_log'] = rfm_without_outliers['Recency'].apply(math.log)
        rfm_without_outliers['Frequency_log'] = rfm_without_outliers['Frequency'].apply(math.log)
        rfm_without_outliers['Monetary_log'] = rfm_without_outliers['Monetary'].apply(math.log)
        
        features = ['Monetary_log', 'Recency_log', 'Frequency_log']
        X_subset = rfm_without_outliers[features]
        st = StandardScaler().fit(X_subset)
        X_scaled = st.transform(X_subset)
        rfm_without_outliers_log = pd.DataFrame(X_scaled, columns=X_subset.columns)
        
        return rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled
    
    # clustering 모델링 
    def fit_clustering(self, X_scaled, n_clusters=4):
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=123)
        self.kmeans.fit(X_scaled)
        self.rfm['Cluster'] = self.kmeans.predict(X_scaled)
    
    # clustering 예측 & join 
    def predict(self, new_data):
        new_data["거래날짜"] = pd.to_datetime(new_data["거래날짜"])
        reference_date = self.data["거래날짜"].max() + datetime.timedelta(days=1)
        new_data['Recency'] = (reference_date - new_data["거래날짜"]).dt.days
        new_data["고객소비액"] = new_data["수량"] * new_data["평균금액"] * (1 - new_data["할인율"]/100) * (1 + new_data["GST"]) + new_data["배송료"]
        
        rfm = new_data[['고객ID', 'Recency']].groupby("고객ID").min().reset_index()
        freq = new_data[['고객ID', '거래ID']].drop_duplicates().groupby('고객ID').count().reset_index()
        freq.columns = ['고객ID', 'Frequency']
        rfm = pd.merge(rfm, freq, how='left')
        monetary = new_data[['고객ID', '고객소비액']].groupby("고객ID").sum().reset_index()
        rfm = rfm.merge(monetary, on="고객ID", how="left").rename(columns={"고객소비액": "Monetary"})
        
        rfm['Recency_log'] = rfm['Recency'].apply(math.log)
        rfm['Frequency_log'] = rfm['Frequency'].apply(math.log)
        rfm['Monetary_log'] = rfm['Monetary'].apply(math.log)
        
        features = ['Monetary_log', 'Recency_log', 'Frequency_log']
        X_subset = rfm[features]
        X_scaled = StandardScaler().fit_transform(X_subset)
        
        # 이상치 처리
        outlier_indices = rfm.loc[
            (rfm['Frequency'] > self.rfm['Frequency'].quantile(0.75) + 1.5 * (self.rfm['Frequency'].quantile(0.75) - self.rfm['Frequency'].quantile(0.25))) |
            (rfm['Frequency'] < self.rfm['Frequency'].quantile(0.25) - 1.5 * (self.rfm['Frequency'].quantile(0.75) - self.rfm['Frequency'].quantile(0.25))) |
            (rfm['Monetary'] > self.rfm['Monetary'].quantile(0.75) + 1.5 * (self.rfm['Monetary'].quantile(0.75) - self.rfm['Monetary'].quantile(0.25))) |
            (rfm['Monetary'] < self.rfm['Monetary'].quantile(0.25) - 1.5 * (self.rfm['Monetary'].quantile(0.75) - self.rfm['Monetary'].quantile(0.25)))
        ].index
        
        # 이상치에 클러스터 값 -1 부여
        rfm.loc[outlier_indices, 'Cluster'] = -1  
        non_outlier_indices = rfm.index.difference(outlier_indices)
        
        rfm.loc[non_outlier_indices, 'Cluster'] = self.kmeans.predict(X_scaled[non_outlier_indices])
        
        return rfm

# # 적용
# processor = RFMProcessor(train) 
# rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
# processor.fit_clustering(X_scaled, n_clusters=4)
# new_data_predictions = processor.predict(train)
# new_data_predictions


# # Visualization Class

# In[211]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class Visualization:
    def __init__(self, data):
        self.data = data

    def plot_clusters(self, labels, x='Recency_log', y='Monetary_log', z='Frequency_log'):
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(self.data[x], self.data[y], self.data[z], c=labels, cmap='viridis', s=60)
        
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.show()

    def plot_boxplots(self):
        fig, axs = plt.subplots(ncols=3, figsize=(20, 7))

        sns.boxplot(data=self.data, x='Cluster', y='Recency', ax=axs[0])
        axs[0].set_title('Recency')

        sns.boxplot(data=self.data, x='Cluster', y='Frequency', ax=axs[1])
        axs[1].set_title('Frequency')
        axs[1].set_ylim(0, 60)

        sns.boxplot(data=self.data, x='Cluster', y='Monetary', ax=axs[2])
        axs[2].set_title('Monetary')
        axs[2].set_ylim(0, 15000)

        plt.show()

# # Usage
# visualization = Visualization(new_data_predictions)
# visualization.plot_clusters(new_data_predictions["Cluster"])
# visualization.plot_boxplots()


# 고객유형 0은 Recency가 높지만, Frequency와 Monetary가 보통 재방문 및 이벤트전략을 펼치는게 좋아보임. (우수고객)
# 
# 고객유형 1은 Recency가 매우 높고 Frequency와 Monetary가 낮음(이탈고객)
# 
# 고객유형 2은 Recency가 높고, Frequency와 Monetary가 보통 (잠재고객)
# 
# 고객유형 3은 고객유형 0과 Frequency와 Monetary는 비슷하지만, Recency가 매우 낮은것으로 보아 최근에 거래가 이루어진 집단(관심고객)
# 
# 고객유형 -1는 이상치 고객으로 매우높은 Frequency와 Monetary값을 가짐(VIP고객)
# 

# # Mapping_cluster function

# In[214]:


def mapping_cluster(df):
    # 클러스터 값과 한글 레이블을 매핑하는 사전
    cluster_map = {
        0: '우수고객',
        1: '이탈고객',
        2: '잠재고객',
        3: '관심고객',
        -1: 'VIP고객'  # 이상치 고객
    }
    
    # 클러스터 값을 한글 레이블로 변환
    df['고객분류'] = df['Cluster'].map(cluster_map)
    
    return df


# In[217]:


# cluster_data = mapping_cluster(new_data_predictions)
# cluster = cluster_data[['고객ID', '고객분류']]


# In[218]:


# bw = bw_df.merge(cluster, on = '고객ID', how = 'left')
# bw


# # 1번째 대시보드를 위한 class

# In[2]:


class first_dash:
    def __init__(self, rfm_clusters):
        self.rfm_clusters = rfm_clusters
        self.rfm_clusters_final = None

    def calculate_cluster_means(self):
        self.rfm_clusters_grouped = self.rfm_clusters.groupby('고객분류').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
        }).round(0).reset_index()

    def calculate_cluster_counts_and_revenue(self):
        self.rfm_clusters_counts = self.rfm_clusters['고객분류'].value_counts().reset_index()
        self.rfm_clusters_counts.columns = ['고객분류', '고객수']

        self.cluster_revenue = self.rfm_clusters.groupby('고객분류')['Monetary'].sum().reset_index()
        self.cluster_revenue.columns = ['고객분류', '매출']

    def merge_cluster_data(self):
        self.rfm_clusters_final = pd.merge(self.rfm_clusters_grouped, self.rfm_clusters_counts, on='고객분류')
        self.rfm_clusters_final = pd.merge(self.rfm_clusters_final, self.cluster_revenue, on='고객분류')

    def calculate_ratios(self):
        self.rfm_clusters_final['고객수 비율'] = (self.rfm_clusters_final['고객수'] / self.rfm_clusters_final['고객수'].sum() * 100).round().astype(str) + '%'
        self.rfm_clusters_final['매출 비율'] = (self.rfm_clusters_final['매출'] / self.rfm_clusters_final['매출'].sum() * 100).round().astype(str) + '%'

    def rearrange_columns(self):
        cols = self.rfm_clusters_final.columns.tolist()
        cols = cols[0:1] + cols[-1:] + cols[1:-1]
        self.rfm_clusters_final = self.rfm_clusters_final[cols]

    def reset_index(self):
        self.rfm_clusters_final.reset_index(drop=True, inplace=True)

    def get_final_dataframe(self):
        self.calculate_cluster_means()
        self.calculate_cluster_counts_and_revenue()
        self.merge_cluster_data()
        self.calculate_ratios()
        self.rearrange_columns()
        self.reset_index()
        return self.rfm_clusters_final

    def calculate_ratios_as_float(self):
        customer_ratio = self.rfm_clusters_final['고객수 비율'].str.rstrip('%').astype('float') / 100.0
        purchase_ratio = self.rfm_clusters_final['매출 비율'].str.rstrip('%').astype('float') / 100.0
        return customer_ratio, purchase_ratio

# 사용
# rfm_clusters = train_bw[['고객ID','Recency','Frequency','Monetary','고객분류']]

# analysis = firstdash(rfm_clusters)
# rfm_clusters_final = analysis.get_final_dataframe()
# rfm_clusters_final


# # 2번째는 train_bw 이용

# # 3번째 대시보드를 위한 class

# In[ ]:


class thrid_dash:
    def __init__(self, df):
        self.df = df.copy()
        self.monthly_clustered_customers = None
        self.monthly_clustered_monetary = None

    def create_clustered_summary(self):
        # Group by month and clusters to get unique customer counts
        self.monthly_clustered_customers = self.df.groupby(["월", "고객분류"])["고객ID"].nunique().unstack()

        # Group by month and clusters to get total sales
        self.monthly_clustered_monetary = self.df.groupby(["월", "고객분류"])["매출"].sum().unstack()

        # Calculate total sales for each month
        total_sales = self.df.groupby("월")["매출"].sum()

        # Add total sales as a new column in the monthly_clustered_monetary DataFrame
        self.monthly_clustered_monetary['total'] = total_sales

        # Add total transactions as a new column in the monthly_clustered_monetary DataFrame
        self.monthly_clustered_customers['total'] = self.monthly_clustered_customers.sum(axis=1)

        # Extracting the month from the index and assigning it to a new column
        self.monthly_clustered_customers['month'] = self.monthly_clustered_customers.index
        self.monthly_clustered_monetary['month'] = self.monthly_clustered_monetary.index

    def get_monthly_clustered_customers(self):
        return self.monthly_clustered_customers

    def get_monthly_clustered_monetary(self):
        return self.monthly_clustered_monetary

# 사용
# clustered_summary = thrid(train_bw)
# clustered_summary.create_clustered_summary()

# monthly_clustered_customers = clustered_summary.get_monthly_clustered_customers()
# monthly_clustered_monetary = clustered_summary.get_monthly_clustered_monetary()

# monthly_clustered_customers
# monthly_clustered_monetary


# # 4번째 대시보드를 위한 class

# In[4]:


class fourth_dash:
    def __init__(self, df):
        self.df = df.copy()
        self.grouped_df = None

    def preprocess(self):
        self.df.loc[self.df['쿠폰코드'] == 0, '쿠폰코드'] = 'non-coupon'

    def group_by_columns(self, group_columns, sum_column):
        self.grouped_df = self.df.groupby(group_columns)[sum_column].sum().reset_index()

    def get_grouped_df(self):
        return self.grouped_df

# # 사용
# grouped_df_processor = fourth_dash(train_bw)
# grouped_df_processor.preprocess()
# grouped_df_processor.group_by_columns(['제품카테고리', '월', '쿠폰상태'], '매출')

# grouped_df = grouped_df_processor.get_grouped_df()
# grouped_df


# # 5번째 대시보드를 위한 class

# In[ ]:


class fifth_dash:
    def __init__(self, df):
        self.df = df.copy()
        self.coupon_sales = None

    def preprocess(self):
        self.df.loc[self.df['쿠폰코드'] == 0, '쿠폰코드'] = 'non-coupon'

    def calculate_coupon_sales(self):
        self.coupon_sales = self.df.groupby(['쿠폰코드', '고객분류', '제품카테고리'])['매출'].sum()
        self.coupon_sales = self.coupon_sales.reset_index().sort_values(by=['쿠폰코드', '매출'], ascending=[True, False])

    def get_coupon_sales(self):
        return self.coupon_sales

# # 사용
# coupon_sales_processor = fifth_dash(train_bw)
# coupon_sales_processor.preprocess()
# coupon_sales_processor.calculate_coupon_sales()

# coupon_sales = coupon_sales_processor.get_coupon_sales()
# coupon_sales

