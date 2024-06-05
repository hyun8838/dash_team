#!/usr/bin/env python
# coding: utf-8

# In[208]:

# 코호트 분석 관련 class

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class CohortAnalysis:
    def __init__(self, total, customer):
        self.total = total
        self.customer = customer

    def calculate_purchase_metrics(self):
        # 최초구매와 최근구매날짜 정의 
        result = self.total.groupby('고객ID').agg(처음거래날짜=('거래날짜', 'min'),
                                                   마지막거래날짜=('거래날짜', 'max')).reset_index()

        # 최초 구매날로부터 구매날짜 계산
        result['전체기간'] = result['마지막거래날짜'] - result['처음거래날짜']

        # 재구매여부 계산
        result['재구매여부'] = result['전체기간'].apply(lambda x: 0 if x == pd.Timedelta(days=0) else 1)

        # 거래횟수 계산
        p_count = self.total.groupby('고객ID').agg(구매횟수=('거래날짜', 'count')).reset_index()
        result = pd.merge(result, p_count, how='left', left_on='고객ID', right_on='고객ID')

        # 고객 table과 병합
        self.total = pd.merge(self.customer, result, how='left', left_on='고객ID', right_on='고객ID')

    def merge_customer_data(self):
        self.total = pd.merge(self.total, self.customer[['고객ID', '최초구매날짜']], on='고객ID')
        
    def merge_customer_data(self):
        self.total = pd.merge(self.total, self.customer[['고객ID', '최초구매날짜']], on='고객ID')    
        
    def calculate_cohort(self):
        self.total['최초구매_월'] = self.total['처음거래날짜'].dt.to_period('M').astype('str').apply(lambda _ : datetime.strptime(_,'%Y-%m'))
        self.total['구매_월'] = self.total['마지막거래날짜'].dt.to_period('M').astype('str').apply(lambda _ : datetime.strptime(_,'%Y-%m'))
        
        cohort = self.total.groupby(['최초구매_월', '구매_월']).agg(n_customers=('고객ID', 'nunique')).reset_index()
        cohort['코호트_기간'] = round((cohort['구매_월'] - cohort['최초구매_월']).apply(lambda x: x / pd.Timedelta(days=30)))

        cohort_size = cohort[cohort['코호트_기간'] == 0][['최초구매_월', 'n_customers']].rename(columns={'n_customers': '코호트_크기'})

        cohort = pd.merge(cohort, cohort_size, on='최초구매_월')
        
        return cohort
        
    def calculate_retention_rate(self, cohort):
        cohort['재구매율'] = cohort['n_customers'] / cohort['코호트_크기']
        cohort['최초구매_월'] = cohort['최초구매_월'].dt.to_period('M')
        
        retention_matrix = cohort.pivot_table(index='최초구매_월', columns='코호트_기간', values='재구매율')
        return retention_matrix
    
    def visualize_retention_rate(self, retention_matrix):
        plt.figure(figsize=(18, 10))
        annot_kws = {"fontsize": 8}
        sns.heatmap(retention_matrix, annot=True, fmt='.1%', cmap='Blues',cbar_kws={'format': '%.0f%%'}, annot_kws=annot_kws)
        plt.title('Cohort Analysis - Retention Rates')
        plt.ylabel('Cohort Group')
        plt.xlabel('Months After First Purchase')
        plt.show()


# 카테고리별 재구매 관련 class

import pandas as pd
import plotly.express as px

class CustomerCategoryAnalysis:
    def __init__(self, online_sales):
        self.online_sales = online_sales

    def calculate_repurchase_periods(self):
        customer_category_groups = self.online_sales.groupby(['고객ID', '제품카테고리'])
        category_repurchase_periods = {}

        for name, group in customer_category_groups:
            group = group.sort_values(by='거래날짜')
            purchase_gaps = group['거래날짜'].diff().dt.days
            purchase_gaps = purchase_gaps.iloc[1:]
            category = name[1]

            if category not in category_repurchase_periods:
                category_repurchase_periods[category] = []
            category_repurchase_periods[category].extend(purchase_gaps)

        average_repurchase_periods = {}
        for category, periods in category_repurchase_periods.items():
            average_repurchase_periods[category] = pd.Series(periods).mean()

        self.average_repurchase_periods = average_repurchase_periods

    def visualize_repurchase_periods(self):
        result_df = pd.DataFrame(list(self.average_repurchase_periods.items()), columns=['제품카테고리', '평균 재구매 주기(일)'])

        klk = self.online_sales.groupby(['고객ID', '거래날짜', '제품카테고리']).size().reset_index(name='n')
        klk = klk.groupby(['고객ID', '제품카테고리']).size().reset_index(name='n2')
        klk = klk[klk['n2'] != 1]
        klk['n2'] = klk['n2'] - 1

        ka = klk.groupby('제품카테고리').agg(
            n=('고객ID', 'size'),
            sum=('n2', 'sum')
        ).reset_index()
        ka['재구매율'] = ka['n'] / ka['sum']

        fig = px.bar(ka, 
                 x='재구매율',
                 y='제품카테고리',  
                 title='재구매율',
                 color='제품카테고리',  
                 orientation='h',
                 text=ka['재구매율'].map(lambda x: f'{x:.3f}'))  
        return fig


# 재구매 여부 예측 관련 class
## 전처리 class
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

class Preprocessor:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        self.categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ])

    def fit_transform(self, df):
        return self.preprocessor.fit_transform(df)

## logistic regression 관련 class
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class RebuyPredictionModel:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.X_train = None
        self.y_train = None
        self.model = None
        self.trans = None

    def train_test_split(self, test_size=0.3):
        train, test = train_test_split(self.data, test_size=test_size)
        return train, test

    def preprocess_data(self, train):
        X_cols = train.drop(columns=[self.target_column]).columns
        numeric_features = train[X_cols].select_dtypes(include=np.number).columns.tolist()
        categorical_features = train[X_cols].select_dtypes(exclude=np.number).columns.tolist()
        trans = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        trans.fit(train[X_cols])
        X_train = trans.transform(train[X_cols])
        num_features = numeric_features
        cat_features = trans.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = list(num_features) + list(cat_features)
        X_train = pd.DataFrame(X_train, columns=all_features)
        y_train = train[self.target_column]
        return X_train, y_train, trans

    def train_model(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def save_model_and_transformation(self, model, trans, model_file="model.pkl", trans_file="transform.pkl"):
        with open(model_file, "wb") as f:
            pkl.dump(model, f)
        with open(trans_file, "wb") as f:
            pkl.dump(trans, f)

    def evaluate_model(self, model, trans, test):
        X_test = trans.transform(test.drop(columns=[self.target_column]))
        pred = model.predict(X_test)
        y_test = test[self.target_column]
        accuracy = np.mean(pred == y_test)
        confusion_matrix = pd.crosstab(y_test, pred)
        return accuracy, confusion_matrix



