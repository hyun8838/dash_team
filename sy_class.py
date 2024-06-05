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

