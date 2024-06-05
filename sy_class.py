#!/usr/bin/env python
# coding: utf-8

# In[208]:

# 코호트 분석 관련 class
# asdf
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
                 text='재구매율')  
        return fig





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

