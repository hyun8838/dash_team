import pandas as pd
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class mj_preprocessing:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

        # 필요없는 rowid, IND 컬럼 삭제
        self.df = self.df.drop(columns=['rowid', 'IND'])

        # 거래날짜를 datetime 형식으로 변환
        self.df["거래날짜"] = pd.to_datetime(self.df["거래날짜"])

    # 쿠폰을 쓰지 않은 구매내력에 대해서는 할인율 = 0 으로 수정
    def update_discount(self):
        self.df.loc[self.df['쿠폰상태'] != 'Used', '할인율'] = 0
    
    # 고객 소비액, 순수매출 변수 추가(통합했을 때를 고려해 병욱이랑 동일하게 작성)
    def total_sales(self):
        self.df["고객소비액"] = self.df["수량"] * self.df["평균금액"] * (1 - self.df["할인율"]/100) * (1 + self.df["GST"]) + self.df["배송료"]
        self.df["매출"] = self.df["수량"] * self.df["평균금액"]
  
    # 함수 적용    
    def apply_my_function(self):
        self.update_discount()
        self.total_sales()
    
    # 데이터프레임 만들기
    def return_dataframe(self):
        return self.df



class mj_visualization:
    def __init__(self, df, rfm):
        self.df = df
        self.rfm = rfm
    
    def calculate_summary(self, dataframe, groupby_col, value_col, count_col='고객ID'):
        summary = dataframe.groupby(groupby_col).agg({
            value_col: 'sum',
            count_col: pd.Series.nunique
        }).rename(columns={count_col: '결제 유저 수', value_col: '매출'})
        summary['ARPPU'] = summary['매출'] / summary['결제 유저 수']
        return summary

    
    def plot_summary(self, summary, x_title, y_title_total, y_title_secondary):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        colors = px.colors.qualitative.Plotly
        # 총 매출액 trace
        fig.add_trace(
            go.Bar(x=summary.index, y=summary['매출'], name='총 매출액', marker=dict(color=colors)),
            secondary_y=False,
        )
        
        # 결제 유저 수 trace
        fig.add_trace(
            go.Scatter(x=summary.index, y=summary['결제 유저 수'], mode='lines+markers', name='결제 유저 수', marker=dict(color='green')),
            secondary_y=True,
        )
        
        # ARPPU trace
        fig.add_trace(
            go.Scatter(x=summary.index, y=summary['ARPPU'], mode='lines+markers', name='ARPPU', marker=dict(color='red')),
            secondary_y=True,
        )
        
        # x축 
        fig.update_xaxes(title_text=x_title)
        
        # 이중 y축
        fig.update_yaxes(title_text=y_title_total, secondary_y=False)
        fig.update_yaxes(title_text=y_title_secondary, secondary_y=True)
        
        fig.update_layout(title=f'{x_title}별 총 매출액, 결제 유저 수 및 ARPPU 비교', legend=dict(x=0.1, y=1.1, orientation='h'))
        fig.show()
        return fig
    
    def month_calculate_and_plot_arppu(self):
        summary = self.calculate_summary(self.df, '월', '매출')
        return self.plot_summary(summary, '월', '월별 매출액', '값 (결제 유저 수 & ARPPU)')
    
    def area_calculate_and_plot_arppu(self):
        summary = self.calculate_summary(self.df, '고객지역', '매출')
        return self.plot_summary(summary, '지역', '총 매출액', '값 (결제 유저 수 & ARPPU)')
    
    def calculate_and_plot_arppu_by_subscription_period_grouped(self):
        bins = [0, 10, 20, 30, 40, 50, np.inf]
        labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50+']
        self.df['가입기간 그룹'] = pd.cut(self.df['가입기간'], bins=bins, labels=labels, right=False)
        summary = self.calculate_summary(self.df, '가입기간 그룹', '매출')
        return self.plot_summary(summary, '가입기간 그룹', '총 매출액', '값 (결제 유저 수 & ARPPU)')
    
    def area_calculate_and_plot_mapbox(self):
        summary = self.calculate_summary(self.df, '고객지역', '매출')
        
        # 고객 지역에 해당되는 경도, 위도 지정
        coordinates = {
            'Chicago': {'lat': 41.8781, 'long': -87.6298},
            'California': {'lat': 36.7783, 'long': -119.4179},
            'New York': {'lat': 40.7128, 'long': -74.0060},
            'New Jersey': {'lat': 40.0583, 'long': -74.4057},
            'Washington DC': {'lat': 38.9072, 'long': -77.0369}
        }

        summary['lat'] = summary.index.map(lambda x: coordinates[x]['lat'])
        summary['long'] = summary.index.map(lambda x: coordinates[x]['long'])

        # mapbox로 작성
        fig = px.scatter_mapbox(summary, lat="lat", lon="long", 
                                color="매출", size="매출",
                                text=summary.index,
                                hover_data=["결제 유저 수", "ARPPU"],
                                zoom=3, height=500)
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(title_text='고객지역별 총 매출액, 결제 유저 수 및 ARPPU')

        return fig
    

    # 클러스터별 매출, 결제유저수, ARPPU 비교하기(plotly 그래프)
    def cluster_calculate_and_plot_arppu(self):
        # 클러스터별 총 매출액, 결제 유저 수, ARPPU 계산
        cluster_summary = self.rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Cluster': 'count'
        }).rename(columns={'Cluster': '결제 유저 수'})
        cluster_summary['총 매출액'] = self.rfm.groupby('Cluster')['Monetary'].sum()
        cluster_summary['ARPPU'] = cluster_summary['총 매출액'] / cluster_summary['결제 유저 수']

        # Plotly 그래프 시각화
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors = px.colors.qualitative.Plotly

        # 총 매출액 trace
        fig.add_trace(
            go.Bar(x=cluster_summary.index, y=cluster_summary['총 매출액'], name='총 매출액', marker=dict(color=colors)),
            secondary_y=False,
        )

        # 결제 유저 수 trace
        fig.add_trace(
            go.Scatter(x=cluster_summary.index, y=cluster_summary['결제 유저 수'], mode='lines+markers', name='결제 유저 수', marker=dict(color='green')),
            secondary_y=True,
        )

        # ARPPU trace
        fig.add_trace(
            go.Scatter(x=cluster_summary.index, y=cluster_summary['ARPPU'], mode='lines+markers', name='ARPPU', marker=dict(color='red')),
            secondary_y=True,
        )

        # Set x-axis title
        fig.update_xaxes(title_text='클러스터')

        # Set y-axes titles
        fig.update_yaxes(title_text='총 매출액', secondary_y=False)
        fig.update_yaxes(title_text='값 (결제 유저 수 & ARPPU)', secondary_y=True)

        # Add titles and adjust layout
        fig.update_layout(title='클러스터별 총 매출액, 결제 유저 수 및 ARPPU 비교', legend=dict(x=0.1, y=1.1, orientation='h'))
        
        return fig



## 연관분석
class mj_apriori:
    def __init__(self, min_support=0.005, min_confidence=0.005, min_lift=1, top_n=5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.top_n = top_n

    def prepare_data(self, df):
        df_apriori = df[['고객ID', '제품카테고리', '수량']]
        apriori_data = df_apriori.groupby('고객ID')['제품카테고리'].apply(list).values.tolist()
        unique_lists = []
        for sublist in apriori_data:
            unique_sublist = []
            for item in sublist:
                if item not in unique_sublist:
                    unique_sublist.append(item)
            unique_lists.append(unique_sublist)
        return unique_lists

    def encode_data(self, dataset):
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        te_df = pd.DataFrame(te_ary, columns=te.columns_)
        return te_df

    def find_frequent_itemsets(self, te_df):
        frequent_itemsets = apriori(te_df, min_support=self.min_support, max_len=3, use_colnames=True, verbose=1)
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].map(lambda x: len(x))
        frequent_itemsets.sort_values('support', ascending=False, inplace=True)
        return frequent_itemsets

    def generate_association_rules(self, frequent_itemsets):
        association_rules_df = association_rules(frequent_itemsets, metric='confidence', min_threshold=self.min_confidence)
        sep = association_rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        sep = sep[sep['support'] > self.min_support]
        sep = sep[sep['lift'] > self.min_lift]
        association = sep.sort_values(by='lift', ascending=False).head(self.top_n)
        return association

    def format_rules(self, association):
        top_rules = pd.DataFrame(columns=["제품", "지지도", "신뢰도", "향상도"])
        for i, row in association.iterrows():
            rule = {
                "제품": list(row['antecedents'])[0] + " => " + list(row['consequents'])[0],
                "지지도": round(row['support'], 2),
                "신뢰도": round(row['confidence'], 2),
                "향상도": round(row['lift'], 2)
            }
            top_rules = pd.concat([top_rules, pd.DataFrame([rule])], ignore_index=True)
        return top_rules

    def apriori_analysis(self, df):
        dataset = self.prepare_data(df)
        te_df = self.encode_data(dataset)
        frequent_itemsets = self.find_frequent_itemsets(te_df)
        association = self.generate_association_rules(frequent_itemsets)
        top_rules = self.format_rules(association)

        # 설명 출력
        print("지지도: 두 제품을 모두 구매한 고객 수의 비율")
        print("신뢰도: A를 구매한 고객 중 B를 구매한 고객의 비율")
        print("향상도: 마케팅 효과 증가율")

        return top_rules
    

