# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
import pickle as pkl
import database_bw as db
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# 로그 데이터를 Random Survival Forest에 맞는 데이터로 변환
class dh_preprocessing:
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df["거래날짜"] = pd.to_datetime(self.df["거래날짜"])
        self.df = self.df.drop(["rowid", "IND", "거래ID", "제품ID", "배송료", "월", "가입기간", "쿠폰코드", 
                                "할인율", "GST", "오프라인비용", "온라인비용", "마케팅비용"], axis = 1)
        
    def generate_new_features(self):
        
        # 고객 별로 기존 변수 추출
        self.gender = self.df.groupby("고객ID")["성별"].first()
        self.location = self.df.groupby("고객ID")["고객지역"].first()
        
        ## 새로운 변수 만들기
        # 고객 별 평균 금액(구매 건수 대비)
        self.mean_price = self.df.groupby("고객ID")["평균금액"].mean()
        # 고객 별 평균 구매 수량
        self.mean_quantity = self.df.groupby("고객ID")["수량"].mean()
        # 고객 별 가장 많이 구매한 제품 카테고리
        self.most_purchased_categories = self.df.groupby("고객ID")["제품카테고리"]\
            .value_counts().unstack(fill_value = 0).idxmax(axis = 1)
        # 고객 별 월 구매 횟수 
        self.months = self.df.groupby(["고객ID", self.df["거래날짜"].dt.to_period("M")])\
            .size().unstack(fill_value = 0)
        # 고객 별 쿠폰 사용 여부
        self.coupon = self.df.groupby("고객ID")["쿠폰상태"].value_counts().unstack(fill_value = 0)\
            .rename(columns = {"Clicked" : "클릭함", "Not Used" : "사용 안함", "Used" : "사용함"})
    
    def generate_y(self):
        
        # 고객 별 이탈 여부 결정
        date = self.df["거래날짜"].max() + pd.Timedelta(1, "days")
        end_date = self.df.groupby("고객ID")["거래날짜"].max()
        start_date = self.df.groupby("고객ID")["거래날짜"].min()
        churned = (((date - end_date).dt.days >= 90).astype(int))
        time = ((end_date - start_date) + pd.Timedelta(1, "days")).dt.days
        # y 변수를 scikit-survival에서 요구하는 structured array로 변경
        self.y = Surv.from_arrays(churned, time, "이탈 여부", "생존 시간")
    
    def apply_my_function(self):
        
        self.generate_new_features()
        self.generate_y()
        
    def return_X_y(self):
        
        # 기존 변수와 새로운 변수를 결합한 데이터프레임 생성
        X = pd.DataFrame({"성별" : self.gender, "지역" : self.location, 
                          "평균 금액" : self.mean_price, "평균 수량" : self.mean_quantity, 
                          "선호 제품군" : self.most_purchased_categories})
        X = pd.concat([X, self.months, self.coupon], axis = 1)
        X.columns = X.columns.astype(str)
        
        return X, self.y
    
# # example
# dh = dh_preprocessing(train)
# dh.apply_my_function()
# X_train, y_train = dh.return_X_y()

# 고객별 이탈률 예측
class churn_prediction:
    
    def __init__(self):
        
        cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(drop = "if_binary", sparse_output = False, handle_unknown = "ignore"))
            ]) 

        num_pipe = Pipeline([
            ("scaling", StandardScaler())
            ])

        trans = ColumnTransformer([
            ("cat", cat_pipe, make_column_selector(dtype_include = "object")),
            ("num", num_pipe, make_column_selector(dtype_include = ["int64", "float64"]))
            ])

        self.pipe = Pipeline([
            ("transform", trans),
            ("rsf", RandomSurvivalForest(random_state = 42))
            ])
        
    def fit(self, X, y):
        
        self.columns = X.columns.tolist()
        self.pipe.fit(X, y)
        
        return self
    
    # 고객 별로 n일 이후의 이탈률을 예측
    def predict_churn_rate(self, X_test, n = 90):
        
        surv_funcs = self.pipe.predict_survival_function(X_test)
        prob_predict = []
        
        for fn in surv_funcs:
            prob = (1 - fn(n)).round(4)
            prob_predict.append(prob)
            
        self.predict = np.array(prob_predict)
        
        return self.predict

    # def feature_importances(self, X_train, y_train):
        
    #     # 시간이 약간 걸림
    #     imp = permutation_importance(self.pipe, X_train, y_train, n_repeats = 5, random_state = 42)
    #     imp_table = pd.DataFrame(imp["importances_mean"], 
    #                              index = self.columns, columns = ["features"])

    #     return imp_table
    
    # 모델의 성능 평가 (cumulative/dynamic AUC, concordance index ipcw)
    def return_metrics(self, X_test, y_train, y_test, times = np.arange(7, 92, 7), tau = 90):
        
        chf_funcs = self.pipe.predict_cumulative_hazard_function(X_test)
        risk_scores = np.row_stack([chf(times) for chf in chf_funcs])
        risk_score_at_tau = [chf(tau) for chf in chf_funcs]
        mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)[1]
        cindex_ipcw = concordance_index_ipcw(y_train, y_test, risk_score_at_tau, tau)[0]
        
        return mean_auc, cindex_ipcw
    
    # 모델의 성능이 저하되었을 경우 최신 데이터를 이용하여 재학습 -> 아직 미구현
    # def refit_model(self, metric):
        
    # 모델과 transformation을 pickle로 저장
    def to_pkl(self, model_file_name = "rsf.pkl", transformer_file_name = "transformer.pkl"):
        
        with open(model_file_name, "wb") as f:
            pkl.dump(self.pipe["rsf"], f)
        with open(transformer_file_name, "wb") as f:
            pkl.dump(self.pipe["transform"], f)
    
    # 최종 결과 테이블을 만들고 OUR_DATABASE.db에 적재 -> 대시보드에 출력 시 db에서 가져와 사용
    def to_result_table(self, X_test, save = False):
        
        self.result_table = pd.DataFrame(
            np.column_stack((X_test, self.predict)),
            columns = X_test.columns.tolist() + ["예측 이탈률"], 
            index = X_test.index
            )
        
        most_purchased_month = self.result_table.filter(like = "2019-").idxmax(axis = 1)
        self.result_table = self.result_table.loc[:, ~self.result_table.columns.str.startswith("2019-")]
        self.result_table.insert(5, "최다 구매 월", most_purchased_month)
        self.result_table = self.result_table.sort_values("예측 이탈률", ascending = False)
        
        if save == True:
            db.create_new_table(self.result_table, "churn_prediction_table")
            
        return self.result_table
    
# # example
# model = churn_prediction()
# model.fit(X_train, y_train)
# churn_predict = model.predict_churn_rate(X_test, 90)
# model.return_metrics(X_test, y_train, y_test)
# model.refit_model()
# model.to_pkl()
# model.to_result_table(X_test)

# 대시보드 시각화 -> 진행중
class dh_visualization:
    
    def __init__(self):
        pass
    
    # db에 저장된 result table을 가져오기
    def get_result_table(self):
        
        self.result_table = db.making_dataframe_our_db("chrun_prediction_table")
        
    ## 대시보드에 출력할 정보와 그래프 만들기
    def caculate_churned_ratio(self, threshold = 0.5):
        
        self.threshold = threshold
        self.total_customer = self.result_table.index.nunique()
        self.is_churned = (self.result_table["예측 이탈률"] >= self.threshold).astype(int)
        self.churned_customer = self.is_churned.index.nunique()
        self.churned_ratio = round(((self.churned_customer / self.total_customer) * 100), 2)
        
    def plot_gender_ratio(self):
        
        total_gender_count = self.result_table["성별"].value_counts().reset_index()
        gender_pie_chart = px.pie(total_gender_count)
        
