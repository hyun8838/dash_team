import dash
from dash import Input, Output, State, html, dcc, dash_table, MATCH, ALL, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
import time as time_pck
import os
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
#----------------------------------
import database_bw
from bw_class import RFMProcessor
import mj_class

import warnings
warnings.filterwarnings('ignore')

from app import app

layout = html.Div(
    style= {'overflow-x':'hidden'},
    children=[
        dmc.Group(
            direction = 'column',
            grow = True,
            position = 'center',
            spacing = 'sm',
            children=[
                dmc.Title(children='ARPPU Analysis', order=3, style={'font-family':'IntegralCF-ExtraBold', 'text-align':'center', 'color':'slategray'}),
                # dmc.Divider(label='Overview', labelPosition='center', size='xl'),
                dmc.Paper(
                    shadow = 'md',
                    m = 'sm',
                    p = 'md',
                    #style = {'width':'90%'},
                    withBorder = True,
                    children=[
                        dmc.Stack(
                            children=[
                                dmc.Stack(
                                    children=[
                                        dmc.Select(
                                            id='arppu-select',
                                            label='Select ARPPU Analysis Type',
                                            style={'width': '50%', 'margin': 'auto'},
                                            data=[
                                                {'label': 'Cluster Analysis', 'value': 'cluster'},
                                                {'label': 'Monthly ARPPU', 'value': 'monthly'},
                                                {'label': 'Area Analysis', 'value': 'area'},
                                                {'label': 'Subscription Period Analysis', 'value': 'subscription'},
                                                {'label': 'Area Analysis(map)', 'value': 'area(map)'}
                                                
                                            ],
                                            value='monthly'
                                        ),
                                    ]
                                ),
                                dcc.Graph(id='arppu-graph'),
                                dmc.Divider(),
                                html.Div(children=[
                                    html.H3("연관분석"),
                                    html.P("ARPPU를 증가시키기 위해서는 한번에 결제를 더 많이 할 수 있도록 유도해야합니다. "
                                           "고객이 구매할 때 다른 제품 및 서비스도 제안할 수 있는 Cross-Selling 방법을 생각해보아야합니다. "
                                           "아래의 표를 통해 A 제품을 구매한 고객이 B제품도 구매했는 지 알아 볼 수 있습니다."),
                                    html.Div(id='apriori-results')
                                ])
                            ]
                        )
                    ]
                ),
                dmc.Space(h=50)
            ]
        )
    ]
)

@app.callback(
    [Output('arppu-graph', 'figure'),Output('apriori-results', 'children')],
    Input('arppu-select', 'value')
)

def process_and_visualize(df):
    df = mj_class.mj_preprocessing(df)
    df.apply_my_function()
    df = df.return_dataframe()

    # RFM 데이터 프레임
    processor = RFMProcessor(df) 
    rfm_without_outliers, rfm_outliers, rfm_without_outliers_log, X_scaled = processor.process_data()
    processor.fit_clustering(X_scaled, n_clusters=4)
    rfm = processor.predict(df)

    viz = mj_class.mj_visualization(df, rfm)
    return viz


def update_arppu_chart(selected_analysis):

    # 지금 여러 DB가 있는데 일단 헷갈려서 trainDB 코드만 작성(추후 수정하기!!!!!)
    df = database_bw.making_dataframe_train_db('train_table')
    viz = process_and_visualize(df)

    apriori_analyzer = mj_class.mj_apriori()
    apriori_results = apriori_analyzer.apriori_analysis(df)

    if selected_analysis == 'cluster':
        graph_figure =  viz.cluster_calculate_and_plot_arppu()
    elif selected_analysis == 'monthly':
        graph_figure = viz.month_calculate_and_plot_arppu()
    elif selected_analysis == 'area':
        graph_figure = viz.area_calculate_and_plot_arppu()
    elif selected_analysis == 'subscription':
        graph_figure = viz.calculate_and_plot_arppu_by_subscription_period_grouped()
    elif selected_analysis == 'area(map)':
        graph_figure = viz.area_calculate_and_plot_mapbox()
    


    apriori_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in apriori_results.columns],
        data=apriori_results.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'height': 'auto',
            'minWidth': '0px', 'maxWidth': '180px',
            'whiteSpace': 'normal'
        }
    )

    return graph_figure, apriori_table
