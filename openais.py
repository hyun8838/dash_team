from openai import OpenAI

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html

from flask import request, redirect, flash
from flask_login import login_required

from tenacity import retry, wait_random_exponential, stop_after_attempt

import os
import sys

GPT_MODEL = "gpt-3.5-turbo-0613"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

html.Div([
    html.H1("Ask me Anything"),
    dcc.Textarea(id='input-box', value='무엇을 도와드릴까요?', style={'width': '100%', 'height': 200}),
    html.Button('Send', id='button', n_clicks=0),
    html.Div(id='output-container-button', children=[]),
])


def redirect_with_intention(intention):
    if intention == "고객 이탈률":
        return dcc.Location(pathname = "/dashboard/page-1")
    elif intention == "재무":
        return dcc.Location(pathname = "/dashboard/page-2")
    elif intention == "마케팅":
        return dcc.Location(pathname = "/dashboard/page-3")
    elif intention == "고객 충성도 (매출액)":
        return dcc.Location(pathname = "/dashboard/page-4")
    elif intention == "성과지표(ROI, APPRU)":
        return dcc.Location(pathname = "/dashboard/page-5")
    else:
        print("요구사항이 명확하지 않습니다. 원하시는 사안을 더 자세하게 말씀해주세요.")
        return dcc.Location(pathname = "/dashboard/")

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools = redirect_with_intention, tool_choice = None, model = GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            tools = tools,
            tool_choice = tool_choice
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e