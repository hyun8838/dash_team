from login import app as flask_app
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from flask_login import login_required, current_user, logout_user
### Import redirect from dash.dependencies
from dash.dependencies import State, ClientsideFunction

from openais import chat_completion_request
from openais import redirect_with_intention
import json


from waitress import serve

from arppu_page import layout as arppu_layout


# Create a Dash instance within the Flask app
app = dash.Dash(server=flask_app, name="Dashboard", url_base_pathname="/dashboard/", external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Admin Dashboard'

# The style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# The styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Admin \n Dashboard", className="display-5"),
        html.Hr(),
        html.P("관리자를 위한 단 하나의 대시보드.", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/dashboard/", active="exact"),
                dbc.NavLink("고객 이탈률", href="/dashboard/page-1", active="exact"),
                dbc.NavLink("재무", href="/dashboard/page-2", active="exact"),
                dbc.NavLink("마케팅", href="/dashboard/page-3", active="exact"),
                dbc.NavLink("고객 충성도 (재구매)", href="/dashboard/page-4", active="exact"),
                dbc.NavLink("성과지표 (APPRU)", href="/dashboard/page-5", active="exact"),
                ### Add logout button
                dbc.NavLink("로그아웃", href="/logout", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
@login_required
def render_page_content(pathname):
    if pathname == "/dashboard/":
        return html.Div([
            html.P("홈페이지"),
            html.Div([
                html.H1("Ask me Anything"),
                dcc.Textarea(id='input-box', value='무엇을 도와드릴까요?', style={'width': '100%', 'height': 200}),
                html.Button('전송', id='button', n_clicks=0),
                html.Div(id='output-container-button', children=[]),
            ])
        ])
    elif pathname == "/dashboard/page-1":
        return html.P("페이지 1")
    elif pathname == "/dashboard/page-2":
        return html.P("페이지 2")
    elif pathname == "/dashboard/page-3":
        return html.P("페이지 3")
    elif pathname == "/dashboard/page-4":
        return html.P("페이지 4")
    elif pathname == "/dashboard/page-5":
        return arppu_layout
    return dcc.Location(pathname="/login", id="redirect-login")
    # If the user tries to reach a different page, return a 404 message
    # return html.Div(
    #     [
    #         html.H1("404: Not found", className="text-danger"),
    #         html.Hr(),
    #         html.P(f"The pathname {pathname} was not recognised..."),
    #     ],
    #     className="p-3 bg-light rounded-3",
    # )

### Add logout callback
@app.callback(Output("redirect-login", "pathname"), [Input("url", "pathname")])
def logout_on_click(pathname):
    if pathname == "/logout":
        logout_user()
        return "/login"
    return pathname
### Add chatbot interaction
tools = [
    {
        "type" : "function",
        "function" : {  # Corrected the spelling of "function"
            "name" : "force_category",
            "description" : "Use this function to (1) Capture which category does the user's intention belongs to. (2) Redirect to the page which belongs to that category",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "intention" :{
                        "type" :"string",
                        "description":"Intention(Category). Possible category is 6. 고객 이탈률, 재무, 마케팅, 재구매 및 기존고객 관리, 성과지표(ROI, APPRU), 해당없음. After the Category is fixed, redirect to the belonging cateogory."
                    }
                },
                "required":["intention"]
            }
        }
    }
]
@app.callback(
    Output('output-container-button', 'children'),
    [Input('button', 'n_clicks')],
    [State('input-box', 'value')]
)
@login_required
def update_output(n_clicks, input_value):
    if n_clicks > 0:
        messages = [{"role": "system", "content": "You infer about which category(intention) the user wants to know about. Then, you use tool to redirect the user with user's intention."}]
        messages.append({"role": "user", "content": input_value})
        response = chat_completion_request(messages, tools=tools)
        if 'conversation_stack' not in globals():
            global conversation_stack
            conversation_stack = []
        for choice in response.choices:
            if choice.message.role == "user":
                conversation_stack.append(html.P(choice.message.content))
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    function_name = tool_call.function.name
                    function_arguments = json.loads(tool_call.function.arguments)
                    if function_name == "force_category":
                        category = function_arguments["intention"]
                        if category == "고객 이탈률":
                            return dcc.Location(pathname="/dashboard/page-1", id="redirect-page-1")
                        elif category == "재무":
                            return dcc.Location(pathname="/dashboard/page-2", id="redirect-page-2")
                        elif category == "마케팅":
                            return dcc.Location(pathname="/dashboard/page-3", id="redirect-page-3")
                        elif category == "재구매 및 기존고객 관리":
                            return dcc.Location(pathname="/dashboard/page-4", id="redirect-page-4")
                        elif category == "성과지표 (APPRU)":
                            return dcc.Location(pathname="/dashboard/page-5", id="redirect-page-5")
                        else:
                            return html.P("요구사항이 명확하지 않습니다. 원하시는 사안을 더 자세하게 말씀해주세요.")
    else:
        return html.P("무엇을 도와드릴까요?")
if __name__ == "__main__":
    flask_app.run(port=8888, debug=True)
    # serve(app.server, host='0.0.0.0', port=8888)
