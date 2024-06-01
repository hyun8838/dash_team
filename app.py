import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
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
        html.P(
            "관리자를 위한 단 하나의 대시보드.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("고객 이탈률", href="/page-1", active="exact"),
                dbc.NavLink("재무", href="/page-2", active="exact"),
                dbc.NavLink("마케팅", href ="/page-3", active="exact"),
                dbc.NavLink("고객 충성도 (재구매)", href="/page-4", active="exact"),
                dbc.NavLink("성과지표 (ROI, APPRU)", href="/page-5", active="exact")
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
def render_page_content(pathname):
    if pathname == "/":
        return html.P("홈페이지")
    elif pathname == "/page-1":
        return html.P("페이지 1")
    elif pathname == "/page-2":
        return html.P("페이지 2")
    elif pathname == "/page-3":
        return html.P("페이지 3")
    elif pathname == "/page-4":
        return html.P("페이지 4")
    elif pathname == "/page-5":
        return html.P("페이지 5")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run_server(port=8888, debug = True)
