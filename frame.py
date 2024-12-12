import dash, os
from dash import dcc, html
import dash_bootstrap_components as dbc

# 侧边栏样式
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "padding": "2rem 0rem 2rem 1rem",
    "backgroundColor": "#fff",
    "zIndex": 1000,  # 确保侧边栏位于页面内容之上
}

# 内容样式
CONTENT_STYLE = {
    "marginLeft": "14rem",
    "marginRight": "0rem",
    "padding": "1rem 1rem 1rem 1rem",
    "position": "relative",  # 确保内容不会覆盖侧边栏
    "backgroundColor": "#F2F2FD",
    "zIndex": 0,  # 内容层级较低，避免遮挡
    "height": "auto"
}

# 侧边栏布局
sidebar = html.Div(
    [
        html.Img(src='assets/images/logo.png', alt='logo', style={'width':'150px', 'padding': '20px 0'}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="bi bi-arrow-90deg-left", style={"marginRight": "5px"}),"选择数据"], href="/", id='select-data-link'),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# 页面内容区域
content = html.Div(id="page-content", style=CONTENT_STYLE)