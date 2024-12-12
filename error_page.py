import dash, os
from dash import dcc, html
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        html.Div(
            html.Img(src='assets/images/404-img.png', alt='404', style={'width':'600px'}), style={'display':'flex', 'justifyContent': 'center','alignItems': 'center'}
        ),
        html.P(['未找到页面，请 “', html.I(className="bi bi-arrow-left-short"), ' 回退” 或 “选择数据”'], style={'fontSize':'20px', 'paddingTop': '30px', 'textAlign': 'center'})
    ])