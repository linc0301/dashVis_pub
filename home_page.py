import dash, os
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State, no_update
from urllib.parse import urlencode

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(
                html.Div(
                    html.Img(src='assets/images/e25.png', alt='E25', style={'width':'100%', 'opacity': '0.3'}),
                ), width=6),
            
            dbc.Col(
                html.Div([
                    dbc.Label('Step 1: 选择 E25 FG 数据集'),
                    dcc.Dropdown(
                        id='data_select',
                        options=[
                            {'label': 'YF4', 'value': 'yf4'},
                            {'label': '合成0930','value':'syn_0930'},
                            {'label': '合成1001','value':'syn_1001'},
                            {'label': 'JY1014','value':'jy1014'},
                            {'label': '合成1016','value':'syn_1016'},
                            {'label': '合成1017','value':'syn_1017'}
                        ],
                        placeholder="...",
                        style={"width": "60%"},
                        clearable=True
                    ), 
                    html.Br(),
                    dbc.Label('Step 2: 选择可视化类型'),
                    dcc.Dropdown(
                        id='viz_type',
                        options=[
                            {'label': '芯片图', 'value': 'C'},
                            {'label': '分臂图', 'value': 'A'}
                        ],
                        placeholder="...",
                        style={"width": "60%"},
                        clearable=True
                    ),
                    html.Br(),
                    dbc.Button("确认", id="confirm-button", color="secondary", type='button'),
    
                    dbc.Toast(
                        html.P('请确保已选择“数据集”和“可视化类型”'),
                        id='select_warning',
                        header=html.Span([html.I(className="bi bi-arrow-right-circle"), " 提示"]),
                        dismissable=True,
                        duration=2000,
                        is_open=False,
                        style={"position": "fixed", "top": 10, "right": 10},
                    )
                ]), width=6)
        ], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "height": "90vh"})
    ], fluid=True)

def register_callbacks(app):
    # 监听Dropdown变化并通过确认按钮跳转
    @app.callback(
        [Output('url', 'href'), Output("select_warning", "is_open")],
        [Input('confirm-button', 'n_clicks')],
        [State('data_select', 'value')],
        [State('viz_type', 'value')]
    )
    def update_url_on_confirm(n_clicks, ds, viz):
        if n_clicks:
            if ds and viz:
                # 构建新的 URL，带有选中的参数
                params = {'dataset': ds, 'viz': viz}
                return f'/analysis?{urlencode(params)}', False
            else:
                return no_update, True
        else:
            return no_update, False
