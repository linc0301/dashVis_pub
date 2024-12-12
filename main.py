import dash, json
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State, ctx, no_update
from dash.exceptions import PreventUpdate
from urllib.parse import urlencode
from frame import *
from utils import *

import home_page
import error_page
import analysisA_page
import analysisC_page

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
app.title = "FG-DevApp"
app._favicon = "images/favicon.png"

# 布局定义
app.layout = html.Div([
    dcc.Location(id="url"), sidebar, content, dcc.Store(id="arm_clear_btn", data={"n_clicks": 0})
])
    
# 主页，分析页和404页的回调
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname"), Input("url", "search")],
    prevent_initial_call=True
)  ## search指参数
def render_page_content(pathname, search):
    if pathname == "/":
        # 首页内容
        return home_page.layout()
    elif pathname == "/analysis":
        # 数据加载中时的内容
        params = {p.split('=')[0]:p.split('=')[1] for p in search[1:].split('&')} if search else 'None'
        if params.get('viz') == 'C':
            return analysisC_page.layout(params['dataset']) 
        elif params.get('viz') == 'A':
            return analysisA_page.layout(params['dataset'])
    # 404页面
    return error_page.layout()

## 注册回调函数
home_page.register_callbacks(app)
analysisA_page.register_callbacks(app)
analysisC_page.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, host="10.49.60.23", port=8024)