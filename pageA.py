import dash, os, uuid
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State, no_update, callback_context
from dash.exceptions import PreventUpdate
from utils import *

config_arm = {
    'scrollZoom': True,  # 启用滚轮缩放
    'displayModeBar': True,
    'modeBarButtonsToRemove': [
        'pan',
        'toImage',  # 隐藏“保存为图像”按钮
        'lasso2d',  # 隐藏套索选择按钮
        'autoScale2d',  # 隐藏自动缩放按钮
        'zoom',
        'zoomIn2d',  # 隐藏放大按钮
        'zoomOut2d'  # 隐藏缩小按钮
    ],
    'displaylogo': False  # 隐藏 Plotly logo
}

config_arm_dnb = {
    'scrollZoom': True,  # 启用滚轮缩放
    'displayModeBar': True,
    'modeBarButtonsToRemove': [
        'pan',
        'toImage',  # 隐藏“保存为图像”按钮
        'lasso2d',  # 隐藏套索选择按钮
        'select',
        'autoScale2d',  # 隐藏自动缩放按钮
        'zoom',
        'zoomIn2d',  # 隐藏放大按钮
        'zoomOut2d'  # 隐藏缩小按钮
    ],
    'displaylogo': False  # 隐藏 Plotly logo
}

def layout(dataset):
    return dbc.Container([
        #dcc.Store(id='arm_dataset_store', data=dataset, storage_type='session'),  # 存储dataset名称
        dcc.Store(id='arm_uuid', storage_type='session'),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H5([f'数据集：',html.Span(dataset,id='dataset_id')], style={"display": "inline-block", "marginRight": "10px"}),
                    dbc.Button(
                        [html.I(className="bi bi-bullseye"), " 点击，加载数据"], 
                        id='load_arm_dataset', 
                        n_clicks=0
                    )
                ]),
                width=12
            )
        ], style={"fontSize": "10pt", "padding": "10px 0"}),
        
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div([
                            html.P(html.B("分臂图（Testing Area）"), style={'display': 'inline-block', 'marginRight': '20px'}),
                            dbc.Button([html.I(className="bi bi-eraser"), " 清空选区"], id='clear_button', n_clicks=0, type='button', style={"backgroundColor": "#ffffff", 'display': 'inline-block', "border": "none", "color":"#666666", "borderRadius": "1rem", "box-shadow": "0 0 20px 0 rgba(0,0,0,0.1)"}),
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', "paddingBottom": "10px"}),
                        dcc.Loading(
                            type="default",
                            delay_hide=1000,
                            children=[dcc.Graph(id='arm_plot', figure=gen_blank_plot("请先加载数据"), config=config_arm)]
                        ),
                        html.Span('请选择展示的DNB数目', style={'padding': '0px 25px 20px 25px'}),
                        dcc.Slider(1, 7, 0.01, value=2, id='size_slider',
                           marks={1: '10', 2: '100', 3: '1000', 4: '10k', 5: '100k', 6: '1M', 7: '10M'}
                        )
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ),
                width=9, style={'height': '600px'}
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(html.B("手动配置选项")),
                        dbc.Label("Color:"),
                        dcc.Dropdown(# 'Error Rate',
                                 options=[
                                     {'label': '真实碱基', 'value': 'real_base'},
                                     {'label': 'Call碱基', 'value': 'call_base'},
                                     {'label': 'Mismatch', 'value': 'mismatch'}],
                                     # {'label': 'T>C', 'value': 'T>C'},
                                     # {'label': 'C>T', 'value': 'C>T'},
                                     # {'label': 'A>C', 'value': 'A>C'},
                                     # {'label': 'C>A', 'value': 'C>A'}],
                                 value='real_base',
                                 placeholder="选择颜色来源",  id="color_s"
                        ),
                        html.Br(),
                        dbc.Label("标签类型:"),
                        dcc.Dropdown(['碱基', 'Cycle'], '碱基', placeholder="选择标签", id="label_s"),
                        html.Br(),
                        dbc.Label("Cycle:"),
                        dbc.Input(id="cycle_s", placeholder="请输入关注的Cycle number...", type="text")
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ),
            width=3),
        ], style={"fontSize": "10pt", "padding": "30px 0"}),
    
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(html.B("所选单个DNB的光曲图"), style={"marginBottom":"0"}),
                        dcc.Loading(
                            type="default",
                            delay_hide=1000,
                            children=[dcc.Graph(id='arm_dnb_plot', figure=gen_blank_plot("请先加载DNB"), config=config_arm_dnb)]
                        )
                    ]), style={"borderRadius": "1rem", "border":"none"},
                ), 
                width=6, style={'height': '400px'}
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(html.B("所框选DNB的光曲图"), style={"marginBottom":"0"}),
                        dcc.Loading(
                            type="default",
                            delay_hide=1000,
                            children=[dcc.Graph(id='arm_region_plot', figure=gen_blank_plot("请先选择区域"), config=config_arm_dnb)]
                        )
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ), 
                width=6
            )
        ], style={"fontSize": "10pt", "padding": "0 0 30px 0"}),
        
        dbc.Toast(
            html.Table(
                html.Tbody([
                    html.Tr([
                        html.Td('鼠标滚轮', style={"width": "80%"}), html.Td('缩放')
                    ]),
                    html.Tr([
                        html.Td('鼠标左键 + 移动'), html.Td('平移')
                    ]),
                    html.Tr([
                        html.Td('选区 + 鼠标左键框选'), html.Td('选中区域')
                    ]),
                    html.Tr([
                        html.Td('鼠标左键点击'), html.Td('显示DNB')
                    ])
                ])
            ),
            header=html.Span([html.I(className="bi bi-mouse"), " 操作说明"]),
            dismissable=True,
            duration=3000,
            style={"position": "fixed", "bottom": 10, "right": 10},
        )
    ], fluid=True)


def register_callbacks(app):
    @app.callback(
        [Output('arm_plot', 'figure', allow_duplicate=True), Output('arm_uuid', 'data')],
        Input('load_arm_dataset', 'n_clicks'),
        #State('arm_dataset_store', 'data'),
        State('dataset_id', 'children'),
        State('arm_uuid', 'data'),
        prevent_initial_call=True  # 避免初次加载时触发回调
    )
    def load_data_and_visualize(n_clicks, dataset,uid):
        if not callback_context.triggered or n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            if uid is None:
                uid = str(uuid.uuid4())
                
            print("load",dataset,uid)
            load_arm_dataset(dataset, uid)
            arm_plot = generate_arm_plot(sample_size=100, color_value='real_base', dataset=dataset, uid=uid)
            return arm_plot, uid
        return gen_blank_plot("请先加载数据"), no_update
    
    ## 修改slider时，更新数据和图
    @app.callback(
        [Output('arm_plot', 'figure', allow_duplicate=True), 
         Output('arm_dnb_plot', 'figure', allow_duplicate=True), 
         Output('arm_region_plot', 'figure', allow_duplicate=True)],
        Input('size_slider','value'),
        Input('color_s', 'value'),
        #State('arm_dataset_store', 'data'),
        State('dataset_id', 'children'),
        State('arm_uuid', 'data'),
        prevent_initial_call = True
    )
    def update_figure(slide_value, color_v, dataset, uid):
        if not callback_context.triggered or not dataset or not uid:
            raise PreventUpdate  # 防止未触发时执行
        print("size=",slide_value)
        print("color=",color_v)
        slide_value = transform_value(slide_value)
        arm_plot = generate_arm_plot(sample_size = slide_value, color_value = color_v, dataset = dataset, uid = uid)
        return arm_plot, gen_blank_plot("请先加载DNB"), gen_blank_plot("请先选择区域")

    
    ## 点击DNB时
    @app.callback(
        Output('arm_plot', 'figure', allow_duplicate=True),
        Input('arm_plot', 'clickData'),          ## 点击数据点，选择DNB
        State('arm_plot', 'figure'),
        State('label_s', 'value'),
        #State('arm_dataset_store', 'data'),
        State('dataset_id', 'children'),
        State('arm_uuid', 'data'),
        prevent_initial_call=True
    )
    def display_highlighted_points(hover_data, figure, label_type, dataset, uid):
        if not callback_context.triggered or not dataset or not uid:
            raise PreventUpdate  # 防止未触发时执行
        if hover_data is None:
            for trace in figure['data']:
                #trace['marker']['color'] = 'gray'
                trace['selectedpoints'] = []
            return figure

        print(hover_data)
    
        highlighted_df = get_arm_highlight_points(hover_data, dataset, uid)
        highlighted_points = highlighted_df[['1-5 mean', '6-12 mean']].values.tolist()
    
        for trace in figure['data']:
            if trace['mode'] == 'markers':
                trace['marker']['color'] = 'gray'
                trace['selectedpoints'] = []

        for i, trace in enumerate(figure['data']):
            if 'hoverlabel' in trace and trace['hoverlabel']['bgcolor'] == 'green':
                del figure['data'][i]
                break
    
        highlighted_trace = {
            'x': [point[0] for point in highlighted_points],
            'y': [point[1] for point in highlighted_points],
            'mode': 'markers',
            'marker': {
                'color': 'green',
                'size': get_arm_personal_data(dataset, uid, 'marker_size') + 2,
                'opacity': 1
            },
            'hoverinfo': 'text',
            'text': [f"Cycle: {highlighted_df.loc[index, 'cycle']}, DNB ID: {highlighted_df.loc[index, 'dnb_id']}" for index in highlighted_df.index],
            'customdata': [highlighted_df.loc[index,'dnb_id'] for index in highlighted_df.index],
            'textposition': 'top center',
            'hoverlabel':{'bgcolor': 'green'},
            'showlegend': False
        }
        
        annotations = []
        highlighted_points_all = highlighted_df[['1-5 mean', '6-12 mean']].values.tolist()
    
        for index, point in enumerate(highlighted_points_all): 
            point_x, point_y = point
            cycle_info = int(highlighted_df.iloc[index]['cycle'])
            MLP_info = highlighted_df.iloc[index]['annotation_MLP']
            if label_type == '碱基':
                text = f"{MLP_info}"
            else:
                text = f"{cycle_info}"
    
            # Create an annotation for the current point
            annotation = go.layout.Annotation(
                x=point_x,
                y=point_y,
                text=text,
                xref='x',
                yref='y',
                font=dict(
                    family='Arial',
                    size=12,
                    color='black'
                ),
                showarrow=True,  # Ensure the annotation has an arrow
                arrowhead=1,
                bordercolor='black',
                borderwidth=1,
                borderpad=4,
                bgcolor='white',
                opacity=0.8
            )
            annotations.append(annotation)
    
        hover_trace_index = 0  
        hover_point_index = hover_data['points'][0]['pointIndex']
        #figure['data'][hover_trace_index]['marker']['color'] = 'red'
        figure['data'][hover_trace_index]['selectedpoints'] = [hover_point_index]
        figure['data'].append(highlighted_trace)
        figure['layout']['annotations'] = annotations
        
        return figure

    @app.callback(
        [
            Output("arm_plot", "figure", allow_duplicate=True),
            Output("arm_dnb_plot", "figure", allow_duplicate=True),
            Output("arm_region_plot", "figure", allow_duplicate=True),
            Output("arm_clear_btn", "data", allow_duplicate=True)
        ],
        Input("clear_button", "n_clicks"),  # 监听新增按钮的点击次数
        State("arm_clear_btn", "data"),
        Input('size_slider','value'),
        Input('color_s', 'value'),
        #State('arm_dataset_store', 'data'),
        State('dataset_id', 'children'),
        State('arm_uuid', 'data'),
        prevent_initial_call=True
    )
    def clear_selection(n_clicks, store_clicks,slide_value,color_v, dataset,uid):
        if n_clicks and n_clicks > store_clicks["n_clicks"]:
            # 点击按钮后清除图表选择 arm_plot = generate_arm_plot(sample_size = slide_value, color_value = color_v, dataset = dataset, uid = uid)
            print(f"Clearing selections after {n_clicks} clicks")
            slide_value = transform_value(slide_value)
            arm_plot = generate_arm_plot(sample_size= slide_value, color_value= color_v, dataset=dataset, uid=uid, need_sample=True)
            return arm_plot , gen_blank_plot("请先选择区域"), gen_blank_plot("清除选择"), {"n_clicks": n_clicks}
        raise PreventUpdate  # 防止重复更新
    
    ## arm dnb
    @app.callback(
        Output('arm_dnb_plot', 'figure'),
        Input('arm_plot', 'clickData'),
        #State('arm_dataset_store', 'data'),
        State('dataset_id', 'children'),
        State('arm_uuid', 'data')
    )
    def display_clicked_data(clickData, dataset, uid):
        if clickData:
            highlighted_df = get_arm_highlight_points(clickData, dataset, uid)
            colors = px.colors.qualitative.Plotly
            fig = go.Figure()
            print(highlighted_df)
    
            for i, row in highlighted_df.iterrows():
                dnb_id = row['dnb_id']
                cycle = row['cycle']
                cycle_h5 = row['cycle'] - 1
                # print(cycle)
                MLP = row['annotation_MLP']
                with h5py.File(get_arm_data(dataset, 'cache_ints_file'), 'r') as f:
                    
                    sig = f['all_data'][dnb_id,cycle_h5,:]
                    color = colors[i % len(colors)]  # 顺序分配颜色
    
                    fig.add_trace(go.Scatter(
                        x=list(range(sig.shape[0])),
                        y=sig,
                        mode='lines+markers',
                        marker=dict(color=color),
                        name=f'cycle {cycle} : {MLP}',
                        text=[f"Cycle {cycle} : {MLP}"] * sig.shape[0],
                        hoverinfo='text+y'
                    ))
            fig.update_layout(
                title="",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                modebar_bgcolor= '#ffffff',
                modebar_color='#666666',
                xaxis_title='Seconds',
                yaxis_title='Intensity',
                margin=dict(l=20, r=20, t=40, b=20)
            )   
            return fig
        return gen_blank_plot("请先加载DNB")
    
    # arm region
    @app.callback(
        Output('arm_region_plot', 'figure'),
        Input('arm_plot', 'selectedData'),
        #State('arm_dataset_store', 'data'),
        State('dataset_id', 'children'),
        State('arm_uuid', 'data')
    )
    def display_selected_data(selectedData, dataset, uid):
        if not callback_context.triggered or not dataset or not uid:
            raise PreventUpdate  # 防止未触发时执行

        sampled_dataframe = get_arm_personal_data(dataset, uid, 'sampled_df')
        if selectedData:
            points = selectedData['points']
            indices = [int(point['customdata'][1]) for point in points]
                
            if indices:
                query_result = sampled_dataframe.loc[indices].index  
                idx1 = sampled_dataframe.loc[query_result, ['dnb_id', 'cycle','annotation_MLP']]
            
                colors = px.colors.qualitative.Plotly
                fig = go.Figure()
    
                for i, row in idx1.iterrows():
                    dnb_id = row['dnb_id']
                    cycle = row['cycle']
                    MLP = row['annotation_MLP']
                    with h5py.File(get_arm_data(dataset, 'cache_ints_file'), 'r') as f:
                        cycle_h5 = cycle -1
                        sig = f['all_data'][dnb_id, cycle_h5, :]
                        color = colors[i % len(colors)]  # 顺序分配颜色
    
                        fig.add_trace(go.Scatter(
                            x=list(range(sig.shape[0])),
                            y=sig,
                            mode='lines+markers',
                            marker=dict(color=color),
                            name=f'cycle {cycle}, {MLP}',
                            text=[f"Cycle {cycle}, dnb {dnb_id}"] * sig.shape[0],
                            hoverinfo='text+y'
                        ))
        
                fig.update_layout(
                    title="",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    modebar_bgcolor= '#ffffff',
                    modebar_color='#666666',
                    xaxis_title='Seconds',
                    yaxis_title='Intensity',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                return fig
        return gen_blank_plot("请先选择区域")