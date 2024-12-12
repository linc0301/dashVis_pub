import dash, os
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State, no_update, callback_context
from dash.exceptions import PreventUpdate
from utils import *


config_chip = {
    'scrollZoom': True,  # 启用滚轮缩放
    'displayModeBar': True,
    'modeBarButtonsToRemove': [
        'pan',
        'toImage',  # 隐藏“保存为图像”按钮
        'lasso2d',  # 隐藏套索选择按钮
        'autoScale2d',  # 隐藏自动缩放按钮
        'zoom',
        'select',
        'zoomIn2d',  # 隐藏放大按钮
        'zoomOut2d'  # 隐藏缩小按钮
    ],
    'displaylogo': False  # 隐藏 Plotly logo
}

## 比上面多一个框选的功能
config_chip_dnb = {
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


def layout(dataset):
    return dbc.Container([
        #dcc.Store(id='chip_dataset_store', data=dataset, storage_type='session'),  # 存储dataset名称

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H5([f'数据集:',html.Span(dataset,id='dataset_id')], style={"display": "inline-block", "marginRight": "10px"}),
                    dbc.Button(
                        [html.I(className="bi bi-bullseye"), " 点击，加载数据"], 
                        id='load_chip_dataset', 
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
                       html.P(html.B("概览图 (All Cycles)"), style={"marginBottom":"0"}),
                       dcc.Loading(
                            type="default",
                            delay_hide=1000,
                            children=[dcc.Graph(id='chip_plot', figure=gen_blank_plot("请先加载数据"), config=config_chip)]    ## eye-plot
                        )
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ), width=5, style={'height': '500px'}),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(html.B(id='cycle_display',children = 'Cycle'), style={"marginBottom":"0"}),
                        dcc.Loading(
                            type="default",
                            delay_hide=1000,
                            children=[dcc.Graph(id='chip_plot2', figure=gen_blank_plot("请先加载数据"), config=config_chip)]   ## plot_mismatch
                        )
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ), width=5, style={'height': '500px'}),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(html.B("手动配置选项")),
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Label("Color:"),
                                dcc.Dropdown(options=[
                                    {'label': 'Non Specific', 'value': 'non_spec_percentage'},
                                    {'label': 'Mismatch Rate', 'value': 'mismatch'},
                                    {'label': 'T>C', 'value': 'T>C'},
                                    {'label': 'G>C', 'value': 'G>C'},
                                    {'label': 'A>C', 'value': 'A>C'},
                                    {'label': 'C>T', 'value': 'C>T'},
                                    {'label': 'G>T', 'value': 'G>T'},
                                    {'label': 'A>T', 'value': 'A>T'},
                                    {'label': 'C>A', 'value': 'C>A'},
                                    {'label': 'G>A', 'value': 'G>A'},
                                    {'label': 'T>A', 'value': 'T>A'},
                                    {'label': 'C>G', 'value': 'C>G'},
                                    {'label': 'A>G', 'value': 'A>G'},
                                    {'label': 'T>G', 'value': 'T>G'}
                                ], 
                                 value = 'non_spec_percentage', id="color_s")
                            ]), style={"borderRadius": "1rem", "border":"none", "boxShadow": "0 0 20px 0 rgba(0,0,0,0.1)"}
                        ),
                        html.Br(),
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Label("Testing Area:"),
                                dbc.Input(id="ta_s",placeholder="请输入关注的Testing Area index ...", debounce=True, type="int"),  ## ta-i
                                html.Br(),
                                dbc.Label("Cycle:"),
                                dbc.Input(id="cycle_s", placeholder="请输入关注的Cycle number...", debounce=True, type="int"),   ## my_slider
                                # html.Br(), 
                                # dbc.Button("筛选", id='chip_filter_button', type='button', style={"marginTop": "10px", "width": "50%", "backgroundColor": "#584FBE", "border": "none", "color":"#ffffff", "borderRadius": "1rem"})
                            ]), style={"borderRadius": "1rem", "border":"none", "boxShadow": "0 0 20px 0 rgba(0,0,0,0.1)"}
                        ),
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ),
            width=2),
        ], style={"fontSize": "10pt", "padding": "30px 0"}),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(html.B("区域图"), style={"marginBottom":"0"}),
                        dcc.Loading(
                            type="default",
                            delay_hide=1000,
                            children=[dcc.Graph(id='ta_plot', figure=gen_blank_plot("请先加载数据"), config=config_chip_dnb)]   ## plot_mismatch
                        )
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ), width=12, style={'height': '500px'}),
        ], style={"fontSize": "10pt", "padding": "30px 0"}),
        
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(html.B("所框选DNB的光曲图"), style={"marginBottom":"0"}),
                        dcc.Loading(
                            type="default",
                            delay_hide=1000,
                            children=[dcc.Graph(id='chip_dnbs_plot', figure=gen_blank_plot("请先加载DNB"), config=config_chip)]   ## select-plot-dnbs
                        )
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ), width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(html.B("所单选DNB的光曲图"), style={"marginBottom":"0"}),
                        dcc.Loading(
                            type="default",
                            delay_hide=1000,
                            children=[dcc.Graph(id='chip_detail_plot', figure=gen_blank_plot("请先选择区域"), config=config_chip)]   ## detail-plot-dnbs
                        )
                    ]), style={"borderRadius": "1rem", "border":"none"}
                ), width=6)
        ], style={"fontSize": "10pt", "padding": "0 0 30px 0"}),

        dbc.Toast(
            html.P('请确保已选择“Testing Area 和 Cycle”'),
            id='chip_select_warning',
            header=html.Span([html.I(className="bi bi-arrow-right-circle"), " 提示"]),
            dismissable=True,
            is_open=False,
            duration=2000,
            style={"position": "fixed", "top": 10, "right": 10},
        ),

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
                    ])
                ])
            ),
            header=html.Span([html.I(className="bi bi-mouse"), " 操作说明"]),
            dismissable=True,
            duration=3000,
            style={"position": "fixed", "bottom": 10, "right": 10},
        )    
    ], fluid=True)


def color_by_all_cycles(dataset, value):
    fig = init_chip_plot(dataset)
    ta_stat_df = get_chip_data(dataset, 'ta_stat_df')
    color_scale = px.colors.sequential.Viridis
    lower_bound = np.percentile(ta_stat_df[value], 10)
    upper_bound = np.percentile(ta_stat_df[value], 90)
    norm_percentage = np.clip(ta_stat_df[value], lower_bound, upper_bound)
    norm_percentage = norm_percentage / norm_percentage.max()

    for i, row in ta_stat_df.iterrows():   ## 变量每个TA
        x = row['y']
        y = row['x']
        percentage = row[value]
        # print(percentage)
        color_index = int(norm_percentage[i] * (len(color_scale) - 1))
        color = color_scale[color_index]
        #print("ca",color_index)
        columns_to_process = ['T>A', 'T>C', 'T>G', 'C>A', 'C>T', 'C>G', 'A>T', 'A>C', 'A>G', 'G>A', 'G>C', 'G>T']
        if value in columns_to_process:
            N_dnb = int(row['mismatch']*row['total_dnb']*row[value])
        else:
            N_dnb = int(row['mismatch']*row['total_dnb'])

        fig.add_trace(go.Scatter(
            x=[x, x + 96*2.5, x + 96*2.5, x, x], 
            y=[y, y, y + 80*2.5, y + 80*2.5, y],
            mode='lines',
            fill='toself',
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.8,
            hoverinfo='text',
            customdata = np.array(row['idx']),
            text=f'id={i}<br>{value}<br> {percentage:.2%}<br>N:{N_dnb}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x + 96*2.5 / 2],
            y=[y + 80*2.5 / 2],
            text=[f'{i}'],
            mode='text',
            textfont=dict(color='black', size=11),
            showlegend=False,
            name="none",
            hoverinfo='skip'
        ))

    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=color_scale,
            cmin=lower_bound*100,
            cmax=upper_bound*100,
            colorbar=dict(
            title='%',
            thickness=20,
            ticksuffix='%'
            )
        ),
        showlegend=False
    )
    fig.add_trace(colorbar_trace)
    return fig


def register_callbacks(app):
    @app.callback(
        [Output('chip_plot', 'figure', allow_duplicate=True), Output('chip_plot2', 'figure', allow_duplicate=True), Output('ta_plot', 'figure', allow_duplicate=True), Output('color_s', 'value', allow_duplicate=True)],
        Input('load_chip_dataset', 'n_clicks'),
        State('dataset_id', 'children'),
        prevent_initial_call=True  # 避免初次加载时触发回调
    )
    def load_data_and_visualize(n_clicks, dataset):
        if not callback_context.triggered or n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            # if judge_chip_dataset(dataset):
            #     raise PreventUpdate
            print("load",dataset)
            load_chip_dataset(dataset)
            chip_plot = init_chip_plot(dataset)
            ts_plot = init_ts_plot(0, 0, dataset)
            return chip_plot, chip_plot, ts_plot, 'non_spec_percentage'
        return gen_blank_plot("请先加载数据"), gen_blank_plot("请先加载数据"), gen_blank_plot("请先加载数据"), 'non_spec_percentage'

    ## 选择上色逻辑
    @app.callback(
        Output('chip_plot', 'figure'),
        Input('color_s','value'),
        #State('chip_dataset_store', 'data'),
        State('dataset_id', 'children'),
        prevent_initial_call=True  # 避免初次加载时触发回调
    )
    def update_color_s(value, dataset):
        if not callback_context.triggered or not value or not dataset:
            raise PreventUpdate  # 防止未触发时执行

        return color_by_all_cycles(dataset, value)

    @app.callback(
        [Output('chip_plot2', 'figure'), Output('cycle_display','children')],
        [Input('color_s','value'), Input('cycle_s', 'value')],
        [State('chip_plot','figure'), 
         #State('chip_dataset_store', 'data')
         State('dataset_id', 'children'),
        ],
        prevent_initial_call=True  # 避免初次加载时触发回调
    )
    def update_color_s2(value, cycle, chip_plot, dataset):
        if not callback_context.triggered or not dataset:
            raise PreventUpdate  # 防止未触发时执行

        ## 如果还是非特异性吸附，则无论是否选择cycle，都会上色
        if value == 'non_spec_percentage':
            return color_by_all_cycles(dataset, value), '概览图(请先输入Cycle)'
        else:
            if cycle:  ##
                fig = init_chip_plot(dataset)
                ta_stat_df = get_chip_data(dataset,'ta_stat_df_cycles')
                user_id = get_chip_data(dataset,'user_id')
                print("call:", user_id)
                ta_stat_df = ta_stat_df[ta_stat_df['cycle']==int(cycle)]
                ta_stat_df = ta_stat_df.reset_index(drop=True)
                ##print('color_s' , ta_stat_df)
                color_scale = px.colors.sequential.Viridis
                lower_bound = np.percentile(ta_stat_df[value], 10)
                upper_bound = np.percentile(ta_stat_df[value], 90)
                norm_percentage = np.clip(ta_stat_df[value], lower_bound, upper_bound)
                norm_percentage = norm_percentage / norm_percentage.max()
                norm_percentage = np.where(np.isnan(norm_percentage), 0, norm_percentage)

                #print(norm_percentage)
                #print(color_scale)

                for i, row in ta_stat_df.iterrows():
                    x = row['y']
                    y = row['x']
                    idx = int(row['idx'])
                    percentage = row[value]
                   # #print(percentage) 
                    color_index = int(norm_percentage[i] * (len(color_scale) - 1))
                    #print(norm_percentage[i])
                    color = color_scale[color_index]
                    #print("c",color_index)
                    columns_to_process = ['T>A', 'T>C', 'T>G', 'C>A', 'C>T',
                      'C>G', 'A>T', 'A>C', 'A>G', 'G>A', 'G>C', 'G>T']
                    if value in columns_to_process:
                        N_dnb = int(row['mismatch']*row['total_dnb']*row[value])
                    else:
                        N_dnb = int(row['mismatch']*row['total_dnb'])
        
                    fig.add_trace(go.Scatter(
                        x=[x, x + 96*2.5, x + 96*2.5, x, x], 
                        y=[y, y, y + 80*2.5, y + 80*2.5, y],
                        mode='lines',
                        fill='toself',
                        line=dict(color=color, width=2),
                        fillcolor=color,
                        opacity=0.8,
                        hoverinfo='text',
                        customdata = np.array(row['idx']),
                        text=f'id={idx}<br>{value}<br>{percentage:.2%}<br>N:{N_dnb}',
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=[x + 96*2.5 / 2],
                        y=[y + 80*2.5 / 2],
                        text=[f'{idx}'],
                        mode='text',
                        textfont=dict(color='black', size=11),
                        showlegend=False,
                        name="none",
                        hoverinfo='skip'
                    ))

                colorbar_trace = go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        colorscale=color_scale,
                        cmin=lower_bound*100,
                        cmax=upper_bound*100,
                        colorbar=dict(
                        title='%',
                        thickness=20,
                        ticksuffix='%'
                        )
                    ),
                    showlegend=False
                )
                fig.add_trace(colorbar_trace)

                return fig, f'概览图(Cycle {cycle} - {dataset})'
            else:
                fig = init_chip_plot(dataset)
                return fig, f'概览图(请先输入Cycle)'
    
    ## 当点击TA时，更新
    @app.callback(
        Output('ta_s', 'value'),
        Input('ta_s', 'value'),
        Input('chip_plot', 'clickData'),
        prevent_initial_call=True  # 避免初次加载时触发回调
    )
    def click_to_select_ta(ta_s, clickData):
        if clickData:
            ta_id = clickData.get('points', [{}])[0].get('customdata')
            if ta_id != ta_s:
                ta_s = ta_id[0]
        return ta_s
    
    # 当用户选择某个cycle或某个testing area时
    @app.callback(
        [Output('ta_plot', 'figure', allow_duplicate=True), Output("chip_select_warning", "is_open", allow_duplicate=True)],
        Input('ta_s', 'value'),         ## 选择某个TA
        Input('cycle_s', 'value'),
        #State('chip_dataset_store', 'data'),
        State('dataset_id', 'children'),
        prevent_initial_call=True
    )
    def update_mismatch_plot(ta_s, cycle, dataset):
        if type(ta_s) == int and cycle:
            cycle = int(cycle)
            
            x_offset, y_offset = get_chip_data(dataset, 'foreground_points')[int(ta_s)]
            fig = init_ts_plot(x_offset, y_offset, dataset)
            x_start, x_end = x_offset - padding_y , x_offset + 96 + padding_y
            y_start, y_end = y_offset - padding_x , y_offset + 80 +padding_x

            dnb_id_df = get_chip_data(dataset, 'dnb_id_df')    ## idx
        
            # load all dnbs
            dnb_id_sub_df = dnb_id_df.iloc[(dnb_id_df.index.get_level_values('row_index') >= y_start) &
                                      (dnb_id_df.index.get_level_values('row_index') <  y_end)&
                                    (dnb_id_df.index.get_level_values('col_index') >= x_start) &
                                      (dnb_id_df.index.get_level_values('col_index') < x_end)]
    
            fig.add_trace( go.Scattergl(
                x=dnb_id_sub_df.index.get_level_values('col_index'),
                y=dnb_id_sub_df.index.get_level_values('row_index'),
                mode='markers',
                hoverinfo='text',
                text = dnb_id_sub_df["dnb_id"],
                customdata = dnb_id_sub_df["dnb_id"],
                marker=dict(color='blue', size=6,opacity=0.3,symbol='square'),
                name =  "loaded DNB"
            ))
            
            merge_df = get_chip_data(dataset, 'merge_df').reset_index().set_index(['dnb_id', 'cycle'])
            #print("wow")
            #print(merge_df)

            # load miscalled
            def get_annotation(idx):
                dnb_id = int(dnb_id_invalid.loc[idx,'dnb_id'])
                return df_sub.loc[(dnb_id, cycle), 'annotation_MLP']
        
            for group in checklist_options:
                print(cycle)
                invalid_values = checklist_options[group]
                invalid_rows = merge_df[merge_df['annotation_MLP'].isin(invalid_values)]
                print("mismatch num",invalid_rows)
               
                df_sub = invalid_rows.iloc[invalid_rows.index.get_level_values('cycle') == cycle]
                print("df_sub", df_sub)
                indice = df_sub.index.get_level_values('dnb_id')
                dnb_id_invalid = dnb_id_sub_df[dnb_id_sub_df['dnb_id'].isin(indice)]
                # print(indice, dnb_id_sub_df)
                print("invalid",dnb_id_invalid.index)
    
                hover_text = [get_annotation(idx) for idx in dnb_id_invalid.index]
                dnb_text = [dnb_id_invalid.loc[idx,'dnb_id'] for idx in dnb_id_invalid.index]
                #print("hover: ", hover_text)
                
                fig.add_trace( go.Scatter(
                    x=dnb_id_invalid.index.get_level_values('col_index'),
                    y=dnb_id_invalid.index.get_level_values('row_index'),
                    mode='markers',
                    marker=dict(color=color_mapping[group], size=6,symbol='cross',opacity=1),
                    hoverinfo='text',
                    text = hover_text,
                    customdata = dnb_text,
                    name = group,
                    line=dict(width=2)
                ))
        
            fig.update_layout( go.Layout(
                xaxis=dict(range=[x_start, x_end],gridcolor='rgba(0,0,0,0)'),
                yaxis=dict(range=[y_start, y_end],gridcolor='rgba(0,0,0,0)'),
                #margin=dict(l=0,r=0,t=0,b=0),
                title='')
            )
    
            print("update! mem:", get_memory_usage(),"MB")
            return fig, False
        elif cycle is not None:
            return no_update, True
        else:
            return no_update, False
    
    ## 当点击TA，显示DNB的详细信息
    @app.callback(
        Output('chip_detail_plot', 'figure'),
        Input('ta_plot', 'clickData'),
        #State('chip_dataset_store', 'data'),
        State('dataset_id', 'children'),
        prevent_initial_call=True  # 避免初次加载时触发回调
    )
    def display_Clicked_data(clickData, dataset):
        merge_df = get_chip_data(dataset, 'merge_df').reset_index().set_index(['cycle', 'dnb_id'])
        cache_all_data_file = get_chip_data(dataset, 'cache_all_data_file')

        if clickData:
            print("click")
            dnb_id = clickData.get('points', [{}])[0].get('customdata')
            if dnb_id is None:
                gen_blank_plot("no dnb loaded here")
            else:
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                df_for_plot = merge_df.iloc[merge_df.index.get_level_values("dnb_id")==dnb_id]
                print("df:",df_for_plot)
                for idx, row in df_for_plot.iterrows():
                    cycle = idx[0]
                    print(cycle)
                    cycle_h5 = cycle - 1
                    MLP = row['annotation_MLP']
                    with h5py.File(cache_all_data_file, 'r') as f:
                        sig = f['all_data'][dnb_id, cycle_h5, :]
                        color = colors[cycle % len(colors)]
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
                title=f"Detailed Data Visualization Id:{dnb_id}",
                xaxis_title='Seconds',
                yaxis_title='Intensity',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                modebar_bgcolor= '#ffffff',
                modebar_color='#666666',
                margin=dict(t=40)
            )
            
            print("Click! mem:", get_memory_usage(),"MB")
            return fig
        return gen_blank_plot("请先从“区域图”选择DNB")
     
    @app.callback(
        Output('chip_dnbs_plot', 'figure'),
        Input('ta_plot', 'selectedData'),
        State('cycle_s','value'),
        #State('chip_dataset_store', 'data'),
        State('dataset_id', 'children'),
        prevent_initial_call=True  # 避免初次加载时触发回调
    )
    def display_Selected_data(selectedData, cycle, dataset):
        merge_df = get_chip_data(dataset, 'merge_df').reset_index().set_index(['cycle', 'dnb_id'])
        cache_all_data_file = get_chip_data(dataset, 'cache_all_data_file')
        
        if selectedData:
            cycle = int(cycle) if cycle is not None else None
            fig = go.Figure()
            # print(selectedData)
            points = selectedData['points']
            indices = [point['customdata'] for point in points] 
        
            if indices:
            # Create a new figure for the signals
                colors = px.colors.qualitative.Plotly
                df_for_plot = merge_df.iloc[(merge_df.index.get_level_values("dnb_id").isin(indices))&
                                            (merge_df.index.get_level_values("cycle")==cycle)]
                #print(indices)
                # print("select!",df_for_plot)
                cycle_h5 = cycle  - 1
                for idx,row in df_for_plot.iterrows():
                    MLP = row['annotation_MLP']
                    dnb_id =  idx[1]
                    
                    # print(MLP)
                    with h5py.File(cache_all_data_file, 'r') as f:
                        sig = f['all_data'][dnb_id,cycle_h5]
                        color = colors[dnb_id % len(colors)] 
                        fig.add_trace(go.Scatter(
                                x=list(range(sig.shape[0])),
                                y=sig,
                                mode='lines+markers',
                                marker=dict(color=color),
                                name=f'dnb {dnb_id} : {MLP}',
                                text=[f"dnb {dnb_id} : {MLP}"] * sig.shape[0],
                                hoverinfo='text+y'
                            ))
                    
            fig.update_layout(
                    title='Intensity Visualization',
                    xaxis_title='Index',
                    yaxis_title='Intensity',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    modebar_bgcolor= '#ffffff',
                    modebar_color='#666666',
                    margin=dict(t=40)
                )
                #show_coords = indices if len(indices) < 5 else indices[:5] + [f'...{len(indices)} points selected']
            print("Select! mem:", get_memory_usage(),"MB")
            return fig
        return gen_blank_plot("请先从“区域图”框选DNB")