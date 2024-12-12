import h5py, os, sys, cv2, pickle
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import time, random, psutil, uuid
from collections import defaultdict
  
'''
配置
'''
ta_file = '/home/CXxie/workspace/basecall/E25/TestArea/gt.npz'

'''
通用函数
'''

def gen_blank_plot(text):
    fig = go.Figure()
    fig.update_layout(
        title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        modebar_bgcolor= '#ffffff',
        modebar_color='#666666',
        xaxis=dict(
            showgrid=False,  # Hide grid
            zeroline=False,  # Hide zero line
            visible=False    # Hide axis
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        ),
        showlegend=False,
        annotations=[
            dict(
                text=text,
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=20)
            )
        ]
    )
    return fig

def init_empty_figure():
    return {
        "data": [],
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{"text": "Loading...", "xref": "paper", "yref": "paper", "showarrow": False}]
        }
    }

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss/1024/1024

'''
分臂图特有函数
'''
color_map = {'match': '#5470c6', 'mismatch': '#ee6666', 'clip': '#6e7079',  'unmapped': '#6e7070', 'T':"pink", 'C': "lightgreen", 'A': "red", 'G': "yellow",
             'T Mismatch': "lime",
'C Mismatch': "coral",
'A Mismatch': "violet",
'G Mismatch': "gold"}

def datasets_arm(dataset):
    return [f'/storage/QShe_share/data/{dataset}/{dataset}_dataframe_cache_TA_4.pkl', f'/storage/QShe_share/data/{dataset}/{dataset}_all_data_cache.h5']

arm_data = dict()
arm_personal_data = defaultdict(dict)

def load_arm_dataset(dataset,uid):
    global arm_data
    if dataset not in arm_data:
        print("assign new data", dataset)
        cache_info_file, cache_ints_file = datasets_arm(dataset)
        info_df = pd.read_pickle(cache_info_file)
        info_df.index = info_df.index.to_numpy()
        arm_data[dataset] = {
            'info_df' : info_df,
            'cache_ints_file' : cache_ints_file,
        }
        arm_personal_data[dataset][uid] = {
            'sampled_df' : None,
            'marker_size' : None,
        }
    else:
        print("grab ",dataset)
        

def get_arm_data(dataset, key):
    global arm_data
    return arm_data[dataset][key]

def get_arm_personal_data(dataset, uid, key):
    global arm_personal_data
    return arm_personal_data[dataset][uid][key]

def set_arm_personal_data(dataset, uid, key, value):
    global arm_personal_data
    arm_personal_data[dataset][uid][key] = value

def getTag(r, value):
    if r['annotation_MLP'] == '*':
        return 'unmapped'
    elif r['annotation_MLP'] == '-':
        return None
    elif r['annotation_MLP'] is not None:
        if value == 'real_base':
            return r['annotation_MLP'][0]
        else:
            return r['annotation_MLP'][-1]

def set_arm_cls(s_df, value):
    if value in ['real_base', 'call_base']:
        return s_df.apply(getTag, args=(value,), axis=1)
    else:
        #return s_df['annotation_MLP'].map(lambda x: 'mismatch' if '>' in x else 'match')
        return s_df['annotation_MLP'].map(lambda x: f'{x[0]} Mismatch' if '>' in x else 'match')
    
def generate_arm_plot(sample_size, color_value, dataset, uid, need_sample=True):
    def sample_arm_df(df, sample_size):
        unique_dnb_ids = df['dnb_id'].unique()
        random_indices = np.random.choice(unique_dnb_ids, size=sample_size, replace=False)
        return df[df['dnb_id'].isin(random_indices)].copy()

    sampled_dataframe = sample_arm_df(get_arm_data(dataset, 'info_df'), sample_size) if need_sample else get_arm_personal_data(dataset, 'sampled_df')
    total_points = len(sampled_dataframe)

    
    sampled_dataframe.loc[:, 'cls'] = set_arm_cls(sampled_dataframe,color_value)

    # marker_size = max(0.5, (1/2)**(np.log10(total_points)-5))
    # marker_size = min(marker_size, 5)
    marker_size = 5

    set_arm_personal_data(dataset, uid, 'sampled_df', sampled_dataframe)
    set_arm_personal_data(dataset, uid, 'marker_size', marker_size)

    arm_plot = go.Figure()
    
    for cls_value in sampled_dataframe['cls'].unique():
        if cls_value:
            filtered_df = sampled_dataframe[sampled_dataframe['cls'] == cls_value]
            arm_plot.add_trace(
                go.Scattergl(
                    x=filtered_df['1-5 mean'],
                    y=filtered_df['6-12 mean'],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=color_map[cls_value],
                        opacity=1,  
                    ),
                    name=cls_value,
                    hoverinfo='text',
                    text=filtered_df.apply(lambda row: f"Cycle: {int(row['cycle'])}, DNB ID: {int(row['dnb_id'])}", axis=1),
                    customdata = filtered_df.apply(lambda row: (row['dnb_id'], row.name), axis=1),
                    showlegend=True,
                    #selected=dict(
                     #   marker=dict(
                            #color='red',  
                      #      opacity=1, 
                       #     size=marker_size + 2  
                        #)
                    #),
                    unselected=dict(
                        marker=dict(
                            color='grey',  
                            opacity=0.2 
                        )
                    )
                )
            )

    arm_plot.update_layout(
        title='',
        xaxis=dict(title='前5秒的光强均值', autorange=True),
        yaxis=dict(title='后7秒的光强均值', autorange=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        modebar_bgcolor= '#ffffff',
        modebar_color='#666666',
        dragmode='pan',  # 默认启用平移模式
        autosize=True,  # 自动调整大小
        clickmode='event+select',
        legend=dict(title=""),
        margin=dict(t=40)
    )

    return arm_plot

def transform_value(value):
    value = int(10 ** value)
    return value

## 获取高亮的点
def get_arm_highlight_points(hover_data, dataset, uid):
    dnb_id = hover_data['points'][0]['customdata'][0]
    #hover_index = hover_data['points'][0]['customdata'][1]
    sampled_df = get_arm_personal_data(dataset, uid, 'sampled_df')
    highlighted_df = sampled_df[sampled_df['dnb_id'] == dnb_id]
    # highlighted_df_rm = highlighted_df[highlighted_df.index != hover_index]
    print(highlighted_df.index)
    return highlighted_df

'''
芯片图特有函数
'''
def datasets_chip(dataset):
    return {
        'mask_mtx_file': f'/storage/QShe_share/data/{dataset}/mask.npy',
        'meta_file': f'/storage/QShe_share/data/{dataset}/{dataset}_meta.pkl',
        'dnb_id_sub_df_file': f'/storage/QShe_share/data/{dataset}/{dataset}_dnb_id_sub_df_4.pkl',
        'cache_TA_long': f'/storage/QShe_share/data/{dataset}/{dataset}_mm_df_ta_long.pkl',
        'cache_dataframe_file': f'/storage/QShe_share/data/{dataset}/{dataset}_dataframe_cache_TA_4.pkl',
        'cache_dataframe_file_all': f'/storage/QShe_share/data/{dataset}/{dataset}_dataframe_cache2_50barcode.pkl',
        'cache_all_data_file': f'/storage/QShe_share/data/{dataset}/{dataset}_all_data_cache.h5',
        'TA_chip_df': f'/storage/QShe_share/data/{dataset}/{dataset}_TA_chip_df.pkl',
        'TA_chip_df_cycles': f'/storage/QShe_share/data/{dataset}/{dataset}_TA_chip_df_cycles.pkl',
    }

chip_data = {}
padding_y = 40
padding_x = 48

checklist_options = {
    'T Mismatch': ['T>A', 'T>C', 'T>G'],
    'C Mismatch': ['C>A', 'C>T', 'C>G'],
    'A Mismatch': ['A>T', 'A>C', 'A>G'],
    'G Mismatch': ['G>A', 'G>C', 'G>T']
}

color_mapping = {
    'T Mismatch':"pink",
    'C Mismatch': "lightgreen",
    'A Mismatch': "red",
    'G Mismatch': "yellow"
}  

def load_chip_dataset(dataset):
    global chip_data
    ds = datasets_chip(dataset)

    if dataset not in chip_data:
        print("assign new data", dataset)
        mask_mtx, foreground_points, foreground_mask = read_mask_mtx(ds['mask_mtx_file'])
        ta_stat_df = pd.read_pickle(ds['TA_chip_df'])
        ta_stat_df_cycles = pd.read_pickle(ds['TA_chip_df_cycles'])
        meta = pd.read_pickle(ds['meta_file'])
        meta = meta[['idx', 'x', 'y']]
        meta.columns = ['dnb_id', 'row_index', 'col_index']
        dnb_id_df = meta.set_index(['row_index', 'col_index'])
        dnb_id_sub_df = pd.read_pickle(ds['dnb_id_sub_df_file'])
        merge_df = pd.read_pickle(ds['cache_TA_long'])
        merge_df = merge_df.rename(columns={'idx': 'dnb_id'})
        # merge_df = merge_df.rename_axis(index={'idx': 'dnb_id'})
        ta_gt = np.load(ta_file)['data']
        gt_shifted = np.roll(ta_gt, shift=1, axis=1)
       
        chip_data[dataset] = {
            'mask_mtx' : mask_mtx,
            'foreground_points' : foreground_points,
            'foreground_mask' : foreground_mask,
            'merge_df' : merge_df,
            'dnb_id_df' : dnb_id_df,
            'dnb_id_sub_df' : dnb_id_sub_df,
            'ta_stat_df' : ta_stat_df,
            'ta_stat_df_cycles' : ta_stat_df_cycles,
            # chip_data'mask_vis' : mask_vis,
            'gt_shifted' : gt_shifted,
            'cache_all_data_file' : ds['cache_all_data_file'],
            'user_id' : str(uuid.uuid4())
        }
        print("user_id renewed", chip_data[dataset]['user_id'])
    else:
        print("grab ",dataset,chip_data[dataset]['user_id'])

def judge_chip_dataset(dataset):
    global chip_data
    return chip_data.get(dataset, None)
    
def get_chip_data(dataset, key):
    global chip_data
    return chip_data[dataset][key]

#def set_chip_data(dataset,key, value):
    # global chip_data
 #   chip_data[dataset,key] = value

"""获取全部mask信息和TA的位置信息"""
def read_mask_mtx(fname):
    x_coords =  [307, 1015, 1723, 2431, 3139, 3847, 4555, 5263]
    y_coords =  [314, 1022, 1730, 2438, 3146, 3854, 4562, 5270, 5978, 6686, 7394, 8102]
    
    mask_mtx = np.load(fname)
    if mask_mtx is None:
        raise ValueError("Failed to load image data")
    # print(f"Data shape: {mask_mtx.shape}")

    if mask_mtx.ndim == 3:
        gray_data = cv2.cvtColor(mask_mtx, cv2.COLOR_BGR2GRAY)
    else:
        gray_data = mask_mtx

    threshold_value = 0.98  # 阈值
    max_binary_value = 1.0  # 数据范围的最大值

    _, binary = cv2.threshold(gray_data, threshold_value, max_binary_value, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    foreground_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    all_points = [(x, y) for x in x_coords for y in y_coords]

    foreground_points = [
        (x, y) for x, y in all_points
        if foreground_mask[y, x] == 0
    ]
    return mask_mtx, foreground_points, foreground_mask


def init_chip_plot(dataset):
    foreground_mask = get_chip_data(dataset, 'foreground_mask')
    
    fig = go.Figure()
    non_zero_points = np.array(np.where(foreground_mask))
    non_zero_points_downsampled = non_zero_points[:,::1000]      
    fig.add_trace(go.Scattergl(
        x=non_zero_points_downsampled[1],
        y=non_zero_points_downsampled[0],
        mode='markers',
        marker_symbol='x',
        marker=dict(
            color='black',
            size=3,
            opacity=0.1
        ),
        showlegend=False,
        name="none",
        hoverinfo="none"
    ))

    fig.update_layout(
        title='',
        dragmode='pan',  # 默认启用平移模式
        autosize=True,  # 自动调整大小
        xaxis=dict(visible=False,range=[0,max(non_zero_points_downsampled[1])]),
        yaxis=dict(visible=False,range=[max(non_zero_points_downsampled[0]),0]),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgb(255, 255, 255)'
    )
    return fig


def init_ts_plot(x_offset, y_offset, dataset):
    gt_shifted = get_chip_data(dataset, 'gt_shifted')
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z = gt_shifted,
        colorscale= [[0, 'rgb(255,255,255)'], [1, 'rgb(192,192,192)']],  
        zauto=False, zmin=0, zmax=1,  
        x=np.arange(gt_shifted.shape[1]) + x_offset,  
        y=np.arange(gt_shifted.shape[0]) + y_offset,
        showscale=False,
        hoverinfo='none'
    ))
    fig.update_layout(
        go.Layout(
            #showlegend=False, 
            dragmode='pan',  # 默认启用平移模式
            xaxis=dict(range = [0-padding_x,96+padding_x],
                zeroline=False,gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(range = [0-padding_y,80+padding_y],
                zeroline=False,gridcolor='rgba(0,0,0,0)'),
            margin=dict(l=20, r=20, t=40, b=20),
            title=''
        )
    )
    return fig