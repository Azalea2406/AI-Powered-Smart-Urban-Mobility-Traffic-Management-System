"""
dashboard.py — Plotly Dash Interactive Dashboard
Stanley College of Engineering and Technology for Women
Run: python dashboard.py  →  Open http://127.0.0.1:8050

Install first:
    pip install dash plotly pandas joblib
"""

import pandas as pd
import numpy as np
import joblib, os

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc   # pip install dash-bootstrap-components

# ── Load data ─────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
df   = pd.read_csv(os.path.join(BASE, 'traffic_data.csv'), parse_dates=['timestamp'])
clf  = joblib.load(os.path.join(BASE, 'model.pkl'))

FEATURES = ['hour','day_of_week','is_weekend','is_peak',
            'avg_speed_kmph','rolling_30m','rolling_1h','rolling_3h']
ROADS    = sorted(df['road_id'].unique())
LABEL_MAP= {0:'Low', 1:'Medium', 2:'High'}
COLOR_MAP= {'Low':'#00e676', 'Medium':'#ffab40', 'High':'#ff5252'}

df['congestion_label'] = df['congestion_level'].map(LABEL_MAP)

DARK = {
    'bg':     '#0a0e1a',
    'panel':  '#111827',
    'border': '#1e2d45',
    'text':   '#e2e8f0',
    'muted':  '#64748b',
    'accent': '#00d4ff',
}

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor=DARK['panel'],
        plot_bgcolor =DARK['panel'],
        font_color   =DARK['text'],
        font_family  ='DM Sans',
        margin=dict(l=16, r=16, t=40, b=16),
        legend=dict(bgcolor=DARK['panel'], bordercolor=DARK['border'])
    )
    fig.update_xaxes(gridcolor=DARK['border'], linecolor=DARK['border'])
    fig.update_yaxes(gridcolor=DARK['border'], linecolor=DARK['border'])
    return fig

# ── Pre-built figures ─────────────────────────────────────────────

def fig_hourly():
    hourly = df.groupby('hour')['vehicle_count'].mean().reset_index()
    hourly['color'] = hourly['hour'].apply(
        lambda h: '#ff5252' if h in [8,9,17,18,19] else '#00d4ff')
    fig = go.Figure(go.Bar(
        x=hourly['hour'], y=hourly['vehicle_count'],
        marker_color=hourly['color'],
        hovertemplate='Hour %{x}:00<br>Avg Vehicles: %{y:.1f}<extra></extra>'
    ))
    fig.update_layout(title='Hourly Traffic Density (🔴 = Peak)', xaxis_title='Hour', yaxis_title='Avg Vehicles')
    return dark_fig(fig)

def fig_pie():
    dist = df['congestion_label'].value_counts().reset_index()
    dist.columns = ['level','count']
    fig = px.pie(dist, names='level', values='count',
                 color='level',
                 color_discrete_map=COLOR_MAP,
                 hole=0.55)
    fig.update_traces(textfont_color='white')
    fig.update_layout(title='Congestion Distribution')
    return dark_fig(fig)

def fig_speed_scatter():
    sample = df.sample(min(500, len(df)), random_state=42)
    fig = px.scatter(sample, x='avg_speed_kmph', y='vehicle_count',
                     color='congestion_label',
                     color_discrete_map=COLOR_MAP,
                     opacity=0.7,
                     labels={'avg_speed_kmph':'Speed (km/h)', 'vehicle_count':'Vehicle Count'})
    fig.update_layout(title='Speed vs Vehicle Count')
    return dark_fig(fig)

def fig_corridors():
    peak_df = df[df['is_peak'] == 1] if df['is_peak'].sum() > 0 else df
    stats = peak_df.groupby('road_id').agg(
        avg_count=('vehicle_count','mean'),
        incidents=('congestion_level', lambda x: (x==2).sum())
    ).reset_index()
    mc = stats['avg_count'].max(); mi = stats['incidents'].max()
    stats['risk_score'] = ((stats['avg_count']/mc)*50 + (stats['incidents']/(mi if mi>0 else 1))*50).round(1)
    stats = stats.sort_values('risk_score', ascending=True).tail(10)
    stats['color'] = stats['risk_score'].apply(
        lambda s: '#ff5252' if s >= 70 else '#ffab40' if s >= 40 else '#00e676')
    fig = go.Figure(go.Bar(
        x=stats['risk_score'], y=stats['road_id'].str.replace('_',' '),
        orientation='h',
        marker_color=stats['color'],
        hovertemplate='%{y}<br>Risk Score: %{x}<extra></extra>'
    ))
    fig.add_vline(x=70, line_dash='dash', line_color='#ff5252', annotation_text='High Risk')
    fig.add_vline(x=40, line_dash='dash', line_color='#ffab40', annotation_text='Medium Risk')
    fig.update_layout(title='Top High-Risk Corridors (Peak Hours)', xaxis_title='Risk Score (0–100)')
    return dark_fig(fig)

def fig_map():
    sample = df.dropna(subset=['latitude','longitude'])
    fig = px.scatter_mapbox(
        sample, lat='latitude', lon='longitude',
        color='congestion_label', size='vehicle_count',
        color_discrete_map=COLOR_MAP,
        mapbox_style='carto-darkmatter',
        zoom=11, center={'lat':17.38, 'lon':78.48},
        hover_data={'road_id':True, 'vehicle_count':True,
                    'avg_speed_kmph':True, 'latitude':False, 'longitude':False},
        title='Hyderabad Traffic Heatmap'
    )
    fig.update_layout(paper_bgcolor=DARK['panel'], font_color=DARK['text'],
                      margin=dict(l=0,r=0,t=40,b=0))
    return fig

def fig_feature_imp():
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    colors = ['#ff5252' if i == imp.idxmax() else '#00d4ff' for i in imp.index]
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))
    fig.update_layout(title='Feature Importance — Random Forest', xaxis_title='Importance Score')
    return dark_fig(fig)

# ── App layout ────────────────────────────────────────────────────
app = Dash(__name__,
           external_stylesheets=[
               dbc.themes.CYBORG,
               'https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap'
           ])
server = app.server  # Required for Render to detect the Flask server
app.title = 'Smart Traffic — Hyderabad'

total   = len(df)
low_pct = round((df['congestion_level']==0).sum() / total * 100, 1)
med_pct = round((df['congestion_level']==1).sum() / total * 100, 1)
hi_pct  = round((df['congestion_level']==2).sum() / total * 100, 1)

CARD_STYLE = {
    'background': DARK['panel'],
    'border': f'1px solid {DARK["border"]}',
    'borderRadius': '12px',
    'padding': '20px 24px',
}
TITLE_STYLE = {
    'fontFamily': 'Rajdhani, sans-serif',
    'color': DARK['accent'],
    'fontSize': '0.85rem',
    'letterSpacing': '1px',
    'textTransform': 'uppercase',
    'marginBottom': '4px'
}
VAL_STYLE_BASE = {
    'fontFamily': 'Rajdhani, sans-serif',
    'fontSize': '2rem',
    'fontWeight': '700',
    'lineHeight': '1.1'
}

app.layout = html.Div(style={'background': DARK['bg'], 'minHeight':'100vh', 'fontFamily':'DM Sans, sans-serif'}, children=[

    # ── Header ──
    html.Div(style={
        'background': 'linear-gradient(135deg,#0a0e1a,#0d1f35)',
        'borderBottom': f'1px solid {DARK["border"]}',
        'padding': '18px 32px',
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'
    }, children=[
        html.H1('🚦 Smart Traffic | Hyderabad', style={
            'fontFamily':'Rajdhani,sans-serif', 'color': DARK['accent'],
            'fontSize':'1.5rem', 'fontWeight':'700', 'letterSpacing':'1px'
        }),
        html.Div('Stanley College of Engineering & Technology for Women — AI-Powered Urban Mobility System',
                 style={'color': DARK['muted'], 'fontSize':'0.8rem', 'textAlign':'center'}),
        html.Div([
            html.Span('● ', style={'color':'#00e676'}),
            html.Span('Dashboard Live', style={'color':'#00e676', 'fontSize':'0.8rem'})
        ])
    ]),

    html.Div(style={'padding':'24px 32px', 'maxWidth':'1400px', 'margin':'0 auto'}, children=[

        # ── Stat Cards ──
        dbc.Row(style={'marginBottom':'20px'}, children=[
            dbc.Col(html.Div(style={**CARD_STYLE, 'borderTop':'3px solid #00d4ff'}, children=[
                html.Div('Total Records', style=TITLE_STYLE),
                html.Div(f'{total:,}', style={**VAL_STYLE_BASE, 'color':'#00d4ff'}),
                html.Div('traffic observations', style={'color':DARK['muted'],'fontSize':'0.78rem'})
            ]), width=3),
            dbc.Col(html.Div(style={**CARD_STYLE, 'borderTop':'3px solid #00e676'}, children=[
                html.Div('Low Congestion', style=TITLE_STYLE),
                html.Div(f'{low_pct}%', style={**VAL_STYLE_BASE, 'color':'#00e676'}),
                html.Div('vehicle count < 200', style={'color':DARK['muted'],'fontSize':'0.78rem'})
            ]), width=3),
            dbc.Col(html.Div(style={**CARD_STYLE, 'borderTop':'3px solid #ffab40'}, children=[
                html.Div('Medium Congestion', style=TITLE_STYLE),
                html.Div(f'{med_pct}%', style={**VAL_STYLE_BASE, 'color':'#ffab40'}),
                html.Div('200 – 400 vehicles', style={'color':DARK['muted'],'fontSize':'0.78rem'})
            ]), width=3),
            dbc.Col(html.Div(style={**CARD_STYLE, 'borderTop':'3px solid #ff5252'}, children=[
                html.Div('High Congestion', style=TITLE_STYLE),
                html.Div(f'{hi_pct}%', style={**VAL_STYLE_BASE, 'color':'#ff5252'}),
                html.Div('vehicle count > 400', style={'color':DARK['muted'],'fontSize':'0.78rem'})
            ]), width=3),
        ]),

        # ── Row 1: Hourly + Pie ──
        dbc.Row(style={'marginBottom':'20px'}, children=[
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                dcc.Graph(figure=fig_hourly(), config={'displayModeBar':False}, style={'height':'280px'})
            ]), width=7),
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                dcc.Graph(figure=fig_pie(), config={'displayModeBar':False}, style={'height':'280px'})
            ]), width=5),
        ]),

        # ── Row 2: Map (full width) ──
        dbc.Row(style={'marginBottom':'20px'}, children=[
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                dcc.Graph(figure=fig_map(), config={'displayModeBar':False}, style={'height':'340px'})
            ]), width=12),
        ]),

        # ── Row 3: Corridors + Feature Imp ──
        dbc.Row(style={'marginBottom':'20px'}, children=[
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                dcc.Graph(figure=fig_corridors(), config={'displayModeBar':False}, style={'height':'300px'})
            ]), width=6),
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                dcc.Graph(figure=fig_feature_imp(), config={'displayModeBar':False}, style={'height':'300px'})
            ]), width=6),
        ]),

        # ── Row 4: Scatter + Live Predictor ──
        dbc.Row(children=[
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                dcc.Graph(figure=fig_speed_scatter(), config={'displayModeBar':False}, style={'height':'320px'})
            ]), width=6),

            # Live Predictor Panel
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.Div('⚡ Live Congestion Predictor', style={**TITLE_STYLE, 'marginBottom':'16px', 'fontSize':'1rem'}),
                dbc.Row([
                    dbc.Col([
                        html.Label('Hour (0–23)', style={'color':DARK['muted'],'fontSize':'0.75rem','letterSpacing':'0.8px'}),
                        dcc.Input(id='p-hour', type='number', value=8, min=0, max=23,
                                  style={'width':'100%','background':'#0a0e1a','color':DARK['text'],
                                         'border':f'1px solid {DARK["border"]}','borderRadius':'8px','padding':'10px'})
                    ], width=6),
                    dbc.Col([
                        html.Label('Avg Speed (km/h)', style={'color':DARK['muted'],'fontSize':'0.75rem','letterSpacing':'0.8px'}),
                        dcc.Input(id='p-speed', type='number', value=22, min=5, max=120,
                                  style={'width':'100%','background':'#0a0e1a','color':DARK['text'],
                                         'border':f'1px solid {DARK["border"]}','borderRadius':'8px','padding':'10px'})
                    ], width=6),
                ], style={'marginBottom':'12px'}),
                dbc.Row([
                    dbc.Col([
                        html.Label('Rolling 30m', style={'color':DARK['muted'],'fontSize':'0.75rem','letterSpacing':'0.8px'}),
                        dcc.Input(id='p-r30', type='number', value=380,
                                  style={'width':'100%','background':'#0a0e1a','color':DARK['text'],
                                         'border':f'1px solid {DARK["border"]}','borderRadius':'8px','padding':'10px'})
                    ], width=4),
                    dbc.Col([
                        html.Label('Rolling 1h', style={'color':DARK['muted'],'fontSize':'0.75rem','letterSpacing':'0.8px'}),
                        dcc.Input(id='p-r1h', type='number', value=420,
                                  style={'width':'100%','background':'#0a0e1a','color':DARK['text'],
                                         'border':f'1px solid {DARK["border"]}','borderRadius':'8px','padding':'10px'})
                    ], width=4),
                    dbc.Col([
                        html.Label('Rolling 3h', style={'color':DARK['muted'],'fontSize':'0.75rem','letterSpacing':'0.8px'}),
                        dcc.Input(id='p-r3h', type='number', value=390,
                                  style={'width':'100%','background':'#0a0e1a','color':DARK['text'],
                                         'border':f'1px solid {DARK["border"]}','borderRadius':'8px','padding':'10px'})
                    ], width=4),
                ], style={'marginBottom':'16px'}),
                html.Button('⚡ PREDICT', id='predict-btn', n_clicks=0, style={
                    'width':'100%', 'padding':'12px',
                    'background':'linear-gradient(135deg,#0066ff,#00d4ff)',
                    'color':'#fff', 'border':'none', 'borderRadius':'8px',
                    'fontFamily':'Rajdhani,sans-serif', 'fontSize':'1rem',
                    'fontWeight':'700', 'letterSpacing':'1px', 'cursor':'pointer'
                }),
                html.Div(id='predict-output', style={'marginTop':'14px'})
            ]), width=6),
        ]),

    ]),

    html.Div('AI-Powered Smart Urban Mobility & Traffic Management System  |  Stanley College of Engineering and Technology for Women, Hyderabad  |  2025–2026',
             style={'textAlign':'center','padding':'16px','color':DARK['muted'],
                    'fontSize':'0.72rem','borderTop':f'1px solid {DARK["border"]}','marginTop':'8px'})
])

# ── Callback: predictor ───────────────────────────────────────────
@callback(
    Output('predict-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('p-hour',  'value'),
    State('p-speed', 'value'),
    State('p-r30',   'value'),
    State('p-r1h',   'value'),
    State('p-r3h',   'value'),
    prevent_initial_call=True
)
def run_prediction(n, hour, speed, r30, r1h, r3h):
    if not n:
        return ''
    hour  = int(hour  or 0)
    speed = float(speed or 30)
    r30   = float(r30  or 0)
    r1h   = float(r1h  or 0)
    r3h   = float(r3h  or 0)

    is_peak    = 1 if hour in [8,9,17,18,19] else 0
    day_of_week= 1   # default Tuesday
    is_weekend = 0

    row  = [[hour, day_of_week, is_weekend, is_peak, speed, r30, r1h, r3h]]
    pred = clf.predict(row)[0]
    prob = clf.predict_proba(row)[0]

    label  = LABEL_MAP[pred]
    conf   = round(prob.max() * 100, 1)
    color  = COLOR_MAP[label]
    icons  = {'Low':'🟢','Medium':'🟡','High':'🔴'}

    return html.Div(style={
        'background': f'rgba({",".join(str(int(c*255)) for c in px.colors.hex_to_rgb(color))},0.08)' if False else '#1a2540',
        'border': f'2px solid {color}',
        'borderRadius': '10px', 'padding': '16px'
    }, children=[
        html.Div(f'{icons[label]} {label} Congestion', style={
            'fontFamily':'Rajdhani,sans-serif', 'fontSize':'1.7rem',
            'fontWeight':'700', 'color': color
        }),
        html.Div(f'Model confidence: {conf}%  ·  Peak hour: {"Yes" if is_peak else "No"}',
                 style={'color': DARK['muted'], 'fontSize':'0.83rem', 'marginTop':'4px'})
    ])


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  📊 Plotly Dash Dashboard")
    print("  Stanley College of Engineering — Hyderabad")
    print("="*55)
    print(f"  ✅ Dataset: {len(df)} records, {df['road_id'].nunique()} roads")
    print("  🌐 Dashboard → http://127.0.0.1:8050")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))