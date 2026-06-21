"""
dashboard.py — Plotly Dash Interactive Dashboard
AI-Powered Smart Urban Mobility & Traffic Management System
Stanley College of Engineering and Technology for Women

Now with: Authentication | Database | Live Data (TomTom API) | Alerts

Run: python dashboard.py  →  Open http://127.0.0.1:8050

Install first:
    pip install dash dash-bootstrap-components plotly pandas joblib requests
"""

import pandas as pd
import numpy as np
import joblib, os
import dash
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc

import database as db
import live_data

# ── Init database & load model ─────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
db.init_db()

clf = joblib.load(os.path.join(BASE, 'model.pkl'))

FEATURES = ['hour','day_of_week','is_weekend','is_peak',
            'avg_speed_kmph','rolling_30m','rolling_1h','rolling_3h']
LABEL_MAP = {0:'Low', 1:'Medium', 2:'High'}
COLOR_MAP = {'Low':'#00e676', 'Medium':'#ffab40', 'High':'#ff5252'}
ALERT_RISK_THRESHOLD = 70

DARK = {
    'bg':     '#0a0e1a',
    'panel':  '#111827',
    'border': '#1e2d45',
    'text':   '#e2e8f0',
    'muted':  '#64748b',
    'accent': '#00d4ff',
}


def get_df():
    df = db.get_all_traffic_records()
    df['congestion_label'] = df['congestion_level'].map(LABEL_MAP)
    return df


def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor=DARK['panel'], plot_bgcolor=DARK['panel'],
        font_color=DARK['text'], font_family='DM Sans',
        margin=dict(l=16, r=16, t=40, b=16),
        legend=dict(bgcolor=DARK['panel'], bordercolor=DARK['border'])
    )
    fig.update_xaxes(gridcolor=DARK['border'], linecolor=DARK['border'])
    fig.update_yaxes(gridcolor=DARK['border'], linecolor=DARK['border'])
    return fig


# ── Figure builders ─────────────────────────────────────────────────
def fig_hourly(df):
    hourly = df.groupby('hour')['vehicle_count'].mean().reset_index()
    hourly['color'] = hourly['hour'].apply(lambda h: '#ff5252' if h in [8,9,17,18,19] else '#00d4ff')
    fig = go.Figure(go.Bar(
        x=hourly['hour'], y=hourly['vehicle_count'], marker_color=hourly['color'],
        hovertemplate='Hour %{x}:00<br>Avg Vehicles: %{y:.1f}<extra></extra>'
    ))
    fig.update_layout(title='Hourly Traffic Density (🔴 = Peak)', xaxis_title='Hour', yaxis_title='Avg Vehicles')
    return dark_fig(fig)


def fig_pie(df):
    dist = df['congestion_label'].value_counts().reset_index()
    dist.columns = ['level','count']
    fig = px.pie(dist, names='level', values='count', color='level',
                 color_discrete_map=COLOR_MAP, hole=0.55)
    fig.update_traces(textfont_color='white')
    fig.update_layout(title='Congestion Distribution')
    return dark_fig(fig)


def fig_speed_scatter(df):
    sample = df.sample(min(500, len(df)), random_state=42)
    fig = px.scatter(sample, x='avg_speed_kmph', y='vehicle_count', color='congestion_label',
                     color_discrete_map=COLOR_MAP, opacity=0.7,
                     labels={'avg_speed_kmph':'Speed (km/h)', 'vehicle_count':'Vehicle Count'})
    fig.update_layout(title='Speed vs Vehicle Count')
    return dark_fig(fig)


def get_corridors(df):
    peak_df = df[df['is_peak'] == 1] if df['is_peak'].sum() > 0 else df
    stats = peak_df.groupby('road_id').agg(
        avg_count=('vehicle_count','mean'),
        incidents=('congestion_level', lambda x: (x==2).sum())
    ).reset_index()
    mc = stats['avg_count'].max(); mi = stats['incidents'].max()
    stats['risk_score'] = ((stats['avg_count']/mc if mc>0 else 0)*50 +
                            (stats['incidents']/(mi if mi>0 else 1))*50).round(1)
    return stats.sort_values('risk_score', ascending=False)


def fig_corridors(df):
    stats = get_corridors(df).sort_values('risk_score', ascending=True).tail(10)
    stats['color'] = stats['risk_score'].apply(
        lambda s: '#ff5252' if s >= 70 else '#ffab40' if s >= 40 else '#00e676')
    fig = go.Figure(go.Bar(
        x=stats['risk_score'], y=stats['road_id'].str.replace('_',' '), orientation='h',
        marker_color=stats['color'],
        hovertemplate='%{y}<br>Risk Score: %{x}<extra></extra>'
    ))
    fig.add_vline(x=70, line_dash='dash', line_color='#ff5252', annotation_text='High Risk')
    fig.add_vline(x=40, line_dash='dash', line_color='#ffab40', annotation_text='Medium Risk')
    fig.update_layout(title='Top High-Risk Corridors (Peak Hours)', xaxis_title='Risk Score (0–100)')
    return dark_fig(fig)


def fig_map(df):
    sample = df.dropna(subset=['latitude','longitude'])
    fig = px.scatter_mapbox(
        sample, lat='latitude', lon='longitude', color='congestion_label', size='vehicle_count',
        color_discrete_map=COLOR_MAP, mapbox_style='carto-darkmatter',
        zoom=11, center={'lat':17.38, 'lon':78.48},
        hover_data={'road_id':True, 'vehicle_count':True, 'avg_speed_kmph':True,
                    'latitude':False, 'longitude':False},
        title='Hyderabad Traffic Heatmap'
    )
    fig.update_layout(paper_bgcolor=DARK['panel'], font_color=DARK['text'], margin=dict(l=0,r=0,t=40,b=0))
    return fig


def fig_feature_imp():
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    colors = ['#ff5252' if i == imp.idxmax() else '#00d4ff' for i in imp.index]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h', marker_color=colors,
                            hovertemplate='%{y}: %{x:.4f}<extra></extra>'))
    fig.update_layout(title='Feature Importance — Random Forest', xaxis_title='Importance Score')
    return dark_fig(fig)


def check_alerts(df):
    stats = get_corridors(df)
    high_risk = stats[stats['risk_score'] >= ALERT_RISK_THRESHOLD]
    for _, row in high_risk.iterrows():
        msg = f"{row['road_id'].replace('_',' ')} has crossed High Risk threshold (score: {row['risk_score']})"
        db.create_alert(row['road_id'], 2, row['risk_score'], msg)
    return db.get_recent_alerts(limit=5)


# ── App ───────────────────────────────────────────────────────────
app = Dash(__name__,
           external_stylesheets=[
               dbc.themes.CYBORG,
               'https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap'
           ],
           suppress_callback_exceptions=True)
server = app.server  # Required for Render deployment
server.secret_key = os.environ.get('SECRET_KEY', 'smart-traffic-hyd-dash-2026')
app.title = 'Smart Traffic — Hyderabad'

CARD_STYLE = {'background': DARK['panel'], 'border': f'1px solid {DARK["border"]}',
              'borderRadius': '12px', 'padding': '20px 24px'}
TITLE_STYLE = {'fontFamily': 'Rajdhani, sans-serif', 'color': DARK['accent'], 'fontSize': '0.85rem',
               'letterSpacing': '1px', 'textTransform': 'uppercase', 'marginBottom': '4px'}
VAL_STYLE_BASE = {'fontFamily': 'Rajdhani, sans-serif', 'fontSize': '2rem', 'fontWeight': '700', 'lineHeight': '1.1'}
INPUT_STYLE = {'width':'100%','background':'#0a0e1a','color':DARK['text'],
               'border':f'1px solid {DARK["border"]}','borderRadius':'8px','padding':'10px'}
LABEL_STYLE = {'color':DARK['muted'],'fontSize':'0.75rem','letterSpacing':'0.8px'}


# ════════════════════════════════════════════════════════════════════
#  LOGIN LAYOUT
# ════════════════════════════════════════════════════════════════════
def login_layout(error=None, mode='login'):
    return html.Div(style={'background': DARK['bg'], 'minHeight':'100vh', 'display':'flex',
                            'alignItems':'center', 'justifyContent':'center',
                            'fontFamily':'DM Sans, sans-serif'}, children=[
        html.Div(style={'background':DARK['panel'], 'border':f'1px solid {DARK["border"]}',
                         'borderRadius':'16px', 'padding':'40px', 'width':'380px',
                         'boxShadow':'0 0 40px rgba(0,212,255,0.08)'}, children=[
            html.H1('🚦 Smart Traffic', style={'fontFamily':'Rajdhani,sans-serif','color':DARK['accent'],
                     'fontSize':'1.6rem','textAlign':'center','marginBottom':'6px'}),
            html.Div('Hyderabad Urban Mobility System', style={'textAlign':'center','color':DARK['muted'],
                     'fontSize':'0.8rem','marginBottom':'24px'}),
            html.Div(error, style={'background':'rgba(255,82,82,0.1)','border':'1px solid #ff5252',
                     'color':'#ff5252','padding':'10px 14px','borderRadius':'8px','fontSize':'0.82rem',
                     'marginBottom':'10px','display':'block' if error else 'none'}),

            html.Label('Username', style={**LABEL_STYLE, 'display':'block','marginTop':'14px','marginBottom':'6px'}),
            dcc.Input(id='auth-username', type='text', style=INPUT_STYLE),

            html.Div([
                html.Label('Email', style={**LABEL_STYLE, 'display':'block','marginTop':'14px','marginBottom':'6px'}),
                dcc.Input(id='auth-email', type='email', style=INPUT_STYLE),
            ], style={'display':'block' if mode=='signup' else 'none'}),

            html.Label('Password', style={**LABEL_STYLE, 'display':'block','marginTop':'14px','marginBottom':'6px'}),
            dcc.Input(id='auth-password', type='password', style=INPUT_STYLE),

            html.Button('Create Account' if mode=='signup' else 'Login', id='auth-submit-btn', n_clicks=0,
                        style={'width':'100%','marginTop':'22px','background':'linear-gradient(135deg,#0066ff,#00d4ff)',
                               'color':'#fff','border':'none','borderRadius':'8px','padding':'13px',
                               'fontFamily':'Rajdhani,sans-serif','fontSize':'1.05rem','fontWeight':'700',
                               'letterSpacing':'1px','cursor':'pointer'}),

            html.Div([
                "Don't have an account? " if mode=='login' else "Already have an account? ",
                dcc.Link('Sign up' if mode=='login' else 'Login',
                         href='/signup' if mode=='login' else '/login',
                         style={'color':DARK['accent'],'textDecoration':'none'})
            ], style={'textAlign':'center','marginTop':'18px','fontSize':'0.82rem','color':DARK['muted']}),

            html.Div('Demo credentials → admin / admin123',
                      style={'background':'rgba(0,212,255,0.08)','border':f'1px solid {DARK["border"]}',
                             'color':DARK['muted'],'padding':'10px 14px','borderRadius':'8px',
                             'fontSize':'0.75rem','marginTop':'18px','textAlign':'center'}),

            dcc.Store(id='auth-mode-store', data=mode),
        ])
    ])


# ════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD LAYOUT
# ════════════════════════════════════════════════════════════════════
def dashboard_layout(username='User'):
    df = get_df()
    total   = len(df)
    low_pct = round((df['congestion_level']==0).sum() / total * 100, 1)
    med_pct = round((df['congestion_level']==1).sum() / total * 100, 1)
    hi_pct  = round((df['congestion_level']==2).sum() / total * 100, 1)
    alerts  = check_alerts(df)
    live_active = (df['source'] == 'live_tomtom').any() if 'source' in df.columns else False

    return html.Div(style={'background': DARK['bg'], 'minHeight':'100vh', 'fontFamily':'DM Sans, sans-serif'}, children=[

        # Header
        html.Div(style={'background': 'linear-gradient(135deg,#0a0e1a,#0d1f35)',
                         'borderBottom': f'1px solid {DARK["border"]}', 'padding': '18px 32px',
                         'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}, children=[
            html.H1('🚦 Smart Traffic | Hyderabad', style={'fontFamily':'Rajdhani,sans-serif',
                     'color': DARK['accent'], 'fontSize':'1.5rem', 'fontWeight':'700', 'letterSpacing':'1px'}),
            html.Div('Stanley College of Engineering & Technology for Women — AI-Powered Urban Mobility System',
                     style={'color': DARK['muted'], 'fontSize':'0.8rem', 'textAlign':'center'}),
            html.Div([
                html.Span(id='live-status-dot', style={'color': '#00e676' if live_active else '#ffab40'}, children='● '),
                html.Span('Live API Connected' if live_active else 'Historical Data Mode',
                          id='live-status-text',
                          style={'color': '#00e676' if live_active else '#ffab40', 'fontSize':'0.8rem'}),
                html.Span(f'  |  👤 {username}', style={'color':DARK['text'], 'fontSize':'0.8rem', 'marginLeft':'14px'}),
                dcc.Link('Logout', href='/logout', style={'color':'#ff5252','fontSize':'0.75rem',
                          'marginLeft':'10px','textDecoration':'none'})
            ])
        ]),

        html.Div(style={'padding':'24px 32px', 'maxWidth':'1400px', 'margin':'0 auto'}, children=[

            # Alert banner
            html.Div(id='alert-banner-wrap', children=render_alert_banner(alerts)),

            # Stat Cards
            dbc.Row(style={'marginBottom':'20px'}, children=[
                dbc.Col(html.Div(style={**CARD_STYLE, 'borderTop':'3px solid #00d4ff'}, children=[
                    html.Div('Total Records', style=TITLE_STYLE),
                    html.Div(f'{total:,}', id='stat-total', style={**VAL_STYLE_BASE, 'color':'#00d4ff'}),
                    html.Div('traffic observations', style={'color':DARK['muted'],'fontSize':'0.78rem'})
                ]), width=3),
                dbc.Col(html.Div(style={**CARD_STYLE, 'borderTop':'3px solid #00e676'}, children=[
                    html.Div('Low Congestion', style=TITLE_STYLE),
                    html.Div(f'{low_pct}%', id='stat-low', style={**VAL_STYLE_BASE, 'color':'#00e676'}),
                    html.Div('vehicle count low', style={'color':DARK['muted'],'fontSize':'0.78rem'})
                ]), width=3),
                dbc.Col(html.Div(style={**CARD_STYLE, 'borderTop':'3px solid #ffab40'}, children=[
                    html.Div('Medium Congestion', style=TITLE_STYLE),
                    html.Div(f'{med_pct}%', id='stat-med', style={**VAL_STYLE_BASE, 'color':'#ffab40'}),
                    html.Div('moderate traffic', style={'color':DARK['muted'],'fontSize':'0.78rem'})
                ]), width=3),
                dbc.Col(html.Div(style={**CARD_STYLE, 'borderTop':'3px solid #ff5252'}, children=[
                    html.Div('High Congestion', style=TITLE_STYLE),
                    html.Div(f'{hi_pct}%', id='stat-high', style={**VAL_STYLE_BASE, 'color':'#ff5252'}),
                    html.Div('heavy traffic', style={'color':DARK['muted'],'fontSize':'0.78rem'})
                ]), width=3),
            ]),

            # Row 1: Hourly + Pie
            dbc.Row(style={'marginBottom':'20px'}, children=[
                dbc.Col(html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id='graph-hourly', figure=fig_hourly(df), config={'displayModeBar':False}, style={'height':'280px'})
                ]), width=7),
                dbc.Col(html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id='graph-pie', figure=fig_pie(df), config={'displayModeBar':False}, style={'height':'280px'})
                ]), width=5),
            ]),

            # Row 2: Map
            dbc.Row(style={'marginBottom':'20px'}, children=[
                dbc.Col(html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id='graph-map', figure=fig_map(df), config={'displayModeBar':False}, style={'height':'340px'})
                ]), width=12),
            ]),

            # Row 3: Corridors + Feature Imp
            dbc.Row(style={'marginBottom':'20px'}, children=[
                dbc.Col(html.Div(style=CARD_STYLE, children=[
                    html.Div(style={'display':'flex','justifyContent':'space-between','alignItems':'center'}, children=[
                        html.Div(),
                        html.Button('🔄 Fetch Live Data', id='live-fetch-btn', n_clicks=0, style={
                            'background':'#0d1626','border':f'1px solid {DARK["border"]}','color':DARK['accent'],
                            'borderRadius':'6px','padding':'6px 14px','fontSize':'0.75rem','cursor':'pointer',
                            'fontFamily':'DM Sans,sans-serif'})
                    ]),
                    dcc.Graph(id='graph-corridors', figure=fig_corridors(df), config={'displayModeBar':False}, style={'height':'300px'}),
                    html.Div(id='live-fetch-status', style={'fontSize':'0.78rem','color':DARK['muted'],'marginTop':'8px'})
                ]), width=6),
                dbc.Col(html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(figure=fig_feature_imp(), config={'displayModeBar':False}, style={'height':'300px'})
                ]), width=6),
            ]),

            # Row 4: Scatter + Live Predictor
            dbc.Row(children=[
                dbc.Col(html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id='graph-scatter', figure=fig_speed_scatter(df), config={'displayModeBar':False}, style={'height':'320px'})
                ]), width=6),

                dbc.Col(html.Div(style=CARD_STYLE, children=[
                    html.Div('⚡ Live Congestion Predictor', style={**TITLE_STYLE, 'marginBottom':'16px', 'fontSize':'1rem'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label('Hour (0–23)', style=LABEL_STYLE),
                            dcc.Input(id='p-hour', type='number', value=8, min=0, max=23, style=INPUT_STYLE)
                        ], width=6),
                        dbc.Col([
                            html.Label('Avg Speed (km/h)', style=LABEL_STYLE),
                            dcc.Input(id='p-speed', type='number', value=22, min=5, max=120, style=INPUT_STYLE)
                        ], width=6),
                    ], style={'marginBottom':'12px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label('Rolling 30m', style=LABEL_STYLE),
                            dcc.Input(id='p-r30', type='number', value=380, style=INPUT_STYLE)
                        ], width=4),
                        dbc.Col([
                            html.Label('Rolling 1h', style=LABEL_STYLE),
                            dcc.Input(id='p-r1h', type='number', value=420, style=INPUT_STYLE)
                        ], width=4),
                        dbc.Col([
                            html.Label('Rolling 3h', style=LABEL_STYLE),
                            dcc.Input(id='p-r3h', type='number', value=390, style=INPUT_STYLE)
                        ], width=4),
                    ], style={'marginBottom':'16px'}),
                    html.Button('⚡ PREDICT', id='predict-btn', n_clicks=0, style={
                        'width':'100%', 'padding':'12px', 'background':'linear-gradient(135deg,#0066ff,#00d4ff)',
                        'color':'#fff', 'border':'none', 'borderRadius':'8px',
                        'fontFamily':'Rajdhani,sans-serif', 'fontSize':'1rem',
                        'fontWeight':'700', 'letterSpacing':'1px', 'cursor':'pointer'}),
                    html.Div(id='predict-output', style={'marginTop':'14px'})
                ]), width=6),
            ]),

        ]),

        html.Div('AI-Powered Smart Urban Mobility & Traffic Management System  |  '
                 'Stanley College of Engineering and Technology for Women, Hyderabad  |  2025–2026',
                 style={'textAlign':'center','padding':'16px','color':DARK['muted'],
                        'fontSize':'0.72rem','borderTop':f'1px solid {DARK["border"]}','marginTop':'8px'}),

        dcc.Interval(id='alert-poll', interval=15000, n_intervals=0),  # re-check alerts every 15s
    ])


def render_alert_banner(alerts):
    if not alerts:
        return html.Div(style={'display':'none'})
    return html.Div(style={'background': 'rgba(255,82,82,0.08)', 'border':'1px solid #ff5252',
                            'borderRadius':'10px', 'padding':'14px 20px', 'marginBottom':'20px'}, children=[
        html.Div('🔔 Active Alerts — High Risk Corridors',
                 style={'fontFamily':'Rajdhani,sans-serif','color':'#ff5252','fontWeight':'700','fontSize':'0.95rem','marginBottom':'6px'}),
        html.Div([html.Div(f"⚠️ {a['message']}", style={'fontSize':'0.82rem','color':DARK['text'],'padding':'3px 0'})
                  for a in alerts])
    ])


# ════════════════════════════════════════════════════════════════════
#  TOP-LEVEL APP LAYOUT (URL ROUTING)
# ════════════════════════════════════════════════════════════════════
app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    dcc.Store(id='session-store', storage_type='session'),
    html.Div(id='page-content')
])


@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    State('session-store', 'data')
)
def display_page(pathname, session_data):
    session_data = session_data or {}

    if pathname == '/logout':
        return login_layout(mode='login')

    if pathname == '/signup':
        return login_layout(mode='signup')

    if session_data.get('logged_in'):
        return dashboard_layout(username=session_data.get('username', 'User'))

    return login_layout(mode='login')


@callback(
    Output('session-store', 'data'),
    Output('url', 'pathname'),
    Output('auth-username', 'value'),
    Input('auth-submit-btn', 'n_clicks'),
    State('auth-username', 'value'),
    State('auth-password', 'value'),
    State('auth-email', 'value'),
    State('url', 'pathname'),
    prevent_initial_call=True
)
def handle_auth_submit(n, username, password, email, current_path):
    if not n or not username or not password:
        return dash.no_update, dash.no_update, dash.no_update

    if current_path == '/signup':
        success, msg = db.create_user(username, email or f'{username}@example.com', password)
        if success:
            return dash.no_update, '/login', None
        return dash.no_update, dash.no_update, dash.no_update

    # login mode
    user = db.verify_user(username, password)
    if user:
        return {'logged_in': True, 'username': user['username'], 'role': user['role']}, '/', None

    return dash.no_update, dash.no_update, dash.no_update


@callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('url', 'pathname'),
    State('session-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def logout_clear(pathname, session_data):
    if pathname == '/logout':
        return '/login'
    return dash.no_update


@callback(
    Output('session-store', 'data', allow_duplicate=True),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def clear_session_on_logout(pathname):
    if pathname == '/logout':
        return {}
    return dash.no_update


# ── Callback: live data fetch button ────────────────────────────────
@callback(
    Output('live-fetch-status', 'children'),
    Output('graph-corridors', 'figure'),
    Output('stat-total', 'children'),
    Output('stat-low', 'children'),
    Output('stat-med', 'children'),
    Output('stat-high', 'children'),
    Output('alert-banner-wrap', 'children'),
    Output('live-status-text', 'children'),
    Output('live-status-dot', 'style'),
    Output('live-status-text', 'style'),
    Input('live-fetch-btn', 'n_clicks'),
    prevent_initial_call=True
)
def fetch_live(n):
    if not n:
        return dash.no_update

    try:
        df_live, status = live_data.fetch_all_roads_live()
        for _, row in df_live.iterrows():
            db.insert_live_record(row.to_dict())

        df = get_df()
        total   = len(df)
        low_pct = round((df['congestion_level']==0).sum() / total * 100, 1)
        med_pct = round((df['congestion_level']==1).sum() / total * 100, 1)
        hi_pct  = round((df['congestion_level']==2).sum() / total * 100, 1)
        alerts  = check_alerts(df)
        live_active = (df['source'] == 'live_tomtom').any() if 'source' in df.columns else False

        live_color = '#00e676' if live_active else '#ffab40'
        live_text  = 'Live API Connected' if live_active else 'Historical Data Mode'

        return (f"{status} — {len(df_live)} roads updated",
                fig_corridors(df), f'{total:,}', f'{low_pct}%', f'{med_pct}%', f'{hi_pct}%',
                render_alert_banner(alerts), live_text,
                {'color': live_color}, {'color': live_color, 'fontSize':'0.8rem'})
    except Exception as e:
        return (f"Error: {e}", dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update)


# ── Callback: predictor (uses State so it only fires on button click) ──
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

    is_peak     = 1 if hour in [8,9,17,18,19] else 0
    day_of_week = 1
    is_weekend  = 0

    row  = [[hour, day_of_week, is_weekend, is_peak, speed, r30, r1h, r3h]]
    pred = clf.predict(row)[0]
    prob = clf.predict_proba(row)[0]

    label = LABEL_MAP[pred]
    conf  = round(prob.max() * 100, 1)
    color = COLOR_MAP[label]
    icons = {'Low':'🟢','Medium':'🟡','High':'🔴'}

    return html.Div(style={
        'background': '#1a2540', 'border': f'2px solid {color}',
        'borderRadius': '10px', 'padding': '16px'
    }, children=[
        html.Div(f'{icons[label]} {label} Congestion', style={
            'fontFamily':'Rajdhani,sans-serif', 'fontSize':'1.7rem', 'fontWeight':'700', 'color': color
        }),
        html.Div(f'Model confidence: {conf}%  ·  Peak hour: {"Yes" if is_peak else "No"}',
                 style={'color': DARK['muted'], 'fontSize':'0.83rem', 'marginTop':'4px'})
    ])


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  📊 Plotly Dash Dashboard")
    print("  Stanley College of Engineering — Hyderabad")
    print("="*55)
    df_check = get_df()
    print(f"  ✅ Database: {len(df_check)} records, {df_check['road_id'].nunique()} roads")
    print(f"  ✅ TomTom Live API: {'Configured' if live_data.is_api_available() else 'Not configured (fallback mode)'}")
    print("  🔐 Login → username: admin | password: admin123")
    print("  🌐 Dashboard → http://127.0.0.1:8050")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=True)
