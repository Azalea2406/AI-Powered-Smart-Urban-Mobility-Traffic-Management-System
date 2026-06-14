"""
app.py — AI-Powered Smart Urban Mobility & Traffic Management System
Stanley College of Engineering and Technology for Women
Run: python app.py  →  Open http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template_string
import joblib, numpy as np, pandas as pd
import os, json

app = Flask(__name__)

# ── Load model & data ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
df    = pd.read_csv(os.path.join(BASE_DIR, 'traffic_data.csv'), parse_dates=['timestamp'])

FEATURES = ['hour','day_of_week','is_weekend','is_peak',
            'avg_speed_kmph','rolling_30m','rolling_1h','rolling_3h']
LABELS   = {0: 'Low', 1: 'Medium', 2: 'High'}
ROADS    = sorted(df['road_id'].unique().tolist())

# ── Helper: corridor risk ──────────────────────────────────────────
def get_corridors():
    peak_df = df[df['is_peak'] == 1]
    if peak_df.empty:
        peak_df = df
    stats = peak_df.groupby('road_id').agg(
        avg_count      = ('vehicle_count','mean'),
        incident_count = ('congestion_level', lambda x: (x==2).sum())
    ).reset_index()
    max_count    = stats['avg_count'].max()
    max_incident = stats['incident_count'].max()
    stats['risk_score'] = (
        (stats['avg_count']      / max_count    if max_count    > 0 else 0) * 50 +
        (stats['incident_count'] / max_incident if max_incident > 0 else 0) * 50
    ).round(1)
    return stats.sort_values('risk_score', ascending=False).head(10).to_dict(orient='records')

# ════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD PAGE
# ════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Traffic Management — Hyderabad</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg:       #0a0e1a;
    --panel:    #111827;
    --border:   #1e2d45;
    --accent:   #00d4ff;
    --green:    #00e676;
    --orange:   #ffab40;
    --red:      #ff5252;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --glow:     0 0 20px rgba(0,212,255,0.15);
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1f35 100%);
    border-bottom: 1px solid var(--border);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }
  header h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem; font-weight: 700;
    color: var(--accent);
    letter-spacing: 1px;
  }
  header h1 span { color: var(--text); font-weight: 400; }
  .live-badge {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.78rem; color: var(--green);
    background: rgba(0,230,118,0.08);
    border: 1px solid rgba(0,230,118,0.25);
    padding: 5px 12px; border-radius: 20px;
  }
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(1.3); }
  }

  /* ── Layout ── */
  main { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }

  /* ── Stat Cards ── */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px; margin-bottom: 24px;
  }
  .stat-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative; overflow: hidden;
    animation: fadeUp 0.5s ease both;
  }
  .stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px;
  }
  .stat-card.blue::before  { background: var(--accent); }
  .stat-card.green::before { background: var(--green); }
  .stat-card.orange::before{ background: var(--orange); }
  .stat-card.red::before   { background: var(--red); }
  .stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .stat-value { font-family: 'Rajdhani', sans-serif; font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
  .stat-card.blue   .stat-value { color: var(--accent); }
  .stat-card.green  .stat-value { color: var(--green); }
  .stat-card.orange .stat-value { color: var(--orange); }
  .stat-card.red    .stat-value { color: var(--red); }
  .stat-sub { font-size: 0.78rem; color: var(--muted); }

  /* ── Grid ── */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  .grid-3 { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 24px; }

  /* ── Panel ── */
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    animation: fadeUp 0.6s ease both;
  }
  .panel-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem; font-weight: 600;
    color: var(--accent);
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }
  .panel-title::before {
    content: '';
    width: 3px; height: 16px;
    background: var(--accent);
    border-radius: 2px;
  }

  /* ── Chart containers ── */
  .chart-wrap { position: relative; height: 240px; }

  /* ── Predict Form ── */
  .form-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
  }
  .form-group label {
    display: block;
    font-size: 0.72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 6px;
  }
  .form-group input, .form-group select {
    width: 100%;
    background: #0a0e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    padding: 10px 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    transition: border-color 0.2s;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,212,255,0.1);
  }
  .form-group select option { background: #111827; }

  .predict-btn {
    width: 100%; margin-top: 16px;
    background: linear-gradient(135deg, #0066ff, #00d4ff);
    color: #fff;
    border: none; border-radius: 8px;
    padding: 13px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .predict-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,212,255,0.3);
  }
  .predict-btn:active { transform: translateY(0); }

  /* ── Result Box ── */
  #result-box {
    margin-top: 16px;
    border-radius: 10px;
    padding: 18px 20px;
    display: none;
    border: 1px solid;
    animation: fadeUp 0.3s ease;
  }
  #result-box.low    { background: rgba(0,230,118,0.08); border-color: var(--green); }
  #result-box.medium { background: rgba(255,171,64,0.08); border-color: var(--orange); }
  #result-box.high   { background: rgba(255,82,82,0.08);  border-color: var(--red); }
  .result-level {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8rem; font-weight: 700;
  }
  .result-low    { color: var(--green); }
  .result-medium { color: var(--orange); }
  .result-high   { color: var(--red); }
  .result-conf { font-size: 0.85rem; color: var(--muted); margin-top: 4px; }

  /* ── Corridor Table ── */
  .corridor-row {
    display: grid;
    grid-template-columns: 1fr 80px 90px;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    align-items: center;
    font-size: 0.88rem;
  }
  .corridor-row:last-child { border-bottom: none; }
  .corridor-name { color: var(--text); }
  .risk-bar-wrap { background: #1a2540; border-radius: 4px; height: 6px; overflow: hidden; }
  .risk-bar { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
  .badge {
    font-size: 0.7rem; font-weight: 600;
    padding: 3px 8px; border-radius: 4px;
    text-align: center; text-transform: uppercase;
  }
  .badge-high   { background: rgba(255,82,82,0.15);  color: var(--red);    border: 1px solid rgba(255,82,82,0.3); }
  .badge-medium { background: rgba(255,171,64,0.15); color: var(--orange); border: 1px solid rgba(255,171,64,0.3); }
  .badge-low    { background: rgba(0,230,118,0.15);  color: var(--green);  border: 1px solid rgba(0,230,118,0.3); }

  @keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
  }

  /* ── Footer ── */
  footer {
    text-align: center; padding: 20px;
    color: var(--muted); font-size: 0.75rem;
    border-top: 1px solid var(--border);
    margin-top: 8px;
  }
</style>
</head>
<body>

<header>
  <h1>🚦 Smart Traffic <span>| Hyderabad</span></h1>
  <div style="font-family:Rajdhani;font-size:0.8rem;color:var(--muted);text-align:center;">
    Stanley College of Engineering & Technology for Women<br>
    <span style="color:var(--accent);">AI-Powered Urban Mobility System</span>
  </div>
  <div class="live-badge"><div class="live-dot"></div> System Online</div>
</header>

<main>

  <!-- STAT CARDS -->
  <div class="stats-row" id="stat-cards">
    <div class="stat-card blue">
      <div class="stat-label">Total Records</div>
      <div class="stat-value" id="s-total">—</div>
      <div class="stat-sub">traffic observations</div>
    </div>
    <div class="stat-card green">
      <div class="stat-label">Low Congestion</div>
      <div class="stat-value" id="s-low">—</div>
      <div class="stat-sub">vehicle count &lt; 200</div>
    </div>
    <div class="stat-card orange">
      <div class="stat-label">Medium Congestion</div>
      <div class="stat-value" id="s-med">—</div>
      <div class="stat-sub">200 – 400 vehicles</div>
    </div>
    <div class="stat-card red">
      <div class="stat-label">High Congestion</div>
      <div class="stat-value" id="s-high">—</div>
      <div class="stat-sub">vehicle count &gt; 400</div>
    </div>
  </div>

  <!-- CHARTS ROW -->
  <div class="grid-2">
    <div class="panel">
      <div class="panel-title">Hourly Traffic Density</div>
      <div class="chart-wrap"><canvas id="hourlyChart"></canvas></div>
    </div>
    <div class="panel">
      <div class="panel-title">Congestion Distribution</div>
      <div class="chart-wrap"><canvas id="pieChart"></canvas></div>
    </div>
  </div>

  <!-- CORRIDORS + PREDICT -->
  <div class="grid-3">

    <!-- Corridor Risk Table -->
    <div class="panel">
      <div class="panel-title">High-Risk Corridors (Peak Hours)</div>
      <div id="corridor-list"></div>
    </div>

    <!-- Live Predictor -->
    <div class="panel">
      <div class="panel-title">Live Predictor</div>
      <div class="form-grid">
        <div class="form-group">
          <label>Hour (0–23)</label>
          <input type="number" id="f-hour" value="8" min="0" max="23">
        </div>
        <div class="form-group">
          <label>Day of Week</label>
          <select id="f-day">
            <option value="0">Monday</option>
            <option value="1">Tuesday</option>
            <option value="2">Wednesday</option>
            <option value="3">Thursday</option>
            <option value="4">Friday</option>
            <option value="5">Saturday</option>
            <option value="6">Sunday</option>
          </select>
        </div>
        <div class="form-group">
          <label>Avg Speed (km/h)</label>
          <input type="number" id="f-speed" value="22" min="5" max="120">
        </div>
        <div class="form-group">
          <label>Rolling 30m avg</label>
          <input type="number" id="f-r30" value="380" min="0">
        </div>
        <div class="form-group">
          <label>Rolling 1h avg</label>
          <input type="number" id="f-r1h" value="420" min="0">
        </div>
        <div class="form-group">
          <label>Rolling 3h avg</label>
          <input type="number" id="f-r3h" value="390" min="0">
        </div>
      </div>
      <button class="predict-btn" onclick="predict()">⚡ PREDICT CONGESTION</button>
      <div id="result-box">
        <div class="result-level" id="result-level"></div>
        <div class="result-conf" id="result-conf"></div>
      </div>
    </div>

  </div>

</main>

<footer>
  AI-Powered Smart Urban Mobility & Traffic Management System &nbsp;|&nbsp;
  Stanley College of Engineering and Technology for Women, Hyderabad &nbsp;|&nbsp;
  Academic Year 2025–2026
</footer>

<script>
// ── Fetch analytics & populate ────────────────────────────────────
async function loadDashboard() {
  const res  = await fetch('/analytics');
  const data = await res.json();

  // Stat cards
  document.getElementById('s-total').textContent = data.total_records;
  document.getElementById('s-low').textContent   = data.low_pct  + '%';
  document.getElementById('s-med').textContent   = data.med_pct  + '%';
  document.getElementById('s-high').textContent  = data.high_pct + '%';

  // Hourly bar chart
  const hours  = data.hourly.map(r => r.hour);
  const counts = data.hourly.map(r => r.avg_count.toFixed(1));
  const barColors = hours.map(h =>
    [8,9,17,18,19].includes(h) ? '#ff5252' : '#00d4ff'
  );

  new Chart(document.getElementById('hourlyChart'), {
    type: 'bar',
    data: {
      labels: hours,
      datasets: [{ data: counts, backgroundColor: barColors, borderRadius: 4, borderSkipped: false }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color:'#64748b', font:{size:10} }, grid: { color:'#1e2d45' } },
        y: { ticks: { color:'#64748b', font:{size:10} }, grid: { color:'#1e2d45' } }
      }
    }
  });

  // Pie chart
  new Chart(document.getElementById('pieChart'), {
    type: 'doughnut',
    data: {
      labels: ['Low','Medium','High'],
      datasets: [{ data: [data.low_pct, data.med_pct, data.high_pct],
        backgroundColor: ['#00e676','#ffab40','#ff5252'],
        borderWidth: 0, hoverOffset: 8 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position:'bottom', labels:{ color:'#e2e8f0', padding:16, font:{size:12} } }
      },
      cutout: '65%'
    }
  });

  // Corridors
  const list = document.getElementById('corridor-list');
  data.corridors.forEach(c => {
    const score  = c.risk_score;
    const cls    = score >= 70 ? 'high' : score >= 40 ? 'medium' : 'low';
    const color  = score >= 70 ? '#ff5252' : score >= 40 ? '#ffab40' : '#00e676';
    list.innerHTML += `
      <div class="corridor-row">
        <div class="corridor-name">${c.road_id.replace(/_/g,' ')}</div>
        <div class="risk-bar-wrap">
          <div class="risk-bar" style="width:${score}%;background:${color}"></div>
        </div>
        <div class="badge badge-${cls}">${cls} ${score}</div>
      </div>`;
  });
}

// ── Predict ───────────────────────────────────────────────────────
async function predict() {
  const hour       = parseInt(document.getElementById('f-hour').value);
  const day_of_week= parseInt(document.getElementById('f-day').value);
  const is_weekend = day_of_week >= 5 ? 1 : 0;
  const is_peak    = [8,9,17,18,19].includes(hour) ? 1 : 0;
  const speed      = parseFloat(document.getElementById('f-speed').value);
  const r30        = parseFloat(document.getElementById('f-r30').value);
  const r1h        = parseFloat(document.getElementById('f-r1h').value);
  const r3h        = parseFloat(document.getElementById('f-r3h').value);

  const res  = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hour, day_of_week, is_weekend, is_peak,
                           avg_speed_kmph: speed,
                           rolling_30m: r30, rolling_1h: r1h, rolling_3h: r3h })
  });
  const data = await res.json();

  const box   = document.getElementById('result-box');
  const level = data.congestion_level.toLowerCase();
  const icons = { low:'🟢', medium:'🟡', high:'🔴' };

  box.className       = level;
  box.style.display   = 'block';
  document.getElementById('result-level').className   = `result-level result-${level}`;
  document.getElementById('result-level').textContent = `${icons[level]} ${data.congestion_level} Congestion`;
  document.getElementById('result-conf').textContent  = `Confidence: ${(data.confidence * 100).toFixed(1)}%  ·  Road: ${data.road_suggested}`;
}

loadDashboard();
</script>
</body>
</html>
"""

# ════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    row  = [[data[f] for f in FEATURES]]
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0]

    # Suggest most congested road at that hour
    hour = data.get('hour', 0)
    road_df = df[df['hour'] == hour] if not df[df['hour'] == hour].empty else df
    suggested = road_df.loc[road_df['vehicle_count'].idxmax(), 'road_id']

    return jsonify({
        'congestion_level': LABELS[int(pred)],
        'confidence':       round(float(prob.max()), 3),
        'road_suggested':   suggested
    })


@app.route('/analytics', methods=['GET'])
def analytics():
    # Hourly averages
    hourly = df.groupby('hour')['vehicle_count'].mean().reset_index()
    hourly.columns = ['hour', 'avg_count']

    # Congestion distribution
    total  = len(df)
    counts = df['congestion_level'].value_counts()
    low_pct  = round(counts.get(0, 0) / total * 100, 1)
    med_pct  = round(counts.get(1, 0) / total * 100, 1)
    high_pct = round(counts.get(2, 0) / total * 100, 1)

    return jsonify({
        'total_records': total,
        'low_pct':  low_pct,
        'med_pct':  med_pct,
        'high_pct': high_pct,
        'hourly':   hourly.to_dict(orient='records'),
        'corridors': get_corridors()
    })


@app.route('/heatmap', methods=['GET'])
def heatmap():
    sample = df[['road_id','latitude','longitude','congestion_level']].dropna()
    return jsonify(sample.to_dict(orient='records'))


@app.route('/risk', methods=['GET'])
def risk():
    return jsonify(get_corridors())


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🚦 Smart Traffic Management System")
    print("  Stanley College of Engineering — Hyderabad")
    print("="*55)
    print("  ✅ Model loaded")
    print(f"  ✅ Dataset loaded — {len(df)} records")
    print("  🌐 Dashboard → http://127.0.0.1:5000")
    print("  📡 API Endpoints:")
    print("     POST /predict   — congestion prediction")
    print("     GET  /analytics — hourly stats + corridors")
    print("     GET  /heatmap   — GPS + congestion data")
    print("     GET  /risk      — top 10 risk corridors")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
