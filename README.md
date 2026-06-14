# 🚦 AI-Powered Smart Urban Mobility & Traffic Management System
**Stanley College of Engineering and Technology for Women**
Academic Year 2025–2026

---

## 📁 Project File Structure

```
your_project_folder/
│
├── model.pkl            ← Trained Random Forest model (from Colab)
├── traffic_data.csv     ← Processed dataset (from Colab)
│
├── app.py               ← Flask API + Browser Dashboard  (Option A)
├── dashboard.py         ← Plotly Dash Dashboard          (Option C)
│
└── requirements.txt     ← Python dependencies
```

---

## ⚙️ ONE-TIME SETUP (Do this once on your PC)

### Step 1 — Install Python 3.9+
Download from https://www.python.org/downloads/

### Step 2 — Install dependencies
Open Command Prompt / Terminal in your project folder:
```
pip install -r requirements.txt
```

---

## 🎯 DEMO — HOW TO SHOW IT

### Option A — Flask Dashboard (Simple & Clean)
```
python app.py
```
Then open browser → **http://127.0.0.1:5000**

Shows:
- Live stat cards (total records, congestion %)
- Hourly traffic bar chart
- Congestion distribution donut chart
- High-risk corridors with risk scores
- **Live predictor form** — type values → get prediction!

---

### Option C — Plotly Dash (Full Interactive)
```
python dashboard.py
```
Then open browser → **http://127.0.0.1:8050**

Shows everything in Option A PLUS:
- **Interactive Hyderabad map** with traffic dots
- Speed vs Vehicle Count scatter plot
- Feature importance chart
- All charts are interactive (hover, zoom, click)

---

## 🎤 DEMO SCRIPT (What to say to judges)

1. **Start with dashboard.py** — most impressive visually
   > "This is our real-time dashboard. It shows live traffic analytics
   >  for 10 major Hyderabad corridors."

2. **Point to the map**
   > "These dots represent GPS-tracked vehicles across Hyderabad.
   >  Red = high congestion, yellow = medium, green = low."

3. **Point to Corridors chart**
   > "Our model identified Kukatpally Road and Secunderabad Road
   >  as the highest-risk corridors during peak hours."

4. **Use the Live Predictor**
   > "Let me show a live prediction. I'll set hour = 8 (morning peak),
   >  speed = 18 km/h (slow traffic), rolling average = 450..."
   > Click PREDICT → shows 🔴 High Congestion
   > "Now I'll change hour to 3 AM, speed = 70..."
   > Click PREDICT → shows 🟢 Low Congestion

5. **Point to Feature Importance**
   > "Our model found that recent rolling averages — the last 30 minutes
   >  and 1 hour — are the strongest predictors of congestion.
   >  This matches real-world traffic behaviour."

---

## 📡 API Endpoints (for technical questions)

| Method | Endpoint     | Description                        |
|--------|--------------|------------------------------------|
| POST   | /predict     | Predict congestion for given input |
| GET    | /analytics   | Hourly stats + corridor data       |
| GET    | /heatmap     | GPS points with congestion labels  |
| GET    | /risk        | Top 10 high-risk corridors         |

### Sample /predict request:
```json
POST http://127.0.0.1:5000/predict
{
  "hour": 8,
  "day_of_week": 1,
  "is_weekend": 0,
  "is_peak": 1,
  "avg_speed_kmph": 22.5,
  "rolling_30m": 380,
  "rolling_1h": 420,
  "rolling_3h": 390
}
```
Response:
```json
{
  "congestion_level": "High",
  "confidence": 0.91,
  "road_suggested": "Kukatpally_Rd"
}
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `model.pkl not found` | Make sure model.pkl is in the same folder as app.py |
| Port already in use | Change `port=5000` to `port=5001` in app.py |
| Map not loading | Needs internet connection (uses Mapbox CDN) |
