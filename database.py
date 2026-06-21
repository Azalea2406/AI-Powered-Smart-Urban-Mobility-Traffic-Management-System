"""
database.py — SQLite Database Setup & Helper Functions
AI-Powered Smart Urban Mobility & Traffic Management System

Creates and manages:
  - users table        (authentication)
  - traffic_records     (historical + live traffic data)
  - alerts table        (congestion threshold notifications)

Run once to initialize: python database.py
"""

import sqlite3
import pandas as pd
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traffic_system.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    """Simple SHA-256 hashing with a static salt (for academic project use)."""
    salt = "smart_traffic_hyd_2026"
    return hashlib.sha256((password + salt).encode()).hexdigest()


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # ── Users table ───────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'viewer',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Traffic records table (replaces flat CSV) ────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS traffic_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            road_id TEXT,
            vehicle_count REAL,
            avg_speed_kmph REAL,
            latitude REAL,
            longitude REAL,
            hour INTEGER,
            day_of_week INTEGER,
            is_weekend INTEGER,
            is_peak INTEGER,
            rolling_30m REAL,
            rolling_1h REAL,
            rolling_3h REAL,
            congestion_level INTEGER,
            source TEXT DEFAULT 'historical'
        )
    """)

    # ── Alerts table ──────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            road_id TEXT,
            congestion_level INTEGER,
            risk_score REAL,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_read INTEGER DEFAULT 0
        )
    """)

    conn.commit()

    # ── Seed a default admin user (only if users table is empty) ──
    cur.execute("SELECT COUNT(*) as cnt FROM users")
    if cur.fetchone()['cnt'] == 0:
        cur.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?,?,?,?)",
            ('admin', 'admin@smarttraffic.com', hash_password('admin123'), 'admin')
        )
        conn.commit()
        print("✅ Default admin user created → username: admin | password: admin123")

    # ── Load traffic_data.csv into traffic_records (only if empty) ──
    cur.execute("SELECT COUNT(*) as cnt FROM traffic_records")
    if cur.fetchone()['cnt'] == 0:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traffic_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['source'] = 'historical'
            df.to_sql('traffic_records', conn, if_exists='append', index=False)
            print(f"✅ Loaded {len(df)} historical records into database")

    conn.close()


def create_user(username, email, password, role='viewer'):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?,?,?,?)",
            (username, email, hash_password(password), role)
        )
        conn.commit()
        return True, "Account created successfully"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    finally:
        conn.close()


def verify_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    conn.close()
    if user and user['password_hash'] == hash_password(password):
        return dict(user)
    return None


def get_all_traffic_records():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM traffic_records", conn)
    conn.close()
    return df


def insert_live_record(record: dict):
    """Insert a single live traffic reading fetched from TomTom API."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO traffic_records
        (timestamp, road_id, vehicle_count, avg_speed_kmph, latitude, longitude,
         hour, day_of_week, is_weekend, is_peak, rolling_30m, rolling_1h, rolling_3h,
         congestion_level, source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record['timestamp'], record['road_id'], record['vehicle_count'],
        record['avg_speed_kmph'], record['latitude'], record['longitude'],
        record['hour'], record['day_of_week'], record['is_weekend'], record['is_peak'],
        record['rolling_30m'], record['rolling_1h'], record['rolling_3h'],
        record['congestion_level'], record.get('source', 'live')
    ))
    conn.commit()
    conn.close()


def create_alert(road_id, congestion_level, risk_score, message):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alerts (road_id, congestion_level, risk_score, message) VALUES (?,?,?,?)",
        (road_id, congestion_level, risk_score, message)
    )
    conn.commit()
    conn.close()


def get_recent_alerts(limit=10):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


if __name__ == '__main__':
    init_db()
    print(f"\n✅ Database initialized at: {DB_PATH}")
    df = get_all_traffic_records()
    print(f"✅ Total traffic records in DB: {len(df)}")
