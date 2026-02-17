"""
================================================================================
APP.PY - F1 Pit Wall Box — Professional Engineering Software
================================================================================
Webapp Streamlit che simula un muretto box durante il GP di Monza 2025.
Layout a 4 quadranti con onboard camera, telemetria live, previsione
sorpassi e classifica in tempo reale.

Funzionalita:
- Onboard camera Leclerc #16 (YouTube embed)
- Telemetria live simulata (velocita, throttle, freno, marcia, RPM, G-force)
- Calcolo probabilita di sorpasso basato su modello ML
- Classifica live con gap

Modello utilizzato: XGBoost con eval_metric='logloss'
Dati training: Monza GP 2022, 2023, 2024
================================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import os

# --- DRIVERS DATA 2024 ---
DRIVERS_2024 = [
    {"pos": 1, "code": "VER", "name": "Verstappen", "team": "Red Bull", "color": "#3671C6"},
    {"pos": 2, "code": "NOR", "name": "Norris", "team": "McLaren", "color": "#FF8000"},
    {"pos": 3, "code": "LEC", "name": "Leclerc", "team": "Ferrari", "color": "#E80020"},
    {"pos": 4, "code": "PIA", "name": "Piastri", "team": "McLaren", "color": "#FF8000"},
    {"pos": 5, "code": "SAI", "name": "Sainz", "team": "Ferrari", "color": "#E80020"},
    {"pos": 6, "code": "HAM", "name": "Hamilton", "team": "Mercedes", "color": "#27F4D2"},
    {"pos": 7, "code": "RUS", "name": "Russell", "team": "Mercedes", "color": "#27F4D2"},
    {"pos": 8, "code": "PER", "name": "Perez", "team": "Red Bull", "color": "#3671C6"},
    {"pos": 9, "code": "ALO", "name": "Alonso", "team": "Aston Martin", "color": "#229971"},
    {"pos": 10, "code": "STR", "name": "Stroll", "team": "Aston Martin", "color": "#229971"},
    {"pos": 11, "code": "HUL", "name": "Hulkenberg", "team": "Haas", "color": "#B6BABD"},
    {"pos": 12, "code": "TSU", "name": "Tsunoda", "team": "RB", "color": "#6692FF"},
    {"pos": 13, "code": "RIC", "name": "Ricciardo", "team": "RB", "color": "#6692FF"},
    {"pos": 14, "code": "ALB", "name": "Albon", "team": "Williams", "color": "#64C4FF"},
    {"pos": 15, "code": "GAS", "name": "Gasly", "team": "Alpine", "color": "#FF87BC"},
    {"pos": 16, "code": "OCO", "name": "Ocon", "team": "Alpine", "color": "#FF87BC"},
    {"pos": 17, "code": "MAG", "name": "Magnussen", "team": "Haas", "color": "#B6BABD"},
    {"pos": 18, "code": "BOT", "name": "Bottas", "team": "Sauber", "color": "#52E252"},
    {"pos": 19, "code": "ZHO", "name": "Zhou", "team": "Sauber", "color": "#52E252"},
    {"pos": 20, "code": "SAR", "name": "Sargeant", "team": "Williams", "color": "#64C4FF"},
]

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="F1 Pit Wall Box — Monza 2025",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD CSS ---
def load_css():
    """Carica gli stili CSS da file esterno."""
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    with open(css_path, 'r', encoding='utf-8') as f:
        return f.read()

st.markdown(f'<style>{load_css()}</style>', unsafe_allow_html=True)


# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    base_path = '../../models' if os.path.exists('../../models') else 'models'
    try:
        with open(f'{base_path}/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{base_path}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return {'model': model, 'scaler': scaler}
    except FileNotFoundError:
        return None


def get_compound_value(compound):
    """
    Converte il tipo di mescola in un valore numerico per il calcolo del vantaggio.
    SOFT = 3 (massima aderenza), MEDIUM = 2, HARD = 1 (minima aderenza).
    """
    return {"SOFT": 3, "MEDIUM": 2, "HARD": 1}.get(compound, 0)


def predict_overtake(resources, attacker, defender):
    """
    Calcola la probabilita di sorpasso usando il modello ML addestrato.
    """
    model, scaler = resources['model'], resources['scaler']

    features = np.array([[
        attacker['position'],
        attacker['lap_time'] - defender['lap_time'],
        attacker['tyre_life'] - defender['tyre_life'],
        get_compound_value(attacker['compound']) - get_compound_value(defender['compound']),
        attacker['lap_time'],
        attacker['tyre_life']
    ]])

    features_scaled = scaler.transform(features)
    return model.predict_proba(features_scaled)[0][1] * 100


def get_telemetry_html():
    """
    Ogni segmento corrisponde a una sezione reale della pista.
    """
    return '''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: transparent;
            font-family: 'Inter', sans-serif;
            color: #f0f6fc;
            overflow: hidden;
        }

        .telem-wrap {
            background: linear-gradient(180deg, rgba(20,20,35,0.98) 0%, rgba(10,10,18,0.99) 100%);
            border: 1px solid rgba(0,212,255,0.2);
            border-top: 2px solid #00d4ff;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }

        .telem-wrap::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(255,255,255,0.007) 1px, rgba(255,255,255,0.007) 2px);
            pointer-events: none;
        }

        .telem-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 14px;
            border-bottom: 1px solid rgba(0,212,255,0.1);
            background: rgba(0,0,0,0.4);
        }

        .telem-label {
            font-family: 'Orbitron', monospace;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 2px;
            color: #00d4ff;
        }

        .live-badge {
            background: #e10600;
            color: white;
            font-family: 'Orbitron', monospace;
            font-size: 8px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 3px;
            letter-spacing: 2px;
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .telem-body {
            padding: 14px;
            position: relative;
            z-index: 1;
        }

        /* Track section indicator */
        .track-section {
            text-align: center;
            padding: 5px 10px;
            margin-bottom: 10px;
            background: rgba(0,0,0,0.35);
            border-radius: 4px;
            border: 1px solid rgba(0,212,255,0.06);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .section-name {
            font-family: 'Orbitron', monospace;
            font-size: 9px;
            font-weight: 600;
            color: #00d4ff;
            letter-spacing: 1.5px;
        }

        .section-type {
            font-family: 'Share Tech Mono', monospace;
            font-size: 8px;
            letter-spacing: 1px;
            padding: 2px 6px;
            border-radius: 2px;
        }

        .section-type.straight { color: #00d26a; background: rgba(0,210,106,0.1); border: 1px solid rgba(0,210,106,0.2); }
        .section-type.braking { color: #ff4757; background: rgba(255,71,87,0.1); border: 1px solid rgba(255,71,87,0.2); }
        .section-type.corner { color: #ffd700; background: rgba(255,215,0,0.1); border: 1px solid rgba(255,215,0,0.2); }
        .section-type.accel { color: #00d4ff; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.2); }

        /* Speed */
        .speed-block {
            text-align: center;
            padding: 10px 0;
            margin-bottom: 12px;
            background: rgba(0,0,0,0.35);
            border-radius: 6px;
            border: 1px solid rgba(0,212,255,0.08);
        }

        .speed-num {
            font-family: 'Orbitron', monospace;
            font-size: 52px;
            font-weight: 900;
            color: #f0f6fc;
            line-height: 1;
            text-shadow: 0 0 25px rgba(0,212,255,0.3);
            transition: color 0.3s;
        }

        .speed-unit {
            font-family: 'Share Tech Mono', monospace;
            font-size: 11px;
            color: #4a5568;
            letter-spacing: 4px;
            margin-top: 2px;
        }

        /* Gear + RPM row */
        .gear-rpm-row {
            display: flex;
            gap: 10px;
            margin-bottom: 14px;
        }

        .stat-box {
            flex: 1;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 4px;
            padding: 8px;
            text-align: center;
        }

        .stat-label {
            font-family: 'Share Tech Mono', monospace;
            font-size: 9px;
            color: #4a5568;
            letter-spacing: 2px;
            margin-bottom: 4px;
        }

        .stat-val {
            font-family: 'Orbitron', monospace;
            font-size: 24px;
            font-weight: 800;
        }

        .stat-val.gear { color: #ffd700; text-shadow: 0 0 10px rgba(255,215,0,0.3); }
        .stat-val.rpm { color: #00d4ff; font-size: 16px; text-shadow: 0 0 8px rgba(0,212,255,0.3); }

        /* Bars */
        .bar-group { margin-bottom: 8px; }

        .bar-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 3px;
        }

        .bar-name {
            font-family: 'Share Tech Mono', monospace;
            font-size: 9px;
            color: #4a5568;
            letter-spacing: 1.5px;
        }

        .bar-pct {
            font-family: 'Orbitron', monospace;
            font-size: 9px;
            font-weight: 600;
            color: #8b949e;
        }

        .bar-track {
            height: 7px;
            background: rgba(255,255,255,0.04);
            border-radius: 2px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.5s ease;
        }

        .bar-fill.throttle {
            background: linear-gradient(90deg, #00d26a, #00ff88);
            box-shadow: 0 0 6px rgba(0,210,106,0.5);
        }

        .bar-fill.brake {
            background: linear-gradient(90deg, #e10600, #ff4757);
            box-shadow: 0 0 6px rgba(225,6,0,0.5);
        }

        /* Bottom row */
        .mini-row {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }

        .mini-stat {
            flex: 1;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 4px;
            padding: 6px 4px;
            text-align: center;
        }

        .mini-val {
            font-family: 'Orbitron', monospace;
            font-size: 13px;
            font-weight: 700;
            color: #f0f6fc;
        }

        .mini-label {
            font-family: 'Share Tech Mono', monospace;
            font-size: 7px;
            color: #4a5568;
            letter-spacing: 1px;
            margin-top: 2px;
        }

        .drs-on { color: #00d26a !important; text-shadow: 0 0 8px rgba(0,210,106,0.6); }
        .drs-off { color: #ff4757 !important; }

        /* Speed color zones */
        .speed-slow { color: #ff4757 !important; }
        .speed-mid { color: #ffd700 !important; }
        .speed-fast { color: #00d26a !important; }
    </style>
    </head>
    <body>
    <div class="telem-wrap">
        <div class="telem-header">
            <span class="telem-label">CAR #16 — TELEMETRY</span>
            <span class="live-badge">LIVE</span>
        </div>
        <div class="telem-body">
            <div class="track-section">
                <span class="section-name" id="section-name">RETTIFILO</span>
                <span class="section-type straight" id="section-type">STRAIGHT</span>
            </div>

            <div class="speed-block">
                <div class="speed-num" id="speed">340</div>
                <div class="speed-unit">KM/H</div>
            </div>

            <div class="gear-rpm-row">
                <div class="stat-box">
                    <div class="stat-label">GEAR</div>
                    <div class="stat-val gear" id="gear">8</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">RPM</div>
                    <div class="stat-val rpm" id="rpm">12400</div>
                </div>
            </div>

            <div class="bar-group">
                <div class="bar-header">
                    <span class="bar-name">THROTTLE</span>
                    <span class="bar-pct" id="throttle-pct">100%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill throttle" id="throttle-bar" style="width:100%"></div>
                </div>
            </div>

            <div class="bar-group">
                <div class="bar-header">
                    <span class="bar-name">BRAKE</span>
                    <span class="bar-pct" id="brake-pct">0%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill brake" id="brake-bar" style="width:0%"></div>
                </div>
            </div>

            <div class="mini-row">
                <div class="mini-stat">
                    <div class="mini-val" id="g-lat">0.2G</div>
                    <div class="mini-label">LAT G</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-val drs-on" id="drs">ON</div>
                    <div class="mini-label">DRS</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-val" id="ers">92%</div>
                    <div class="mini-label">ERS</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-val" id="fuel">42.1</div>
                    <div class="mini-label">FUEL KG</div>
                </div>
            </div>
        </div>
    </div>

    <script>
    (function() {
        function r(a,b){return Math.floor(Math.random()*(b-a+1))+a;}
        function rf(a,b){return (Math.random()*(b-a)+a).toFixed(1);}

        /*
         * MONZA CIRCUIT MODEL — Autodromo Nazionale di Monza
         * Each segment: [name, type, ticks, speedMin, speedMax, throttle, brake, gear, rpmMin, rpmMax, gLat, drs, ers]
         * type: 'straight', 'braking', 'corner', 'accel'
         * ticks = how many 500ms intervals this segment lasts
         */
        var monza = [
            // === RETTIFILO (Main Straight) — ~1100m, DRS Zone 1 ===
            ["RETTIFILO",            "straight", 23, 320, 345,  98, 0,   8, 12000, 12800, 0.2, true,  95],
            // === T1-T2 BRAKING (Variante del Rettifilo approach) ===
            ["BRAKING T1",           "braking",  3,  180, 250,   0, 92,  5,  9000, 10500, 1.2, false, 70],
            // === VARIANTE DEL RETTIFILO (Chicane T1-T2) ===
            ["VAR. RETTIFILO",       "corner",   4,   70,  95,  30, 15,  2,  7500,  8800, 3.8, false, 55],
            // === EXIT T2 (Acceleration out of chicane) ===
            ["EXIT T2",              "accel",    3,  120, 200,  85,  0,  4,  9000, 10500, 1.0, false, 45],
            // === CURVA GRANDE APPROACH ===
            ["CURVA GRANDE APP.",    "straight", 4,  260, 300,  95,  0,  7, 11500, 12200, 0.3, false, 60],
            // === CURVA GRANDE (High-speed right-hander) ===
            ["CURVA GRANDE",         "corner",   5,  225, 255,  70, 5,   6, 10500, 11500, 3.2, false, 50],
            // === APPROACH LESMO 1 ===
            ["APPROACH LESMO",       "accel",    3,  260, 295,  92,  0,  7, 11000, 12000, 0.4, false, 55],
            // === LESMO 1 BRAKING ===
            ["BRAKING LESMO 1",      "braking",  2,  170, 220,   0, 78,  4,  8500, 10000, 1.5, false, 48],
            // === LESMO 1 (Fast right) ===
            ["LESMO 1",              "corner",   3,  148, 165,  55, 10,  3,  8000,  9200, 2.8, false, 42],
            // === LESMO 1 EXIT ===
            ["LESMO 1 EXIT",         "accel",    2,  170, 230,  88,  0,  5,  9500, 10800, 0.8, false, 38],
            // === LESMO 2 BRAKING ===
            ["BRAKING LESMO 2",      "braking",  2,  155, 195,   0, 72,  4,  8200,  9500, 1.3, false, 35],
            // === LESMO 2 (Tight right) ===
            ["LESMO 2",              "corner",   3,  135, 155,  48, 12,  3,  7800,  9000, 3.1, false, 30],
            // === LESMO 2 EXIT / SERRAGLIO STRAIGHT ===
            ["SERRAGLIO",            "accel",    4,  190, 290,  95,  0,  6, 10000, 11800, 0.5, false, 40],
            // === ASCARI APPROACH ===
            ["ASCARI APPROACH",      "straight", 3,  300, 320,  98,  0,  7, 11800, 12500, 0.2, false, 55],
            // === ASCARI BRAKING ===
            ["BRAKING ASCARI",       "braking",  2,  175, 230,   0, 82,  4,  8500, 10000, 1.8, false, 50],
            // === VARIANTE ASCARI (Chicane) ===
            ["VAR. ASCARI",          "corner",   4,  160, 195,  45, 10,  4,  8500,  9800, 3.5, false, 45],
            // === ASCARI EXIT ===
            ["ASCARI EXIT",          "accel",    3,  210, 280,  92,  0,  6, 10500, 11500, 0.7, false, 50],
            // === RETTILINEO BACK STRAIGHT — DRS Zone 2, longest straight ===
            ["BACK STRAIGHT",        "straight", 9,  310, 340,  98,  0,  8, 12000, 12700, 0.2, true,  85],
            // === PARABOLICA BRAKING (Curva Alboreto) ===
            ["BRAKING PARABOLICA",   "braking",  3,  160, 230,   0, 85,  4,  8500, 10000, 1.4, false, 65],
            // === PARABOLICA APEX ===
            ["PARABOLICA",           "corner",   4,  200, 230,  60, 5,   5, 10000, 11000, 3.6, false, 55],
            // === PARABOLICA EXIT (progressively faster) ===
            ["PARABOLICA EXIT",      "accel",    4,  240, 300,  90,  0,  6, 10800, 12000, 1.5, false, 70],
            // === PIT STRAIGHT ACCELERATION (back toward start/finish) ===
            ["PIT STRAIGHT",         "accel",    4,  290, 330,  96,  0,  7, 11500, 12400, 0.3, false, 85]
        ];

        var segIdx = 0;
        var tickInSeg = 0;
        var fuel = 42.1;

        function getSeg() { return monza[segIdx]; }

        setInterval(function() {
            var seg = getSeg();
            var name     = seg[0];
            var type     = seg[1];
            var dur      = seg[2];
            var spdMin   = seg[3];
            var spdMax   = seg[4];
            var thrBase  = seg[5];
            var brkBase  = seg[6];
            var gearBase = seg[7];
            var rpmMin   = seg[8];
            var rpmMax   = seg[9];
            var gLatBase = parseFloat(seg[10]);
            var drs      = seg[11];
            var ersBase  = seg[12];

            // Progress through segment (0.0 to 1.0)
            var prog = tickInSeg / dur;

            // Add realistic variation
            var jitter = function(v, pct) { return v + (Math.random() - 0.5) * v * pct; };

            var speed = Math.round(spdMin + (spdMax - spdMin) * prog + (Math.random()-0.5)*12);
            speed = Math.max(60, Math.min(350, speed));

            var throttle = Math.round(jitter(thrBase, 0.06));
            throttle = Math.max(0, Math.min(100, throttle));

            var brake = Math.round(jitter(brkBase, 0.08));
            brake = Math.max(0, Math.min(100, brake));

            var gear = gearBase;
            var rpm = r(rpmMin, rpmMax);

            var gLat = (gLatBase + (Math.random()-0.5)*0.6).toFixed(1);
            gLat = Math.max(0, parseFloat(gLat)).toFixed(1);

            var ers = Math.round(jitter(ersBase, 0.1));
            ers = Math.max(0, Math.min(100, ers));

            // Fuel slowly decreasing
            fuel = Math.max(0, fuel - 0.003);

            // Update DOM
            var el;

            // Track section
            el = document.getElementById('section-name'); if(el) el.textContent = name;
            el = document.getElementById('section-type');
            if(el) {
                var typeLabel = {straight:'STRAIGHT',braking:'BRAKING',corner:'CORNER',accel:'ACCEL'}[type] || type;
                el.textContent = typeLabel;
                el.className = 'section-type ' + type;
            }

            // Speed with color coding
            el = document.getElementById('speed');
            if(el) {
                el.textContent = speed;
                el.className = 'speed-num' + (speed < 150 ? ' speed-slow' : speed < 250 ? ' speed-mid' : ' speed-fast');
            }

            el = document.getElementById('gear'); if(el) el.textContent = gear;
            el = document.getElementById('rpm'); if(el) el.textContent = rpm;
            el = document.getElementById('throttle-pct'); if(el) el.textContent = throttle+'%';
            el = document.getElementById('throttle-bar'); if(el) el.style.width = throttle+'%';
            el = document.getElementById('brake-pct'); if(el) el.textContent = brake+'%';
            el = document.getElementById('brake-bar'); if(el) el.style.width = brake+'%';
            el = document.getElementById('g-lat'); if(el) el.textContent = gLat+'G';

            el = document.getElementById('drs');
            if(el) {
                el.textContent = drs ? 'ON' : 'OFF';
                el.className = 'mini-val ' + (drs ? 'drs-on' : 'drs-off');
            }

            el = document.getElementById('ers'); if(el) el.textContent = ers+'%';
            el = document.getElementById('fuel'); if(el) el.textContent = fuel.toFixed(1);

            // Advance tick
            tickInSeg++;
            if (tickInSeg >= dur) {
                tickInSeg = 0;
                segIdx = (segIdx + 1) % monza.length;
            }
        }, 500);
    })();
    </script>
    </body>
    </html>
    '''


def render_leaderboard():
    """Genera la classifica live con gap dal leader."""
    gaps = ["Leader", "+2.3", "+4.1", "+5.8", "+8.2", "+12.5", "+14.1", "+18.3",
            "+22.7", "+25.4", "+28.9", "+32.1", "+35.6", "+38.2", "+41.5",
            "+44.8", "+48.3", "+52.1", "+56.7", "+61.2"]

    rows = ""
    for i, driver in enumerate(DRIVERS_2024):
        gap_text = gaps[i] if i < len(gaps) else f"+{i * 3:.1f}"
        rows += f'<div class="leaderboard-row"><span class="leaderboard-pos">{driver["pos"]}</span><span class="leaderboard-team-dot" style="background:{driver["color"]};color:{driver["color"]};"></span><span class="leaderboard-driver">{driver["code"]}</span><span class="leaderboard-gap">{gap_text}</span></div>'

    return f'''
    <div class="pit-panel standings-panel">
        <div class="panel-header">
            <span class="panel-label standings">Race Classification</span>
            <span class="live-badge">LIVE</span>
        </div>
        <div class="leaderboard-container">
            {rows}
        </div>
    </div>
    '''


# --- MAIN APP ---
def main():
    resources = load_resources()

    # === HEADER ===
    st.markdown('''
    <div class="header-bar">
        <div>
            <p class="header-title"><span class="live-dot"></span>MONZA 2025 — PIT WALL</p>
            <p class="header-subtitle">Overtake Probability Analysis System</p>
        </div>
        <div class="header-session">
            <span>RACE</span> &nbsp;|&nbsp;
            LAP <span class="lap-num">32</span> / <span class="lap-num">53</span> &nbsp;|&nbsp;
            TRACK: 42°C &nbsp;|&nbsp;
            AIR: 28°C &nbsp;|&nbsp;
            DRY
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if resources is None:
        st.error("Model not found. Run training pipeline first.")
        st.stop()

    # Initialize swap state
    if 'is_defense' not in st.session_state:
        st.session_state.is_defense = False

    # ================================================================
    # TOP ROW: Onboard Video (left) + Telemetry (right)
    # ================================================================
    top_left, top_right = st.columns([3, 2])

    with top_left:
        # Onboard panel via st.markdown (no JS needed)
        st.markdown('''
        <div class="pit-panel onboard-panel">
            <div class="panel-header">
                <span class="panel-label ferrari">CAR #16 — LECLERC — ONBOARD</span>
                <span class="live-badge">LIVE</span>
            </div>
            <div class="video-wrapper">
                <iframe
                    src="https://www.youtube.com/embed/SE4GN7_5Tq0?autoplay=1&mute=1&loop=1&playlist=SE4GN7_5Tq0&controls=0&modestbranding=1&rel=0&showinfo=0"
                    allow="autoplay; encrypted-media"
                    allowfullscreen>
                </iframe>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with top_right:
        # Telemetria: usa components.html() per eseguire il JavaScript
        components.html(get_telemetry_html(), height=480, scrolling=False)

    st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)

    # ================================================================
    # BOTTOM ROW: Prediction Tool (left) + Leaderboard (right)
    # ================================================================
    bottom_left, bottom_right = st.columns([3, 1])

    # ===== PREDICTION TOOL =====
    with bottom_left:
        # Panel header
        st.markdown('''
        <div class="pred-system-header">
            <div class="pred-system-header-inner">
                <span class="pred-sys-label">OVERTAKE PREDICTION SYSTEM</span>
                <span class="pred-sys-meta">ML MODEL v2.1 &nbsp;|&nbsp; XGBOOST &nbsp;|&nbsp; MONZA 2022-2024</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # --- DRIVER INPUT CARDS ---
        driver_col1, gap_col, driver_col2 = st.columns([5, 2, 5])

        # === YOUR DRIVER (left card) ===
        with driver_col1:
            st.markdown('''
            <div class="driver-card attacker-card">
                <div class="driver-card-header attacker-header">
                    <span class="driver-card-icon">▶</span>
                    <span class="driver-card-title">YOUR DRIVER</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            my_compound = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="comp")
            my_position = st.slider("Grid Position", 2, 20, 5, key="pos")
            my_tyre_life = st.slider("Tyre Wear (laps)", 1, 40, 12, key="tyre")

            lt1, lt2 = st.columns(2)
            with lt1:
                my_lap_minutes = st.number_input("Lap Min", 1, 2, 1, 1, key="my_min")
            with lt2:
                my_lap_seconds = st.number_input("Lap Sec", 0.0, 59.9, 24.5, 0.1, key="my_sec")
            my_lap_time = my_lap_minutes * 60 + my_lap_seconds

            # Lap time readout
            st.markdown(f'''
            <div class="lap-readout attacker-readout">
                <span class="lap-readout-label">LAP TIME</span>
                <span class="lap-readout-value">{my_lap_minutes}:{my_lap_seconds:05.2f}</span>
            </div>
            ''', unsafe_allow_html=True)

        # === CENTER: GAP + MODE ===
        with gap_col:
            st.markdown('''
            <div class="gap-center-block">
                <div class="gap-versus">VS</div>
            </div>
            ''', unsafe_allow_html=True)

            gap = st.number_input("Gap (s)", 0.1, 10.0, 0.8, 0.1, key="gap", label_visibility="collapsed")

            st.markdown(f'''
            <div class="gap-display-block">
                <span class="gap-display-label">INTERVAL</span>
                <span class="gap-display-value">{gap:.1f}</span>
                <span class="gap-display-unit">SEC</span>
            </div>
            ''', unsafe_allow_html=True)

            if st.button("⇄  SWAP ROLES", key="swap", use_container_width=True):
                st.session_state.is_defense = not st.session_state.is_defense
                st.rerun()

            mode_class = "defense" if st.session_state.is_defense else "attack"
            mode_text = "DEFENSE MODE" if st.session_state.is_defense else "ATTACK MODE"
            mode_icon = "🛡" if st.session_state.is_defense else "⚔"
            st.markdown(f'''
            <div class="mode-block">
                <span class="mode-badge {mode_class}">{mode_icon} {mode_text}</span>
            </div>
            ''', unsafe_allow_html=True)

        # === OPPONENT (right card) — SYMMETRIC ===
        with driver_col2:
            st.markdown(f'''
            <div class="driver-card defender-card">
                <div class="driver-card-header defender-header">
                    <span class="driver-card-icon">◀</span>
                    <span class="driver-card-title">OPPONENT — P{my_position - 1}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            opp_compound = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=2, key="opp_comp")
            opp_position_display = my_position - 1
            opp_tyre_life = st.slider("Tyre Wear (laps)", 1, 40, 25, key="opp_tyre")

            lt1, lt2 = st.columns(2)
            with lt1:
                opp_lap_minutes = st.number_input("Lap Min", 1, 2, 1, 1, key="opp_min")
            with lt2:
                opp_lap_seconds = st.number_input("Lap Sec", 0.0, 59.9, 25.2, 0.1, key="opp_sec")
            opp_lap_time = opp_lap_minutes * 60 + opp_lap_seconds

            # Lap time readout
            st.markdown(f'''
            <div class="lap-readout defender-readout">
                <span class="lap-readout-label">LAP TIME</span>
                <span class="lap-readout-value">{opp_lap_minutes}:{opp_lap_seconds:05.2f}</span>
            </div>
            ''', unsafe_allow_html=True)

        # === CALCULATE BUTTON ===
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            calculate = st.button("ANALYZE  OVERTAKE  PROBABILITY", use_container_width=True)

        # === RESULTS ===
        if calculate:
            if st.session_state.is_defense:
                attacker = {'position': my_position - 1, 'compound': opp_compound,
                            'tyre_life': opp_tyre_life, 'lap_time': opp_lap_time}
                defender = {'compound': my_compound, 'tyre_life': my_tyre_life, 'lap_time': my_lap_time}
                context = "OF BEING OVERTAKEN"
            else:
                attacker = {'position': my_position, 'compound': my_compound,
                            'tyre_life': my_tyre_life, 'lap_time': my_lap_time}
                defender = {'compound': opp_compound, 'tyre_life': opp_tyre_life, 'lap_time': opp_lap_time}
                context = "OF OVERTAKING"

            prob = predict_overtake(resources, attacker, defender)

            if prob > 60:
                level, label, bar_color = "high", "HIGH PROBABILITY", "#00d26a"
            elif prob > 40:
                level, label, bar_color = "medium", "UNCERTAIN", "#ffd700"
            else:
                level, label, bar_color = "low", "LOW PROBABILITY", "#ff4757"

            delta = attacker['lap_time'] - defender['lap_time']
            delta_sign = "FASTER" if delta < 0 else "SLOWER"
            tyre_diff = attacker['tyre_life'] - defender['tyre_life']
            compound_adv = get_compound_value(attacker['compound']) - get_compound_value(defender['compound'])
            compound_txt = "SOFTER" if compound_adv > 0 else ("HARDER" if compound_adv < 0 else "EQUAL")

            st.markdown(f'''
            <div class="result-readout {level}">
                <div class="result-main">
                    <div class="result-prob-block">
                        <span class="result-prob-value {level}">{prob:.1f}<span class="result-prob-pct">%</span></span>
                        <span class="result-prob-label">{label}</span>
                        <span class="result-prob-context">{context}</span>
                    </div>
                    <div class="result-bar-block">
                        <div class="result-bar-track">
                            <div class="result-bar-fill" style="width:{min(prob,100):.0f}%;background:{bar_color};box-shadow:0 0 12px {bar_color}40;"></div>
                        </div>
                    </div>
                </div>
                <div class="result-stats-row">
                    <div class="result-stat">
                        <span class="result-stat-value" style="color:{'#00d26a' if delta < 0 else '#ff4757'};">{abs(delta):.2f}s</span>
                        <span class="result-stat-label">DELTA • {delta_sign}</span>
                    </div>
                    <div class="result-stat-divider"></div>
                    <div class="result-stat">
                        <span class="result-stat-value" style="color:#c9d1d9;">{tyre_diff:+d} laps</span>
                        <span class="result-stat-label">TYRE WEAR DIFF</span>
                    </div>
                    <div class="result-stat-divider"></div>
                    <div class="result-stat">
                        <span class="result-stat-value" style="color:#ffd700;">{compound_txt}</span>
                        <span class="result-stat-label">COMPOUND ADV</span>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    # ===== LEADERBOARD (right) =====
    with bottom_right:
        st.markdown(render_leaderboard(), unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer-text">F1 PIT WALL BOX • Overtake Prediction System • Model trained on Monza 2022-2024 • ML Course Project</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()