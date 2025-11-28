
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import scikit-learn as sk
# 🎨 Color palette
from colors import BACKGROUND_COLOR, TITLE_COLOR, TEXT_COLOR, ACCENT_COLOR

# Extra colors for this activity
PASS_COLOR = "#3b82f6"        # blue for total passes
SUCCESS_COLOR = "#22c55e"     # green for successful passes
RADAR_LINE_COLOR = "#22d3ee"  # cyan line
RADAR_FILL_COLOR = "rgba(34,211,238,0.30)"  # translucent cyan fill

# ------------------------------
# CSS helper
# ------------------------------
def inject_css(css_file: str):
    """Read styles.css and inject it + color variables into the page."""
    try:
        with open(css_file) as f:
            css = f.read()

        st.markdown(
            f"""
            <style>
            :root {{
                --bg-color: {BACKGROUND_COLOR};
                --title-color: {TITLE_COLOR};
                --text-color: {TEXT_COLOR};
                --accent-color: {ACCENT_COLOR};
            }}
            {css}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.warning("⚠️ styles.css not found. Make sure it is in the same folder as the app.")

# ------------------------------
# Configuración de la página
# ------------------------------
st.set_page_config(
    page_title="Premier League Player Stats 24/25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS after set_page_config
inject_css("styles.css")

# Título principal (usando clase para aplicar color desde CSS)
st.markdown(
    '<h1 class="main-title">🏴 Premier League Player Stats 24/25 Dashboard</h1>',
    unsafe_allow_html=True,
)

# ------------------------------
# Carga de datos con caché
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("epl_player_stats_24_25.csv")

    numeric_cols = ["Goals", "Assists", "Clean Sheets", "Passes", "Passes%"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

try:
    df = load_data()
    st.success(f"✅ Datos cargados: {len(df)} registros")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------------------
# MÉTRICAS PRINCIPALES
# ------------------------------
st.markdown('<h2 class="section-title">📊 Main KPIs</h2>', unsafe_allow_html=True)
met_col1, met_col2, met_col3, met_col4 = st.columns(4)

name_col = "Player Name"

# 1) Player with most goals
with met_col1:
    goals_col = "Goals"
    goals_series = pd.to_numeric(df[goals_col], errors="coerce")
    valid_goals = goals_series.dropna()

    if not valid_goals.empty:
        idx_max_goals = valid_goals.idxmax()
        top_scorer_name = str(df.loc[idx_max_goals, name_col])
        top_scorer_goals = int(valid_goals.loc[idx_max_goals])
        st.metric("Top Scorer (Goals)", top_scorer_name, f"{top_scorer_goals} goals")
    else:
        st.metric("Top Scorer (Goals)", "N/A", "0 goals")

# 2) Player with most assists
with met_col2:
    assists_col = "Assists"
    assists_series = pd.to_numeric(df[assists_col], errors="coerce")
    valid_assists = assists_series.dropna()

    if not valid_assists.empty:
        idx_max_assists = valid_assists.idxmax()
        top_assist_name = str(df.loc[idx_max_assists, name_col])
        top_assist_value = int(valid_assists.loc[idx_max_assists])
        st.metric("Most Assists", top_assist_name, f"{top_assist_value} assists")
    else:
        st.metric("Most Assists", "N/A", "0 assists")

# 3) Player with most clean sheets
with met_col3:
    clean_col = "Clean Sheets"
    clean_series = pd.to_numeric(df[clean_col], errors="coerce")
    valid_clean = clean_series.dropna()

    if not valid_clean.empty:
        idx_max_clean = valid_clean.idxmax()
        top_clean_name = str(df.loc[idx_max_clean, name_col])
        top_clean_value = int(valid_clean.loc[idx_max_clean])
        st.metric("Most Clean Sheets", top_clean_name, f"{top_clean_value} clean sheets")
    else:
        st.metric("Most Clean Sheets", "N/A", "0 clean sheets")

# 4) Player with highest number of successful passes
with met_col4:
    succ_pass_col = "Successful Passes"
    succ_pass_series = pd.to_numeric(df[succ_pass_col], errors="coerce")
    valid_succ_pass = succ_pass_series.dropna()

    if not valid_succ_pass.empty:
        idx_max_succ = valid_succ_pass.idxmax()
        top_succ_name = str(df.loc[idx_max_succ, name_col])
        top_succ_value = int(valid_succ_pass.loc[idx_max_succ])
        st.metric("Most Successful Passes", top_succ_name, f"{top_succ_value} successful passes")
    else:
        st.metric("Most Successful Passes", "N/A", "0 successful passes")

st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------------------
# FILTROS: CLUB Y JUGADOR
# ------------------------------
st.markdown('<h2 class="section-title">🎯 Player Filter</h2>', unsafe_allow_html=True)

filt_col1, filt_col2 = st.columns(2)

with filt_col1:
    club_list = sorted(df["Club"].dropna().unique().tolist())
    selected_club = st.selectbox("Club", club_list)

with filt_col2:
    players_in_club = df[df["Club"] == selected_club]["Player Name"].dropna().unique().tolist()
    selected_player = st.selectbox("Player", players_in_club)

player_df = df[(df["Club"] == selected_club) & (df["Player Name"] == selected_player)]

st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------------------
# GRÁFICA: PASSES vs SUCCESSFUL PASSES
# ------------------------------
st.markdown('<h2 class="section-title">📈 Passes vs Successful Passes</h2>', unsafe_allow_html=True)

if not player_df.empty:
    passes_value = float(player_df["Passes"].iloc[0])
    succ_pass_value = float(player_df["Successful Passes"].iloc[0])

    chart_df = pd.DataFrame({
        "Metric": ["Passes", "Successful Passes"],
        "Value": [passes_value, succ_pass_value]
    })

    # 🔵🟢 Blue for passes, green for successful passes
    fig_passes = px.bar(
        chart_df,
        x="Metric",
        y="Value",
        text="Value",
        title=f"Passes vs Successful Passes - {selected_player} ({selected_club})",
        color="Metric",
        color_discrete_map={
            "Passes": PASS_COLOR,
            "Successful Passes": SUCCESS_COLOR,
        },
    )
    fig_passes.update_traces(textposition="outside")

    # Make title and area easier to read
    fig_passes.update_layout(
        yaxis_title="Number of passes",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor="#020617",  # a bit darker than plot area
        font_color=TEXT_COLOR,
        title_font=dict(color=TITLE_COLOR, size=22),
        legend_title_text="Metric",
    )

    st.plotly_chart(fig_passes, use_container_width=True)
else:
    st.info("No hay datos para el jugador seleccionado.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------------------
# RADAR: TEAM COMPARISON
# ------------------------------
rad_f1, rad_f2 = st.columns(2)

club_list = sorted(df["Club"].dropna().unique().tolist())

with rad_f1:
    team1 = st.selectbox("Team 1", club_list, key="team1")

with rad_f2:
    default_idx = 1 if len(club_list) > 1 else 0
    team2 = st.selectbox("Team 2", club_list, index=default_idx, key="team2")


def safe_pct(num_series, den_series):
    num = pd.to_numeric(num_series, errors="coerce").fillna(0)
    den = pd.to_numeric(den_series, errors="coerce").fillna(0)
    num_sum = num.sum()
    den_sum = den.sum()
    if den_sum > 0:
        return (num_sum / den_sum) * 100
    else:
        return 0.0


def clean_percent_series(series):
    return pd.to_numeric(series.astype(str).str.rstrip('%'), errors="coerce")


def compute_team_metrics(df, club_name):
    sub = df[df["Club"] == club_name].copy()

    tck_pct = safe_pct(sub["gDuels Won"], sub["Ground Duels"])
    pass_pct = safe_pct(sub["Successful Passes"], sub["Passes"])
    shot_conv = safe_pct(sub["Goals"], sub["Shots"])

    if "Saves %" in sub.columns:
        sub_gk = sub[sub["Position"] == "GKP"].copy()
        if not sub_gk.empty:
            saves_series = clean_percent_series(sub_gk["Saves %"])
            saves_series = saves_series.dropna()
            saves_pct = float(saves_series.mean()) if not saves_series.empty else 0.0
        else:
            saves_pct = 0.0
    else:
        saves_pct = 0.0

    if "Successful Crosses" in sub.columns and "Crosses" in sub.columns:
        crosses_pct = safe_pct(sub["Successful Crosses"], sub["Crosses"])
    else:
        crosses_pct = 0.0

    metrics = {
        "Tck %": tck_pct,
        "Pas %": pass_pct,
        "Shot conv %": shot_conv,
        "Saves %": saves_pct,
        "Crosses %": crosses_pct,
    }
    return metrics


def make_radar_figure(metrics_dict, title):
    radar_df = pd.DataFrame({
        "Metric": list(metrics_dict.keys()),
        "Value": list(metrics_dict.values())
    })

    fig = px.line_polar(
        radar_df,
        r="Value",
        theta="Metric",
        line_close=True,
        title=title,
    )

    # 🌌 Cleaner radar style like the example image
    fig.update_traces(
        fill="toself",
        fillcolor=RADAR_FILL_COLOR,
        mode="lines+markers+text",
        text=radar_df["Value"].round(1),
        textposition="top center",
        line_color=RADAR_LINE_COLOR,
        marker_color=RADAR_LINE_COLOR,
        marker_size=6,
    )

    fig.update_layout(
        polar=dict(
            bgcolor="#020617",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#1f2937",
                gridwidth=1,
                showline=False,
                tickfont=dict(color="#9ca3af"),
            ),
            angularaxis=dict(
                gridcolor="#1f2937",
                tickfont=dict(color="#e5e7eb"),
            ),
        ),
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        title_font=dict(color=TITLE_COLOR, size=18),
    )
    return fig


metrics_team1 = compute_team_metrics(df, team1)
metrics_team2 = compute_team_metrics(df, team2)

rad_c1, rad_c2 = st.columns(2)

with rad_c1:
    fig_team1 = make_radar_figure(metrics_team1, f"{team1} - General Performance")
    st.plotly_chart(fig_team1, use_container_width=True)

with rad_c2:
    fig_team2 = make_radar_figure(metrics_team2, f"{team2} - General Performance")
    st.plotly_chart(fig_team2, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------------------
# GRÁFICA: MINUTOS POR JUGADOR (POR CLUB)
# ------------------------------
st.markdown('<h2 class="section-title">⏱️ Player Minutes per Club</h2>', unsafe_allow_html=True)

g3_col1, g3_col2 = st.columns(2)

with g3_col1:
    sort_order = st.selectbox("Order", ["Descending", "Ascending"], key="minutes_order")

with g3_col2:
    club_for_minutes = st.selectbox(
        "Club",
        sorted(df["Club"].dropna().unique().tolist()),
        key="minutes_club",
    )

asc = True if sort_order == "Ascending" else False

club_df = df[df["Club"] == club_for_minutes].copy()

if club_df.empty:
    st.info("No hay datos para el club seleccionado.")
else:
    minutes_by_player = (
        club_df.groupby("Player Name")["Minutes"]
        .sum()
        .reset_index()
        .sort_values("Minutes", ascending=asc)
    )

    # 🔥 Gradient: red -> yellow -> green based on minutes played
    fig_minutes = px.bar(
        minutes_by_player,
        x="Minutes",
        y="Player Name",
        orientation="h",
        title=f"Total Minutes per Player - {club_for_minutes} ({sort_order})",
        text="Minutes",
        color="Minutes",
        color_continuous_scale="RdYlGn",  # built-in Plotly colorscale
    )

    fig_minutes.update_layout(
        yaxis={'categoryorder': 'total ascending' if asc else 'total descending'},
        xaxis_title="Minutes",
        yaxis_title="Player",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        title_font=dict(color=TITLE_COLOR, size=22),
    )

    # Hide the color bar to keep it clean, we just use color as cue
    fig_minutes.update_coloraxes(showscale=False)

    fig_minutes.update_traces(textposition="outside")

    st.plotly_chart(fig_minutes, use_container_width=True)

