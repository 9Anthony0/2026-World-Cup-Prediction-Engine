"""
2026 FIFA World Cup Prediction Engine — Streamlit Dashboard
=============================================================
Premium dark-themed dashboard with live Monte Carlo simulation,
interactive charts, group breakdowns, and team deep-dives.
Mirrors the design language of the Masters Predictor app.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import time

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
import team_data
import world_cup_engine
importlib.reload(team_data)
importlib.reload(world_cup_engine)

from world_cup_engine import (
    run_monte_carlo,
    simulate_match,
    simulate_tournament,
    simulate_single_bracket,
    calculate_expected_goals,
    calculate_expected_goals_composite,
    ELO_RATINGS,
    GROUPS,
    BASE_GOALS_PER_TEAM,
    PENALTY_DAMPING,
)
from team_data import TEAM_PROFILES, DEFAULT_WEIGHTS, FACTOR_LABELS

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="2026 FIFA World Cup Predictor",
    page_icon=":trophy:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# CUSTOM CSS — FIFA World Cup Deep Navy / Gold Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────── */
    .stApp {
        background: linear-gradient(175deg, #0a0c1a 0%, #0d1025 40%, #0e1117 100%);
    }

    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }

    /* ── Hero Header ────────────────────────────────── */
    .hero-container {
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b69 25%, #44318d 50%, #1a0a2e 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(212, 175, 55, 0.3);
        box-shadow: 0 20px 60px rgba(44, 27, 105, 0.4), 0 0 40px rgba(44, 27, 105, 0.2);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 20%, rgba(212, 175, 55, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        bottom: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(220, 50, 50, 0.06) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #D4AF37;
        text-shadow: 0 2px 20px rgba(212, 175, 55, 0.3);
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: rgba(255, 255, 255, 0.85);
        margin-top: 0.5rem;
        font-weight: 300;
        letter-spacing: 0.3px;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(212, 175, 55, 0.12);
        border: 1px solid rgba(212, 175, 55, 0.35);
        color: #D4AF37;
        padding: 0.3rem 0.9rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin-top: 1rem;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }

    /* ── Cards ──────────────────────────────────────── */
    .stat-card {
        background: linear-gradient(145deg, rgba(26, 31, 46, 0.95), rgba(20, 25, 38, 0.95));
        border: 1px solid rgba(212, 175, 55, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }

    .stat-card:hover {
        border-color: rgba(212, 175, 55, 0.3);
        box-shadow: 0 8px 40px rgba(44, 27, 105, 0.2);
        transform: translateY(-2px);
    }

    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #D4AF37;
        line-height: 1;
    }

    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: rgba(232, 232, 232, 0.6);
        margin-top: 0.4rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Section Headers ───────────────────────────── */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #E8E8E8;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(212, 175, 55, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Sidebar ───────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1025, #0e1117);
        border-right: 1px solid rgba(212, 175, 55, 0.1);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #D4AF37 !important;
        font-family: 'Outfit', sans-serif;
    }

    /* ── Tabs ──────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: rgba(232, 232, 232, 0.6);
        border-radius: 10px;
        padding: 10px 20px;
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        color: #D4AF37 !important;
        background: rgba(44, 27, 105, 0.3) !important;
        border-color: rgba(212, 175, 55, 0.3) !important;
    }

    /* ── Selectbox / Slider labels ─────────────────── */
    .stSelectbox label, .stSlider label, .stMultiSelect label, .stNumberInput label {
        color: rgba(232, 232, 232, 0.8) !important;
        font-family: 'Inter', sans-serif;
    }

    /* ── Metrics ───────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(26, 31, 46, 0.95), rgba(20, 25, 38, 0.95));
        border: 1px solid rgba(212, 175, 55, 0.12);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stMetric"] label {
        color: rgba(232, 232, 232, 0.6) !important;
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-size: 0.75rem !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #D4AF37 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ── Group Card ─────────────────────────────────── */
    .group-card {
        background: linear-gradient(145deg, rgba(26, 31, 46, 0.9), rgba(16, 18, 30, 0.95));
        border: 1px solid rgba(212, 175, 55, 0.1);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }

    .group-card:hover {
        border-color: rgba(212, 175, 55, 0.25);
        box-shadow: 0 4px 25px rgba(44, 27, 105, 0.15);
    }

    .group-letter {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #D4AF37;
        margin-bottom: 0.6rem;
    }

    .group-team {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: rgba(232, 232, 232, 0.85);
        padding: 0.25rem 0;
        display: flex;
        justify-content: space-between;
    }

    .group-team-elo {
        color: rgba(212, 175, 55, 0.6);
        font-size: 0.8rem;
    }

    /* ── Hide default Streamlit elements ────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E8E8E8"),
    margin=dict(l=40, r=40, t=50, b=40),
)

FIFA_COLORS = [
    "#D4AF37", "#6C63FF", "#FF6B6B", "#4ECDC4", "#00A86B",
    "#FF69B4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
    "#C2B280", "#FF8C42",
]


def fifa_gradient(n):
    """Generate a World Cup-themed color list."""
    base = [
        "#D4AF37", "#6C63FF", "#FF6B6B", "#4ECDC4", "#00A86B",
        "#FF69B4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
        "#F97316", "#8B5CF6", "#06B6D4", "#F43F5E", "#22D3EE",
        "#A78BFA", "#FB923C", "#34D399", "#F472B6", "#818CF8",
    ]
    return (base * ((n // len(base)) + 1))[:n]


# ══════════════════════════════════════════════════════════════
# SESSION STATE — Run simulation on first load or param change
# ══════════════════════════════════════════════════════════════
def run_simulation(iterations, seed, weights=None):
    """Run the Monte Carlo engine and cache results in session state."""
    with st.spinner(f"Simulating {iterations:,} tournaments..."):
        start = time.perf_counter()
        df = run_monte_carlo(iterations=iterations, seed=seed, verbose=False,
                             profiles=TEAM_PROFILES, weights=weights)
        elapsed = time.perf_counter() - start
    return df, elapsed


# ══════════════════════════════════════════════════════════════
# SIDEBAR — Configuration Panel
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">&#9917;</div>
        <div style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #D4AF37; margin-top: 0.5rem;">
            World Cup Predictor
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: rgba(232,232,232,0.5); margin-top: 0.2rem;">
            2026 &middot; USA &middot; Mexico &middot; Canada
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Simulation Controls")
    sim_iterations = st.select_slider(
        "Simulations",
        options=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
        format_func=lambda x: f"{x:,}",
    )

    sim_seed = st.number_input("Random Seed", value=42, min_value=0, max_value=999999,
                                help="Set seed for reproducible results. Change to explore variance.")

    run_btn = st.button("Run Simulation", type="primary", width="stretch")

    st.markdown("---")

    # Factor weight controls
    st.markdown("### Sports Factor Weights")
    st.caption("Adjust the influence of sports-related indicators")

    user_weights = {}
    sports_keys = ["elo", "star_player_rating", "squad_depth", "tournament_pedigree", "recent_form", 
                   "offensive_rating", "defensive_rating", "manager_experience", "host_advantage"]
    econ_keys = ["gdp_pc", "population", "avg_temp"]

    for key in sports_keys:
        default_val = DEFAULT_WEIGHTS.get(key, 0.0)
        label = FACTOR_LABELS.get(key, key)
        user_weights[key] = st.slider(
            label,
            min_value=0.0, max_value=0.50,
            value=float(default_val),
            step=0.01,
            key=f"w_{key}",
        )

    st.markdown("### Klement Econometric Weights")
    st.caption("Adjust the influence of Joachim Klement's econometric factors")
    for key in econ_keys:
        default_val = DEFAULT_WEIGHTS.get(key, 0.0)
        label = FACTOR_LABELS.get(key, key)
        user_weights[key] = st.slider(
            label,
            min_value=0.0, max_value=0.50,
            value=float(default_val),
            step=0.01,
            key=f"w_{key}",
        )

    # Normalize weights so they sum to 1.0
    total_w = sum(user_weights.values())
    if total_w > 0:
        user_weights = {k: v / total_w for k, v in user_weights.items()}

    st.markdown("---")
    st.markdown("### Model Parameters")
    st.markdown(f"""
    **Multi-Factor Poisson Model**
    - Base rate: `{BASE_GOALS_PER_TEAM}` goals/team
    - Penalty damping: `{PENALTY_DAMPING}`
    - Factors: `{len(DEFAULT_WEIGHTS)}` indicators

    **Tournament Format**
    - 48 teams, 12 groups of 4
    - Top 2 + 8 best 3rd place
    - Single-elimination knockout
    """)

    st.markdown("---")

    # Group overview in sidebar
    st.markdown("### Groups at a Glance")
    for group_letter, teams in sorted(GROUPS.items()):
        team_lines = "".join(
            f'<div class="group-team"><span>{t}</span><span class="group-team-elo">{ELO_RATINGS[t]}</span></div>'
            for t in teams
        )
        st.markdown(f"""
        <div class="group-card">
            <div class="group-letter">Group {group_letter}</div>
            {team_lines}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:rgba(232,232,232,0.3); font-size:0.75rem;'>"
        f"v2.0 | {len(ELO_RATINGS)} teams | Multi-Factor Poisson</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# RUN SIMULATION (on button click or first load)
# ══════════════════════════════════════════════════════════════
if run_btn or "results_df" not in st.session_state:
    results_df, sim_time = run_simulation(sim_iterations, sim_seed, weights=user_weights)
    st.session_state["results_df"] = results_df
    st.session_state["sim_time"] = sim_time
    st.session_state["sim_n"] = sim_iterations
    st.session_state["sim_seed"] = sim_seed
    st.session_state["user_weights"] = user_weights

results_df = st.session_state["results_df"]
sim_time = st.session_state["sim_time"]
sim_n = st.session_state["sim_n"]


# ══════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════
top_team = results_df.iloc[0]

st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">2026 FIFA World Cup Predictor</div>
    <div class="hero-subtitle">
        Multi-Factor Monte Carlo Engine &nbsp;|&nbsp; 8 Predictive Indicators &nbsp;|&nbsp; Poisson Goal Model
    </div>
    <div class="hero-badge">
        {sim_n:,} Simulations &middot; {len(ELO_RATINGS)} Teams &middot; 12 Groups &middot; {sim_time:.1f}s Runtime
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top Stats Row ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Predicted Winner", top_team["Team"])
with col2:
    st.metric("Win Probability", f"{top_team['Win World Cup %']:.1f}%")
with col3:
    st.metric("Top Elo Rating", f"{int(top_team['Elo'])}")
with col4:
    st.metric("Simulation Speed", f"{sim_n / sim_time:,.0f}/sec")


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab_consensus, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Tournament Odds",
    "Predicted Bracket",
    "Bracket Predictor",
    "Group Analysis",
    "Team Profiles",
    "Head-to-Head",
    "Knockout Scenarios",
])


# ──────────────────────────────────────────────────────────────
# TAB 1: TOURNAMENT ODDS (Leaderboard)
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Tournament Probability Leaderboard</div>', unsafe_allow_html=True)

    n_show = st.slider("Number of teams to display", 10, 48, 20, key="lb_slider")
    display_df = results_df.head(n_show).copy()

    # ── Win Probability Horizontal Bar Chart ──
    fig_bar = go.Figure()
    bar_data = display_df.head(min(25, n_show))

    fig_bar.add_trace(go.Bar(
        y=bar_data["Team"],
        x=bar_data["Win World Cup %"],
        orientation='h',
        marker=dict(
            color=[f"rgba({max(0,180 - i*6)}, {max(0, 100 - i*3)}, {min(255, 55 + i*10)}, 0.85)" for i in range(len(bar_data))],
            line=dict(color="rgba(212, 175, 55, 0.35)", width=1),
        ),
        text=[f"{v:.1f}%" for v in bar_data["Win World Cup %"]],
        textposition="outside",
        textfont=dict(color="#D4AF37", size=12, family="Outfit"),
    ))

    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Win World Cup Probability", font=dict(size=18, family="Outfit", color="#D4AF37")),
        height=max(450, len(bar_data) * 35),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        xaxis=dict(title="Probability (%)", range=[0, max(bar_data["Win World Cup %"]) * 1.4]),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width="stretch")

    # ── Multi-stage stacked bar ──
    st.markdown("#### Stage-by-Stage Advancement Probabilities")

    top_16 = results_df.head(16)
    fig_stages = go.Figure()

    stages = [
        ("Group Stage %", "Group Stage", "#4ECDC4"),
        ("Quarterfinals %", "Quarterfinals", "#6C63FF"),
        ("Semifinals %", "Semifinals", "#FF6B6B"),
        ("Final %", "Final", "#FF69B4"),
        ("Win World Cup %", "Champion", "#D4AF37"),
    ]

    for col_name, label, color in stages:
        fig_stages.add_trace(go.Bar(
            y=top_16["Team"],
            x=top_16[col_name],
            name=label,
            orientation='h',
            marker_color=color,
            marker_line=dict(color="rgba(255,255,255,0.05)", width=0.5),
        ))

    fig_stages.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Advancement Probability by Stage (Top 16)", font=dict(size=18, family="Outfit", color="#D4AF37")),
        barmode='group',
        height=600,
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis_title="Probability (%)",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15,
            bgcolor="rgba(26, 31, 46, 0.8)",
            bordercolor="rgba(212, 175, 55, 0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig_stages, width="stretch")

    # ── Full Data Table ──
    st.markdown("#### Full Prediction Table")
    st.dataframe(
        display_df[["Team", "Group", "Elo", "Group Stage %", "Quarterfinals %",
                     "Semifinals %", "Final %", "Win World Cup %"]],
        width="stretch",
        hide_index=True,
        height=min(700, n_show * 38 + 40),
    )


# ──────────────────────────────────────────────────────────────
# TAB: CONSENSUS PREDICTED BRACKET
# ──────────────────────────────────────────────────────────────
with tab_consensus:
    st.markdown('<div class="section-header">Consensus Predicted Bracket</div>', unsafe_allow_html=True)
    st.markdown("The statistically expected tournament outcome computed deterministically by the engine based on current weights.")

    consensus_bracket = simulate_single_bracket(
        profiles=TEAM_PROFILES,
        weights=st.session_state.get("user_weights", None),
        deterministic=True,
    )
    
    champion = consensus_bracket["champion"]
    st.markdown(f"""
    <div class="stat-card" style="text-align:center; margin-bottom:1.5rem;">
        <div class="stat-label">CONSENSUS PREDICTED CHAMPION</div>
        <div class="stat-value" style="font-size:3rem; color:#D4AF37;">&#127942; {champion} &#127942;</div>
    </div>
    """, unsafe_allow_html=True)

    # Group Stage Results
    st.markdown("#### Predicted Group Standings")
    grp_cols = st.columns(4)
    for idx, (grp_letter, standings) in enumerate(sorted(consensus_bracket["group_results"].items())):
        with grp_cols[idx % 4]:
            rows_html = ""
            for pos, s in enumerate(standings):
                marker = "" if pos >= 2 else " style='color:#4ECDC4;font-weight:600;'"
                if pos == 2 and s["team"] in [r["winner"] for r in consensus_bracket.get("r32_results", []) if True]:
                    marker = " style='color:#FFEAA7;'"
                rows_html += f"<div class='group-team'{marker}><span>{s['team']}</span><span class='group-team-elo'>{s['points']}pts | GD:{s['gd']:+d}</span></div>"
            st.markdown(f"""
            <div class="group-card">
                <div class="group-letter">Group {grp_letter}</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

    # Knockout Bracket Visualization
    st.markdown("#### Predicted Knockout Bracket (Deterministic)")

    def render_consensus_match(result, round_name):
        t1, t2 = result["team1"], result["team2"]
        g1, g2 = result["goals1"], result["goals2"]
        w = result["winner"]
        pen = " (pen)" if result.get("penalties", False) else ""
        w1_style = "color:#D4AF37;font-weight:700;" if w == t1 else "color:rgba(232,232,232,0.5);"
        w2_style = "color:#D4AF37;font-weight:700;" if w == t2 else "color:rgba(232,232,232,0.5);"
        return f"""
        <div style="background:rgba(26,31,46,0.8);border:1px solid rgba(212,175,55,0.1);border-radius:10px;padding:0.6rem 1rem;margin:0.3rem 0;font-family:'Inter',sans-serif;font-size:0.85rem;">
            <div style="{w1_style}">{t1} <span style="float:right;">{g1}</span></div>
            <div style="{w2_style}">{t2} <span style="float:right;">{g2}</span></div>
            <div style="color:rgba(232,232,232,0.3);font-size:0.7rem;text-align:center;">{round_name}{pen}</div>
        </div>
        """

    # R32
    st.markdown("##### Round of 32")
    r32_cols = st.columns(4)
    for i, result in enumerate(consensus_bracket["r32_results"]):
        with r32_cols[i % 4]:
            st.markdown(render_consensus_match(result, "R32"), unsafe_allow_html=True)

    # R16
    st.markdown("##### Round of 16")
    r16_cols = st.columns(4)
    for i, result in enumerate(consensus_bracket["r16_results"]):
        with r16_cols[i % 4]:
            st.markdown(render_consensus_match(result, "R16"), unsafe_allow_html=True)

    # QF
    st.markdown("##### Quarterfinals")
    qf_cols = st.columns(4)
    for i, result in enumerate(consensus_bracket["qf_results"]):
        with qf_cols[i]:
            st.markdown(render_consensus_match(result, "QF"), unsafe_allow_html=True)

    # SF
    st.markdown("##### Semifinals")
    sf_cols = st.columns(2)
    for i, result in enumerate(consensus_bracket["sf_results"]):
        with sf_cols[i]:
            st.markdown(render_consensus_match(result, "SF"), unsafe_allow_html=True)

    # Final
    st.markdown("##### FINAL")
    final_r = consensus_bracket["final_result"]
    st.markdown(render_consensus_match(final_r, "FINAL"), unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;padding:1rem;">
        <span style="font-family:'Outfit',sans-serif;font-size:2.2rem;color:#D4AF37;font-weight:800;">🏆 {champion} 🏆</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 2: BRACKET PREDICTOR
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Bracket Predictor</div>', unsafe_allow_html=True)
    st.markdown("Simulate a single complete tournament and view the full knockout bracket.")

    bp_col1, bp_col2 = st.columns([1, 3])
    with bp_col1:
        bracket_seed = st.number_input("Bracket Seed", value=42, min_value=0, max_value=999999, key="bracket_seed")
        gen_bracket = st.button("Generate Bracket", type="primary", key="gen_bracket")

    if gen_bracket or "bracket" not in st.session_state:
        bracket = simulate_single_bracket(
            profiles=TEAM_PROFILES,
            weights=st.session_state.get("user_weights", None),
            seed=bracket_seed,
        )
        st.session_state["bracket"] = bracket

    bracket = st.session_state.get("bracket")
    if bracket:
        champion = bracket["champion"]
        with bp_col2:
            st.markdown(f"""
            <div class="stat-card" style="text-align:center;">
                <div class="stat-label">PREDICTED CHAMPION</div>
                <div class="stat-value" style="font-size:2.8rem;">{champion}</div>
            </div>
            """, unsafe_allow_html=True)

        # Group Stage Results
        st.markdown("#### Group Stage Results")
        grp_cols = st.columns(4)
        for idx, (grp_letter, standings) in enumerate(sorted(bracket["group_results"].items())):
            with grp_cols[idx % 4]:
                rows_html = ""
                for pos, s in enumerate(standings):
                    marker = "" if pos >= 2 else " style='color:#4ECDC4;font-weight:600;'"
                    if pos == 2 and s["team"] in [r["winner"] for r in bracket.get("r32_results", []) if True]:
                        marker = " style='color:#FFEAA7;'"
                    rows_html += f"<div class='group-team'{marker}><span>{s['team']}</span><span class='group-team-elo'>{s['points']}pts | GD:{s['gd']:+d}</span></div>"
                st.markdown(f"""
                <div class="group-card">
                    <div class="group-letter">Group {grp_letter}</div>
                    {rows_html}
                </div>
                """, unsafe_allow_html=True)

        # Knockout Bracket Visualization
        st.markdown("#### Knockout Bracket")

        def render_match(result, round_name):
            t1, t2 = result["team1"], result["team2"]
            g1, g2 = result["goals1"], result["goals2"]
            w = result["winner"]
            pen = " (pen)" if result.get("penalties", False) else ""
            w1_style = "color:#D4AF37;font-weight:700;" if w == t1 else "color:rgba(232,232,232,0.5);"
            w2_style = "color:#D4AF37;font-weight:700;" if w == t2 else "color:rgba(232,232,232,0.5);"
            return f"""
            <div style="background:rgba(26,31,46,0.8);border:1px solid rgba(212,175,55,0.1);border-radius:10px;padding:0.6rem 1rem;margin:0.3rem 0;font-family:'Inter',sans-serif;font-size:0.85rem;">
                <div style="{w1_style}">{t1} <span style="float:right;">{g1}</span></div>
                <div style="{w2_style}">{t2} <span style="float:right;">{g2}</span></div>
                <div style="color:rgba(232,232,232,0.3);font-size:0.7rem;text-align:center;">{round_name}{pen}</div>
            </div>
            """

        # R32
        st.markdown("##### Round of 32")
        r32_cols = st.columns(4)
        for i, result in enumerate(bracket["r32_results"]):
            with r32_cols[i % 4]:
                st.markdown(render_match(result, "R32"), unsafe_allow_html=True)

        # R16
        st.markdown("##### Round of 16")
        r16_cols = st.columns(4)
        for i, result in enumerate(bracket["r16_results"]):
            with r16_cols[i % 4]:
                st.markdown(render_match(result, "R16"), unsafe_allow_html=True)

        # QF
        st.markdown("##### Quarterfinals")
        qf_cols = st.columns(4)
        for i, result in enumerate(bracket["qf_results"]):
            with qf_cols[i]:
                st.markdown(render_match(result, "QF"), unsafe_allow_html=True)

        # SF
        st.markdown("##### Semifinals")
        sf_cols = st.columns(2)
        for i, result in enumerate(bracket["sf_results"]):
            with sf_cols[i]:
                st.markdown(render_match(result, "SF"), unsafe_allow_html=True)

        # Final
        st.markdown("##### FINAL")
        final_r = bracket["final_result"]
        st.markdown(render_match(final_r, "FINAL"), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;">
            <span style="font-family:'Outfit',sans-serif;font-size:2rem;color:#D4AF37;font-weight:800;">&#127942; {champion} &#127942;</span>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 3: GROUP ANALYSIS
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Group Stage Breakdown</div>', unsafe_allow_html=True)

    # Group selector
    selected_groups = st.multiselect(
        "Select groups to analyze",
        options=sorted(GROUPS.keys()),
        default=sorted(GROUPS.keys())[:4],
        key="group_select",
    )

    if selected_groups:
        # Create a grid of group charts
        n_cols = min(len(selected_groups), 3)
        cols = st.columns(n_cols)

        for idx, grp in enumerate(selected_groups):
            with cols[idx % n_cols]:
                grp_teams = GROUPS[grp]
                grp_data = results_df[results_df["Team"].isin(grp_teams)].sort_values(
                    "Group Stage %", ascending=False
                )

                fig_grp = go.Figure()

                fig_grp.add_trace(go.Bar(
                    x=grp_data["Team"],
                    y=grp_data["Group Stage %"],
                    name="Advance",
                    marker_color="#4ECDC4",
                    text=[f"{v:.0f}%" for v in grp_data["Group Stage %"]],
                    textposition="outside",
                    textfont=dict(size=11, color="#4ECDC4"),
                ))

                fig_grp.add_trace(go.Bar(
                    x=grp_data["Team"],
                    y=grp_data["Win World Cup %"],
                    name="Win WC",
                    marker_color="#D4AF37",
                    text=[f"{v:.1f}%" for v in grp_data["Win World Cup %"]],
                    textposition="outside",
                    textfont=dict(size=11, color="#D4AF37"),
                ))

                fig_grp.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(
                        text=f"Group {grp}",
                        font=dict(size=16, family="Outfit", color="#D4AF37"),
                    ),
                    height=350,
                    barmode="group",
                    showlegend=(idx == 0),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=-0.25,
                        font=dict(size=10),
                    ),
                    xaxis=dict(tickfont=dict(size=10)),
                    yaxis=dict(title="Probability (%)", range=[0, 100]),
                )
                st.plotly_chart(fig_grp, width="stretch")

        # Elo distribution across groups
        st.markdown("#### Average Group Strength (Elo)")

        group_elos = []
        for grp_letter, teams in sorted(GROUPS.items()):
            avg_elo = np.mean([ELO_RATINGS[t] for t in teams])
            max_elo = max(ELO_RATINGS[t] for t in teams)
            min_elo = min(ELO_RATINGS[t] for t in teams)
            group_elos.append({"Group": grp_letter, "Avg Elo": avg_elo, "Max": max_elo, "Min": min_elo})

        elo_df = pd.DataFrame(group_elos).sort_values("Avg Elo", ascending=False)

        fig_elo = go.Figure()

        # Error bar style showing range
        fig_elo.add_trace(go.Bar(
            x=[f"Group {g}" for g in elo_df["Group"]],
            y=elo_df["Avg Elo"],
            marker=dict(
                color=elo_df["Avg Elo"],
                colorscale=[[0, "#2d1b69"], [0.5, "#6C63FF"], [1, "#D4AF37"]],
                line=dict(color="rgba(212, 175, 55, 0.3)", width=1),
            ),
            text=[f"{v:.0f}" for v in elo_df["Avg Elo"]],
            textposition="outside",
            textfont=dict(color="#D4AF37", size=12, family="Outfit"),
            error_y=dict(
                type="data",
                symmetric=False,
                array=elo_df["Max"] - elo_df["Avg Elo"],
                arrayminus=elo_df["Avg Elo"] - elo_df["Min"],
                color="rgba(212, 175, 55, 0.4)",
                thickness=2,
                width=6,
            ),
        ))

        fig_elo.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Group Strength Comparison (Avg Elo with Min/Max Range)",
                       font=dict(size=16, family="Outfit", color="#D4AF37")),
            height=400,
            yaxis=dict(title="Elo Rating", range=[1550, 2150]),
            showlegend=False,
        )
        st.plotly_chart(fig_elo, width="stretch")


# ──────────────────────────────────────────────────────────────
# TAB 4: TEAM PROFILES (Enhanced Deep-Dive)
# ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Team Profiles & Analysis</div>', unsafe_allow_html=True)

    dd_col1, dd_col2 = st.columns([1, 2])

    with dd_col1:
        selected_team = st.selectbox(
            "Select a Team",
            results_df["Team"].tolist(),
            key="dd_team",
        )

    team_row = results_df[results_df["Team"] == selected_team].iloc[0]
    team_elo = ELO_RATINGS[selected_team]
    profile = TEAM_PROFILES.get(selected_team, {})

    with dd_col2:
        st.markdown(f"""
        <div class="stat-card">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div class="stat-label">Group Stage</div>
                    <div class="stat-value">{team_row['Group Stage %']:.1f}%</div>
                </div>
                <div>
                    <div class="stat-label">Quarterfinals</div>
                    <div class="stat-value">{team_row['Quarterfinals %']:.1f}%</div>
                </div>
                <div>
                    <div class="stat-label">Semifinals</div>
                    <div class="stat-value">{team_row['Semifinals %']:.1f}%</div>
                </div>
                <div>
                    <div class="stat-label">Final</div>
                    <div class="stat-value">{team_row['Final %']:.1f}%</div>
                </div>
                <div>
                    <div class="stat-label">Win World Cup</div>
                    <div class="stat-value">{team_row['Win World Cup %']:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if profile:
        # Team Info Cards Row
        info_c1, info_c2, info_c3, info_c4 = st.columns(4)
        with info_c1:
            st.metric("Elo Rating", f"{team_elo:,}")
        with info_c2:
            st.metric("Group", f"Group {team_row['Group']}")
        with info_c3:
            st.metric("Manager", profile.get("manager", "N/A"))
        with info_c4:
            st.metric("Formation", profile.get("formation", "N/A"))

        # Econometric & Climate Info Cards Row
        econ_c1, econ_c2, econ_c3 = st.columns(3)
        with econ_c1:
            gdp_val = profile.get("gdp_pc", 0)
            st.metric("GDP per Capita (USD)", f"${gdp_val:,.0f}" if gdp_val else "N/A")
        with econ_c2:
            pop_val = profile.get("population", 0)
            st.metric("Population (Millions)", f"{pop_val:.1f}M" if pop_val else "N/A")
        with econ_c3:
            temp_val = profile.get("avg_temp", 14.0)
            st.metric("Average Temperature", f"{temp_val:.1f}°C" if temp_val is not None else "N/A")

        # Style
        st.markdown(f"**Playing Style:** {profile.get('style', 'N/A')}")

        # Radar Chart — 11 Factor Spider Chart
        st.markdown("#### Factor Radar Chart")
        
        # Calculate Klement normalizations for display in radar chart (scaled 0-100)
        norm_gdp = np.log(max(profile.get("gdp_pc", 1000.0), 10.0)) / np.log(90000.0)
        norm_pop = np.log(max(profile.get("population", 1.0), 0.01)) / np.log(350.0)
        norm_temp = np.exp(-((profile.get("avg_temp", 14.0) - 14.0) ** 2) / 100.0)
        norm_gdp = min(max(norm_gdp, 0.0), 1.0)
        norm_pop = min(max(norm_pop, 0.0), 1.0)

        radar_keys = [
            "elo", "star_player_rating", "squad_depth", "tournament_pedigree",
            "offensive_rating", "defensive_rating", "manager_experience", "recent_form",
            "gdp_pc", "population", "avg_temp"
        ]
        radar_labels = [FACTOR_LABELS.get(k, k) for k in radar_keys]
        radar_values = []
        for k in radar_keys:
            if k == "elo":
                val = min(100.0, (profile.get("elo", 1500) / 2200.0) * 100.0)
            elif k == "recent_form":
                val = profile.get(k, 0.5) * 100.0
            elif k == "gdp_pc":
                val = norm_gdp * 100.0
            elif k == "population":
                val = norm_pop * 100.0
            elif k == "avg_temp":
                val = norm_temp * 100.0
            else:
                val = profile.get(k, 0.0)
            radar_values.append(val)
        radar_values.append(radar_values[0])  # close the polygon
        radar_labels.append(radar_labels[0])

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_labels,
            fill='toself',
            fillcolor='rgba(108, 99, 255, 0.2)',
            line=dict(color='#6C63FF', width=2),
            marker=dict(color='#D4AF37', size=6),
            name=selected_team,
        ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                                tickfont=dict(size=9, color="rgba(232,232,232,0.4)"),
                                gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(tickfont=dict(size=11, color="#E8E8E8"),
                                 gridcolor="rgba(255,255,255,0.08)"),
            ),
            showlegend=False,
            height=420,
            title=dict(text=f"{selected_team} — 11-Factor Profile",
                       font=dict(size=16, family="Outfit", color="#D4AF37")),
        )
        st.plotly_chart(fig_radar, width="stretch")

        # Key Players and Strengths/Weaknesses side by side
        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown("#### Key Players")
            players = profile.get("key_players", [])
            for p in players:
                rating_color = "#D4AF37" if p.get('rating', 0) >= 85 else "#6C63FF" if p.get('rating', 0) >= 78 else "#4ECDC4"
                st.markdown(f"""
                <div style="background:rgba(26,31,46,0.8);border:1px solid rgba(212,175,55,0.08);border-radius:10px;padding:0.6rem 1rem;margin:0.4rem 0;display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-family:'Inter',sans-serif;font-weight:600;color:#E8E8E8;">{p['name']}</span>
                        <span style="color:rgba(232,232,232,0.4);font-size:0.8rem;margin-left:0.5rem;">{p['position']}</span>
                    </div>
                    <span style="font-family:'Outfit',sans-serif;font-weight:700;color:{rating_color};font-size:1.2rem;">{p.get('rating', 'N/A')}</span>
                </div>
                """, unsafe_allow_html=True)

        with pc2:
            st.markdown("#### Strengths")
            for s in profile.get("strengths", []):
                st.markdown(f"""
                <div style="background:rgba(0,168,107,0.08);border-left:3px solid #00A86B;padding:0.5rem 0.8rem;margin:0.3rem 0;border-radius:0 8px 8px 0;font-family:'Inter',sans-serif;font-size:0.85rem;color:rgba(232,232,232,0.85);">
                    {s}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Weaknesses")
            for w in profile.get("weaknesses", []):
                st.markdown(f"""
                <div style="background:rgba(255,107,107,0.08);border-left:3px solid #FF6B6B;padding:0.5rem 0.8rem;margin:0.3rem 0;border-radius:0 8px 8px 0;font-family:'Inter',sans-serif;font-size:0.85rem;color:rgba(232,232,232,0.85);">
                    {w}
                </div>
                """, unsafe_allow_html=True)

        # World Cup History
        wc = profile.get("wc_history", {})
        if wc:
            st.markdown("#### World Cup History")
            wc_c1, wc_c2, wc_c3 = st.columns(3)
            with wc_c1:
                st.metric("Appearances", wc.get("appearances", "N/A"))
            with wc_c2:
                st.metric("Best Finish", wc.get("best", "N/A"))
            with wc_c3:
                st.metric("Titles", wc.get("titles", 0))

    # Probability funnel
    st.markdown("#### Advancement Funnel")
    stages_data = {
        "Stage": ["Group Stage", "Quarterfinals", "Semifinals", "Final", "Champion"],
        "Probability": [
            team_row["Group Stage %"],
            team_row["Quarterfinals %"],
            team_row["Semifinals %"],
            team_row["Final %"],
            team_row["Win World Cup %"],
        ],
    }

    fig_funnel = go.Figure(go.Funnel(
        y=stages_data["Stage"],
        x=stages_data["Probability"],
        textinfo="value+percent initial",
        texttemplate="%{value:.1f}%",
        marker=dict(
            color=["#4ECDC4", "#6C63FF", "#FF6B6B", "#FF69B4", "#D4AF37"],
            line=dict(width=2, color="rgba(255,255,255,0.1)"),
        ),
        connector=dict(line=dict(color="rgba(212, 175, 55, 0.2)", width=2)),
        textfont=dict(family="Outfit", size=14, color="white"),
    ))

    fig_funnel.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{selected_team} -- Tournament Path Probabilities",
            font=dict(size=18, family="Outfit", color="#D4AF37"),
        ),
        height=400,
    )
    st.plotly_chart(fig_funnel, width="stretch")

    # Team context
    st.markdown("#### Team Profile")
    ctx_col1, ctx_col2, ctx_col3, ctx_col4 = st.columns(4)
    with ctx_col1:
        st.metric("Elo Rating", f"{team_elo:,}")
    with ctx_col2:
        st.metric("Group", f"Group {team_row['Group']}")
    with ctx_col3:
        # Elo percentile
        all_elos = sorted(ELO_RATINGS.values())
        percentile = (sum(1 for e in all_elos if e <= team_elo) / len(all_elos)) * 100
        st.metric("Elo Percentile", f"{percentile:.0f}th")
    with ctx_col4:
        # Group mates
        group_teams = GROUPS[team_row["Group"]]
        rivals = [t for t in group_teams if t != selected_team]
        avg_rival_elo = np.mean([ELO_RATINGS[t] for t in rivals])
        st.metric("Avg Rival Elo", f"{avg_rival_elo:.0f}")

    # Expected goals matchup chart
    st.markdown("#### Expected Goals vs Group Opponents")

    group_mates = [t for t in GROUPS[team_row["Group"]] if t != selected_team]
    matchup_data = []
    for opp in group_mates:
        lam_for, lam_against = calculate_expected_goals_composite(
            selected_team, opp, TEAM_PROFILES, weights=user_weights
        )
        matchup_data.append({
            "Opponent": opp,
            "xG For": round(lam_for, 2),
            "xG Against": round(lam_against, 2),
            "xG Diff": round(lam_for - lam_against, 2),
        })

    matchup_df = pd.DataFrame(matchup_data)

    fig_xg = go.Figure()
    fig_xg.add_trace(go.Bar(
        x=matchup_df["Opponent"], y=matchup_df["xG For"],
        name=f"{selected_team} xG", marker_color="#4ECDC4",
    ))
    fig_xg.add_trace(go.Bar(
        x=matchup_df["Opponent"], y=matchup_df["xG Against"],
        name="Opponent xG", marker_color="#FF6B6B",
    ))

    fig_xg.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Expected Goals per Match -- {selected_team}",
                   font=dict(size=16, family="Outfit", color="#D4AF37")),
        barmode="group",
        height=350,
        yaxis=dict(title="Expected Goals"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_xg, width="stretch")

    st.dataframe(matchup_df, width="stretch", hide_index=True)


# ──────────────────────────────────────────────────────────────
# TAB 5: HEAD-TO-HEAD SIMULATOR
# ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Head-to-Head Match Simulator</div>', unsafe_allow_html=True)
    st.markdown("Simulate a single match between any two teams using the Poisson engine.")

    h2h_col1, h2h_col2 = st.columns(2)
    all_team_names = sorted(ELO_RATINGS.keys())

    with h2h_col1:
        h2h_team1 = st.selectbox("Team 1", all_team_names, index=all_team_names.index("Argentina"), key="h2h1")
    with h2h_col2:
        h2h_team2 = st.selectbox("Team 2", all_team_names, index=all_team_names.index("Brazil"), key="h2h2")

    if h2h_team1 != h2h_team2:
        h2h_sims = st.slider("Number of simulated matches", 1000, 50000, 10000, step=1000, key="h2h_n")

        if st.button("Simulate Head-to-Head", key="h2h_btn"):
            rng = np.random.default_rng(42)
            wins1, wins2, draws = 0, 0, 0
            goals1_total, goals2_total = [], []

            for _ in range(h2h_sims):
                result = simulate_match(
                    h2h_team1, h2h_team2, ELO_RATINGS, allow_draw=True, rng=rng,
                    profiles=TEAM_PROFILES, weights=user_weights
                )
                goals1_total.append(result["goals1"])
                goals2_total.append(result["goals2"])
                if result["goals1"] > result["goals2"]:
                    wins1 += 1
                elif result["goals2"] > result["goals1"]:
                    wins2 += 1
                else:
                    draws += 1

            w1_pct = wins1 / h2h_sims * 100
            w2_pct = wins2 / h2h_sims * 100
            d_pct = draws / h2h_sims * 100

            # Results metrics
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.metric(f"{h2h_team1} Wins", f"{w1_pct:.1f}%")
            with r_col2:
                st.metric("Draw", f"{d_pct:.1f}%")
            with r_col3:
                st.metric(f"{h2h_team2} Wins", f"{w2_pct:.1f}%")

            # Stacked bar for outcome
            fig_h2h = go.Figure()
            fig_h2h.add_trace(go.Bar(
                x=[w1_pct], y=["Match Result"], orientation='h',
                name=h2h_team1, marker_color="#6C63FF",
                text=f"{w1_pct:.1f}%", textposition="inside",
                textfont=dict(size=14, family="Outfit", color="white"),
            ))
            fig_h2h.add_trace(go.Bar(
                x=[d_pct], y=["Match Result"], orientation='h',
                name="Draw", marker_color="#555",
                text=f"{d_pct:.1f}%", textposition="inside",
                textfont=dict(size=14, family="Outfit", color="white"),
            ))
            fig_h2h.add_trace(go.Bar(
                x=[w2_pct], y=["Match Result"], orientation='h',
                name=h2h_team2, marker_color="#D4AF37",
                text=f"{w2_pct:.1f}%", textposition="inside",
                textfont=dict(size=14, family="Outfit", color="#1a1f2e"),
            ))
            fig_h2h.update_layout(
                **PLOTLY_LAYOUT,
                barmode="stack",
                height=150,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.6),
                yaxis=dict(visible=False),
                xaxis=dict(visible=False, range=[0, 100]),
            )
            st.plotly_chart(fig_h2h, width="stretch")

            # Goal distribution histograms
            st.markdown("#### Goal Distribution")
            g_col1, g_col2 = st.columns(2)

            with g_col1:
                fig_g1 = go.Figure()
                fig_g1.add_trace(go.Histogram(
                    x=goals1_total, nbinsx=max(goals1_total) + 1,
                    marker=dict(color="rgba(108, 99, 255, 0.6)", line=dict(color="#6C63FF", width=1)),
                    name=h2h_team1,
                ))
                avg_g1 = np.mean(goals1_total)
                fig_g1.add_vline(x=avg_g1, line_dash="dash", line_color="#D4AF37",
                                 annotation_text=f"Avg: {avg_g1:.2f}",
                                 annotation_font_color="#D4AF37")
                fig_g1.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text=f"{h2h_team1} Goals", font=dict(size=14, family="Outfit", color="#6C63FF")),
                    height=300,
                    xaxis_title="Goals",
                    yaxis_title="Frequency",
                )
                st.plotly_chart(fig_g1, width="stretch")

            with g_col2:
                fig_g2 = go.Figure()
                fig_g2.add_trace(go.Histogram(
                    x=goals2_total, nbinsx=max(goals2_total) + 1,
                    marker=dict(color="rgba(212, 175, 55, 0.6)", line=dict(color="#D4AF37", width=1)),
                    name=h2h_team2,
                ))
                avg_g2 = np.mean(goals2_total)
                fig_g2.add_vline(x=avg_g2, line_dash="dash", line_color="#6C63FF",
                                 annotation_text=f"Avg: {avg_g2:.2f}",
                                 annotation_font_color="#6C63FF")
                fig_g2.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text=f"{h2h_team2} Goals", font=dict(size=14, family="Outfit", color="#D4AF37")),
                    height=300,
                    xaxis_title="Goals",
                    yaxis_title="Frequency",
                )
                st.plotly_chart(fig_g2, width="stretch")

            # Most common scorelines
            st.markdown("#### Most Common Scorelines")
            from collections import Counter
            scorelines = Counter(zip(goals1_total, goals2_total))
            top_scores = scorelines.most_common(10)
            score_df = pd.DataFrame([
                {"Scoreline": f"{h2h_team1} {g1} - {g2} {h2h_team2}",
                 "Frequency": count,
                 "Probability": f"{count / h2h_sims * 100:.1f}%"}
                for (g1, g2), count in top_scores
            ])
            st.dataframe(score_df, width="stretch", hide_index=True)

    else:
        st.info("Select two different teams to simulate a head-to-head match.")


# ──────────────────────────────────────────────────────────────
# TAB 6: KNOCKOUT SCENARIOS
# ──────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Knockout Stage Analysis</div>', unsafe_allow_html=True)

    # Quarterfinal probability treemap
    st.markdown("#### Quarterfinal Contenders")

    qf_data = results_df[results_df["Quarterfinals %"] > 0].copy()
    qf_data = qf_data.sort_values("Quarterfinals %", ascending=False).head(24)

    fig_tree = go.Figure(go.Treemap(
        labels=qf_data["Team"],
        parents=["" for _ in range(len(qf_data))],
        values=qf_data["Quarterfinals %"],
        text=[f"{v:.1f}%" for v in qf_data["Quarterfinals %"]],
        textinfo="label+text",
        textfont=dict(family="Outfit", size=14),
        marker=dict(
            colors=qf_data["Quarterfinals %"],
            colorscale=[[0, "#1a0a2e"], [0.3, "#2d1b69"], [0.6, "#6C63FF"], [1.0, "#D4AF37"]],
            line=dict(color="rgba(0,0,0,0.3)", width=2),
        ),
    ))

    fig_tree.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Quarterfinal Probability (Top 24 Teams)",
                   font=dict(size=18, family="Outfit", color="#D4AF37")),
        height=500,
    )
    st.plotly_chart(fig_tree, width="stretch")

    # Championship contender scatter
    st.markdown("#### Championship Contenders: Elo vs Win Probability")

    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=results_df["Elo"],
        y=results_df["Win World Cup %"],
        mode="markers+text",
        text=results_df["Team"],
        textposition="top center",
        textfont=dict(size=9, color="rgba(232,232,232,0.7)", family="Inter"),
        marker=dict(
            size=results_df["Win World Cup %"] * 3 + 6,
            color=results_df["Win World Cup %"],
            colorscale=[[0, "#1a0a2e"], [0.3, "#6C63FF"], [0.7, "#FF6B6B"], [1.0, "#D4AF37"]],
            colorbar=dict(title="Win %", tickfont=dict(color="#E8E8E8")),
            line=dict(color="rgba(212, 175, 55, 0.3)", width=1),
        ),
        hovertemplate="<b>%{text}</b><br>Elo: %{x}<br>Win %: %{y:.1f}%<extra></extra>",
    ))

    fig_scatter.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Elo Rating vs Championship Probability",
                   font=dict(size=18, family="Outfit", color="#D4AF37")),
        xaxis=dict(title="Elo Rating", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Win World Cup %", gridcolor="rgba(255,255,255,0.05)"),
        height=550,
        showlegend=False,
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # Deep run probability heatmap
    st.markdown("#### Deep Run Probability Heatmap (Top 20)")

    top20 = results_df.head(20)
    heatmap_cols = ["Group Stage %", "Quarterfinals %", "Semifinals %", "Final %", "Win World Cup %"]
    heatmap_labels = ["Groups", "QF", "SF", "Final", "Win"]

    fig_heat = go.Figure(go.Heatmap(
        z=top20[heatmap_cols].values,
        x=heatmap_labels,
        y=top20["Team"].values,
        colorscale=[[0, "#0d1025"], [0.2, "#1a0a2e"], [0.4, "#2d1b69"], [0.6, "#6C63FF"], [0.8, "#FF6B6B"], [1, "#D4AF37"]],
        text=[[f"{v:.1f}" for v in row] for row in top20[heatmap_cols].values],
        texttemplate="%{text}%",
        textfont=dict(size=10, family="Inter"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Prob %", tickfont=dict(color="#E8E8E8")),
    ))

    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Tournament Path Heatmap", font=dict(size=18, family="Outfit", color="#D4AF37")),
        height=600,
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis=dict(side="top", tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_heat, width="stretch")
