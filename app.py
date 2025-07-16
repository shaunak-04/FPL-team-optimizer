#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="FPL Team Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for FPL styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #00FF87 0%, #60EFFF 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: #2D0A31;
    }
    
    .player-card {
        background: linear-gradient(135deg, #37003C 0%, #00FF87 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .player-card:hover {
        transform: translateY(-5px);
    }
    
    .captain-badge {
        background: #FFD700;
        color: #37003C;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .vice-badge {
        background: #C0C0C0;
        color: #37003C;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .formation-display {
        background: linear-gradient(135deg, #37003C 0%, #00FF87 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        min-height: 600px;
        position: relative;
    }
    
    .pitch-lines {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            linear-gradient(to right, rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        border-radius: 15px;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #37003C 0%, #00FF87 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .bench-area {
        background: linear-gradient(135deg, #1E1E1E 0%, #37003C 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)



# Load models and data (with error handling)
@st.cache_resource
def load_models():
    """Load ML models"""
    try:
        models = {
            'gk': joblib.load("gk_model.pkl"),
            'def': joblib.load("def_model.pkl"),
            'mid': joblib.load("mid_model.pkl"),
            'fwd': joblib.load("fwd_model.pkl")
        }
        return models
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure all .pkl files are in the same directory.")
        return None

@st.cache_data
def load_data():
    """Load FPL data"""
    try:
        data = {
            'gk': pd.read_csv("fpl_gk_data.csv"),
            'def': pd.read_csv("fpl_def_data.csv"),
            'mid': pd.read_csv("fpl_mid_data.csv"),
            'fwd': pd.read_csv("fpl_fwd_data.csv")
        }
        return data
    except FileNotFoundError:
        st.error("❌ Data files not found. Please ensure all CSV files are in the same directory.")
        return None

def predict_for_gameweek(data, models, season, gameweek):
    """Predict player points for NEXT gameweek"""
    def prepare(df, model):
        current_gw_data = df[(df['season'] == season) & (df['gameweek'] == gameweek)].copy()
        
        if current_gw_data.empty:
            return pd.DataFrame(columns=['name', 'team', 'value', 'predicted_next_points', 'actual_next_points'])
        
        ids = current_gw_data[['name', 'team', 'value']].copy()
        
        if 'future_points' in current_gw_data.columns:
            ids['actual_next_points'] = current_gw_data['future_points']
        
        train_columns = [col for col in model.feature_names_in_ if col != 'future_points']
        missing_cols = set(train_columns) - set(current_gw_data.columns)
        
        if missing_cols:
            return pd.DataFrame(columns=['name', 'team', 'value', 'predicted_next_points', 'actual_next_points'])
        
        X = current_gw_data[train_columns].fillna(0)
        preds = model.predict(X)
        
        result = ids.copy()
        result['predicted_next_points'] = preds
        result['value'] = result['value'] / 10
        
        result['predicted_next_points'] = pd.to_numeric(result['predicted_next_points'], errors='coerce').fillna(0)
        result['value'] = pd.to_numeric(result['value'], errors='coerce').fillna(4.0)
        
        if 'actual_next_points' in result.columns:
            result['actual_next_points'] = pd.to_numeric(result['actual_next_points'], errors='coerce').fillna(0)
        
        result = result[np.isfinite(result['predicted_next_points']) & np.isfinite(result['value'])]
        return result.reset_index(drop=True)
    
    return (prepare(data['gk'], models['gk']), 
            prepare(data['def'], models['def']), 
            prepare(data['mid'], models['mid']), 
            prepare(data['fwd'], models['fwd']))

def optimize_fpl_team(gk, defs, mids, fwds, budget=100.0, formation='3-5-2'):
    """Optimize FPL team selection"""
    if any(df.empty for df in [gk, defs, mids, fwds]):
        return pd.DataFrame(), None, None
    
    all_players = pd.concat([
        gk.assign(position='GK'), defs.assign(position='DEF'),
        mids.assign(position='MID'), fwds.assign(position='FWD')
    ], ignore_index=True)
    
    def_start, mid_start, fwd_start = map(int, formation.split('-'))
    
    prob = LpProblem("FPL_Squad_Optimization", LpMaximize)
    n_players = len(all_players)
    
    squad_vars = [LpVariable(f"squad_{i}", cat=LpBinary) for i in range(n_players)]
    start_vars = [LpVariable(f"start_{i}", cat=LpBinary) for i in range(n_players)]
    captain_vars = [LpVariable(f"captain_{i}", cat=LpBinary) for i in range(n_players)]
    vice_vars = [LpVariable(f"vice_{i}", cat=LpBinary) for i in range(n_players)]
    
    prob += lpSum([
        start_vars[i] * float(all_players.iloc[i]['predicted_next_points']) +
        captain_vars[i] * float(all_players.iloc[i]['predicted_next_points'])
        for i in range(n_players)
    ])
    
    # Constraints
    prob += lpSum([squad_vars[i] * float(all_players.iloc[i]['value']) for i in range(n_players)]) <= budget
    prob += lpSum(squad_vars) == 15
    prob += lpSum(start_vars) == 11
    prob += lpSum(captain_vars) == 1
    prob += lpSum(vice_vars) == 1
    
    for i in range(n_players):
        prob += start_vars[i] <= squad_vars[i]
        prob += captain_vars[i] <= start_vars[i]
        prob += vice_vars[i] <= start_vars[i]
        prob += captain_vars[i] + vice_vars[i] <= 1
    
    # Position constraints
    gk_indices = [i for i in range(n_players) if all_players.iloc[i]['position'] == 'GK']
    def_indices = [i for i in range(n_players) if all_players.iloc[i]['position'] == 'DEF']
    mid_indices = [i for i in range(n_players) if all_players.iloc[i]['position'] == 'MID']
    fwd_indices = [i for i in range(n_players) if all_players.iloc[i]['position'] == 'FWD']
    
    prob += lpSum([squad_vars[i] for i in gk_indices]) == 2
    prob += lpSum([squad_vars[i] for i in def_indices]) == 5
    prob += lpSum([squad_vars[i] for i in mid_indices]) == 5
    prob += lpSum([squad_vars[i] for i in fwd_indices]) == 3
    
    prob += lpSum([start_vars[i] for i in gk_indices]) == 1
    prob += lpSum([start_vars[i] for i in def_indices]) == def_start
    prob += lpSum([start_vars[i] for i in mid_indices]) == mid_start
    prob += lpSum([start_vars[i] for i in fwd_indices]) == fwd_start
    
    # Max 3 players per team
    for team in all_players['team'].unique():
        team_indices = [i for i in range(n_players) if all_players.iloc[i]['team'] == team]
        prob += lpSum([squad_vars[i] for i in team_indices]) <= 3
    
    prob.solve()
    
    if prob.status != 1:
        return pd.DataFrame(), None, None
    
    # Extract results
    squad_indices = [i for i in range(n_players) if squad_vars[i].value() == 1]
    start_indices = [i for i in range(n_players) if start_vars[i].value() == 1]
    captain_index = [i for i in range(n_players) if captain_vars[i].value() == 1][0]
    vice_index = [i for i in range(n_players) if vice_vars[i].value() == 1][0]
    
    squad_df = all_players.iloc[squad_indices].copy()
    squad_df['in_starting_xi'] = squad_df.index.isin(start_indices)
    squad_df['is_captain'] = squad_df.index == captain_index
    squad_df['is_vice_captain'] = squad_df.index == vice_index
    
    squad_df['effective_predicted_points'] = squad_df['predicted_next_points'].copy()
    squad_df.loc[squad_df['is_captain'], 'effective_predicted_points'] *= 2
    
    squad_df = squad_df.sort_values(['in_starting_xi', 'position', 'predicted_next_points'], 
                                   ascending=[False, True, False]).reset_index(drop=True)
    
    return squad_df, all_players.iloc[captain_index], all_players.iloc[vice_index]

def create_formation_visualization(squad_df, formation):
    """Create interactive formation visualization"""
    if squad_df.empty:
        return None
    
    starting_xi = squad_df[squad_df['in_starting_xi']].copy()
    
    # Formation positions
    formation_positions = {
        '3-5-2': {
            'GK': [(0.5, 0.1)],
            'DEF': [(0.2, 0.3), (0.5, 0.3), (0.8, 0.3)],
            'MID': [(0.1, 0.55), (0.35, 0.55), (0.5, 0.65), (0.65, 0.55), (0.9, 0.55)],
            'FWD': [(0.35, 0.85), (0.65, 0.85)]
        },
        '3-4-3': {
            'GK': [(0.5, 0.1)],
            'DEF': [(0.2, 0.3), (0.5, 0.3), (0.8, 0.3)],
            'MID': [(0.25, 0.55), (0.45, 0.55), (0.55, 0.55), (0.75, 0.55)],
            'FWD': [(0.2, 0.85), (0.5, 0.85), (0.8, 0.85)]
        },
        '4-5-1': {
            'GK': [(0.5, 0.1)],
            'DEF': [(0.15, 0.3), (0.38, 0.3), (0.62, 0.3), (0.85, 0.3)],
            'MID': [(0.1, 0.55), (0.3, 0.55), (0.5, 0.65), (0.7, 0.55), (0.9, 0.55)],
            'FWD': [(0.5, 0.85)]
        },
        '4-4-2': {
            'GK': [(0.5, 0.1)],
            'DEF': [(0.15, 0.3), (0.38, 0.3), (0.62, 0.3), (0.85, 0.3)],
            'MID': [(0.2, 0.55), (0.45, 0.55), (0.55, 0.55), (0.8, 0.55)],
            'FWD': [(0.35, 0.85), (0.65, 0.85)]
        }
    }
    
    positions = formation_positions.get(formation, formation_positions['3-5-2'])
    
    fig = go.Figure()
    
    # Add pitch background
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        fillcolor="rgba(0, 255, 135, 0.1)",
        line=dict(color="rgba(255, 255, 255, 0.3)", width=2)
    )
    
    # Add players
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = starting_xi[starting_xi['position'] == pos]
        pos_positions = positions[pos]
        
        for i, (_, player) in enumerate(pos_players.iterrows()):
            if i < len(pos_positions):
                x, y = pos_positions[i]
                
                # Player circle
                color = "#FFD700" if player['is_captain'] else "#C0C0C0" if player['is_vice_captain'] else "#00FF87"
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color=color,
                        line=dict(color='white', width=2)
                    ),
                    text=player['name'].split()[0],
                    textposition="middle center",
                    textfont=dict(color='#37003C', size=10, family="Arial Black"),
                    hovertemplate=f"<b>{player['name']}</b><br>" +
                                f"Team: {player['team']}<br>" +
                                f"Value: £{player['value']:.1f}m<br>" +
                                f"Predicted: {player['predicted_next_points']:.1f} pts<br>" +
                                f"{'(C)' if player['is_captain'] else '(VC)' if player['is_vice_captain'] else ''}<extra></extra>",
                    showlegend=False
                ))
    
    fig.update_layout(
        title="Formation View",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def display_player_card(player, is_starting=True):
    """Display a player card"""
    badge = ""
    if player['is_captain']:
        badge = '<span class="captain-badge">C</span>'
    elif player['is_vice_captain']:
        badge = '<span class="vice-badge">VC</span>'
    
    card_html = f"""
    <div class="player-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: #00FF87;">{player['name']} {badge}</h4>
                <p style="margin: 0; opacity: 0.8;">{player['team']} | {player['position']}</p>
            </div>
            <div style="text-align: right;">
                <h3 style="margin: 0; color: #FFD700;">£{player['value']:.1f}m</h3>
                <p style="margin: 0; opacity: 0.8;">{player['predicted_next_points']:.1f} pts</p>
            </div>
        </div>
    </div>
    """
    return card_html

def create_stats_visualization(squad_df):
    """Create stats visualization"""
    if squad_df.empty:
        return None, None
    
    # Position distribution
    pos_counts = squad_df['position'].value_counts()
    fig_pos = px.pie(
        values=pos_counts.values,
        names=pos_counts.index,
        title="Squad Composition",
        color_discrete_map={
            'GK': '#FF6B6B',
            'DEF': '#4ECDC4',
            'MID': '#45B7D1',
            'FWD': '#96CEB4'
        }
    )
    fig_pos.update_traces(textposition='inside', textinfo='percent+label')
    fig_pos.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Team distribution
    team_counts = squad_df['team'].value_counts().head(10)
    fig_team = px.bar(
        x=team_counts.index,
        y=team_counts.values,
        title="Players by Team",
        color=team_counts.values,
        color_continuous_scale='Viridis'
    )
    fig_team.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="Team"),
        yaxis=dict(title="Number of Players")
    )
    
    return fig_pos, fig_team

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚽ FPL Team Predictor</h1>
        <p>AI-Powered Fantasy Premier League Team Selection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models
    models = load_models()
    data = load_data()
    
    if models is None or data is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎮 Team Configuration")
    
    season = st.sidebar.selectbox(
        "Season",
        options=["2024-25", "2023-24","2022-23"],
        index=0
    )
    
    current_gameweek = st.sidebar.slider(
        "Current Gameweek",
        min_value=1,
        max_value=38,
        value=7,
        help="Gameweek to use for predictions"
    )
    
    target_gameweek = st.sidebar.slider(
        "Target Gameweek",
        min_value=current_gameweek + 1,
        max_value=38,
        value=current_gameweek + 1,
        help="Gameweek to predict for"
    )
    
    budget = st.sidebar.slider(
        "Budget (£m)",
        min_value=80.0,
        max_value=120.0,
        value=100.0,
        step=0.5,
        help="Team budget in millions"
    )
    
    formation = st.sidebar.selectbox(
        "Formation",
        options=["3-5-2", "3-4-3", "4-5-1", "4-4-2"],
        index=0,
        help="Team formation"
    )
    
    # Generate team button
    if st.sidebar.button("🚀 Generate Optimal Team", type="primary"):
        with st.spinner("🔮 Predicting optimal team..."):
            # Get predictions
            gk_p, def_p, mid_p, fwd_p = predict_for_gameweek(
                data, models, season, current_gameweek
            )
            
            # Optimize team
            squad_df, captain, vice = optimize_fpl_team(
                gk_p, def_p, mid_p, fwd_p, budget, formation
            )
            
            # Store in session state
            st.session_state.squad_df = squad_df
            st.session_state.captain = captain
            st.session_state.vice = vice
            st.session_state.formation = formation
    
    # Display results
    if hasattr(st.session_state, 'squad_df') and not st.session_state.squad_df.empty:
        squad_df = st.session_state.squad_df
        captain = st.session_state.captain
        vice = st.session_state.vice
        formation = st.session_state.formation
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <h3>💰 Total Cost</h3>
                <h2>£{squad_df['value'].sum():.1f}m</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            predicted_points = squad_df[squad_df['in_starting_xi']]['effective_predicted_points'].sum()
            st.markdown(f"""
            <div class="stats-card">
                <h3>🔮 Predicted Points</h3>
                <h2>{predicted_points:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <h3>👑 Captain</h3>
                <h2>{captain['name']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stats-card">
                <h3>🥈 Vice Captain</h3>
                <h2>{vice['name']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Formation visualization
        st.subheader("🏟️ Formation View")
        formation_fig = create_formation_visualization(squad_df, formation)
        if formation_fig:
            st.plotly_chart(formation_fig, use_container_width=True)
        
        # Team details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Starting XI")
            starting_xi = squad_df[squad_df['in_starting_xi']]
            
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                pos_players = starting_xi[starting_xi['position'] == pos]
                if not pos_players.empty:
                    st.markdown(f"**{pos}:**")
                    for _, player in pos_players.iterrows():
                        st.markdown(display_player_card(player), unsafe_allow_html=True)
        
        with col2:
            st.subheader("🪑 Bench")
            bench = squad_df[~squad_df['in_starting_xi']]
            
            for _, player in bench.iterrows():
                st.markdown(display_player_card(player, is_starting=False), unsafe_allow_html=True)
        
        # Analytics
        st.subheader("📊 Team Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pos, fig_team = create_stats_visualization(squad_df)
            if fig_pos:
                st.plotly_chart(fig_pos, use_container_width=True)
        
        with col2:
            if fig_team:
                st.plotly_chart(fig_team, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Detailed Squad List")
        
        # Format display table
        display_df = squad_df.copy()
        display_df['Status'] = display_df.apply(
            lambda x: 'Captain' if x['is_captain'] else 'Vice' if x['is_vice_captain'] else 'Starting' if x['in_starting_xi'] else 'Bench',
            axis=1
        )
        
        display_cols = ['name', 'team', 'position', 'value', 'predicted_next_points', 'Status']
        display_df = display_df[display_cols]
        display_df.columns = ['Player', 'Team', 'Position', 'Value (£m)', 'Predicted Points', 'Status']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Export functionality
        st.subheader("💾 Export Team")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download as CSV"):
                csv = squad_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"fpl_team_gw{target_gameweek}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Copy Team to Clipboard"):
                team_text = f"FPL Team for GW{target_gameweek}\n"
                team_text += f"Formation: {formation}\n"
                team_text += f"Budget: £{squad_df['value'].sum():.1f}m\n\n"
                
                starting_xi = squad_df[squad_df['in_starting_xi']]
                team_text += "Starting XI:\n"
                for pos in ['GK', 'DEF', 'MID', 'FWD']:
                    pos_players = starting_xi[starting_xi['position'] == pos]
                    if not pos_players.empty:
                        team_text += f"{pos}: "
                        team_text += ", ".join([f"{p['name']} ({p['team']})" for _, p in pos_players.iterrows()])
                        team_text += "\n"
                
                st.text_area("Team Summary", team_text, height=200)
    
    else:
        st.info("👈 Configure your team settings in the sidebar and click 'Generate Optimal Team' to start!")
        
        # Show sample formation
        st.subheader("🏟️ Sample Formation")
        sample_fig = go.Figure()
        sample_fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=1, y1=1,
            fillcolor="rgba(0, 255, 135, 0.1)",
            line=dict(color="rgba(255, 255, 255, 0.3)", width=2)
        )
        sample_fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(sample_fig, use_container_width=True)

if __name__ == "__main__":
    main()

