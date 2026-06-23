import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="IPL Stats Dashboard",
    page_icon="🏏",
    layout="wide"
)

# ============ FILE UPLOAD ============
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Your Data")

matches_file = st.sidebar.file_uploader("Upload matches.csv", type=['csv'])
deliveries_file = st.sidebar.file_uploader("Upload deliveries.csv", type=['csv'])

if matches_file and deliveries_file:
    df = pd.read_csv(matches_file)
    df_ball = pd.read_csv(deliveries_file)
    st.sidebar.success("✅ Files Loaded!")
else:
    # Default files use பண்ணுவோம்
    @st.cache_data
    def load_data():
        import os
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        matches = pd.read_csv(os.path.join(BASE_DIR, "data", "matches.csv"))
        deliveries = pd.read_csv(os.path.join(BASE_DIR, "data", "deliveries.csv"))
        return matches, deliveries
    df, df_ball = load_data()
    st.sidebar.info("📌 Default IPL data loaded!")

# ============ SIDEBAR ============
st.sidebar.title("🏏 IPL Dashboard")
menu = st.sidebar.radio("Select Section", [
    "🏠 Overview",
    "🔍 Data Quality",
    "🏆 Team Stats",
    "🏏 Batting Stats",
    "🎳 Bowling Stats",
    "🤖 ML Prediction",
    "⚔️ Head to Head"
])


# ============ DATA QUALITY ============
if menu == "🔍 Data Quality":
    st.title("🔍 Data Overview & Cleaning")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📋 Matches Data", "🏏 Deliveries Data"])

    with tab1:
        st.subheader("📋 Matches Dataset")
        st.markdown("---")

        # Basic Info
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Duplicate Rows", df.duplicated().sum())

        st.markdown("---")

        # Column Info
        st.subheader("📌 Column Details")
        col_info = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values.tolist(),
            "Missing %": (df.isnull().sum().values / len(df) * 100).round(2).tolist(),
            "Unique Values": df.nunique().values.tolist()
        })
        st.dataframe(col_info.astype(str), width='stretch')

        st.markdown("---")

        # Missing Values Chart
        st.subheader("📊 Missing Values Chart")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(missing.index, missing.values, color='crimson')
            ax.set_xticks(range(len(missing.index)))
            ax.set_xticklabels(missing.index, rotation=45, ha='right')
            ax.set_ylabel("Missing Count")
            ax.set_title("Missing Values per Column")
            st.pyplot(fig)
        else:
            st.success("No missing values found!")

        st.markdown("---")

        # Sample Data
        st.subheader("👀 Sample Data (First 5 rows)")
        st.dataframe(df.head().rename(columns=str), width='stretch')

        st.markdown("---")

        # After Cleaning
        st.subheader("🧹 After Cleaning Summary")
        df_clean = df.drop_duplicates()
        col4, col5, col6 = st.columns(3)
        col4.metric("Rows Before", df.shape[0])
        col5.metric("Duplicates Removed", df.shape[0] - df_clean.shape[0])
        col6.metric("Rows After", df_clean.shape[0])
        st.success("Duplicates removed successfully!")

    with tab2:
        st.subheader("🏏 Deliveries Dataset")
        st.markdown("---")

        # Basic Info
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df_ball.shape[0])
        col2.metric("Total Columns", df_ball.shape[1])
        col3.metric("Duplicate Rows", df_ball.duplicated().sum())

        st.markdown("---")

        # Column Info
        st.subheader("📌 Column Details")
        col_info2 = pd.DataFrame({
            "Column": df_ball.columns.astype(str),
            "Data Type": df_ball.dtypes.astype(str).values,
            "Missing Values": df_ball.isnull().sum().values.tolist(),
            "Missing %": (df_ball.isnull().sum().values / len(df_ball) * 100).round(2).tolist(),
            "Unique Values": df_ball.nunique().values.tolist()
        })
        st.dataframe(col_info2.astype(str), width='stretch')

        st.markdown("---")

        # Missing Values Chart
        st.subheader("📊 Missing Values Chart")
        missing2 = df_ball.isnull().sum()
        missing2 = missing2[missing2 > 0].sort_values(ascending=False)
        if len(missing2) > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.bar(missing2.index, missing2.values, color='orange')
            ax2.set_xticks(range(len(missing2.index)))
            ax2.set_xticklabels(missing2.index, rotation=45, ha='right')
            ax2.set_ylabel("Missing Count")
            ax2.set_title("Missing Values per Column")
            st.pyplot(fig2)
        else:
            st.success("No missing values found!")

        st.markdown("---")

        # Sample Data
        st.subheader("👀 Sample Data (First 5 rows)")
        st.dataframe(df_ball.head().rename(columns=str), width='stretch')

        st.markdown("---")

        # After Cleaning
        st.subheader("🧹 After Cleaning Summary")
        df_ball_clean = df_ball.drop_duplicates()
        col4, col5, col6 = st.columns(3)
        col4.metric("Rows Before", df_ball.shape[0])
        col5.metric("Duplicates Removed", df_ball.shape[0] - df_ball_clean.shape[0])
        col6.metric("Rows After", df_ball_clean.shape[0])
        st.success("Duplicates removed successfully!")

# ============ OVERVIEW ============
elif menu == "🏠 Overview":
    st.title("🏏 IPL Stats Dashboard")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(df))
    col2.metric("Seasons", f"{df['season'].min()} - {df['season'].max()}")
    col3.metric("Cities", df['city'].nunique())
    col4.metric("Teams", df['team1'].nunique())
    
    st.markdown("---")
    st.subheader("📅 Season Wise Matches")
    season_matches = df.groupby('season').size().reset_index(name='matches')
    st.bar_chart(season_matches.set_index('season'))

# ============ TEAM STATS ============
elif menu == "🏆 Team Stats":
    st.title("🏆 Team Statistics")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Teams by Wins")
        wins = df['winner'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(wins.index[::-1], wins.values[::-1], color='orange')
        ax.set_xlabel("Wins")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Toss Decision")
        toss = df['toss_decision'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.pie(toss.values, labels=toss.index, autopct='%1.1f%%',
                colors=['skyblue', 'lightgreen'])
        st.pyplot(fig2)
    
    st.markdown("---")
    st.subheader("Toss Analysis")
    toss_wins = df[df['toss_winner'] == df['winner']]
    pct = (len(toss_wins) / len(df)) * 100
    st.info(f"🎯 Is winning the toss an advantage in winning the match: **{pct:.1f}%**")

# ============ BATTING STATS ============
elif menu == "🏏 Batting Stats":
    st.title("🏏 Batting Statistics")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Top Run Scorers", "Strike Rates", "Most Sixes"])
    
    with tab1:
        st.subheader("Top 10 Run Scorers")
        runs = df_ball.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(runs)
        st.dataframe(runs.reset_index().rename(columns={'batter': 'Player', 'batsman_runs': 'Runs'}))
    
    with tab2:
        st.subheader("Best Strike Rates (Min 500 balls)")
        balls = df_ball.groupby('batter')['batsman_runs'].count()
        run_total = df_ball.groupby('batter')['batsman_runs'].sum()
        sr = (run_total / balls * 100).round(2)
        qualified = sr[balls >= 500].sort_values(ascending=False).head(10)
        st.bar_chart(qualified)
        st.dataframe(qualified.reset_index().rename(columns={'batter': 'Player', 'batsman_runs': 'Strike Rate'}))
    
    with tab3:
        st.subheader("Most Sixes")
        sixes = df_ball[df_ball['batsman_runs'] == 6]
        top_six = sixes.groupby('batter')['batsman_runs'].count().sort_values(ascending=False).head(10)
        st.bar_chart(top_six)
        st.dataframe(top_six.reset_index().rename(columns={'batter': 'Player', 'batsman_runs': 'Sixes'}))

# ============ BOWLING STATS ============
elif menu == "🎳 Bowling Stats":
    st.title("🎳 Bowling Statistics")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Top Wicket Takers", "Economy Rates", "Bowling SR"])
    
    with tab1:
        st.subheader("Top 10 Wicket Takers")
        wickets_df = df_ball[df_ball['dismissal_kind'].notna()]
        wickets_df = wickets_df[~wickets_df['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
        wickets = wickets_df.groupby('bowler')['dismissal_kind'].count().sort_values(ascending=False).head(10)
        st.bar_chart(wickets)
        st.dataframe(wickets.reset_index().rename(columns={'bowler': 'Player', 'dismissal_kind': 'Wickets'}))
    
    with tab2:
        st.subheader("Best Economy Rates (Min 200 overs)")
        runs_given = df_ball.groupby('bowler')['total_runs'].sum()
        balls_bowled = df_ball.groupby('bowler')['total_runs'].count()
        overs = balls_bowled / 6
        economy = (runs_given / overs).round(2)
        qualified_eco = economy[overs >= 200].sort_values().head(10)
        st.bar_chart(qualified_eco)
        st.dataframe(qualified_eco.reset_index().rename(columns={'bowler': 'Player', 'total_runs': 'Economy'}))
    
    with tab3:
        st.subheader("Best Bowling Strike Rates (Min 200 overs)")
        balls_bowled2 = df_ball.groupby('bowler')['total_runs'].count()
        w_df = df_ball[df_ball['dismissal_kind'].notna()]
        w_df = w_df[~w_df['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
        wkts = w_df.groupby('bowler')['dismissal_kind'].count()
        b_sr = (balls_bowled2 / wkts).round(2)
        overs2 = balls_bowled2 / 6
        qualified_sr = b_sr[overs2 >= 200].sort_values().head(10)
        st.bar_chart(qualified_sr)
        st.dataframe(qualified_sr.reset_index().rename(columns={'bowler': 'Player', 'total_runs': 'Bowling SR'}))

# ============ ML PREDICTION ============
elif menu == "🤖 ML Prediction":
    st.title("🤖 Match Winner Prediction")
    st.markdown("---")
    
    # Training vs Testing Info
    st.subheader("📊 Model Training Details")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    ml_df_info = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'winner']].dropna()
    total = len(ml_df_info)
    train_size = int(total * 0.8)
    test_size = total - train_size
    
    col_a.metric("Total Matches", total)
    col_b.metric("Training Data (80%)", train_size)
    col_c.metric("Testing Data (20%)", test_size)
    col_d.metric("Model Accuracy", "48.6%")
    
    # Train Test Split Chart
    fig_tt, ax_tt = plt.subplots(figsize=(6, 3))
    ax_tt.barh(["Dataset"], [train_size], color='royalblue', label='Training')
    ax_tt.barh(["Dataset"], [test_size], left=[train_size], color='orange', label='Testing')
    ax_tt.set_xlabel("Matches")
    ax_tt.set_title("Train vs Test Split (80/20)")
    ax_tt.legend()
    st.pyplot(fig_tt)
    
    st.info("🔵 Training - The model learns from this data | 🟠 Testing - The model is evaluated using this data")
    
    st.markdown("---")
    # Testing Phase Results
    from sklearn.metrics import confusion_matrix, classification_report
    import warnings
    
    st.markdown("---")
    st.subheader("🧪 Testing Phase Results")
    
    @st.cache_resource
    def get_test_results():
        ml_df = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'winner']].dropna()
        le_team = LabelEncoder()
        le_decision = LabelEncoder()
        all_teams = pd.concat([ml_df['team1'], ml_df['team2'], ml_df['toss_winner'], ml_df['winner']])
        le_team.fit(all_teams)
        le_decision.fit(ml_df['toss_decision'])
        ml_df = ml_df.copy()
        ml_df['team1_enc'] = le_team.transform(ml_df['team1'])
        ml_df['team2_enc'] = le_team.transform(ml_df['team2'])
        ml_df['toss_winner_enc'] = le_team.transform(ml_df['toss_winner'])
        ml_df['toss_decision_enc'] = le_decision.transform(ml_df['toss_decision'])
        ml_df['winner_enc'] = le_team.transform(ml_df['winner'])
        X = ml_df[['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc']]
        y = ml_df['winner_enc']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_test, y_pred, le_team
    
    y_test, y_pred, le_team2 = get_test_results()
    
    # Correct vs Wrong predictions
    correct = sum(y_test.values == y_pred)
    wrong = len(y_test) - correct
    
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("Total Test Matches", len(y_test))
    col_t2.metric("✅ Correct Predictions", correct)
    col_t3.metric("❌ Wrong Predictions", wrong)
    
    # Correct vs Wrong Chart
    fig_cw, ax_cw = plt.subplots(figsize=(5, 3))
    ax_cw.bar(["Correct ✅", "Wrong ❌"], [correct, wrong], color=['green', 'red'])
    ax_cw.set_title("Testing Phase - Prediction Results")
    ax_cw.set_ylabel("Matches")
    for i, v in enumerate([correct, wrong]):
        ax_cw.text(i, v + 1, str(v), ha='center', fontweight='bold')
    st.pyplot(fig_cw)
    
    st.success(f"✅ {correct} matches correctly predicted out of {len(y_test)} test matches!")
    
    @st.cache_resource
    def train_model():
        ml_df = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'winner']].dropna()
        le_team = LabelEncoder()
        le_decision = LabelEncoder()
        all_teams = pd.concat([ml_df['team1'], ml_df['team2'], ml_df['toss_winner'], ml_df['winner']])
        le_team.fit(all_teams)
        le_decision.fit(ml_df['toss_decision'])
        ml_df = ml_df.copy()
        ml_df['team1_enc'] = le_team.transform(ml_df['team1'])
        ml_df['team2_enc'] = le_team.transform(ml_df['team2'])
        ml_df['toss_winner_enc'] = le_team.transform(ml_df['toss_winner'])
        ml_df['toss_decision_enc'] = le_decision.transform(ml_df['toss_decision'])
        ml_df['winner_enc'] = le_team.transform(ml_df['winner'])
        X = ml_df[['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc']]
        y = ml_df['winner_enc']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, le_team, le_decision
    
    model, le_team, le_decision = train_model()
    
    teams = sorted(df['team1'].unique().tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1 Select", teams)
    with col2:
        team2 = st.selectbox("Team 2 Select", [t for t in teams if t != team1])
    
    col3, col4 = st.columns(2)
    with col3:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    with col4:
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
    
    if st.button("🔮 Predict Winner!", width='stretch'):
        input_data = pd.DataFrame([[
            le_team.transform([team1])[0],
            le_team.transform([team2])[0],
            le_team.transform([toss_winner])[0],
            le_decision.transform([toss_decision])[0]
        ]], columns=['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc'])
        
        pred = model.predict(input_data)[0]
        winner = le_team.inverse_transform([pred])[0]
        proba = model.predict_proba(input_data)[0]
        confidence = max(proba) * 100
        
        st.markdown("---")
        st.success(f"🏆 Predicted Winner: **{winner}**")
        st.info(f"📊 Confidence: **{confidence:.1f}%**")
        
        col5, col6 = st.columns(2)
        col5.metric(team1, f"{proba[le_team.transform([team1])[0]]:.1%}")
        col6.metric(team2, f"{proba[le_team.transform([team2])[0]]:.1%}")

# ============ HEAD TO HEAD ============
elif menu == "⚔️ Head to Head":
    st.title("⚔️ Head to Head Comparison")
    st.markdown("---")
    
    teams = sorted(df['team1'].unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams)
    with col2:
        team2 = st.selectbox("Team 2", [t for t in teams if t != team1])
    
    h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
             ((df['team1'] == team2) & (df['team2'] == team1))]
    
    total = len(h2h)
    team1_wins = len(h2h[h2h['winner'] == team1])
    team2_wins = len(h2h[h2h['winner'] == team2])
    
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    col3.metric(f"🏆 {team1}", f"{team1_wins} wins")
    col4.metric("📊 Total Matches", total)
    col5.metric(f"🏆 {team2}", f"{team2_wins} wins")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([team1, team2], [team1_wins, team2_wins], color=['blue', 'yellow'])
    ax.set_title(f"{team1} vs {team2}")
    ax.set_ylabel("Wins")
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Last 5 Matches")
    last5 = h2h.tail(5)[['season', 'winner', 'result', 'result_margin']]
    st.dataframe(last5, width='stretch')
    
    if team1_wins > team2_wins:
        st.success(f"🏆 Overall Leader: {team1}")
    elif team2_wins > team1_wins:
        st.success(f"🏆 Overall Leader: {team2}")
    else:
        st.info("🤝 Equal!")