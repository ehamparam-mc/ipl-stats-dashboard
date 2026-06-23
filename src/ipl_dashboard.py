import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ DATA LOAD ============
df = pd.read_csv("data/matches.csv")
df_ball = pd.read_csv("data/deliveries.csv")

# ============ FUNCTIONS ============ (Module 2 - Functions!)

def show_overview():
    print("\n========== IPL OVERVIEW ==========")
    print(f"📅 Seasons   : {df['season'].min()} to {df['season'].max()}")
    print(f"🏏 Total Matches : {len(df)}")
    print(f"🏙️  Cities    : {df['city'].nunique()}")
    print(f"🏟️  Venues    : {df['venue'].nunique()}")
    print(f"👑 Teams     : {df['team1'].nunique()}")

def show_team_wins():
    print("\n========== TOP TEAMS BY WINS ==========")
    # Module 4 - GroupBy + Sorting!
    wins = df['winner'].value_counts().head(10)
    for i, (team, count) in enumerate(wins.items(), 1):
        bar = "█" * (count // 10)
        print(f"{i:2}. {team:<35} {bar} {count}")

def show_toss_analysis():
    print("\n========== TOSS ANALYSIS ==========")
    # Module 7 - Statistics!
    toss_wins = df[df['toss_winner'] == df['winner']]
    percentage = (len(toss_wins) / len(df)) * 100
    print(f"🎯 Does winning the toss increase the chances of winning the match : {percentage:.1f}%")
    
    print("\nToss Decision Preference:")
    decision = df['toss_decision'].value_counts()
    for d, count in decision.items():
        print(f"  {d:<10} : {count} times")

def show_player_of_match():
    print("\n========== PLAYER OF THE MATCH ==========")
    # Module 4 - value_counts!
    top_players = df['player_of_match'].value_counts().head(10)
    for i, (player, count) in enumerate(top_players.items(), 1):
        stars = "⭐" * (count // 5)
        print(f"{i:2}. {player:<25} {count} awards {stars}")

def show_season_stats():
    print("\n========== SEASON WISE MATCHES ==========")
    # Module 4 - GroupBy!
    season_matches = df.groupby('season').size().reset_index(name='matches')
    for _, row in season_matches.iterrows():
        bar = "▓" * (row['matches'] // 3)
        print(f"  {str(row['season']):<10} : {bar} {row['matches']}")

def show_chart():
    print("\n📊 Chart window ...")
    wins = df['winner'].value_counts().head(8)
    
    plt.figure(figsize=(12, 5))
    
    # Bar Chart - Module 5!
    plt.subplot(1, 2, 1)
    plt.bar(wins.index, wins.values, color='orange')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 8 Teams by Wins')
    plt.ylabel('Wins')
    
    # Pie Chart - Module 5!
    plt.subplot(1, 2, 2)
    toss = df['toss_decision'].value_counts()
    plt.pie(toss.values, labels=toss.index, autopct='%1.1f%%', colors=['skyblue','lightgreen'])
    plt.title('Toss Decision')
    
    plt.tight_layout()
    plt.show()

# ============ MAIN MENU ============
def main():
    while True:
        print("\n" + "="*45)
        print("        🏏 IPL STATS DASHBOARD 🏏")
        print("="*45)
        print("  1. IPL Overview")
        print("  2. Top Teams by Wins")
        print("  3. Toss Analysis")
        print("  4. Player of the Match Leaders")
        print("  5. Season Wise Stats")
        print("  6. Show Charts 📊")
        print("  ── BATTING STATS ──")
        print("  7. Top Run Scorers 🏏")
        print("  8. Best Strike Rates ⚡")
        print("  9. Most Sixes 6️⃣")
        print("  10. Batting Chart 📊")
        print("  ── BOWLING STATS ──")
        print("  11. Top Wicket Takers 🎳")
        print("  12. Best Economy Rates 💰")
        print("  13. Best Bowling Strike Rates ⚡")
        print("  14. Bowling Chart 📊")
        print("  ── ML PREDICTION ──")
        print("  15. Match Winner Predict 🤖")
        print("  -- HEAD TO HEAD --")
        print("  16. Team vs Team Compare 🏆")
        print("  0. Exit")
        print("="*45)
        
        choice = input("  Enter your choice: ")
        
        if choice == '1':
            show_overview()
        elif choice == '2':
            show_team_wins()
        elif choice == '3':
            show_toss_analysis()
        elif choice == '4':
            show_player_of_match()
        elif choice == '5':
            show_season_stats()
        elif choice == '6':
            show_chart()
        elif choice == '7':
            show_top_batsmen()
        elif choice == '8':
            show_top_strikers()
        elif choice == '9':
            show_most_sixes()
        elif choice == '10':
            show_batting_chart()    
        elif choice == '11':
            show_top_wicket_takers()
        elif choice == '12':
            show_best_economy()
        elif choice == '13':
            show_best_bowling_sr()
        elif choice == '14':
            show_bowling_chart()   
        elif choice == '15':
            show_prediction()    
        elif choice == '16':
            show_head_to_head()    
        elif choice == '0':
            print("\n👋 Dashboard closed! ")
            break
        else:
            print("❌ Wrong choice! Please enter 0-6")

            
# ============ BATTING STATS ============
# Module 4 - Pandas!

def show_top_batsmen():
    print("\n========== TOP RUN SCORERS ==========")
    # Module 4 - GroupBy + Sorting!
    runs = df_ball.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
    for i, (player, run) in enumerate(runs.items(), 1):
        bar = "█" * (run // 200)
        print(f"{i:2}. {player:<25} {bar} {run}")

def show_top_strikers():
    print("\n========== BEST STRIKE RATES (Min 500 balls) ==========")
    # Module 7 - Statistics!
    balls = df_ball.groupby('batter')['batsman_runs'].count()
    runs = df_ball.groupby('batter')['batsman_runs'].sum()
    sr = (runs / balls * 100).round(2)
    # Filter - min 500 balls only
    qualified = sr[balls >= 500].sort_values(ascending=False).head(10)
    for i, (player, strike) in enumerate(qualified.items(), 1):
        print(f"{i:2}. {player:<25} SR: {strike}")

def show_most_sixes():
    print("\n========== MOST SIXES ==========")
    # Module 4 - Filtering!
    sixes = df_ball[df_ball['batsman_runs'] == 6]
    top_six = sixes.groupby('batter')['batsman_runs'].count().sort_values(ascending=False).head(10)
    for i, (player, count) in enumerate(top_six.items(), 1):
        balls = "6️⃣" * (count // 20)
        print(f"{i:2}. {player:<25} {balls} {count} sixes")

def show_batting_chart():
    print("\n📊 Batting Chart ...")
    runs = df_ball.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    plt.bar(runs.index, runs.values, color='royalblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Run Scorers in IPL')
    plt.ylabel('Total Runs')
    plt.tight_layout()
    plt.show()            

# ============ BOWLING STATS ============

def show_top_wicket_takers():
    print("\n========== TOP WICKET TAKERS ==========")
    # Module 4 - Filtering + GroupBy!
    wickets_df = df_ball[df_ball['dismissal_kind'].notna()]
    wickets_df = wickets_df[~wickets_df['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
    wickets = wickets_df.groupby('bowler')['dismissal_kind'].count().sort_values(ascending=False).head(10)
    for i, (player, count) in enumerate(wickets.items(), 1):
        bar = "🎳" * (count // 20)
        print(f"{i:2}. {player:<25} {bar} {count} wickets")

def show_best_economy():
    print("\n========== BEST ECONOMY RATES (Min 200 overs) ==========")
    # Module 7 - Statistics!
    runs_given = df_ball.groupby('bowler')['total_runs'].sum()
    balls_bowled = df_ball.groupby('bowler')['total_runs'].count()
    overs = balls_bowled / 6
    economy = (runs_given / overs).round(2)
    # Min 200 overs filter
    qualified = economy[overs >= 200].sort_values().head(10)
    for i, (player, eco) in enumerate(qualified.items(), 1):
        print(f"{i:2}. {player:<25} Economy: {eco}")

def show_best_bowling_sr():
    print("\n========== BEST BOWLING STRIKE RATES (Min 200 overs) ==========")
    # Module 7 - Statistics!
    balls_bowled = df_ball.groupby('bowler')['total_runs'].count()
    wickets_df = df_ball[df_ball['dismissal_kind'].notna()]
    wickets_df = wickets_df[~wickets_df['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
    wickets = wickets_df.groupby('bowler')['dismissal_kind'].count()
    bowling_sr = (balls_bowled / wickets).round(2)
    overs = balls_bowled / 6
    qualified = bowling_sr[overs >= 200].sort_values().head(10)
    for i, (player, sr) in enumerate(qualified.items(), 1):
        print(f"{i:2}. {player:<25} Bowling SR: {sr}")

def show_bowling_chart():
    print("\n📊 Bowling Chart ...")
    wickets_df = df_ball[df_ball['dismissal_kind'].notna()]
    wickets_df = wickets_df[~wickets_df['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
    wickets = wickets_df.groupby('bowler')['dismissal_kind'].count().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    plt.bar(wickets.index, wickets.values, color='crimson')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Wicket Takers in IPL')
    plt.ylabel('Total Wickets')
    plt.tight_layout()
    plt.show()

# ============ ML PREDICTION ============ (Module 9!)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_model():
    print("\n⏳ ML Model training ... ")
    
    # Module 6 - Data Preprocessing!
    ml_df = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'winner']].dropna()
    
    # Module 6 - Encoding Categorical Data!
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
    
    # Features & Target
    X = ml_df[['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc']]
    y = ml_df['winner_enc']
    
    # Module 6 - Train Test Split!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Module 9 - Random Forest!
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Module 11 - Model Evaluation!
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model Ready! Accuracy: {accuracy*100:.1f}%")
    return model, le_team, le_decision

def show_prediction():
    print("\n========== MATCH WINNER PREDICTION 🤖 ==========")
    
    # Train 
    model, le_team, le_decision = train_model()
    
    # Available teams
    teams = sorted(df['team1'].unique().tolist())
    print("\n📋 Available Teams:")
    for i, team in enumerate(teams, 1):
        print(f"  {i:2}. {team}")
    
    # User input
    print("\n Enter team number :")
    try:
        t1_idx = int(input("  Team 1 number: ")) - 1
        t2_idx = int(input("  Team 2 number: ")) - 1
        team1 = teams[t1_idx]
        team2 = teams[t2_idx]
        
        print(f"\nToss winner?")
        print(f"  1. {team1}")
        print(f"  2. {team2}")
        toss_idx = int(input("  Choice: ")) - 1
        toss_winner = [team1, team2][toss_idx]
        
        print(f"\nToss decision:")
        print(f"  1. bat")
        print(f"  2. field")
        dec_idx = int(input("  Choice: ")) - 1
        toss_decision = ['bat', 'field'][dec_idx]
        
        # Prediction!
        t1_enc = le_team.transform([team1])[0]
        t2_enc = le_team.transform([team2])[0]
        tw_enc = le_team.transform([toss_winner])[0]
        td_enc = le_decision.transform([toss_decision])[0]
        
        input_data = pd.DataFrame([[t1_enc, t2_enc, tw_enc, td_enc]], 
                      columns=['team1_enc', 'team2_enc', 
                               'toss_winner_enc', 'toss_decision_enc'])
        pred = model.predict(input_data)[0]
        winner = le_team.inverse_transform([pred])[0]
        
        proba = model.predict_proba(input_data)[0]
        confidence = max(proba) * 100
        
        print(f"\n{'='*45}")
        print(f"  🏏 {team1}")
        print(f"       VS")
        print(f"  🏏 {team2}")
        print(f"{'='*45}")
        print(f"  🎯 Toss Winner  : {toss_winner}")
        print(f"  📋 Decision     : {toss_decision}")
        print(f"{'='*45}")
        print(f"  🏆 PREDICTED WINNER : {winner}")
        print(f"  📊 Confidence       : {confidence:.1f}%")
        print(f"{'='*45}")
        
    except (ValueError, IndexError):
        print("❌ Wrong input! Enter valid number !")


# ============ HEAD TO HEAD ============

def show_head_to_head():
    print("\n========== HEAD TO HEAD COMPARISON ==========")
    
    teams = sorted(df['team1'].unique().tolist())
    print("\nAvailable Teams:")
    for i, team in enumerate(teams, 1):
        print(f"  {i:2}. {team}")
    
    try:
        t1_idx = int(input("\n  Team 1 number: ")) - 1
        t2_idx = int(input("  Team 2 number: ")) - 1
        team1 = teams[t1_idx]
        team2 = teams[t2_idx]
        
        # Filter matches between two teams
        h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
                 ((df['team1'] == team2) & (df['team2'] == team1))]
        
        total = len(h2h)
        team1_wins = len(h2h[h2h['winner'] == team1])
        team2_wins = len(h2h[h2h['winner'] == team2])
        no_result = total - team1_wins - team2_wins
        
        # Win percentage
        t1_pct = (team1_wins / total * 100) if total > 0 else 0
        t2_pct = (team2_wins / total * 100) if total > 0 else 0
        
        # Last 5 matches
        last5 = h2h.tail(5)[['season', 'winner', 'result_margin', 'result']]
        
        print(f"\n{'='*50}")
        print(f"  🏏 {team1}")
        print(f"       VS")
        print(f"  🏏 {team2}")
        print(f"{'='*50}")
        print(f"  📊 Total Matches : {total}")
        print(f"  🏆 {team1:<30} : {team1_wins} wins ({t1_pct:.1f}%)")
        print(f"  🏆 {team2:<30} : {team2_wins} wins ({t2_pct:.1f}%)")
        if no_result > 0:
            print(f"  🤝 No Result     : {no_result}")
        
        # Win bar
        t1_bar = "█" * int(t1_pct // 5)
        t2_bar = "█" * int(t2_pct // 5)
        print(f"\n  {team1[:15]:<15} {t1_bar}")
        print(f"  {team2[:15]:<15} {t2_bar}")
        
        print(f"\n  --- Last 5 Matches ---")
        for _, row in last5.iterrows():
            margin = f"{int(row['result_margin'])} {row['result']}" if pd.notna(row['result_margin']) else "N/R"
            print(f"  {str(row['season']):<10} Winner: {str(row['winner']):<30} ({margin})")
        
        print(f"\n  🎯 Overall Leader: ", end="")
        if team1_wins > team2_wins:
            print(f"{team1} 💪")
        elif team2_wins > team1_wins:
            print(f"{team2} 💪")
        else:
            print("Equal! 🤝")
        print(f"{'='*50}")
        
    except (ValueError, IndexError):
        print("Wrong input! valid number Enter valid number !")

main()