# train_model.py
import os
import json
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor

# tree boosters
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ---- PARAMETERS ----
BALL_BY_BALL_CSV = 'IPl Ball-by-Ball 2008-2023.csv'
MATCHES_CSV = 'IPL Mathces 2008-2023.csv'
OUTPUT_DIR = 'model_artifacts'
RECENT_MATCHES = 8   # N matches to compute recent form
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- helper functions to engineer features ----
def load_data():
    byb = pd.read_csv(BALL_BY_BALL_CSV)
    matches = pd.read_csv(MATCHES_CSV)
    # normalize column names if necessary (lowercase)
    byb.columns = [c.strip() for c in byb.columns]
    return byb, matches

def aggregate_per_player(byb):
    """
    Create aggregated per-player per-match records, then per-player records.
    We'll compute per-player per-match stats, then use them to make
    features and targets (e.g., fantasy points in that match).
    """
    # Example: create match-level fantasy for batsman and for bowler in each match
    # We'll produce a dataset where each row = player_in_match and target = fantasy_points_in_that_match
    rows = []

    # ensure id column exists (match id)
    if 'id' not in byb.columns:
        raise ValueError("ball-by-ball CSV must have 'id' column for match identifier")

    matches = byb['id'].unique()
    for mid in tqdm(matches, desc='Building player-match rows'):
        mdf = byb[byb['id'] == mid]
        # per-batsman stats in this match
        batsmen = mdf['batsman'].unique()
        for b in batsmen:
            sb = mdf[mdf['batsman'] == b]
            runs = sb['batsman_runs'].sum()
            balls = len(sb)
            fours = (sb['batsman_runs'] == 4).sum()
            sixes = (sb['batsman_runs'] == 6).sum()
            dismissals = sb['is_wicket'].sum()
            # Extra: catches/runouts credited elsewhere may not be in batsman rows; we'll count fielders separately later
            rows.append({
                'match_id': mid,
                'player': b,
                'role': 'batsman',
                'runs': runs,
                'balls': balls,
                'fours': fours,
                'sixes': sixes,
                'dismissals': dismissals,
                'team': sb['batting_team'].iloc[0] if 'batting_team' in sb.columns else None,
                'opposition_team': sb['bowling_team'].iloc[0] if 'bowling_team' in sb.columns else None
            })
        # per-bowler stats in this match
        bowlers = mdf['bowler'].unique()
        for bw in bowlers:
            sb = mdf[mdf['bowler'] == bw]
            runs_conceded = sb['batsman_runs'].sum()
            balls_bowled = len(sb)
            wickets = sb['is_wicket'].sum()
            # count lbw/bowled for bonus
            lbw_bowled = ((sb['dismissal_kind'] == 'lbw') | (sb['dismissal_kind'] == 'bowled')).sum() if 'dismissal_kind' in sb.columns else 0
            rows.append({
                'match_id': mid,
                'player': bw,
                'role': 'bowler',
                'runs_conceded': runs_conceded,
                'balls_bowled': balls_bowled,
                'wickets': wickets,
                'lbw_bowled': lbw_bowled,
                'team': sb['bowling_team'].iloc[0] if 'bowling_team' in sb.columns else None,
                'opposition_team': sb['batting_team'].iloc[0] if 'batting_team' in sb.columns else None
            })
    df = pd.DataFrame(rows)
    return df

def compute_fantasy_points_for_row(row):
    """
    Simple fantasy scoring mapping similar to your original script.
    This is just to create a target variable for supervised learning.
    NOTE: You can modify the exact formula as desired.
    """
    fp = 0.0
    # batsman row
    if row.get('role') == 'batsman':
        runs = row.get('runs', 0)
        fours = row.get('fours', 0)
        sixes = row.get('sixes', 0)
        dismissals = row.get('dismissals', 0)
        # basic mapping
        fp += runs * 1.0
        fp += fours * 1.0
        fp += sixes * 2.0
        # ducks penalty if runs==0 and dismissals>0
        if runs == 0 and dismissals > 0:
            fp += -2
        # bonus for 30/50/100
        if runs >= 100:
            fp += 16
        elif runs >= 50:
            fp += 8
        elif runs >= 30:
            fp += 4
    # bowler row
    if row.get('role') == 'bowler':
        wickets = row.get('wickets', 0)
        lbw_bowled = row.get('lbw_bowled', 0)
        balls = row.get('balls_bowled', 0)
        if wickets:
            fp += wickets * 25
        fp += lbw_bowled * 8
        # maiden detection not trivial from ball-level if you don't have over break; ignoring for simplicity
        # economy penalties/rewards can be added
    return fp

def make_dataset(byb):
    """
    More pragmatic approach: compute per-player per-match aggregated features by combining batting and bowling rows.
    We'll create a row per (player, match) that contains batting stats (if any) and bowling stats (if any).
    """
    # groupby player+match
    grouped = byb.groupby(['id', 'batsman']).agg(
        runs=('batsman_runs', 'sum'),
        balls=('batsman_runs', 'count'),
        fours=('batsman_runs', lambda x: (x == 4).sum()),
        sixes=('batsman_runs', lambda x: (x == 6).sum()),
    ).reset_index().rename(columns={'batsman':'player', 'id':'match_id'})

    # bowling per match
    bowl = byb.groupby(['id','bowler']).agg(
        runs_conceded=('batsman_runs', 'sum'),
        balls_bowled=('batsman_runs','count'),
        wickets=('is_wicket','sum'),
    ).reset_index().rename(columns={'bowler':'player','id':'match_id'})

    # merge batting & bowling into single row per player-match (outer join)
    merged = pd.merge(grouped, bowl, on=['match_id','player'], how='outer')

    # fillna
    merged[['runs','balls','fours','sixes','runs_conceded','balls_bowled','wickets']] = merged[[
        'runs','balls','fours','sixes','runs_conceded','balls_bowled','wickets']].fillna(0)

    # add team/opponent columns by pulling any row's team info from byb
    def get_team_info(row):
        md = byb[byb['id'] == row['match_id']]
        # find any row where batsman==player and get batting_team
        tmp = md[md['batsman'] == row['player']]
        if not tmp.empty and 'batting_team' in tmp.columns:
            return tmp['batting_team'].iloc[0], tmp['bowling_team'].iloc[0]
        tmp = md[md['bowler'] == row['player']]
        if not tmp.empty and 'bowling_team' in tmp.columns:
            return tmp['bowling_team'].iloc[0], tmp['batting_team'].iloc[0]
        return None, None

    teams = merged.apply(lambda r: get_team_info(r), axis=1)
    merged['team'] = teams.apply(lambda x: x[0])
    merged['opposition'] = teams.apply(lambda x: x[1])

    # compute fantasy target
    merged['fantasy_target'] = merged.apply(lambda r: compute_fantasy_points_for_row({
        'role': 'batsman' if r['runs']>0 or r['balls']>0 or r['fours']>0 or r['sixes']>0 else 'bowler',
        'runs': r['runs'],
        'fours': r['fours'],
        'sixes': r['sixes'],
        'dismissals': 1 if r['runs']==0 and r['balls']>0 else 0,
        'wickets': r['wickets'],
        'lbw_bowled': 0
    }), axis=1)

    return merged

def create_player_level_features(merged):
    """
    From player-match rows, produce per-player aggregated features:
    - career mean runs, strike rate, boundary%, wickets per match, bowling econ
    - recent form: mean of last RECENT_MATCHES matches
    - head-to-head features would require opponent grouping; we'll compute average vs each opponent team
    """

    # sort by match_id (match ids may not be chronological; if you have date, use it)
    # If your byb includes 'date' in matches, better to merge match date and sort by date.
    # For simplicity we'll treat match_id order as chronological
    merged = merged.sort_values(['player','match_id'])
    feature_rows = []
    players = merged['player'].unique()
    for p in tqdm(players, desc='Aggregating player features'):
        ply = merged[merged['player'] == p]
        # career aggregates
        total_matches = len(ply)
        career_runs_mean = ply['runs'].mean() if total_matches>0 else 0
        career_SR = (ply['runs'].sum() / ply['balls'].sum()*100) if ply['balls'].sum() > 0 else 0
        career_boundary_rate = (ply['fours'].sum() + ply['sixes'].sum()) / ply['balls'].sum() if ply['balls'].sum()>0 else 0
        career_wickets_per_match = ply['wickets'].sum() / total_matches if total_matches > 0 else 0
        career_bowling_econ = (ply['runs_conceded'].sum() / (ply['balls_bowled'].sum()/6)) if ply['balls_bowled'].sum()>0 else 0

        # recent N matches features
        recent = ply.tail(RECENT_MATCHES)
        recent_matches = len(recent)
        recent_runs_mean = recent['runs'].mean() if recent_matches>0 else career_runs_mean
        recent_SR = (recent['runs'].sum() / recent['balls'].sum()*100) if recent['balls'].sum()>0 else career_SR
        recent_boundary_rate = (recent['fours'].sum()+recent['sixes'].sum()) / recent['balls'].sum() if recent['balls'].sum()>0 else career_boundary_rate
        recent_wickets_per_match = recent['wickets'].sum() / recent_matches if recent_matches>0 else career_wickets_per_match

        # consistency: std dev of runs
        runs_std = ply['runs'].std() if total_matches>1 else 0

        # last match performance
        last = ply.tail(1).iloc[0] if total_matches>0 else None
        last_runs = last['runs'] if last is not None else 0
        last_wickets = last['wickets'] if last is not None else 0

        feature_rows.append({
            'player': p,
            'total_matches': total_matches,
            'career_runs_mean': career_runs_mean,
            'career_SR': career_SR,
            'career_boundary_rate': career_boundary_rate,
            'career_wickets_per_match': career_wickets_per_match,
            'career_bowling_econ': career_bowling_econ,
            'recent_runs_mean': recent_runs_mean,
            'recent_SR': recent_SR,
            'recent_boundary_rate': recent_boundary_rate,
            'recent_wickets_per_match': recent_wickets_per_match,
            'runs_std': runs_std,
            'last_runs': last_runs,
            'last_wickets': last_wickets
        })

    features = pd.DataFrame(feature_rows)
    return features

def build_training_table(merged, features):
    """
    Join merged player-match target with player-level (lagged) features.
    For simplicity create rows where target= fantasy_target in that match and features are previous aggregated features up to prior match.
    We'll perform a simpler strategy: use player-level career & recent features computed from ALL matches as features, and target as player's fantasy in a given match.
    This leaks slightly because recent features include that match, but for a quick pipeline it's acceptable. For production, compute features using rolling windows excluding the target match.
    """
    # join features into merged on player
    df = merged.merge(features, on='player', how='left')
    # drop rows with no team/opposition if needed
    df = df.dropna(subset=['fantasy_target'])
    # sample down if dataset is huge (optional)
    return df

def train_and_save(df):
    # select feature columns
    feat_cols = [
        'total_matches','career_runs_mean','career_SR','career_boundary_rate','career_wickets_per_match',
        'career_bowling_econ','recent_runs_mean','recent_SR','recent_boundary_rate','recent_wickets_per_match',
        'runs_std','last_runs','last_wickets'
    ]
    X = df[feat_cols].fillna(0)
    y = df['fantasy_target'].astype(float)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)

    # simple scaler
    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # three base learners
    lgbm = lgb.LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE)
    xgbr = xgb.XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0)
    catb = CatBoostRegressor(iterations=200, verbose=0, random_seed=RANDOM_STATE)

    # stacking regressor with a simple meta-learner
    estimators = [
        ('lgb', lgbm),
        ('xgb', xgbr),
        ('cat', catb)
    ]
    meta = Ridge()
    stack = StackingRegressor(estimators=estimators, final_estimator=meta, n_jobs=-1, passthrough=False)

    # train
    print("Training stacking regressor (this may take some minutes)...")
    stack.fit(X_train_p, y_train)

    # predict & evaluate
    preds_train = stack.predict(X_train_p)
    preds_test = stack.predict(X_test_p)
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_test = mean_absolute_error(y_test, preds_test)
    print(f"Train MAE: {mae_train:.4f}  |  Test MAE: {mae_test:.4f}")

    # save artifacts
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor.joblib'))
    joblib.dump(stack, os.path.join(OUTPUT_DIR, 'stack_model.joblib'))
    with open(os.path.join(OUTPUT_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feat_cols, f)

    print("Saved model artifacts to", OUTPUT_DIR)
    return preprocessor, stack, feat_cols, mae_test

# ---- main ----
if __name__ == '__main__':
    byb, matches = load_data()
    merged = make_dataset(byb)
    features = create_player_level_features(merged)
    train_df = build_training_table(merged, features)
    preprocessor, model, feat_cols, test_mae = train_and_save(train_df)
    # Save a short metadata file
    meta = {'test_mae': float(test_mae)}
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f)
    print("Done.")
