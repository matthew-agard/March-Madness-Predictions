"""Data Pipeline Helper Functions

This script is used as a module in the March_Madness_Predictions Jupyter notebooks.

The following functions are present:
    * regular_season_stats
    * coach_team_performance
    * all_team_season_data
    * hist_tournament_games
    * dataset_pipeline
    * feature_pipeline
    * round_pipeline
    * bracket_pipeline

Requires a minimum of the 'pandas' library, as well as the 'data_fetch', 'data_clean',
'data_merge', and 'feature_engineering' helper modules, being present in your environment to run.
"""

import pandas as pd

from sys import path
path.append('../fetch')
from data_fetch import get_team_data, get_coach_rankings_data, get_hist_bracket
path.append('../model')
from model_evaluation import model_predictions

from data_clean import clean_basic_stats, clean_adv_stats, clean_coach_ranking_stats, clean_merged_season_stats, clean_tourney_data, clean_curr_round_data, fill_playin_teams, clean_bracket
from data_merge import merge_clean_team_stats, merge_clean_coaches_rankings, merge_clean_tourney_games
from feature_engineering import totals_to_game_average, records_wl_pct, encode_confs, bidirectional_rounds_str_numeric, matchups_to_underdog_relative, scale_features, create_bracket_round, create_bracket_winners


def regular_season_stats(year):
    """Fetch and clean all regular season stats

    Parameters
    ----------
    year : int
        Calendar year

    Returns
    -------
    clean_season_basic_df : DataFrame
        Cleaned basic regular season stats for all teams in given year
    clean_reg_season_df : DataFrame
        All cleaned regular season stats for all teams in given year
    """
    # Fetch & clean basic regular season stats
    season_basic_df = get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/men/{year}-school-stats.html",
                                    attrs={'id': 'basic_school_stats'})
    clean_season_basic_df = clean_basic_stats(season_basic_df)
    
    # Fetch & clean advanced regular season stats
    season_adv_df = get_team_data(url=f"https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-school-stats.html", 
                                attrs={'id': 'adv_school_stats'})
    clean_season_adv_df = clean_adv_stats(season_adv_df)

    # Merge all cleaned regular season stats
    clean_reg_season_df = merge_clean_team_stats(clean_season_basic_df, clean_season_adv_df)

    return clean_season_basic_df, clean_reg_season_df


def coach_team_performance(year, stats_df):
    """Fetch and clean coach & team performance, merge onto team stats

    Parameters
    ----------
    year : int
        Calendar year
    stats_df : DataFrame
        Cleaned regular season stats for all teams in given year

    Returns
    -------
    all_reg_season_df : DataFrame
        Complete data for all regular season team and coach stats
    """
    # Fetch & clean coach performance data
    coaches_rankings_df = get_coach_rankings_data(year)
    clean_coaches_rankings_df = clean_coach_ranking_stats(coaches_rankings_df)

    # Merge coach data to all regular season data
    all_reg_season_df = merge_clean_coaches_rankings(stats_df, clean_coaches_rankings_df)

    return all_reg_season_df


def all_team_season_data(year):
    """Create dataset for all regular season team and coach stats

    Parameters
    ----------
    year : int
        Calendar year

    Returns
    -------
    all_season_stats_df : DataFrame
        Complete data for all regular season team and coach stats
    clean_season_basic_df : DataFrame
        Cleaned basic regular season stats for all teams in given year
    """
    # Fetch, clean, and merge regular season team stats
    clean_season_basic_df, team_season_stats_df = regular_season_stats(year)

    # Fetch and clean coach & team performance, merge them to team stats
    all_season_stats_df = coach_team_performance(year, team_season_stats_df)

    return all_season_stats_df, clean_season_basic_df


def hist_tournament_games(year, all_stats):
    """Fetch and clean all tournament data for a given year

    Parameters
    ----------
    year : int
        Calendar year
    all_stats : DataFrame
        Complete data for all regular season team and coach stats

    Returns
    -------
    mm_data_df : DataFrame
        Complete dataset for given year
    """
    # Reclean all team names & season stats (prior to merging of tournament games)
    clean_all_season_stats_df = clean_merged_season_stats(year, all_stats)
    
    # Fetch tournament game data
    mm_games_df = get_hist_bracket(year)
    
    # Clean & merge regular season data to tournament games (if they exist for given year)
    if not mm_games_df.empty:
        clean_mm_df = clean_tourney_data(mm_games_df, clean_all_season_stats_df)
        mm_data_df = merge_clean_tourney_games(year, clean_mm_df, clean_all_season_stats_df)
    else:
        mm_data_df = pd.DataFrame()

    return mm_data_df


def dataset_pipeline(years):
    """Create complete dataset over the range of years passed as an input

    Parameters
    ----------
    years : list
        Range of years to include in constructing the dataset

    Returns
    -------
    all_data_df : DataFrame
        Complete dataset
    """
    all_data_df = pd.DataFrame()

    for year in years:
        # Fetch. clean, and merge all regular season team and coach data    
        all_season_stats_df, clean_season_basic_df = all_team_season_data(year)

        # Merge tournament data to regular season data to create complete dataset for given year
        year_mm_data_df = hist_tournament_games(year, all_season_stats_df)

        # Concatenate current year's data to DataFrame containing remainder of dataset
        all_data_df = pd.concat([all_data_df, year_mm_data_df], ignore_index=True)

    return all_data_df


def feature_pipeline(dataset_type, data_cuts, basic_stats_cols):
    """Engineer features for complete dataset

    Parameters
    ----------
    primary_df : DataFrame
        Dataset to engineer; always used to transform StandardScaler()
    fit_df : DataFrame
        Dataset used to fit StandardScaler()

    Returns
    -------
    full_feature_df : DataFrame
        Complete dataset with re-engineered features
    """
    all_data_df = data_cuts['FULL']
    primary_df = data_cuts[dataset_type]
    fit_df = data_cuts['TRAIN']

    # Reclassify the 'Round' feature accordingly (if it's even present)
    try:
        bidirectional_rounds_str_numeric(primary_df)
    except KeyError:
        pass

    # Convert team regular season stats from season totals to per game averages
    totals_to_game_average(primary_df, basic_stats_cols)

    # Convert regular season conference record to a percentage
    records_wl_pct(primary_df)

    # Convert categorical conference values to numeric values
    encode_confs(primary_df, fit_df=all_data_df)

    # Convert favorite-underdog features to a single class of underdog relative feature (for primary & fit df's)
    matchups_to_underdog_relative(primary_df)

    # 'Center the data' for all numerical features; improves models' signal processing abilities
    full_feature_df = scale_features(primary_df, fit_df)

    return full_feature_df


def round_pipeline(year, curr_round, all_curr_matchups, 
                    curr_season_basic_df, clean_curr_season_data, data_cuts, null_drops):
    """Generate a round to be used for in the creation of an entire bracket

    Parameters
    ----------
    year : int
        Calendar year
    curr_round : int
        Tournament/Bracket round
    all_curr_matchups : list
        Tournament matchups used for model prediction; 1 round per index
    clean_curr_season_data : DataFrame
        Complete data for all regular season team and coach stats
    fit_df : DataFrame
        Dataset used to fit StandardScaler()
    null_drops : list
        Set of features to drop from whole dataset prior to model prediction

    Returns
    -------
    all_round_data : DataFrame
        Complete dataset for generated/selected round
    curr_X : DataFrame
        Tournament matchups used for model prediction; single round
    school_matchups_df : DataFrame
        Summarized tournament data to show in Jupyter notebook
    """    
    # Generate predicted\selected round
    if curr_round not in [0, 1]:
        generated_round = create_bracket_round(all_curr_matchups[curr_round-1])
    else:
        generated_round = all_curr_matchups[curr_round]

    # Ensure matchup seeds are integers for proper favorite-underdog identification
    generated_round[['Seed', 'Seed.1']] = generated_round[['Seed', 'Seed.1']].astype(int)

    # Cleaned tournament matchup dataset
    cleaned_generated_round = clean_tourney_data(generated_round, clean_curr_season_data)

    # Merge all team season data to teams in matchups
    all_round_data = merge_clean_tourney_games(year, cleaned_generated_round, clean_curr_season_data)

    # Store appropriate data for bracket DataFrame creation
    teams = ['Team_Favorite', 'Team_Underdog']
    school_matchups_df = all_round_data[teams]
    school_matchups_df['Round'] = [curr_round] * len(school_matchups_df)

    # Prepare DataFrame for prediction via feature pipeline preprocessing
    all_round_data.drop(teams + null_drops, axis=1, inplace=True)
    data_cuts['PREDICT'] = all_round_data

    curr_X = feature_pipeline('PREDICT', data_cuts, curr_season_basic_df.columns)

    return all_round_data, curr_X, school_matchups_df


def bracket_pipeline(year, play_in, first_round, model, indices, null_drops):
    """Generate a bracket as a prediction of the current year's tournament

    Parameters
    ----------
    year : int
        Current calendar year
    play_in : DataFrame
        Scraped matchups from the play-in round (non-generated)
    first_round : DataFrame
        Scraped matchups from the first round (non-generated)
    model : sklearn.base.BaseEstimator
        Model of choice for tournament matchup predictions
    fit_df : DataFrame
        Dataset used to fit StandardScaler()
    null_drops : list
        Set of features to drop from whole dataset prior to model prediction

    Returns
    -------
    bracket_preds : DataFrame
        Completely generated, properly formatted bracket
    """  
    # Get all team & coach season stats
    all_curr_season_data, curr_season_basic_df = all_team_season_data(year)
    clean_curr_season_data = clean_merged_season_stats(year, all_curr_season_data)

    # Initialize lists for use in generating/storing rounds
    all_curr_matchups = [play_in, first_round]
    all_curr_rounds = [play_in, first_round]

    for curr_round in range(7):
        # Get all data needed for current generated/selected round    
        all_round_data, curr_X, school_matchups_df = round_pipeline(year, curr_round, all_curr_matchups, curr_season_basic_df,
                                                                    clean_curr_season_data, indices, null_drops)
        # Create predictions
        school_matchups_df['Underdog_Upset'] = model_predictions(model, curr_X)
        
        # Clean current round for use in generating a subsequent round (if needed)
        curr_X, school_matchups_df = clean_curr_round_data(all_round_data, curr_X, school_matchups_df)
        
        # Store summary data for current round and matchup data for next round's predictions        
        if curr_round in [0, 1]:
            all_curr_matchups[curr_round] = curr_X
            all_curr_rounds[curr_round] = school_matchups_df
        else:
            all_curr_matchups.append(curr_X)
            all_curr_rounds.append(school_matchups_df)

        # Fill first round nulls with play-in winners
        if curr_round == 0:
            fill_playin_teams(all_curr_matchups)

    # Clean generated bracket to be nicely formatted for Jupyter notebook visualization
    bracket_preds = clean_bracket(all_curr_matchups, all_curr_rounds)
    create_bracket_winners(bracket_preds)

    return bracket_preds