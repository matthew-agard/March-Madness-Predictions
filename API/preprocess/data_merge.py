"""Data Merge Helper Functions

This script is used as a helper module in the data_pipeline script.
The following functions are present:
    * merge_clean_team_stats
    * merge_clean_ratings
    * merge_clean_coaches
    * merge_clean_tourney_games

Requires a minimum of the 'pandas' library being present in your environment to run.
"""

import pandas as pd
from data_integrity import season_team_to_coach_team_dict


def merge_clean_team_stats(basic_df, adv_df):
    """Merge basic and advanced season stats (by team)

    Parameters
    ----------
    basic_df : DataFrame
        Teams' basic regular season stats
    adv_df : DataFrame
        Teams' advanced regular season stats

    Returns
    -------
    season_team_stats_df : DataFrame
        All teams' regular season stats
    """
    # Merge on the school name
    season_team_stats_df = pd.merge(basic_df, adv_df, on='School')

    # Strip the 'NCAA' tag from the teams with a tournament berth
    if season_team_stats_df['School'].str.contains('NCAA').any():
        season_team_stats_df['School'] = season_team_stats_df['School'].apply(lambda school: school[:-5])
    
    return season_team_stats_df


def merge_clean_ratings(team_stats_df, ratings_df):
    """Merge season ratings to season stats

    Parameters
    ----------
    team_stats_df : DataFrame
        All teams' regular season stats
    ratings_df : DataFrame
        Teams' regular season ratings stats

    Returns
    -------
    season_stats_ratings_df : DataFrame
        Newly-merged DataFrame of a teams' regular season stats with their regular season ranking
    """
    # Merge on the school name
    season_stats_ratings_df = pd.merge(team_stats_df, ratings_df, 
                                        left_on='School', right_on='Team').drop('Team', axis=1)
    
    return season_stats_ratings_df


def merge_clean_coaches(stats_ratings_df, coaches_df):
    """Merge coach performance to teams' season stats and ratings

    Parameters
    ----------
    stats_ratings_df : DataFrame
        All teams' regular season stats and ratings
    coaches_df : DataFrame
        Teams' coach performances historically in the tournament

    Returns
    -------
    all_season_stats_df : DataFrame
        Newly-merged DataFrame of a teams' regular season stats, regular season ranking, and coach performance
    """
    # Change team names accordingly to ensure successful merging with team stats
    stats_ratings_df['School'].replace(season_team_to_coach_team_dict, inplace=True)

    # Merge on the school name
    all_season_stats_df = pd.merge(stats_ratings_df, coaches_df,
                                    left_on='School', right_on='Coach_Team').drop('Coach_Team', axis=1)

    return all_season_stats_df


def merge_clean_tourney_games(mm_df, all_season_df):
    """Merge all team data onto tournament matchups DataFrame

    Parameters
    ----------
    mm_df : DataFrame
        All teams' tournament matchups
    all_season_df : DataFrame
        All teams' data

    Returns
    -------
    all_data_df : DataFrame
        Completed dataset
    """
    # Merge favorites' season data onto tournament matchups DataFrame
    favorites_data_df = pd.merge(mm_df, all_season_df, 
                                left_on='Team_Favorite', right_on='School').drop('School', axis=1)
    
    # Merge underdogs' season data onto tournament matchups DataFrame
    # Account for duplicate stats column names with suffix labeling
    all_data_df = pd.merge(favorites_data_df, all_season_df, suffixes=("_Favorite", "_Underdog"),
                            left_on='Team_Underdog', right_on='School').drop('School', axis=1)

    return all_data_df