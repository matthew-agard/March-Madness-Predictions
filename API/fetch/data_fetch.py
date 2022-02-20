"""Data Fetch Helper Functions

This script is used as a helper module in the data_pipeline script; 
also used as a module in the March_Madness_Predictions Jupyter notebooks.

The following functions are present:
    * get_team_data
    * get_ratings_data
    * get_coach_data
    * get_null_rows
    * get_feature_null_counts
    * get_hist_bracket
    * get_current_bracket

Requires a minimum of the 'pandas' and 're' libraries, as well as the 'web_scraper_types',
'data_merge' and 'data_integrity' helper modules, being present in your environment to run.
"""

import pandas as pd
import re
from merge_fetch import playin_regions_list, merge_raw_tourney_games
from web_scraper_types import bs4_web_scrape, pandas_web_scrape, bracket_web_scrape


def get_team_data(url, attrs, header=1):
    """Fetch team data (season stats, historical tournament performance)

    Parameters
    ----------
    url : str
        URL path to data
    attrs : dict
        Characteristics to idenitfy HTML element of interest
    header : int, optional
        Row in raw data to use for column headers (default=1)

    Returns
    -------
    teams_df[0] : DataFrame
        Web-scraped data points read into a DataFrame
    """
    try:
        # Read team data into dataframe
        teams_df = pandas_web_scrape(url, attrs, header)
    except ValueError:
        # Catch error with empty DataFrame is requested team data doesn't exist
        teams_df = [pd.DataFrame()]
    
    return teams_df[0]


def get_ratings_data(url):
    """Fetch team season ratings

    Parameters
    ----------
    url : str
        URL path to data

    Returns
    -------
    ratings_df : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw HTML and scrape its data
    raw_html = bs4_web_scrape(url)
    table = raw_html.find("table", attrs={"id": "ratings"})
    rows = table.find_all("tr")

    # Prepare DataFrame
    ratings_df = pd.DataFrame(columns=['Team', 'Top_25', 'SRS'])

    # Iterate over raw data to extract team and rank HTML elements
    for i, row in enumerate(rows):
        if row.find('a'):
            # Get team name
            team = row.find('a')

            # Get team simple rating system (SRS) value
            srs = row.find("td", attrs={"data-stat": "srs"})
            
            # Identify Top 25 teams using ternary operator to produce binary output
            ratings_df.loc[i] = [team.text, 1 if (len(ratings_df) < 25) else 0, srs.text]
            
    return ratings_df


def get_coach_data(url):
    """Fetch team coach performance

    Parameters
    ----------
    url : str
        URL path to data

    Returns
    -------
    coaches_df : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw HTML and scrape its data
    raw_html = bs4_web_scrape(url)
    table = raw_html.find("table", attrs={"id": "coaches"})
    rows = table.find_all("tr")

    # Prepare DataFrame
    coaches_df = pd.DataFrame(columns=['Coach_Team', 'Coach_Start', 'MM', 'S16', 'F4', 'Champs', 'Conf'])

    # Iterate over raw data to extract coach tournament appearances HTML elements
    for i, row in enumerate(rows):
        if(row.find('a')):
            coach_team = row.find_all('a')[1]
            year_start = row.find("td", attrs={"data-stat": "since"})
            mm_apps = row.find("td", attrs={"data-stat": "ncaa_car"})
            sw16_apps = row.find("td", attrs={"data-stat": "sw16_car"})
            f4_apps = row.find("td", attrs={"data-stat": "ff_car"})
            champ_wins = row.find("td", attrs={"data-stat": "champ_car"})
            conf = row.find("td", attrs={"data-stat": "conference"})

            coaches_df.loc[i] = [
                coach_team.text, year_start.text, mm_apps.text, sw16_apps.text, f4_apps.text, champ_wins.text, conf.text
            ]

    coaches_df.sort_values(by=['Coach_Team', 'Coach_Start'], inplace=True)
    coaches_df.drop('Coach_Start', axis=1, inplace=True)

    return coaches_df.drop_duplicates(subset='Coach_Team', keep='last')


def get_null_rows(null_fills, df):
    """Fetch rows with any nulls; used for imputing new values

    Parameters
    ----------
    null_fills : list
        Collection of features where nulls reside
    df : DataFrame
        Fully merged dataset

    Returns
    -------
    DataFrame
        Cross-section of df; contains the rows where nulls reside for features in null_fills list
    """
    rows = df[df[null_fills].isnull().any(axis=1)]
    return rows[['Year'] + null_fills + ['Underdog_Upset']]


def get_feature_null_counts(df):
    """Count number of nulls for each feature containing any nulls

    Parameters
    ----------
    df : DataFrame
        Fully merged dataset

    Returns
    -------
    DataFrame
        Structure containing all features with nulls, sorted in descending order by their number of nulls
    """
    nulls = df.isnull().sum().sort_values(ascending=False)
    return nulls[nulls > 0]


def get_playin_matchups(url, year):
    # Fetch raw HTML
    raw_html = bs4_web_scrape(url)

    # Used for iterating over all possible combinations of play-in regions
    playin_regions = playin_regions_list
    playin_classes = ['current', '']
    
    # Initialize data structures to store scraped data
    seeds_list, teams_scores_list = [], []

    for pi_class in playin_classes:
        for i, playin_region in enumerate(playin_regions):
            # Scrape all bracket data
            bracket_raw = raw_html.find("div", attrs={'id': playin_region, 'class': pi_class})

            try:
                # Extract play-in matchups from bracket web scrape data
                playin_raw = bracket_raw.find("p")
                
                # Get play-in teams' seeds
                seeds_raw = playin_raw.find_all("strong")
                seeds_list = seeds_list + [seed.text for seed in seeds_raw if ((seed.text).isdigit()) and (int(seed.text) <= 16)]

                # Get play-in teams' names & game scores
                teams_scores_raw = playin_raw.find_all("a")
                teams_scores_list = teams_scores_list + [team_score.text for team_score in teams_scores_raw]
            
            # Catch the error from trying to scrape data from a non-existent HTML element
            except AttributeError:
                continue

    # Initialize rounds_list accordingly
    rounds_list = (['Play-In'] * (len(seeds_list) // 2))

    # Merge all play-in games into a single DataFrame
    playin_df = merge_raw_tourney_games(year, seeds_list, teams_scores_list, rounds_list)
    return playin_df


def get_tourney_matchups(url, year):
    # Scrape tournament matchup data (excluding play-ins)
    raw_html = bs4_web_scrape(url)
    tourney_regions = raw_html.find_all("div", attrs={'id': 'bracket'})
    
    # Initialize DataFrame to store scraped data
    tourney_df = pd.DataFrame()

    # Iterate over all 4 tournament regions and Final Four
    for i, tourney_region in enumerate(tourney_regions):
        # Get all teams' seeds
        seeds = tourney_region.find_all("span")
        seeds_list = [data.text for data in seeds if ("at ") not in data.text][:-1]
        
        # Get all teams' names and scores
        teams_scores = tourney_region.find_all("a")
        teams_scores_list = [data.text for data in teams_scores if ("at ") not in data.text][:-1]
        
        # If the condition below is met, teams_scores_list must contain Final Four data
        if len(teams_scores_list) == 12:
            # Initialize rounds_list accordingly
            rounds_list = (['Final Four'] * 2) + ['National Championship']
        # If the condition below is met, teams_scores_list must contain regional data
        else:
            # We can expect len(teams_scores_list) == 60 when regional data is present.
            # The only exception to this rule is 2021, where COVID caused the cancellation of 1 game.
            if (year == 2021) and (len(teams_scores_list) != 60):
                # Insert missing scores from COVID cancellation game
                teams_scores_list.insert(25, "1")
                teams_scores_list.insert(27, "0")
            # Initialize rounds_list accordingly
            rounds_list = (['First Round'] * 8) + (['Second Round'] * 4) + (['Sweet Sixteen'] * 2) + ['Elite Eight']

        games_df = merge_raw_tourney_games(year, seeds_list, teams_scores_list, rounds_list)
        # Concatenate all regional DataFrames into a single DataFrame
        tourney_df = pd.concat([tourney_df, games_df], ignore_index=True)

    return tourney_df


def get_hist_bracket(url, year):
    playin_df = get_playin_matchups(url, year)
    tourney_df = get_tourney_matchups(url, year)

    full_tourney_df = pd.concat([playin_df, tourney_df], ignore_index=True)
    return full_tourney_df


def get_current_bracket(url):
    """Fetch current tournament bracket matchups

    Parameters
    ----------
    url : str
        URL path to data

    Returns
    -------
    current_bracket : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw data and prepare DataFrame
    raw_html = bracket_web_scrape(url, attrs={"id": "bracket"})
    current_bracket = pd.DataFrame(columns=['Seed', 'Team', 'Seed.1', 'Team.1'])

    # Iterate over raw data to extract team and their seeds
    for i, game in enumerate(raw_html):
        game_string = game.find('dt')

        teams = [name['title'] for name in game_string.find_all('a')]

        seeds = re.findall(r'\d+', game_string.text) 
        seeds = list(map(int, seeds))

        try:
            # Read team matchups into dataframe
            current_bracket.loc[i] = [seeds[0], teams[0], seeds[1], teams[1]]
        except IndexError:
            # Catch error where 1st Round awaits First Four winners
            if len(teams) > 0:
                current_bracket.loc[i] = [seeds[0], teams[0], 0, None]
                
    current_bracket.index = range(len(current_bracket))
    return current_bracket