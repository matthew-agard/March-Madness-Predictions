"""Data Fetch Helper Functions

This script is used as a helper module in the data_pipeline script; 
also used as a module in the March_Madness_Predictions Jupyter notebooks.

The following functions are present:
    * get_team_data
    * get_ratings_data
    * get_coach_rankings_data
    * get_null_rows
    * get_feature_null_counts
    * get_hist_bracket
    * get_current_bracket

Requires a minimum of the 'pandas' and 're' libraries, as well as the 'web_scraper_types',
'data_merge' and 'data_integrity' helper modules, being present in your environment to run.
"""

import pandas as pd
from merge_fetch import ratings_team_to_coach_team_dict, playin_regions_list, merge_raw_tourney_games
from web_scraper_types import bs4_web_scrape, pandas_web_scrape

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
        # Catch error with empty DataFrame if requested team data doesn't exist
        teams_df = [pd.DataFrame()]
    
    return teams_df[0]


def get_ratings_data(year):
    """Fetch team season ratings

    Parameters
    ----------
    year : int
        Calendar year

    Returns
    -------
    ratings_df : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw HTML and scrape its data
    raw_html = bs4_web_scrape(f"https://www.sports-reference.com/cbb/seasons/men/{year}-polls.html")
    table = raw_html.find("table", attrs={"id": "ap-polls"})
    rows = table.find_all("tr")

    # Prepare DataFrame
    ratings_df = pd.DataFrame(columns=['Top_25_Team'])

    # Iterate over raw data to extract team and rank HTML elements
    for i, row in enumerate(rows):
        try:
            latest_rating = row.find_all("td")[-1]
            team_name = row.find("a")

            if latest_rating.text != '':
                ratings_df.loc[i] = [team_name.text]
        except IndexError:
            pass

    return ratings_df


def get_coach_rankings_data(year):
    """Fetch team coach performance

    Parameters
    ----------
    year : int
        Calendar year

    Returns
    -------
    coaches_rankings_df : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw HTML and scrape its data
    raw_html = bs4_web_scrape(f"https://www.sports-reference.com/cbb/seasons/men/{year}-coaches.html")
    table = raw_html.find("table", attrs={"id": "coaches"})
    rows = table.find_all("tr")

    # Prepare DataFrames
    coaches_rankings_df = pd.DataFrame(columns=['Coach_Team', 'Conf', 'Top_25', 'Coach_Start', 'MM', 'S16', 'F4', 'Champs'])

    ratings_df = get_ratings_data(year)
    ratings_df['Top_25_Team'].replace(ratings_team_to_coach_team_dict, inplace=True)

    # Iterate over raw data to extract coach tournament appearances HTML elements
    for i, row in enumerate(rows):
        if(row.find('a')):
            coach_team = row.find_all('a')[1]
            conf = row.find("td", attrs={"data-stat": "conference"})
            top_25 = 1 if coach_team.text in ratings_df['Top_25_Team'].values else 0
            year_start = row.find("td", attrs={"data-stat": "since"})
            mm_apps = row.find("td", attrs={"data-stat": "ncaa_car"})
            sw16_apps = row.find("td", attrs={"data-stat": "sw16_car"})
            f4_apps = row.find("td", attrs={"data-stat": "ff_car"})
            champ_wins = row.find("td", attrs={"data-stat": "champ_car"})          

            coaches_rankings_df.loc[i] = [
                coach_team.text, conf.text, top_25, year_start.text, 
                mm_apps.text, sw16_apps.text, f4_apps.text, champ_wins.text
            ]

    coaches_rankings_df.sort_values(by=['Coach_Team', 'Coach_Start'], inplace=True)
    coaches_rankings_df.drop('Coach_Start', axis=1, inplace=True)

    return coaches_rankings_df.drop_duplicates(subset='Coach_Team', keep='last')


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


def get_playin_matchups(year):
    # Fetch raw HTML
    raw_html = bs4_web_scrape(f'https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html')

    # Used for iterating over all possible combinations of play-in regions
    playin_regions = playin_regions_list
    playin_classes = ['current', '']
    
    # Initialize data structures to store scraped data
    seeds_list, teams_scores_list = [], []

    for pi_class in playin_classes:
        for playin_region in playin_regions:
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


def get_tourney_matchups(year):
    # Scrape tournament matchup data (excluding play-ins)
    raw_html = bs4_web_scrape(f'https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html')
    tourney_regions = raw_html.find_all("div", attrs={'id': 'bracket'})
    
    # Initialize DataFrame to store scraped data
    tourney_df = pd.DataFrame()

    # Iterate over all 4 tournament regions and Final Four
    for tourney_region in tourney_regions:
        # Get all teams' seeds
        seeds = tourney_region.find_all("span")
        seeds_list = [data.text for data in seeds if ("at ") not in data.text][:-1]
        
        # Get all teams' names and scores
        teams_scores = tourney_region.find_all("a")
        teams_scores_list = [data.text for data in teams_scores if ("at ") not in data.text][:-1]
        
        # If the condition below is met, teams_scores_list must contain Final Four data
        if len(teams_scores_list) == 60:
            # We can expect len(teams_scores_list) == 60 when regional data is present.
            # The only exception to this rule is 2021, where COVID caused the cancellation of 1 game.
            if (year == 2021) and (len(teams_scores_list) != 60):
                # Insert missing scores from COVID cancellation game
                teams_scores_list.insert(25, "1")
                teams_scores_list.insert(27, "0")
            # Initialize rounds_list accordingly
            rounds_list = (['First Round'] * 8) + (['Second Round'] * 4) + (['Sweet Sixteen'] * 2) + ['Elite Eight']
        # If the condition below is met, teams_scores_list must contain Final Four data
        else:
            # We can expect len(teams_scores_list) == 12 when Final Four data is present.
            rounds_list = (['Final Four'] * 2) + ['National Championship']

        games_df = merge_raw_tourney_games(year, seeds_list, teams_scores_list, rounds_list)
        # Concatenate all regional DataFrames into a single DataFrame
        tourney_df = pd.concat([tourney_df, games_df], ignore_index=True)

    return tourney_df


def get_hist_bracket(year):
    playin_df = get_playin_matchups(year)
    tourney_df = get_tourney_matchups(year)

    full_tourney_df = pd.concat([playin_df, tourney_df], ignore_index=True)
    return full_tourney_df


def get_current_bracket(year):
    """Fetch current tournament bracket matchups

    Returns
    -------
    current_bracket : DataFrame
        Curated data points read into a DataFrame
    """
    # Fetch raw data and prepare DataFrame
    raw_html = bs4_web_scrape(f'https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html')
    matchup_regions = raw_html.find_all("div", attrs={'id': 'bracket'})
    current_bracket = pd.DataFrame(columns=['Seed', 'Team', 'Seed.1', 'Team.1'])

    # Iterate over raw data to extract team and their seeds
    for i, matchups in enumerate(matchup_regions):
        # Get all teams' seeds
        seeds = matchups.find_all("span")
        seeds_list = [data.text for data in seeds if ("at ") not in data.text]
        # Clean First Four team seeds; will need to be manually added to CSV
        seeds_list = [int(seed) if seed and seed.lower() != 'tbd' else 0 for seed in seeds_list]
        
        # Get all teams' names
        teams = matchups.find_all("a")
        teams_list = [data.text for data in teams if ("at ") not in data.text]
        # Clean First Four team names; will need to be manually added to CSV
        teams_list = [team if team not in ['Play-In', 'tbd'] else None for team in teams_list]

        # Read team matchups into dataframe
        for j in range(0, len(teams_list), 2):
            current_bracket.loc[(i*len(teams_list)) + j] = [seeds_list[j], teams_list[j], seeds_list[j+1], teams_list[j+1]]
                
    current_bracket.index = range(len(current_bracket))
    return current_bracket