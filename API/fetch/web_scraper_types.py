"""Web Scraper Helper Functions

This script is used as a helper module in the data_fetch script.
The following functions are present:
    * pandas_web_scrape
    * bs4_web_scrape

Requires a minimum of the 'pandas', 'requests', 'BeautifulSoup' 
and 'time' libraries being present  in your environment to run.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from time import sleep


def pandas_web_scrape(url, attrs, header):
    """Pandas web scraper

    Parameters
    ----------
    url : str
        URL path to data
    attrs : dict
        characteristics to idenitfy HTML element of interest
    header : int
        row in raw data to use for column headers

    Returns
    -------
    arr : list
        Collection of all webpage data points (by row)
    """
    # Configure scraper and get table data
    arr = pd.read_html(url, attrs=attrs, header=header)

    # 3 second delay (request rate limit)
    sleep(3)

    return arr


def bs4_web_scrape(url):
    """BeautifulSoup table web scraper

    Parameters
    ----------
    url : str
        URL path to data
    attrs : dict
        characteristics to idenitfy HTML element of interest

    Returns
    -------
    soup : BeautifulSoup
        Raw webpage HTML
    """
    # Configure scraper
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    # 3 second delay (request rate limit)
    sleep(3)

    return soup