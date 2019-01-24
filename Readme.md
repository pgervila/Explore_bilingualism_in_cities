## Quantifying language choice with Twitter

The project aims to quantify language choice in bilingual environments through evidence 
gathered from Twitter users

It contains:
  - 4 Python classes:
    - __StreamTweetData__: class with methods to access the Twitter API to stream specific tweets 
    from a number of followers of specified accounts linked to a country/city/account
    - __ProcessTweetData__: class with methods to post-process tweets by language 
    and do statistical analysis of the data.
    - __PlotTweetData__: class with methods to visualize results from ProcessTweetData class
    - __InterCityComparison__: class to visualize and compare processed data 
    from different cities/countries
  - It is also possible to perform a linguistic random walk through the city and draw linguistic conclusions,
    by jumping between the networks of random residents in a given country or city
 

### Steps to get started:

 In order to use the code, users will need to create a Twitter account and provide OAuth settings and 
an access token: __consumer_key, consumer_secret,
access_token, access_token_secret__ in order to have access to Twitter API. These are automatically
generated once [registration](https://dev.twitter.com/apps) on Twitter API is completed. 

Keys and tokens must be stored as python variables in a file called `twitter_pwd.py` . 
A dummy file with fake keys and tokens is provided.

A number of relevant accounts for a number of cities are provided as class attributes 
of the main class. These cities are : __Barcelona, Brussels, Kiev and Riga__

__StreamTweetData__ class must be initialized specifying both a file and a country/city. If the specified city
is not among the hard-coded ones, a list of corresponding root-accounts must also be provided

1. Retrieve and save a number of followers from a given account using method 'get_account_network'. 
Specify if followers must be city or country-wide residents. repeat for all desired accounts
2. Retrieve and save a specified number of (re)tweets from each follower using 'get_tweets_from_followers'. Keep only tweets whose cleaned text is long enough for reliable
 lang detection. In addition, keep only tweets whose detected lang is the same as that specified in tweet metadata provided by API 
3. Compute and store a list of all unique followers for a given city/ country using method 'filter_root_accs_unique_followers'. 
4. Initialize __ProcessTweetData__ class to post-process tweets data and create a pandas DataFrame that summarizes all information from tweets using method 'process_data'
5. Initialize __PlotTweetData__ and use its plot methods to visualize stats per root account



   
### Where to get help with this project

Contact me on pgervila@gmail.com for any question or doubt concerning this repo. Check 
also my [blog](https://pgervila.github.io/) for more detailed descriptions
   

