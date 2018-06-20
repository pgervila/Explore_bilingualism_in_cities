## Quantifying language choice with Twitter

The project aims to quantify language choice in bilingual environments through evidence 
gathered from Twitter users

It contains:
  - A Python class with methods that access the Twitter API to stream and store tweets 
  from a number of followers of specified accounts
  - Methods to post-process tweets by language and do statistical analysis of the data. 
  - Plotting methods to visualize conclusions
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

Class must be initialized specifying both a file and a country/city. If the specified city
is not among the hard-coded ones, a list of corresponding root-accounts must also be provided

1. Retrieve and save a number of followers from a given account using method 'get_account_network'. 
Specify if followers must be city or country-wide residents. repeat for all desired accounts
2. Retrieve and save a specified number of (re)tweets from each follower using 'get_tweets_from_followers'. Keep only tweets whose cleaned text is long enough for reliable
 lang detection. In addition, keep only tweets whose detected lang is the same as that specified in tweet metadata provided by API 
3. Compute and store a list of all unique followers for a given city/ country using method 'filter_root_accs_unique_followers'. 
4. Post-process tweets data and create a pandas DataFrame that summarizes all information from tweets using method 'process_data'
5. Use plot methods to visualize stats per root account






   
### Where users can get help with your project

Contact me on pgervila@gmail.com for any question or doubt concerning this repo
   

