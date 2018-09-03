# Standard Lib
import os
import time
import re
from collections import defaultdict

# Scientific Python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Twitter API for python
import tweepy
from tweepy import OAuthHandler
# import secret codes to access Twitter API
from twitter_pwd import access_token, access_token_secret, consumer_key, consumer_secret

# language detection
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# stats
from statsmodels.stats import proportion

# progress_bar
import pyprind


class StreamTweetData:
    """ 
        Get tweets from random relevant followers that live in a given city
        and return data on language use 
    """

    city_root_accounts = dict()

    city_root_accounts['Kiev'] = {'KyivOperativ', 'kyivmetroalerts', 'auto_kiev',
                                  'ukrpravda_news', 'Korrespondent', '5channel', 'VWK668',
                                  'patrolpoliceua', 'ServiceSsu', 'segodnya_life'}

    city_root_accounts['Barcelona'] = {'TMB_Barcelona':'CAT', 'bcn_ajuntament':'CAT',
                                       'LaVanguardia':'SPA', 'hola':'SPA', 'diariARA':'CAT',
                                       'elperiodico':'SPA', 'meteocat':'CAT', 'mossos':'CAT',
                                       'sport':'SPA', 'VilaWeb':'CAT'}

    city_root_accounts['Brussels'] = {'STIBMIVB':'B'}
    city_root_accounts['Riga'] = {'Rigassatiksme_', 'nilsusakovs'}

    key_words = {'Barcelona': {'country': ['Catalu'], 'city': ['Barcel']},
                 'Kiev': {'country': ['Україна', 'Ukraine', 'Украина'],
                          'city': ['Kiev', 'Kyiv', 'Київ', 'Киев']},
                 'Brussels': {'country': ['Belg'], 'city': ['Bruxel', 'Brussel']},
                 'Riga': {'country': ['Latvija', 'Латвия', 'Latvia'],
                          'city': ['Rīg', 'Rig', 'Рига']}
                }

    langs_for_postprocess = {'Kiev': ['uk', 'ru', 'en'], 'Barcelona': ['ca', 'es', 'en'],
                             'Brussels': ['fr', 'nl', 'en', 'es', 'it', 'de', 'tr', 'pt', 'el', 'da', 'ar'],
                             'Riga': ['lv', 'ru', 'en']}

    def __init__(self, data_file_name, city, city_accounts=None, city_key_words=None, city_langs=False):
        """ Args:
                * data_file_name: string. Name of database file where data is stored from previous computations.
                    If file name is not found in current directory, a new empty file is created with
                    the same specified name
                * city: string. Name of city where bilingualism is to be analyzed
                * city_accounts: set of strings. Strings are Twitter accounts related to city
                * city_key_words: list of strings. Strings are expressions
                    to recognize the city in different languages or spellings
                * city_langs: list of strings. Strings must be valid identifiers of languages.
                    Use reference https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
        """
        if not os.path.exists(data_file_name):
            open(data_file_name, 'w+').close()

        # initialize key arguments
        self.data_file_name = data_file_name
        # check if city is already in class attributes
        if city not in self.city_root_accounts:
            if city_accounts:
                self.city_root_accounts[city] = city_accounts
            else:
                raise Exception("If a new city is specified, root accounts for this city "
                                "must be provided through 'city_accounts' argument ")
            if city_key_words:
                self.key_words[city] = {'city': city_key_words}
            else:
                raise Exception(" If a new city is specified, 'city_key_words' arg must be specified  ")
            if city_langs:
                self.langs_for_postprocess[city] = city_langs
            else:
                raise Exception(" If a new city is specified, 'city_langs' arg must be specified ")
        self.city = city

        # initialize instance attributes from database
        self.load_root_accs_unique_followers()
        self.load_tweets_from_followers()
        self.load_available_acc_nodes()

        # set_up Twitter API
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def get_account_network(self, root_account_name, rel_type='followers', max_num=100,
                            min_num_tweets=60, min_num_followers=50, only_city=False,
                            limited_search=False, avoid_repeat=None, cursor_on=False,
                            overwrite=False):
        """ Given an account by its account_name, find users that are linked to it
            via a specified relation type 'rel_type'.
            Args:
                * root_account_name: string. Twitter account name
                * rel_type: string. Specifies relation type (default is 'followers',
                    alternative value is 'friends')
                * max_num: integer. Maximum number of 'related' users considered
                * key_words: list of strings. Used to filter retrieved users by location,
                    if specified
                * min_num_tweets: minimum number of tweets a related-user needs to have
                    in order to be included in the list
                * min_num_followers: minimum number of followers a related-user needs to have
                    in order to be included in the list
                * only_city: boolean. True if only followers that are also city residents need to be taken into account.
                    If False, also country-wide residents ( not necessarily city-residents ) will be considered
                * limited_search: boolean. True .Defaults to False
                * avoid_repeat: list of strings. List of accounts that do not have to be added
                * cursor_on: boolean. Set to True if computation needs restart from an
                    intermediate point ( to avoid starting from beginning again )
                * overwrite: boolean. If True, followers data is written on already existing node
            Returns:
                * list_users: a list of account_names.
        """
        # start progress bar
        pbar = pyprind.ProgBar(max_num)
        list_users = []
        # very important to set count=200 MAX VALUE -> Max 3000 accounts per 15 minutes interval
        if not cursor_on:
            cursor = tweepy.Cursor(getattr(self.api, rel_type, 0),
                                   screen_name=root_account_name, count=200)
        else:
            node = '/'.join(['', self.city, root_account_name, rel_type])
            df_old = pd.read_hdf(self.data_file_name, node)
            cursor_id = df_old.cursor_id.values[-1]
            cursor = tweepy.Cursor(self.api.followers, screen_name=root_account_name,
                                   count=200, cursor=cursor_id)
        users = cursor.items(max_num)
        while True:
            try:
                user = next(users)
                if only_city:
                    locs = '|'.join(self.key_words[self.city]['city'])
                else:
                    locs = '|'.join(self.key_words[self.city]['country'] +
                                    self.key_words[self.city]['city'])
                patt = re.compile(locs)
                found_loc = re.findall(patt, user._json['location'])
                if (found_loc and user.statuses_count >= min_num_tweets and
                        not user.protected and user.followers_count >= min_num_followers):
                    user._json.update({'cursor_id': cursor.iterator.next_cursor})
                    if avoid_repeat:
                        if user._json['screen_name'] not in avoid_repeat:
                            list_users.append(user._json)
                    else:
                        list_users.append(user._json)
                    if len(list_users) > 2 and limited_search:
                        break
            except tweepy.TweepError as e:
                if 'Read timed out' in str(e):
                    print('fallen here')
                    print(e)
                    time.sleep(3)
                else:
                    time.sleep(60 * 16)
                    continue  # user = next(users)
            except StopIteration:
                break
            tmp_flws = pd.DataFrame(list_users)
            tmp_flws.to_pickle('_'.join(['tmp_flws', root_account_name]))
            pbar.update()
        if not limited_search:
            node = '/'.join(['', self.city, root_account_name, rel_type])
            df_new = pd.DataFrame(list_users)
            if not cursor_on:
                with pd.HDFStore(self.data_file_name) as f:
                    if node not in f.keys():
                        df_new.to_hdf(self.data_file_name, node)
                        # update list unique followers
                    else:
                        if overwrite:
                            df_new.to_hdf(self.data_file_name, node)
                        else:
                            df_new.to_pickle('_'.join(['tmp_flws', root_account_name]))
            else:
                df_old = df_old.append(df_new, ignore_index=True)
                df_old.to_hdf(self.data_file_name, node)

        return list_users

    def get_account_tweets(self, id_account, max_num_twts=60):
        """ Given an account id,
            method retrieves a specified maximum number of tweets written or retweeted by account owner.
            It returns them in a list.
            Args:
                * id_account: string. Id that identifies the twitter account
                * max_num_twts: integer. Maximum number of tweets to be retrieved for each account
            Returns:
                * list_tweets: list including info of all retrieved tweets in JSON format"""
        list_tweets = []
        timeline = tweepy.Cursor(self.api.user_timeline, id=id_account,
                                 count=200, include_rts=True).items(max_num_twts)
        while True:
            try:
                tw = next(timeline)
                list_tweets.append(tw)
            except tweepy.TweepError as e:
                if '401' in str(e):
                    print(e)
                    time.sleep(2)
                    break
                elif '404' in str(e):
                    print(e)
                    time.sleep(2)
                    break
                else:
                    time.sleep(60 * 15)
                    continue
            except StopIteration:
                break
        return list_tweets

    def get_tweets_from_followers(self, list_accounts, max_num_accounts=None,
                                  max_num_twts=60, save=True, random_walk=False):
        """ Given a list of accounts by their ids, method gets
            tweets texts per each account and their corresponding lang,
            with a maximum number of tweets per account equal to param 'max_num_twts'.
            All URLs and tweet account names are removed from tweet
            texts since they are not relevant for language identification

            * list_accounts: list of ids (strings)
            * max_num_accounts: integer. Specify in order to select only the first n accounts of list_accounts.
                Default None
            * max_num_twts: integer. Maximum number of tweets that need to be retrieved for each account.
                Default 60
            * save: boolean. True if data is to be saved . Default True
            * random_walk: boolean. Specify if tweets are being retrieved during a random walk,
                from accounts that do not belong to unique root-account followers. Default False
        """
        pbar = pyprind.ProgBar(len(list_accounts))
        texts_tweets, langs_tweets, authors_tweets, authors_id_tweets = [], [], [], []

        if max_num_accounts:
            list_accounts = list_accounts[:max_num_accounts]
        for idx, id_author in enumerate(list_accounts):
            twts = self.get_account_tweets(id_author, max_num_twts=max_num_twts)
            texts_tweets.extend([re.sub(r"(@\s?[^\s]+|https?://?[^\s]+)", "", tw.text)
                                 for tw in twts])
            langs_tweets.extend([tw.lang for tw in twts])
            authors_id_tweets.extend([id_author for _ in twts])
            authors_tweets.extend([tw.user.screen_name for tw in twts])
            if not idx % 25:
                # save temp data to avoid losses if something goes wrong...
                temp_df = pd.DataFrame({'tweets': texts_tweets,
                                        'lang': langs_tweets,
                                        'screen_name': authors_tweets,
                                        'id_str': authors_id_tweets})
                temp_df.to_pickle('_'.join([self.city, 'TMP_tweets_from_followers']))
            pbar.update()
        df_tweets = pd.DataFrame({'tweets': texts_tweets,
                                  'lang': langs_tweets,
                                  'screen_name': authors_tweets,
                                  'id_str': authors_id_tweets})
        if not df_tweets.empty:
            df_tweets['tweets'] = df_tweets['tweets'].str.replace(r"RT ", "")
            df_tweets['tweets'] = df_tweets['tweets'].str.replace(r"[^\w\s'’,.!?]+", " ")
            df_tweets['lang_detected'] = df_tweets['tweets'].apply(self.detect_refined)
        else:
            print('There are no new tweets left to be retrieved. All remaining ones are protected')
            return

        if save:
            if not random_walk:
                df_tweets.to_hdf(self.data_file_name,
                                 '/'.join(['', self.city, 'tweets_from_followers']))
            else:
                with pd.HDFStore('city_random_walks.h5') as f:
                    nodes = f.keys()
                ixs_saved_walks = []
                pattern = r"".join([self.city, "/random_walk_", "(\d+)"])
                for n in nodes:
                    try:
                        ixs_saved_walks.append(int(re.findall(pattern, n)[0]))
                    except:
                        continue
                if not ixs_saved_walks:
                    df_tweets.to_hdf(self.data_file_name,
                                     '/'.join(['', self.city, 'random_walk_1']))
                else:
                    i = max(ixs_saved_walks)
                    df_tweets.to_hdf(self.data_file_name,
                                     '/'.join(['', self.city, 'random_walk_' + str(i + 1)]))
        else:
            return df_tweets

    def get_num_flws_per_acc(self):
        """
            Stream and save number of followers of each root account from Twitter
            Output:
                * pandas series with number of followers per each account saved to database
        """
        # define database node name to store num_flws_per_acc
        node = "/".join(['', self.city, 'num_flws_main_accs'])
        num_flws_per_acc = {}
        for acc in self.city_root_accounts[self.city]:
            acc_info = self.api.get_user(acc)
            num_flws_per_acc[acc] = acc_info.followers_count
        # make pandas series and save to hdf
        num_flws_per_acc = pd.Series(num_flws_per_acc)
        num_flws_per_acc.to_hdf(self.data_file_name, node)

    def get_pct_resids_per_acc(self, sample_size=6000):
        """
            Method to quantify proportion of root account followers
            that are explicit city residents. It provides an estimate based on
            average proportion from the sample size
            Args:
                * sample_size: integer. Number of followers to stream to build the sample
                    to estimate the percentage of residents per account
            Output:
                * Method saves a pandas Series with account names as index and
                    percentages as values to a database node
        """
        pct_residents_per_acc = {}
        if not self.av_acc_nodes:
            self.get_available_nodes()
        for root_account in self.av_acc_nodes:
            list_resid_flws = self.get_account_network(root_account, max_num=sample_size, only_city=True)
            pct_residents_per_acc[root_account] = len(list_resid_flws) / sample_size
        pct_residents_per_acc = pd.Series(pct_residents_per_acc)
        node = "/".join(['', self.city, 'pct_resid_root_accs'])
        pct_residents_per_acc.to_hdf(self.data_file_name, node)

    def filter_root_accs_unique_followers(self, save=True, min_num_flws_per_acc=50,
                                          min_num_twts_per_acc=60, twts_to_flws_ratio=30):
        """
            Method to read followers from all already-computed root-account nodes and
            then compute a list of all unique followers for a given city. Method filters out those users
            that do not meet the specified conditions in terms of minimum followers and tweets per account
            Args:
                * save: boolean. If True, filtered followers are saved to hdf file. Default True
                * min_num_flws_per_acc: integer. Minimum number of followers a user
                    needs to have in order to consider it relevant. Default 50
                * min_num_twts_per_acc: integer. Minimum number of times a user
                    needs to have tweeted in order to consider it relevant. Default 60
                * twts_to_flws_ratio: integer. Maximum allowed ratio of the number of tweets
                    published by user to the number of followers the same user has.
                    This ratio is a measure of impact. Defaults to 30
            Output:
                * Unique followers are stored in instance attribute 'unique_flws'. If requested,
                    they are also saved to '/unique_followers' name
        """
        filter_words = '|'.join(self.key_words[self.city]['city'])
        # initialize frame
        df_unique_flws = pd.DataFrame()
        with pd.HDFStore(self.data_file_name) as f:
            pattern = r"".join([str(self.city), '/\w+', '(/followers)'])
            for n in f.keys():
                if re.findall(pattern, n):
                    df = pd.read_hdf(f, n)
                    df = df[(df['followers_count'] > min_num_flws_per_acc) &
                            (df['statuses_count'] / df['followers_count'] < twts_to_flws_ratio) &
                            (df['location'].str.contains(filter_words)) &
                            (df['statuses_count'] >= min_num_twts_per_acc) &
                            (~df['protected'])]
                    df_unique_flws = df_unique_flws.append(df)
            self.unique_flws = df_unique_flws.drop_duplicates('id_str').reset_index()
            if save:
                self.unique_flws.to_hdf(self.data_file_name,
                                        '/'.join(['', self.city, 'unique_followers']))

    def update_tweets_from_followers(self):
        """ Download tweets from newly detected followers and append them to saved data """
        # get sets
        self.filter_root_accs_unique_followers(save=True)
        all_flws = set(self.unique_flws.id_str)
        try:
            saved_tweets = pd.read_hdf(self.data_file_name,
                                       '/'.join(['', self.city, 'tweets_from_followers']))
        except KeyError:
            saved_tweets = pd.DataFrame()
        if not saved_tweets.empty:
            flws_with_twts = set(saved_tweets.id_str)
            # compute set difference
            new_flws = all_flws.difference(flws_with_twts)
            # get tweets from new followers if any
            if new_flws:
                new_twts = self.get_tweets_from_followers(new_flws, save=False)
                # append new tweets
                saved_tweets = saved_tweets.append(new_twts, ignore_index=True)
                # save
                saved_tweets.to_hdf(self.data_file_name,
                                    '/'.join(['', self.city, 'tweets_from_followers']))
        else:
            self.get_tweets_from_followers(all_flws, save=True)

    def load_root_accs_unique_followers(self):
        """
            Method that loads all root accounts' unique followers from database and
            assigns them to class attribute as a pandas dataframe
            Output:
                * pandas dataframe with unique followers is assigned to 'unique_flws' instance attribute
        """
        node = '/'.join(['', self.city, 'unique_followers'])
        try:
            self.unique_flws = pd.read_hdf(self.data_file_name, node)
        except KeyError:
            self.unique_flws = None

    def load_tweets_from_followers(self, filter=True):
        """
            Method to load all tweets from followers of all root accounts into a pandas dataframe
            Args:
                * filter: boolean. True if only tweets relevant for stats are to be
                    considered (lang correctly identified, lang is one of the requested ones, ...).
                    Default True
            Output:
                * Assigns dataframe to 'tweets_from_followers' instance attribute. If data is not in
                    the database, it assigns None
        """
        node = '/'.join(['', self.city, 'tweets_from_followers'])
        try:
            tff = pd.read_hdf(self.data_file_name, node)
            langs = self.langs_for_postprocess[self.city]
            if filter:
                if self.city == 'Barcelona':
                    tff.lang[tff.lang == 'und'] = 'ca'
                tff = tff[tff.lang.isin(langs)]
                tff = tff[tff.lang == tff.lang_detected]
            self.tweets_from_followers = tff
        except KeyError:
            self.tweets_from_followers = None

    def load_available_acc_nodes(self):
        """ Method to load all account nodes available in saved database.
            Output:
                * Resulting nodes will be saved as an instance attribute in self.av_acc_nodes
        """
        with pd.HDFStore(self.data_file_name) as f:
            self.av_acc_nodes = []
            pattern = r"".join([self.city, '/(\w+)', '/followers'])
            for n in f.keys():
                acc = re.findall(pattern, n)
                if acc and acc[0] in self.city_root_accounts[self.city]:
                    self.av_acc_nodes.append(acc[0])

    @staticmethod
    def detect_refined(txt):
        """ Method to deal with exceptions when detecting tweet languages
            Args:
                * txt : string. Tweet text
            Output:
                * string with tweet language or 'Undefined' label if insufficient text is present
        """
        if len(txt.split()) > 2 and len(txt) > 10:
            try:
                return detect(txt)
            except LangDetectException:
                return 'Undefined'
        else:
            return 'Undefined'

    def optimize_saving_space(self):
        """
            Method to rewrite hdf data file in order to optimize file size.
            It creates a new file and deletes old one. It keeps original file name
        """
        with pd.HDFStore(self.data_file_name) as f:
            for n in f.keys():
                data = pd.read_hdf(f, n)
                data.to_hdf('new_city_random_walks.h5', n)
        os.remove(self.data_file_name)
        os.rename('new_city_random_walks.h5', self.data_file_name)


class ProcessTweetData(StreamTweetData):

    def __init__(self, data_file_name, city, update=False):
        """
        Args:
            * data_file_name: string. Name of database file where data is stored from previous computations.
                    If file name is not found in current directory, a new empty file is created with
                    the same specified name
            * city: string. Name of city where bilingualism is to be analyzed
            * update: boolean. True if new data is available in database and 'process_data' needs to be called
                        to create a pandas DataFrame that summarizes all information. Default False
        """
        super().__init__(data_file_name, city)

        # set values from saved data
        self.define_num_resids_per_acc()

        # initialize class-specific instance attributes
        self.data_stats = pd.DataFrame()
        self.lang_settings_per_root_acc = defaultdict(dict)
        self.stats_per_root_acc = dict()
        self.weighted_samples_distribs = dict()

        if update:
            self.process_data(max_num_langs=3)
        else:
            data_file_path = os.path.join(self.city, self.city + '_data_stats.json')
            self.data_stats = pd.read_json(data_file_path)

        # compute root accs stats
        self.compute_stats_per_root_acc(save=False, load_data=True)

    def load_sample_size_per_root_acc(self):
        """
            Method to read the number of followers per account RELEVANT
            for statistic analysis of the languages used in (re)tweets
        """

        if self.data_stats.empty:
            try:
                self.data_stats = pd.read_json(self.city + '_data_stats.json')
            except ValueError:
                print('Requested file is not in current directory !')
        self.sample_size_per_root_acc = {acc: self.data_stats[self.data_stats[acc]].shape[0]
                                         for acc in self.av_acc_nodes}

    def load_num_flws_per_acc(self):
        """ Method to load the number of followers for each account """
        node = "/".join(['', self.city, 'num_flws_main_accs'])
        return pd.read_hdf(self.data_file_name, node)

    def load_pct_resids_per_acc(self, sample_size=6000):
        """
            Method to load average percentage of residents per account
            It adds standard deviation to each proportion value
            Output:
                * returns a two-columns dataframe with the requested data
        """
        node = "/".join(['', self.city, 'pct_resid_root_accs'])
        df = pd.read_hdf(self.data_file_name, node).sort_index().to_frame()
        df.columns = ['pct_resids']
        # get standard deviation of each mean proportion
        df['std_pct_resids'] = df['pct_resids'].apply(proportion.std_prop, args=(sample_size,))
        return df

    def process_data(self, num_tweets_for_stats=40, save=True, max_num_langs=2):
        """
            Method that defines a pandas DataFrame that summarizes all language-related information
            from available tweets. The DataFrame has a column with users id
            and root account columns with boolean values that define whether users
            are followers of root account or not. For each user and for each considered language,
            the average percentage of tweets in the language and the corresponding 95% confidence intervals
            are calculated, using a sample of tweets of size N = 'num_tweets_for_stats'

            Arguments:
                * num_tweets_for_stats: integer >= 40 and <= 60. Number of tweets that will be taken into
                    account for each follower.
                * save: boolean. Specifies whether to save the processed data or not. Defaults to True.
                * max_num_langs: integer. Maximum number of languages to be processed. Default 2

            Output:
                * Method sets value for self.data_stats and saves file to specified directory

        """
        langs_selected = self.langs_for_postprocess[self.city]
        self.load_tweets_from_followers()
        tff = self.tweets_from_followers
        # define function to select only users with > num_tweets_for_stats
        fun_filter = lambda x: len(x) >= num_tweets_for_stats
        tff = tff.groupby('screen_name', as_index=False).filter(fun_filter)
        # define fun to select num_tweets_for_stats per user
        fun_select = lambda obj: obj.iloc[:num_tweets_for_stats, :]
        tff = tff.groupby('screen_name', as_index=False).apply(fun_select)
        # rearrange dataframe usig pivot table
        self.data_stats = tff.pivot_table(aggfunc={'lang': 'count'},
                                          index=['id_str', 'screen_name'],
                                          columns='lang_detected').fillna(0.)
        self.data_stats = self.data_stats.lang.reset_index()
        self.data_stats = self.data_stats[['id_str', 'screen_name'] + langs_selected]
        self.data_stats['tot_lang_counts'] = self.data_stats[langs_selected].sum(axis=1)

        # generate boolean columns with followers for each account
        for root_acc in self.av_acc_nodes:
            root_acc_ids = pd.read_hdf(self.data_file_name,
                                       '/'.join(['', self.city, root_acc, 'followers'])).id_str
            self.data_stats[root_acc] = self.data_stats.id_str.isin(root_acc_ids)

        for lang in self.langs_for_postprocess[self.city][:max_num_langs]:
            # Replace with confint
            mean = lang + '_mean'
            max_cint = lang + '_max_cint'
            min_cint = lang + '_min_cint'

            # get expected value for each account and language
            self.data_stats[mean] = self.data_stats[lang] / self.data_stats['tot_lang_counts']
            # compute confidence intervals
            intervs = self.data_stats.apply(lambda x: proportion.proportion_confint(x[lang],
                                                                                    x['tot_lang_counts'],
                                                                                    method='jeffreys'), axis=1)
            # split 2d-tuples into 2 columns
            intervs = intervs.apply(pd.Series)
            self.data_stats[min_cint], self.data_stats[max_cint] = intervs[0], intervs[1]

        if save:
            data_file = os.path.join(self.city, self.city + '_data_stats.json')
            self.data_stats.to_json(data_file)

    def compute_stats_per_root_acc(self, save=True, load_data=False):
        """
            Method to compute basic stats of language use (mean, median and their conf intervals ...)
            of the sample of followers of each root account
            Args:
                * save: boolean. Default True
                * load_data: boolean
            Output:
                * sets value to instance attribute 'stats_per_root_acc'

        """
        if load_data:
            try:
                data_file = os.path.join(self.city, self.city + '_stats_per_root_acc.pickle')
                self.stats_per_root_acc = pd.read_pickle(data_file)
            except FileNotFoundError:
                print('data file is not available in specified directory')
        else:
            data_frames_per_lang = dict()
            for lang in self.langs_for_postprocess[self.city]:
                self.stats_per_root_acc = dict()
                for root_acc in self.av_acc_nodes:
                    data = self.data_stats[self.data_stats[root_acc]]
                    if data.shape[0] >= 100:
                        # compute confint of the mean
                        conf_int = proportion.proportion_confint(data[lang].sum(),
                                                                 data.tot_lang_counts.sum(),
                                                                 method='jeffreys', alpha=0.01)
                        mean_conf_int = pd.Series(conf_int,
                                             index=['min_confint', 'max_confint'])
                        # compute confint of the median
                        median_conf_int = pd.Series(self.compute_median_conf_int(root_acc, lang),
                                                    index=['median_inf', 'median_sup'])
                        # merge mean and median confidence intervals
                        conf_int = mean_conf_int.append(median_conf_int)

                        self.stats_per_root_acc[root_acc] = (data[lang] / data.tot_lang_counts).describe().append(conf_int)
                data_frames_per_lang[lang] = pd.DataFrame(self.stats_per_root_acc).transpose()
            self.stats_per_root_acc = pd.concat(data_frames_per_lang, axis=1)

            if save:
                data_file_path = os.path.join(self.city, self.city + '_stats_per_root_acc.pickle')
                self.stats_per_root_acc.to_pickle(data_file_path)

    def define_num_resids_per_acc(self):
        """
        Method to obtain estimate of number of explicit residents in city per each account
        Args:
            * update: boolean. If False, method loads already available results
        Output:
            * Method assigns 4-columns pandas dataframe to instance attribute 'num_resids_per_acc'
        """

        self.num_resids_per_acc = self.load_num_flws_per_acc().to_frame()
        self.num_resids_per_acc.columns = ['flws']
        # join number of followers followers with mean percentage of residents and its stdev
        self.num_resids_per_acc = self.num_resids_per_acc.join(self.load_pct_resids_per_acc())
        self.num_resids_per_acc['avg_resids'] = (self.num_resids_per_acc['flws'] *
                                                 self.num_resids_per_acc['pct_resids'])

    def compute_rand_num_resids_per_acc(self):
        """
            Method to compute a random sample of the number of city residents
            per account using mean value and its standard deviation
            Output:
                * pandas Series with random number of residents per account
        """
        random_sample = np.random.normal(self.num_resids_per_acc['pct_resids'],
                                         self.num_resids_per_acc['std_pct_resids'])
        return random_sample * self.num_resids_per_acc['flws']

    def compute_lang_settings_stats_per_root_acc(self, city_only=True):
        """ Find distribution of lang settings for each root account
            and , if requested, for users residents in the city only
            Args:
                * city_only: boolean. True if settings have to be retrieved only for users
                    from class instance city. Default True
            Output:
                * sets value to instance attribute 'lang_settings_per_root_acc'
        """
        # TODO: group data by lang using hierarchical columns instead of column suffix

        stats_per_lang = dict()
        acc_data = defaultdict(dict)
        for root_acc in self.av_acc_nodes:
            node = "/".join([self.city, root_acc, 'followers'])
            df = pd.read_hdf('city_random_walks.h5', node)
            if city_only:
                df = df[df.location.str.contains("|".join(self.key_words[self.city]['city']))]
            counts = df.lang.value_counts()
            sample_means = counts / counts.sum()
            for lang in self.langs_for_postprocess[self.city]:
                acc_data[lang][root_acc] = {'num_accs': counts.sum()}
                sample_mean = sample_means[lang]
                acc_data[lang][root_acc]['mean'] = 100 * sample_mean
                min_confint, max_confint = proportion.proportion_confint(counts[lang], counts.sum(),
                                                                         alpha=0.01, method='jeffreys')
                acc_data[lang][root_acc]['min_confint'] = 100 * min_confint
                acc_data[lang][root_acc]['max_confint'] = 100 * max_confint

        for lang in self.langs_for_postprocess[self.city]:
            stats_per_lang[lang] = pd.DataFrame(acc_data[lang]).transpose()

        self.lang_settings_per_root_acc = pd.concat(stats_per_lang, axis=1)

    def quantify_accs_similarity(self, ref_key, other_key):
        """
            Method to compute the degree of similarity between followers of ref_key account
            as compared to those of another account specified in 'other_key'. It computes the
            percentage of common followers relative to reference account
        """
        s_ref = set(self.data_stats[self.data_stats[ref_key]].id_str)
        s2 = set(self.data_stats[self.data_stats[other_key]].id_str)
        s_int = s_ref.intersection(s2)
        return len(s_int) / len(s_ref)

    def compute_weighted_sample_users(self, rand_seed=42, min_size=100, use_avg_resids=False):
        """
            Method to obtain a weighted sample of followers from all accounts.
            Each account will contribute with a sample of followers.
            A minimum of 'min_size' followers is considered for the least popular account.
            Rest of accounts sample sizes are computed multiplying the minimum size
            by a factor that is the ratio of resident followers of each root account to that of the least
            popular root account in the city
            Args:
                * rand_seed: integer. Seed to reproduce random sampling
                * min_size: minimum sample size from the account with least residents
                * use_avg_resids: boolean. True if weights are to be computed based on average residents per account,
                    false if residents per account are to be considered a random variable that has to be sampled
                    every time the function is called
            Output:
                * numpy array of unique ids of weighted sample
        """
        # TODO : add new column with weighted sample to data_stats dataframe ???

        # set random seed
        np.random.seed(rand_seed)
        # Scale num residents of each account relative to account with the least residents ( divide by num residents
        # of the account with the least residents )
        if use_avg_resids:
            resids_per_acc = self.num_resids_per_acc['avg_resids']
        else:
            resids_per_acc = self.compute_rand_num_resids_per_acc()
        sample_size_per_acc = (min_size * resids_per_acc /
                               resids_per_acc[resids_per_acc.argmin()]).astype(int)

        # construct global weighted sample of users ids
        ids_weighted_sample = []
        for (acc, sample_size) in sample_size_per_acc.iteritems():
            acc_sample = self.data_stats[self.data_stats[acc]].sample(sample_size).id_str
            ids_weighted_sample.extend(acc_sample)
        ids_weighted_sample = np.unique(ids_weighted_sample)

        return ids_weighted_sample

    def compute_weighted_sample_props(self, lang, rand_seed=42, min_size=100, use_avg_resids=False):
        """
            Method to obtain, for a given lang, the weighted distribution of users' lang proportions
            Args:
                * lang: string. Use same code as in rest of module
                * rand_seed: positive integer. Seed to reproduce random sample
                * min_size: positive integer. Minimum sample size from the account with least residents
                * use_avg_resids: boolean. True if weights are to be computed based on average residents per account,
                    false if residents per account are to be considered a random variable that has to be sampled
                    every time the function is called
            Output:
                * pandas Series with percentage values of language use for selected followers
        """
        # get random weighted sample
        users = self.compute_weighted_sample_users(rand_seed=rand_seed, min_size=min_size,
                                                   use_avg_resids=use_avg_resids)
        # get mean column for specified language
        df_column = '{}_mean'.format(lang)
        return self.data_stats.loc[self.data_stats.id_str.isin(users)][df_column]

    def compute_stats_props_distrib(self, lang, distrib=None):
        """
            Method that computes, for a given language, the median and mean user of a given 1D distribution or
            collection of 1D lang proportions distributions, as well as percentage of users that mostly use
            the requested languages.
            Args:
                * lang: string. Language for which stats are requested
                * distrib: numpy array (1D or 2D). Collection of language proportions that define a distribution.
                    If None, the weighted distribution from all accounts with average residents is considered.
                    Default None
            Output:
                * numpy array with the following ordered values for each column:
                   - percentage of use of the requested language made by median user of the weighted distribution
                   - percentage of use of the requested language made by average user of the weighted distribution
                   - percentage of users of the weighted distribution that mostly use the requested language
                   - percentage of users that use the requested language as much as other languages
        """
        if distrib is None:
            distrib = self.compute_weighted_sample_props(lang, use_avg_resids=True)

        if distrib.ndim == 2:
            median = np.median(distrib, axis=1)
            mean = np.mean(distrib, axis=1)
            pct_more_in_lang = 100 * (distrib >= 0.55).sum(axis=1) / distrib.shape[1]
            pct_as_much_in_lang = 100 * ((distrib >= 0.45) & (distrib < 0.55)).sum(axis=1) / distrib.shape[1]
            stats = np.column_stack((median, mean, pct_more_in_lang, pct_as_much_in_lang))
        else:
            median = np.median(distrib)
            mean = np.mean(distrib)
            pct_more_in_lang = 100 * (distrib >= 0.55).sum() / distrib.shape[0]
            pct_as_much_in_lang = 100 * ((distrib >= 0.45) & (distrib < 0.55)).sum() / distrib.shape[0]
            stats = np.array([median, mean, pct_more_in_lang, pct_as_much_in_lang])

        return stats

    def compute_median_conf_int(self, acc, lang, num_samples=1000):

        """
            Method to obtain, for a specified language and account, the distribution of the median
            language choice of the account followers'. A 95% confidence interval is calculated
            Args:
                * acc: string. Name of root account
                * lang: string. Language for which the median is to be estimated
                * num_samples: integer. Number of samples to be generated for each proportion value

        """
        # TODO: if no account specified, apply procedure to weighted sample ( get weighted sample)

        # get computed proportions from followers for given account
        props = self.data_stats[self.data_stats[acc]][lang + "_mean"].values
        # knowing size is always 40, get standard deviation of the proportion for each account follower
        stds = proportion.std_prop(props, 40)
        # generate new props using mean and std for each follower (assuming normal distrib)
        generated_props = np.random.normal(props, stds, size=(num_samples, props.size))
        # compute medians of each generated sample ( need to clip if generated values
        # are less than zero or more than one)
        generated_medians = np.median(np.clip(generated_props, 0, 1), axis=1)
        # compute confidence interval 95%
        mean_medians = np.mean(generated_medians)
        std_medians = np.std(generated_medians)
        return mean_medians - 1.96 * std_medians, mean_medians + 1.96 * std_medians

    def resample_weighted_sample(self, num_w_samps=100, num_props_samps=100, min_size=100):
        """
            Method to simulate random samples of the weighted average of accounts.
            Both weights and users' lang proportions are repeatedly sampled.
            Observed lang proportions and its standard deviations are used
            to generate a normal distributions for each proportion and for each relevant lang
            Args:
                * num_w_samps: integer. Requested number of resamples of accounts' weights
                * num_props_samps: integer. Requested number of resamples for each follower language proportion
                * min_size: integer. Minimum size of the smallest sample (because of limited number of
                    available followers per root account, resample weights need to scale and fit with respect
                    to changing relative sizes. If the requested size of the least important account at each
                    resample is too high, there might not be sufficient available followers for the most relevant
                    account, i.e those with higher weights )
            Output:
                * assigns result of num_w_samps * num_props_samps experiments to a pandas DataFrame
                 with 4 columns : median, mean, pct_more_in_lang, pct_as_much_in_lang. The 'weighted_samples_distribs'
                 instance attribute is set to the DataFrame value, with a different key for each language
        """

        for lang in self.langs_for_postprocess[self.city][:3]:
            list_w_samps = []
            for seed in np.random.randint(10000, size=num_w_samps):
                # get lang ratios for a given seed
                props = self.compute_weighted_sample_props(lang, rand_seed=seed,
                                                           min_size=min_size).values
                # compute standard deviation for each proportion from sample ratio and number of samples (40)
                stds = proportion.std_prop(props, 40)
                # simulate new probabilistic proportions
                generated_props = np.random.normal(props, stds, size=(num_props_samps, props.size))

                stats = self.compute_stats_props_distrib(lang, distrib=generated_props)

                list_w_samps.append(stats)

            distribs = np.concatenate(list_w_samps)
            self.weighted_samples_distribs[lang] = pd.DataFrame(distribs,
                                                                columns=['medians', 'means',
                                                                         'pct_more_in_lang', 'pct_as_much_in_lang'])
        self.weighted_samples_distribs = pd.concat(self.weighted_samples_distribs, axis=1)

    def random_walk(self):
        """
            Select a list of accounts by randomly walking
            all main followers' friends and followers
        """
        # load main followers
        self.load_root_accs_unique_followers()

        # get random sample from main followers
        sample = np.random.choice(self.unique_flws['screen_name'], 10, replace=False)

        # get a random follower and friend from each account from sample
        # ( check they do not belong to already met accounts and main followers !!)
        all_flws = []
        for acc in sample:
            # look for friend and follower
            list_flws = self.get_account_network(acc, min_num_tweets=10,
                                                 only_city=True,
                                                 limited_search=True, avoid_repeat=all_flws)
            all_flws.extend(list_flws)
            list_friends = self.get_account_network(acc, min_num_tweets=10,
                                                    rel_type='friends', only_city=True,
                                                    limited_search=True, avoid_repeat=all_flws)
            all_flws.extend(list_flws)
        self.random_walk_accounts = pd.DataFrame(all_flws)
        print('starting to retrieve tweets')
        self.get_tweets_from_followers(self.random_walk_accounts["id_str"],
                                       max_num_twts=20, save=True,
                                       random_walk=True)


class PlotTweetData(ProcessTweetData):

    def __init__(self, data_file_name, city=None):
        super().__init__(data_file_name, city)
        mpl.rcParams['figure.figsize'] = (15, 10)

    def plot_flws_resids_per_acc(self):
        """
            Method to plot number of total followers and of city-resident followers per account
        """
        # define plot style
        mpl.style.use('seaborn')
        import matplotlib.ticker as mpltick

        # define plot data
        # get idxs to sort accs by num residents
        arg_sorted = self.num_resids_per_acc['avg_resids'].argsort()
        # sort data
        num_flws_per_acc = self.load_num_flws_per_acc().iloc[arg_sorted]
        num_resids_per_acc = self.num_resids_per_acc['avg_resids'].iloc[arg_sorted]
        data = [num_flws_per_acc, num_resids_per_acc]

        # define plot
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        frames = [ax, ax2]
        bar_width = 0.6
        colors = ['black', 'orange']
        X = np.arange(0, 2 * num_flws_per_acc.size, 2)
        labels = ['num_flws', 'num_resid_flws']
        errors_flag = [False, True]

        # define errors for resident estimates

        # node1 = "/".join(['', self.city, 'pct_resid_root_accs'])
        # pct_resids_per_acc = pd.read_hdf(self.data_file_name, node1).sort_index()

        df_errors = pd.DataFrame({key: proportion.proportion_confint(val * 6000, 6000)
                                  for key, val in self.pct_resids_per_acc.iteritems()}).T
        df_errors.columns = ['min_confint', 'max_confint']
        df_errors = df_errors.iloc[arg_sorted].mul(self.load_num_flws_per_acc().iloc[arg_sorted], axis=0)
        errors = df_errors['max_confint'].subtract(self.num_resids_per_acc['avg_resids'], axis=0).iloc[arg_sorted]

        for i, (data, frame, color, label, plot_error) in enumerate(zip(data, frames, colors, labels, errors_flag)):
            if plot_error:
                error = errors.values
                errs = [error, error]
                frame.bar(X + i * bar_width, data, yerr=errs, align='center', edgecolor='black', color=color,
                          width=bar_width, label=label)
            else:
                frame.bar(X + i * bar_width, data, align='center', edgecolor='black', color=color,
                          width=bar_width, label=label)

        # set up axis1
        labels_font_size = 8
        ax.set_xticks(X + bar_width / 2)
        ax.set_xticklabels(num_flws_per_acc.index, rotation=45, fontsize=labels_font_size)
        ax.set_ylabel('Total_followers', color=colors[0])
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_formatter(mpltick.EngFormatter(unit='', places=None))
        ax.tick_params(axis='y', labelsize=8)
        for t in ax.get_yticklabels():
            t.set_color(colors[0])
        ax.xaxis.grid(False)
        ax.yaxis.grid(linestyle='--', alpha=0.6)

        # set up axis2
        ax2.set_ylabel('Resident_followers', color=colors[1])
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax2.yaxis.set_major_formatter(mpltick.EngFormatter(unit='', places=None))
        ax2.tick_params(axis='y', labelsize=8)
        for t in ax2.get_yticklabels():
            t.set_color(colors[1])
        ax2.grid(False)

        # set legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=9)

        # title
        plt.title('Number of followers and number of residents per account in {}'.format(self.city),
                  family='serif', fontsize=10)

        # save figure
        plt.tight_layout()
        fig_name = 'num_flws_resids_per_acc_in_{}'.format(self.city)
        plt.savefig(os.path.join(self.city, 'figures', fig_name))

        plt.show()

    def plot_lang_settings_per_acc(self, city_only=True, max_num_langs=3,
                                   single_account=False, min_num_accs=200):
        """
            Method to visualize statistics on language settings in accounts from followers of root accounts
            Args:
                * city_only: boolean. True if only followers whose location explicitly mentions
                    the city are to be taken into account. False if followers are country-wide,
                    not explicitly from city.  Default True
                * min_num_accs: integer. Minimum number of available followers per root account
                    to consider statistics for it as relevant
                * max_num_langs: integer > 0. Maximum number of languages considered for each root account
                * single_account: boolean. If True, plot will be for a single root account only
        """

        self.compute_lang_settings_stats_per_root_acc(city_only=city_only)
        # set plot style
        mpl.style.use('seaborn')

        if single_account:
            sorted_lang_settings = self.lang_settings_per_root_acc.T.iloc[:, 0][:, 'mean'].sort_values(ascending=False)
            sorted_langs = sorted_lang_settings.index
            fig, ax = plt.subplots()
            bar_width = 0.6
            for i, lang in enumerate(sorted_langs[:max_num_langs]):
                plot_data = self.lang_settings_per_root_acc[lang]
                data = plot_data['mean']
                err_up = (plot_data['max_confint'] - plot_data['mean']).abs()
                err_down = (plot_data['min_confint'] - plot_data['mean']).abs()
                ax.bar(i, data, yerr=[err_down, err_up],
                       align='center', edgecolor='black', color='blue', alpha=0.7,
                       width=bar_width, capsize=3)
            labels_font_size = 8
            ax.set_xticks(np.arange(max_num_langs))
            ax.set_xticklabels(sorted_langs[:max_num_langs], fontsize=labels_font_size)
            ax.set_xlabel('Language', fontsize=labels_font_size, fontweight='bold')
            ax.set_ylabel('Percentage of followers, %', fontsize=labels_font_size, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=labels_font_size)
            ax.xaxis.grid(False)
            ax.yaxis.grid(linestyle='--', alpha=0.8)
            title = 'Twitter language settings of @{} followers'.format(self.lang_settings_per_root_acc.index[0])
            if city_only:
                title = title + ' from {}.'.format(self.city)
            sample_size = int(self.lang_settings_per_root_acc.iloc[0][lang, 'num_accs'])
            ax.set_title(title + ' (Sample size: {})'.format(sample_size),
                         family='serif', fontsize=10)
            plt.tight_layout()

            if city_only:
                fig_name = 'lang_settings_for_@{}_followers_in_{}'.format(self.lang_settings_per_root_acc.index[0],
                                                                          self.city)
            else:
                fig_name = 'lang_settings_for_@{}_followers'.format(self.lang_settings_per_root_acc.index[0])
            plt.savefig(os.path.join(self.city, 'figures', fig_name))
        else:
            # sort data for better visualization
            self.lang_settings_per_root_acc = self.lang_settings_per_root_acc.sort_values(
                by=(self.langs_for_postprocess[self.city][0], 'mean'))

            bar_width = 0.4
            colors = ['green', 'blue', 'red', 'yellow', 'orange']

            fig, ax = plt.subplots()

            for i, lang in enumerate(self.langs_for_postprocess[self.city][:max_num_langs]):
                plot_data = self.lang_settings_per_root_acc[lang][self.lang_settings_per_root_acc[lang].num_accs >=
                                                                  min_num_accs]
                X = np.arange(0, 2 * plot_data.index.shape[0], 2)
                data = plot_data['mean']
                err_up = (plot_data['max_confint'] - plot_data['mean']).abs()
                err_down = (plot_data['min_confint'] - plot_data['mean']).abs()
                ax.bar(X + i * bar_width, data, yerr=[err_down, err_up], width=bar_width,
                       align='center', edgecolor='black', label=lang, color=colors[i], alpha=0.7,
                       capsize=3)
            labels_font_size = 8
            ax.set_xticks(X + bar_width / 2)
            ax.set_xticklabels(plot_data.index, rotation=45, fontsize=labels_font_size)
            ax.set_ylabel('percentage, %', fontsize=labels_font_size)
            ax.tick_params(axis='both', which='major', labelsize=labels_font_size)
            ax.legend(fontsize=10, loc='best')
            ax.set_title('Twitter language settings of root-account followers in ' + self.city,
                         family='serif', fontsize=10)
            ax.xaxis.grid(False)
            ax.yaxis.grid(linestyle='--', alpha=0.6)
            plt.tight_layout()
            # save figure
            fig_name = 'lang_settings_in_{}'.format(self.city)
            plt.savefig(os.path.join(self.city, 'figures', fig_name))

            plt.show()

            # self.get_lang_settings_stats_per_root_acc(city_only=False)

    def plot_lang_pcts_per_acc(self, single_account=False, max_num_langs=3):
        """
            Method to visualize percentages of the three main languages of (re)tweets by followers of root accounts
                * single_account: boolean. True if data is for a single account only. Default False.
        """
        if not isinstance(self.stats_per_root_acc, pd.DataFrame):
            self.compute_stats_per_root_acc()

        if single_account:
            bar_width = 0.6
            sorted_lang_props = self.stats_per_root_acc.T.iloc[:, 0][:, 'mean'].sort_values(ascending=False)
            sorted_langs = sorted_lang_props.index
            fig, ax = plt.subplots()

            for i, lang in enumerate(sorted_langs[:max_num_langs]):
                plot_data = self.stats_per_root_acc[lang]
                data = 100 * plot_data['mean']
                err_up = 100 * (plot_data['max_confint'] - plot_data['mean']).abs()
                err_down = 100 * (plot_data['min_confint'] - plot_data['mean']).abs()
                ax.bar(i, data, yerr=[err_down, err_up],
                       align='center', edgecolor='black', color='blue', alpha=0.7,
                       width=bar_width, capsize=3)
            labels_font_size = 8
            ax.set_xticks(np.arange(max_num_langs))
            ax.set_xticklabels(sorted_langs[:max_num_langs], fontsize=labels_font_size)
            ax.set_xlabel('Language', fontsize=labels_font_size, fontweight='bold')
            ax.set_ylabel('Percentage of tweets, %', fontsize=labels_font_size, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=labels_font_size)
            ax.xaxis.grid(False)
            ax.yaxis.grid(linestyle='--', alpha=0.8)
            lang = self.langs_for_postprocess[self.city][0]
            sample_size = int(self.stats_per_root_acc[lang]['count'][0])
            ax.text(.7, .9, "Sample size: {} tweets per user, {} users".format(40, sample_size),
                    transform=ax.transAxes, fontsize=7)

            ax.set_title('Languages of (re)tweets by @{} '
                         'followers from {}'.format(self.stats_per_root_acc.index[0], self.city),
                         family='serif', fontsize=10)
            plt.tight_layout()
            # save figure
            fig_name = 'lang_ptcs_tweets_by_@{}_followers_in_{}'.format(self.stats_per_root_acc.index[0], self.city)
            plt.savefig(os.path.join(self.city, 'figures', fig_name))

        else:
            # get stats from relevant root accounts sorted by mean value
            self.stats_per_root_acc = self.stats_per_root_acc.sort_values(
                by=(self.langs_for_postprocess[self.city][0], 'mean'))

            mpl.style.use('seaborn')
            fig, ax = plt.subplots()
            #
            bar_width = 0.4
            colors = ['green', 'blue', 'red']
            X = np.arange(0, 2 * self.stats_per_root_acc.shape[0], 2)
            for i, lang in enumerate(self.langs_for_postprocess[self.city]):
                plot_data = self.stats_per_root_acc[lang]
                data = 100 * plot_data['mean']
                err_up = 100 * (plot_data.max_confint - plot_data['mean'])
                err_down = 100 * (plot_data.min_confint - plot_data['mean']).abs()
                ax.bar(X + i * bar_width, data, yerr=[err_down, err_up], width=bar_width,
                       align='center', edgecolor='black', label=lang, color=colors[i], alpha=0.7,
                       capsize=2)
            labels_font_size = 8
            ax.set_xticks(X + bar_width / 2)
            ax.set_xticklabels(plot_data.index, rotation=45, fontsize=labels_font_size)
            ax.set_ylabel('Percentage of tweets, %', fontsize=labels_font_size)
            ax.tick_params(axis='both', which='major', labelsize=labels_font_size)
            ax.legend(fontsize=10, loc='best')
            ax.set_title('Language of (re)tweets by account followers from ' + self.city,
                         family='serif', fontsize=10)
            ax.xaxis.grid(False)
            ax.yaxis.grid(linestyle='--', alpha=0.6)
            plt.tight_layout()
            # save figure
            fig_name = 'percentage_of_each_lang_per_account_in_{}'.format(self.city)
            plt.savefig(os.path.join(self.city, 'figures', fig_name))
            plt.show()

    def plot_lang_distribs_per_acc(self):
        """
            Method to plot the distributions of fraction
            of tweets in each language from followers of each root account
        """
        if not isinstance(self.stats_per_root_acc, pd.DataFrame):
            self.compute_stats_per_root_acc()
        # get relevant root accounts sorted by mean value
        lang_to_sort_by = self.langs_for_postprocess[self.city][0]
        ticks_labels = self.stats_per_root_acc[lang_to_sort_by].sort_values(by='mean').index.tolist()

        data = {self.langs_for_postprocess[self.city][0]: [],
                self.langs_for_postprocess[self.city][1]: []}
        for lang in self.langs_for_postprocess[self.city][:2]:
            for root_acc in ticks_labels:
                acc_data = self.data_stats[self.data_stats[root_acc]]
                acc_data = (acc_data[lang] / acc_data.tot_lang_counts).values
                data[lang].append(acc_data)



        mpl.style.use('seaborn')
        # plt.figure()

        fig, ax = plt.subplots()

        data_a = data[self.langs_for_postprocess[self.city][0]]
        data_b = data[self.langs_for_postprocess[self.city][1]]

        bar_width = 0.5
        tick_dist = 4
        # colors = ['#D7191C', '#2C7BB6']
        colors = ['green', 'blue']
        bpl = ax.boxplot(data_a, positions=np.array(range(len(data_a))) * tick_dist - bar_width,
                         sym='', widths=bar_width, showmeans=True, patch_artist=True,
                         whiskerprops=dict(linestyle='--', linewidth=0.5, alpha=0.5),
                         medianprops=dict(linewidth=2, color='black'),
                         meanprops=dict(marker="o", markeredgecolor='black', markerfacecolor="None"))
        bpr = ax.boxplot(data_b, positions=np.array(range(len(data_b))) * tick_dist + bar_width,
                         sym='', widths=bar_width, showmeans=True, patch_artist=True,
                         whiskerprops=dict(linestyle='--', linewidth=0.5, alpha=0.5),
                         medianprops=dict(linewidth=2, color='black'),
                         meanprops=dict(marker="o", markeredgecolor='black', markerfacecolor="None"))

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            # plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            # plt.setp(bp['medians'], color=color)
        set_box_color(bpl, colors[0])
        set_box_color(bpr, colors[1])

        # draw temporary red and blue lines and use them to create a legend
        ax.plot([], c=colors[0], label=self.langs_for_postprocess[self.city][0])
        ax.plot([], c=colors[1], label=self.langs_for_postprocess[self.city][1])
        ax.legend(loc='best')

        # grid
        ax.xaxis.grid(False)
        ax.yaxis.grid(linestyle='--', alpha=0.5)

        X = np.arange(0, len(ticks_labels) * tick_dist, tick_dist)

        labels_font_size = 8
        ax.set_xticks(X)
        ax.set_xticklabels(ticks_labels, rotation=45, fontsize=labels_font_size)
        ax.set_xlim(-tick_dist, len(ticks_labels) * tick_dist)
        ax.set_ylabel('fraction of tweets in lang', fontsize=labels_font_size)
        ax.tick_params(axis='both', which='major', labelsize=labels_font_size)
        ax.set_title('Language choice distributions from followers of root-accounts in ' + self.city,
                     family='serif', fontsize=10)

        fig.tight_layout()
        # save figure
        fig_name = 'lang_distribs_per_acc_in_{}'.format(self.city)
        plt.savefig(os.path.join(self.city, 'figures', fig_name))

        plt.show()

    def plot_hist_comparison(self, account=None, num_bins=20, alpha=0.5, max_num_langs=2, min_size=100):
        """
            Method to compare language choice distribution for a given account
            in the form of histograms and their cumulative frequency. If no account is provided,
            method will automatically compute a weighted average of all model accounts and plot comparison
            for chosen languages
            Args:
                * account: string. Account name. If None, the weighted sample of all accounts is considered
                * num_bins: integer. Number of bins of the histograms. Default 20
                * alpha: 0 < float < 1. Matplotlib transparency parameter
                * max_num_langs: integer <= 3. Number of languages under consideration. Default 2
                * min_size: integer. minimum size of account with the least residents
        """
        mpl.style.use('seaborn')
        langs = self.langs_for_postprocess[self.city][:max_num_langs]
        colors = ['green', 'blue', 'red']
        if account:
            data = self.data_stats[self.data_stats[account]]
        else:
            data = pd.DataFrame(
                {'{}_mean'.format(langs[0]): self.compute_weighted_sample_props(langs[0], min_size=min_size).values,
                 '{}_mean'.format(langs[1]): self.compute_weighted_sample_props(langs[1], min_size=min_size).values,
                 '{}_mean'.format(langs[2]): self.compute_weighted_sample_props(langs[2], min_size=min_size).values})

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for lang, color in zip(langs, colors):
            key = '{}_mean'.format(lang)
            plot_data = 100 * data[key]
            ax1.hist(plot_data, bins=num_bins, normed=True, alpha=alpha,
                     color=color, edgecolor='black', label=lang)
        ax1.grid(linestyle='--', alpha=0.6)
        ax1.xaxis.grid(linestyle='--', alpha=0.6)
        ax1.set_ylabel('normed frequency\n(histogram)', fontsize=10, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=8)
        if account:
            ax1.set_title('Distribution of linguistic choice of @{} followers from {}'.format(account, self.city),
                          family='serif', fontsize=10)
        else:
            # get sample size
            samp_size = data.shape[0]
            ax1.set_title('Distribution of linguistic choice of weighted sample '
                          'of followers in {} (sample size: {})'.format(self.city, samp_size),
                          family='serif', fontsize=10)
        ax1.legend(loc='best')

        for lang, color in zip(langs, colors):
            key = '{}_mean'.format(lang)
            plot_data = 100 * data[key]
            ax2.hist(plot_data, bins=num_bins, normed=True, histtype='step',
                     color=color, cumulative=1, alpha=1, label=lang)
        ax2.set_xlabel('percentage of tweets in language, %', fontsize=10, fontweight='bold')
        ax2.set_ylabel('frequency\n(cum histogram)', fontsize=10, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.grid(linestyle='--', alpha=0.6)
        ax2.legend(loc='upper left')
        fig.tight_layout()
        # save figure
        if account:
            fig_name = 'hist_lang_choice_@{}_followers'.format(account)
        else:
            fig_name = 'hist_lang_choice_weighted_sample_followers'.format(account)
        plt.savefig(os.path.join(self.city, 'figures', fig_name))

        plt.show()

    def plot_weighted_sample_stats(self, min_size=25):

        if not isinstance(self.weighted_samples_distribs, pd.DataFrame):
            self.resample_weighted_sample(min_size=min_size)

        mpl.style.use('seaborn')

        bar_width = 0.1
        colors = ['green', 'blue']

        fig, ax = plt.subplots(2, 1)

        for i, xlabels, ylabel in zip(range(2),
                                      [['median_follower', 'mean_follower'],
                                       ['pct_more_in_lang', 'pct_as_much_in_lang']],
                                      ['fraction of tweets', 'percentage of users']):

            data = self.weighted_samples_distribs['ca']
            data = data.values[:, :2] if not i else data.values[:, 2:]
            plot_data = ax[i].boxplot(data, positions=np.arange(2) - 0.1, widths=bar_width,
                                      patch_artist=True, medianprops=dict(linewidth=2, color='black'),
                                      labels=xlabels, whis=[1, 99])
            plt.setp(plot_data['boxes'][0], color=colors[0])
            plt.setp(plot_data['boxes'][1], color=colors[0])

            data = self.weighted_samples_distribs['es']
            data = data.values[:, :2] if not i else data.values[:, 2:]
            plot_data = ax[i].boxplot(data, positions=np.arange(2) + 0.1, widths=bar_width,
                                      patch_artist=True, labels=xlabels,
                                      medianprops=dict(linewidth=2, color='black'), whis=[1, 99])
            plt.setp(plot_data['boxes'][0], color=colors[1])
            plt.setp(plot_data['boxes'][1], color=colors[1])

            ax[i].set_ylabel(ylabel)
            ax[i].xaxis.grid(False)
            ax[i].yaxis.grid(linestyle='--', alpha=0.5)

        # draw temporary red and blue lines and use them to create a legend
        for i in range(2):
            ax[i].plot([], c=colors[0], label=self.langs_for_postprocess[self.city][0])
            ax[i].plot([], c=colors[1], label=self.langs_for_postprocess[self.city][1])
            ax[i].legend(loc='best')

        fig.suptitle("Probabilistic distributions of weighted average account",
                     fontsize=12, family='serif')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_name = 'weighted_sample_indicators_distribs_in_{}'.format(self.city)
        plt.savefig(os.path.join(self.city, 'figures', fig_name))
        plt.show()

    def __repr__(self):
        return "_".join([type(self).__name__, self.city])


class InterCityComparison:

    def __init__(self, cities, filename='city_random_walks.h5'):
        """
            Method to compare data from accounts of different cities
            Args:
                * cities: tuple or list of strings. Strings are city names
        """
        self.city_objects = {}
        for city in cities:
            self.city_objects[city] = PlotTweetData(filename, city)

        mpl.rcParams['figure.figsize'] = (15, 10)

    def compare_plot_boxes(self, root_accs, lang_ix=0):
        """
            Compare distributions for a given language of followers of accounts from different cities
            Plots both boxplots and Cumulative Distribution Functions to visualize differences in
            distributions
            Args:
                * root_accs: tuple or list of strings. Strings are root account names
        """

        self.data_accs = {}
        self.langs = {}

        # process data
        data_to_plot = []
        for acc in root_accs:
            # find city account belongs to
            for city in self.city_objects:
                if acc in self.city_objects[city].av_acc_nodes:
                    acc_city = city
                    break
            self.langs[acc_city] = self.city_objects[acc_city].langs_for_postprocess[acc_city][lang_ix]
            key = self.langs[acc_city] + '_mean'
            bool_idx = self.city_objects[acc_city].data_stats[acc]
            self.data_accs[acc] = self.city_objects[acc_city].data_stats[bool_idx][key].values
            data_to_plot.append(self.data_accs[acc])

        # plot data
        mpl.style.use('seaborn')
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
        colors, alpha = ['green', 'blue'], 0.7
        box_width = 0.4
        box_pos = np.arange(len(root_accs))
        # define boxplots
        bplot = ax.boxplot(data_to_plot, positions=box_pos, widths=box_width,
                           boxprops=dict(linewidth=2, color='black'), vert=False,
                           showmeans=True, medianprops=dict(linewidth=2, color='black'),
                           meanprops=dict(marker="^", markerfacecolor="red", markersize=8),
                           patch_artist=True)
        # color boxes
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)

        ax.xaxis.grid(True, linestyle='--', alpha=alpha)
        ax.yaxis.grid(False)
        labels_font_size = 10
        ax.set_yticks(box_pos)
        # box labels
        tlabs = [acc + '\n ({})'.format(lang[1]) for acc, lang in zip(root_accs, self.langs.items())]
        ax.set_yticklabels(tlabs, rotation=45, fontsize=labels_font_size)
        ax.tick_params(axis='y', which='major', labelsize=labels_font_size)
        ax.plot(0.9, 1.4, "^", markersize=8, color='red')
        ax.text(0.92, 1.36, "mean")
        accs = ['@{}'.format(acc) for acc in root_accs]
        ax.set_title('Language choice distributions from followers of {} accounts'.format(', '.join(accs)),
                      family='serif', fontsize=10)

        # plot CDFs
        num_bins=20
        for data, color, acc, (city, lang) in zip(data_to_plot, colors, root_accs, self.langs.items()):
            # key = '{}_mean'.format(lang)
            # plot_data = 100 * data[key]
            ax2.hist(data, bins=num_bins, normed=True, histtype='step',
                     color=color, cumulative=1, alpha=1, label='@{}({})'.format(acc, lang))
        ax2.set_xlabel('fraction of tweets in lang', fontsize=labels_font_size)
        ax2.set_ylabel('cumulative fraction of followers')
        ax2.grid(True, linestyle='--', alpha=alpha)
        ax2.legend(loc='lower right')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.025, wspace=0.025)
        # save figure
        fig_name = 'Language choice distributions from followers of {} accounts'.format(', '.join(accs))
        plt.savefig(os.path.join(acc_city, 'figures', fig_name))









