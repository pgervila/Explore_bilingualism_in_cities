import os
import time
import re
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Twitter API for python
import tweepy
from tweepy import Stream, OAuthHandler, StreamListener
# import secret codes to access Twitter API
from twitter_pwd import access_token, access_token_secret, consumer_key, consumer_secret

# language detection
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# stats
from statsmodels.stats import proportion

# progress_bar
import pyprind


class RandomWalkCityTweets:
    
    """ Get tweets from random relevant followers that live in a given city
        and return data on language use """

    city_root_accounts = dict()

    city_root_accounts['Kiev'] = {'KyivOperativ', 'kyivmetroalerts', 'nashkiev', 'auto_kiev',
                                 'Leshchenkos', 'poroshenko', 'Vitaliy_Klychko', 'kievtypical',
                                 'ukrpravda_news', 'HromadskeUA','lb_ua', 'Korrespondent',
                                 'LIGAnet', 'radiosvoboda', '5channel', 'tsnua', 'VWK668',
                                 'Gordonuacom', 'zn_ua', 'patrolpoliceua', 'KievRestaurants', 'ServiceSsu',
                                 'MVS_UA', 'segodnya_life', 'kp_ukraine', 'vesti'}

    city_root_accounts['Barcelona'] = {'TMB_Barcelona':'CAT', 'bcn_ajuntament':'CAT',
                                       'LaVanguardia':'SPA', 'hola':'SPA', 'diariARA':'CAT', 'elperiodico':'SPA',
                                       'meteocat':'CAT', 'mossos':'CAT',
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

    def __init__(self, data_file_name, city, city_accounts=None, city_key_words=None, update=False, city_langs=False):
        """ Args:
                * data_file_name: string. Name of database file where data is stored from previous computations.
                    If file name is not found in current directory, a new empty file is created with
                    the same specified name
                * city: string. Name of city where bilingualism is to be analyzed
                * city_accounts: set of strings. Strings are Twitter accounts related to city
                * city_key_words: list of strings. Strings are expresisons
                    to recognize the city in different languages or spellings
                * city_langs: list of strings. Strings must be valid identifiers of languages.
                    Use reference https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
                * update: boolean. True if new data is available in database and 'process_data' needs to be called
                    to create a pandas DataFrame that summarizes all information. Default False
        """
        if not os.path.exists(data_file_name):
            open(data_file_name, 'w+').close()

        # initialize key arguments
        self.data_file_name = data_file_name
        if city in self.city_root_accounts:
            self.city = city
        else:
            if city_accounts:
                self.city_root_accounts[city] = city_accounts
            else:
                raise Exception ("If a new city is specified, root accounts for this city "
                                 "must be provided through 'city_accounts' argument ")
            if city_key_words:
                self.key_words[city] = {'city': city_key_words}
            else:
                raise Exception (" If a new city is specified, 'city_key_words' arg must be specified  ")
            if city_langs:
                self.langs_for_postprocess[city] = city_langs
            else:
                raise Exception (" If a new city is specified, 'city_langs' arg must be specified ")

        # initialize instance attributes
        self.unique_flws = None
        self.tweets_from_followers = None
        self.av_nodes = None
        self.data_stats = pd.DataFrame()
        self.lang_settings_per_root_acc = defaultdict(dict)
        self.stats_per_root_acc = dict()

        # set_up Twitter API
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        if update:
            self.process_data()
        else:
            self.data_stats = pd.read_json(self.city + '_data_stats.json')
            self.get_available_nodes()

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
                    continue #user = next(users)
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
            print ('There are no new tweets left to be retrieved. All remaining ones are protected')
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

    def get_proportion_of_residents(self, root_account):
        """
            Method to quantify proportion of root account followers
            that are explicit city residents
            Args:
                * root_account:
        """
        locs = '|'.join(self.key_words[self.city]['city'])
        self.get_account_network(root_account)
        # TODO : how to store proportion per account until all props are available ??
    
    def filter_root_accs_unique_followers(self, save=True, min_num_flws_per_acc=50,
                                          min_num_twts_per_acc=60, twts_to_flws_ratio=30):
        """
            Method to read followers from all already-computed root-account nodes and
            then compute a list of all unique followers for a given city
            Args:
                * save: boolean. If True, followers are saved to hdf file. Default True
                * min_num_flws_per_acc: integer. Minimum number of followers an account
                    needs to have in order to consider it relevant. Default 50
                * min_num_twts_per_acc: integer. Minimum number of times an account
                    needs to have tweeted in order to consider it relevant. Default 60
                * twts_to_flws_ratio: integer. Maximum allowed ratio of the number of tweets
                    published by account to the number of followers the account has.
                    This ratio is a measure of impact. Defaults to 30
            Output:
                * Unique followers are stored in instance attribute self.unique_flws. If requested,
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
                
    def load_root_accs_unique_followers(self):
        """
            Load all root accounts' unique followers from hdf file and assign them to class attribute
            as a pandas Dataframe object
            Output:
                * Unique followers are saved to self.unique_flws instance attribute
        """
        self.unique_flws = pd.read_hdf(self.data_file_name,
                                       '/'.join(['', self.city, 'unique_followers']))

    def get_available_nodes(self):
        """ Method to load all nodes available in saved database.
            Output:
                * Resulting nodes will be saved as an instance attribute in self.av_nodes
        """
        with pd.HDFStore(self.data_file_name) as f:
            self.av_nodes = []
            pattern = r"".join([self.city, '/(\w+)', '/followers'])
            for n in f.keys():
                acc = re.findall(pattern, n)
                if acc and acc[0] in self.city_root_accounts[self.city]:
                    self.av_nodes.append(acc[0])

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

    def load_tweets_from_followers(self, filter=True):
        """ """
        tff = pd.read_hdf(self.data_file_name,
                          '/'.join(['', self.city, 'tweets_from_followers']))
        langs = self.langs_for_postprocess[self.city]
        if filter:
            if self.city == 'Barcelona':
                tff.lang[tff.lang == 'und'] = 'ca'
            tff = tff[tff.lang.isin(langs)]
            tff = tff[tff.lang == tff.lang_detected]
        self.tweets_from_followers = tff

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

    def optimize_saving_space(self):
        """
            Method to rewrite hdf data file in order to optimize file size.
            It keeps original file name
        """
        with pd.HDFStore(self.data_file_name) as f:
            for n in f.keys():
                data = pd.read_hdf(f, n)
                data.to_hdf('new_city_random_walks.h5', n)
        os.remove(self.data_file_name)
        os.rename('new_city_random_walks.h5', self.data_file_name)

    @staticmethod
    def detect_refined(txt):
        """ Method to deal with exceptions when detecting tweet languages
            Args:
                * txt : string. Tweet text
            Output:
                * tweet language or 'Undefined' label if insufficent text is present"""
        if len(txt.split()) > 2 and len(txt) > 10:
            try:
                return detect(txt)
            except LangDetectException:
                return 'Undefined'
        else:
            return 'Undefined'

    def get_num_flws_per_city_acc(self):
        """
            Get number of followers per each account of a given city
            Args:
                * city : string. Name of the city
            Output:
                * pandas series saved in data file
        """
        num_flws_per_acc = {}
        for acc in self.city_root_accounts[self.city]:
            acc_info = self.api.get_user(acc)
            num_flws_per_acc[acc] = acc_info.followers_count
        # make pandas series and save to hdf
        num_flws_per_acc = pd.Series(num_flws_per_acc)
        node = "/".join(['', self.city, 'num_flws_main_accs'])
        num_flws_per_acc.to_hdf(self.data_file_name, node)

    def process_data(self, num_tweets_for_stats=40, save=True):
        """
            Method to post-process tweet data and create a pandas DataFrame
            that summarizes all information

            Arguments:
                * num_tweets_for_stats: integer >= 40 and <= 60. Number of tweets that will be taken into
                    account for each follower.
                * save: boolean. Specifies whether to save the processed data or not. Defaults to True.

            Output:
            * It sets value for self.data_stats

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
        # get nodes with available tweets
        self.get_available_nodes()
        # generate boolean columns with followers for each account
        for root_acc in self.av_nodes:
            root_acc_ids = pd.read_hdf(self.data_file_name,
                                       '/'.join(['', self.city, root_acc, 'followers'])).id_str
            self.data_stats[root_acc] = self.data_stats.id_str.isin(root_acc_ids)

        for lang in self.langs_for_postprocess[self.city][:2]:
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
            self.data_stats.to_json(self.city + '_data_stats.json')

    def get_stats_per_root_acc(self, save=True, load_data=False):
        """
            Method to compute basic stats (mean, median, conf intervals ...)
            for each root account
        """
        if load_data:
            try:
                self.stats_per_root_acc = pd.read_json(self.city + '_stats_per_root_acc.json')
            except ValueError:
                print('data file is not available in current directory')
        else:
            data_frames_per_lang = dict()
            for lang in self.langs_for_postprocess[self.city]:
                self.stats_per_root_acc = dict()
                for root_acc in self.av_nodes:
                    data = self.data_stats[self.data_stats[root_acc]]
                    if data.shape[0] >= 100:
                        conf_int = proportion.proportion_confint(data[lang].sum(),
                                                                 data.tot_lang_counts.sum(),
                                                                 method='jeffreys', alpha=0.01)
                        conf_int = pd.Series(conf_int, index=['min_confint', 'max_confint'])
                        self.stats_per_root_acc[root_acc] = (data[lang] / data.tot_lang_counts).describe().append(conf_int)
                data_frames_per_lang[lang] = pd.DataFrame(self.stats_per_root_acc).transpose()
            self.stats_per_root_acc = pd.concat(data_frames_per_lang, axis=1)

            if save:
                self.stats_per_root_acc.reset_index().to_json(self.city + '_stats_per_root_acc.json')

    def get_lang_settings_stats_per_root_acc(self, city_only=True):
        """ Find distribution of lang settings for each root account
            and , if requested, for users residents in the city only
            Args:
                * city_only: boolean. True if settings have to be retrieved only for users from class instance city
                    Default True
            Output:
                * sets value to instance attribute 'lang_settings_per_root_acc'
        """
        # TODO: group data by lang using hierarchical columns instead of column suffix
        # get nodes if not available
        if not self.av_nodes:
            self.get_available_nodes()

        stats_per_lang = dict()
        acc_data = defaultdict(dict)
        for root_acc in self.av_nodes:
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

    def get_sample_size_per_root_acc(self, print_sizes=False):
        """ Method to read number of relevant followers per account
            for statistic analysis
            Args:
                * print_sizes: boolean. False if no printing of results is required
        """
        if not self.av_nodes:
            self.get_available_nodes()
        if self.data_stats.empty:
            try:
                self.data_stats = pd.read_json(self.city + '_data_stats.json')
            except ValueError:
                print('Requested file is not in current directory !')

        self.sample_size_per_root_acc = {acc: self.data_stats[self.data_stats[acc]].shape[0]
                                         for acc in self.av_nodes}
        if print_sizes:
            for acc in self.av_nodes:
                self.sample_size_per_root_acc
                print(acc, self.data_stats[self.data_stats[acc]].shape[0])

    def get_common_pct(self, ref_key, other_key):
        """ Get degree of similarity between follower sample of ref_key account
            as compared to other_acc : percentage of common followers relative to reference
            account
        """
        s_ref = set(self.data_stats[self.data_stats[ref_key]].id_str)
        s2 = set(self.data_stats[self.data_stats[other_key]].id_str)
        s_int = s_ref.intersection(s2)
        return len(s_int) / len(s_ref)

    def plot_lang_settings_comparison(self, min_num_accs=200, max_num_langs=3, single_account=False):
        """
            Method to visualize statistics on language settings in accounts from followers of root accounts
            Args:
                * min_num_accs: integer. Minimum number of available followers per root account
                    to consider statistics for it as relevant
                * max_num_langs: integer > 0. Maximum number of languages considered for each root account
                * single_account: boolean. If True, plot will be for a single root account only
        """

        self.get_lang_settings_stats_per_root_acc()

        self.lang_settings_per_root_acc = self.lang_settings_per_root_acc.sort_values(
            by=(self.langs_for_postprocess[self.city][0], 'mean'))

        bar_width = 0.4
        colors = ['green', 'blue', 'red', 'yellow', 'orange']

        mpl.style.use('seaborn')

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
        ax.set_xticks(X + bar_width / 2)
        ax.set_xticklabels(plot_data.index, rotation=45, fontsize=8)
        ax.set_ylabel('percentage', fontsize=8)
        ax.legend(fontsize=10, loc='best')
        ax.set_title('Twitter language settings of root-account followers from ' + self.city,
                     family='serif', fontsize=10)
        ax.grid(linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig('lang_settings_in_' + self.city)
        plt.show()

        # self.get_lang_settings_stats_per_root_acc(city_only=False)

    def plot_lang_props_per_acc(self):
        """ TODO : description needed !!! """
        if not isinstance(self.stats_per_root_acc, pd.DataFrame):
            self.get_stats_per_root_acc()
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
            data = plot_data['mean']
            err_up = plot_data.max_confint - plot_data['mean']
            err_down = (plot_data.min_confint - plot_data['mean']).abs()
            ax.bar(X + i * bar_width, data, yerr=[err_down, err_up], width=bar_width,
                   align='center', edgecolor='black', label=lang, color=colors[i], alpha=0.7,
                   capsize=2)
        ax.set_xticks(X + bar_width / 2)
        ax.set_xticklabels(plot_data.index, rotation=45, fontsize=8)
        ax.set_ylabel('proportion of tweets', fontsize=8)
        ax.legend(fontsize=10, loc='best')
        ax.set_title('Language of (re)tweets by followers from ' + self.city + ' per account',
                     family='serif', fontsize=10)
        ax.grid(linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig('percentage_of_each_lang_per_account_in_' + self.city)
        plt.show()

    def plot_lang_distribs_per_acc(self):
        """
            Method to plot the distributions of fraction
            of tweets in each language from followers of each root account
        """
        if not isinstance(self.stats_per_root_acc, pd.DataFrame):
            self.get_stats_per_root_acc()
        # get relevant root accounts sorted by mean value
        ticks = self.stats_per_root_acc[self.langs_for_postprocess[self.city][0]].sort_values(by='mean').index.tolist()

        data = {self.langs_for_postprocess[self.city][0]: [],
                self.langs_for_postprocess[self.city][1]: []}
        for lang in self.langs_for_postprocess[self.city][:2]:
            for root_acc in ticks:
                acc_data = self.data_stats[self.data_stats[root_acc]]
                acc_data = (acc_data[lang] / acc_data.tot_lang_counts).values
                data[lang].append(acc_data)

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            #plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            #plt.setp(bp['medians'], color=color)

        mpl.style.use('seaborn')
        plt.figure()
        data_a = data[self.langs_for_postprocess[self.city][0]]
        data_b = data[self.langs_for_postprocess[self.city][1]]

        bar_width = 0.5
        tick_dist = 4
        #colors = ['#D7191C', '#2C7BB6']
        colors = ['green', 'blue']
        bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a))) * tick_dist - bar_width,
                          sym='', widths=bar_width, showmeans=True, patch_artist=True,
                          whiskerprops=dict(linestyle='--', linewidth=0.5, alpha=0.5),
                          medianprops=dict(linewidth=2, color='black'),
                          meanprops=dict(marker="o", markeredgecolor ='black', markerfacecolor="None"))
        bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * tick_dist + bar_width,
                          sym='', widths=bar_width, showmeans=True, patch_artist=True,
                          whiskerprops=dict(linestyle='--', linewidth=0.5, alpha=0.5),
                          medianprops=dict(linewidth=2, color='black'),
                          meanprops=dict(marker="o", markeredgecolor ='black', markerfacecolor="None"))
        set_box_color(bpl, colors[0])
        set_box_color(bpr, colors[1])

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c=colors[0], label=self.langs_for_postprocess[self.city][0])
        plt.plot([], c=colors[1], label=self.langs_for_postprocess[self.city][1])
        plt.legend(loc='best')

        plt.grid(linestyle='--', alpha=0.4)

        X = np.arange(0, len(ticks) * tick_dist, tick_dist)
        plt.xticks(X, ticks, rotation=45, fontsize=8)
        plt.xlim(-tick_dist, len(ticks) * tick_dist)

        plt.ylabel('fraction of tweets in given lang', fontsize=8)
        plt.title('Language choice distributions from followers of root-accounts in ' + self.city,
                  family='serif', fontsize=10)

        plt.tight_layout()
        plt.savefig('lang_distribs_per_acc_in_' + self.city)
        plt.show()

    def plot_hist_per_root_acc(self):
        pass


class PlotTweetData(RandomWalkCityTweets):

    def __init__(self, data_file_name, city):
        super().__init__()

    def plot_stats(self, lang):
        if not isinstance(self.stats_per_root_acc, pd.DataFrame):
            try:
                self.stats_per_root_acc = pd.read_json(self.city + '_stats_per_root_acc.json')
            except ValueError:
                self.get_stats_per_root_acc()
        if not isinstance(self.lang_settings_per_root_acc, pd.DataFrame ):
            self.get_lang_settings_stats_per_root_acc()
        cols_1 = [lang + stat for stat in ['_SE', '_mean', '_median']]
        cols_2 = [lang + stat for stat in ['_mean', '_SE']]
        df1 = self.stats_per_root_acc[cols_1].sort_values(by=lang + '_mean')
        df2 = self.lang_settings_per_root_acc[cols_2]
        df_merged = df1.join(df2, lsuffix='_lang_twts', rsuffix='_lang_sett')

        X = np.arange(df_merged.shape[0])
        plt.bar(X, df_merged[lang + '_mean_lang_twts'], yerr=3 * df_merged[lang + '_SE_lang_twts'],
                align='center', color='y', ecolor='black', edgecolor='black', width=0.25, label="(re)tweets_mean")
        plt.bar(X + 0.25, df_merged[lang + '_mean_lang_sett'], yerr=3 * df_merged[lang + '_SE_lang_sett'],
                align='center', color='g', ecolor='black', edgecolor='black', width=0.25, label="lang settings")
        plt.xticks(X + 0.125, df_merged.index, rotation=45, fontsize=10)
        plt.hlines(df_merged[lang + '_median'], xmin=X - 0.125, xmax=X + 0.125, color='red', lw=3,
                   label='(re)tweets_median')
        plt.ylabel('% ' + lang, fontsize=10)
        plt.legend(fontsize=10, loc='lower right')
        plt.title( 'language choice of average follower per Twitter account in ' + self.city, fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig(lang + '_stats')
        plt.show()

    def plot_comparison_stats(self, lang1, lang2):
        if not isinstance(self.stats_per_root_acc, pd.DataFrame):
            try:
                self.stats_per_root_acc = pd.read_json(self.city + '_stats_per_root_acc.json')
            except ValueError:
                self.get_stats_per_root_acc()
        cols_1 = [lang1 + stat for stat in ['_SE', '_mean', '_median']]
        cols_2 = [lang2 + stat for stat in ['_SE', '_mean', '_median']]

        df1 = self.stats_per_root_acc[cols_1].sort_values(by=lang1 + '_mean')
        df2 = self.stats_per_root_acc[cols_2]
        df_merged = df1.join(df2)
        x = np.arange(df_merged.shape[0])
        plt.bar(x, df_merged[lang1 + '_mean'], yerr=3 * df_merged[lang1 + '_SE'],
                align='center', color='y', ecolor='black', edgecolor='black', width=0.25, label=lang1 + "_(re)tweets_mean")
        plt.bar(x + 0.25, df_merged[lang2 + '_mean'], yerr=3 * df_merged[lang2 + '_SE'],
                align='center', color='g', ecolor='black', edgecolor='black', width=0.25, label=lang2 + "_(re)tweets_mean")
        plt.xticks(x + 0.125, df_merged.index, rotation=45, fontsize=10)
        plt.hlines(df_merged[lang1 + '_median'], xmin=x - 0.125, xmax=x + 0.125, color='red', lw=3,
                   label=lang1 + '_(re)tweets_median')
        plt.hlines(df_merged[lang2 + '_median'], xmin=x + 0.125, xmax=x + 0.375, color='yellow', lw=3,
                   label=lang2 + '_(re)tweets_median')
        plt.ylabel('% tweets in language', fontsize=10)
        plt.legend(fontsize=10, loc='lower right')
        plt.title('language choice of average follower per Twitter account in ' + self.city, fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.savefig('_'.join([lang1, lang2, 'stats']))
        plt.show()

    def plot_comparison_boxplot(self):
        pass

class StreamTweetData:
    pass

class ProcessTweetData:
    pass

class PlotTweetData:
    pass



