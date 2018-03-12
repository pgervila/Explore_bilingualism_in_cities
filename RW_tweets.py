import os
import tweepy
from tweepy import Stream, OAuthHandler, StreamListener
import time
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# language detection
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

#stats
from statsmodels.stats import proportion

#progress_bar
import pyprind


#import secret codes
from twitter_pwd import access_token, access_token_secret, consumer_key, consumer_secret

class RandomWalkCityTweets:
    
    """ Get tweets from random relevant followers that live in a given city
        and return data on language use """

    city_root_accounts = dict()

    city_root_accounts['Kiev'] = {'KyivOperativ', 'kyivmetroalerts', 'nashkiev', 'auto_kiev',
                                 'Leshchenkos', 'poroshenko', 'Vitaliy_Klychko', 'kievtypical',
                                 'ukrpravda_news', 'HromadskeUA','lb_ua', 'Korrespondent',
                                 'LIGAnet', 'radiosvoboda', '5channel', 'tsnua', 'VWK668',
                                 'Gordonuacom', 'zn_ua', 'patrolpoliceua', 'KievRestaurants', 'ServiceSsu',
                                 'MVS_UA'}

    city_root_accounts['Barcelona'] = {'TMB_Barcelona', 'bcn_ajuntament', 'barcelona_cat',
                                       'LaVanguardia', 'VilaWeb', 'diariARA', 'elperiodico',
                                       'elperiodico_cat', 'elpuntavui', 'meteocat', 'mossos',
                                       'sport', 'VilaWeb'}

    city_root_accounts['Brussels'] = {'STIBMIVB'}
    city_root_accounts['Riga'] = {'Rigassatiksme_', 'nilsusakovs'}

    langs_for_postprocess = {'Kiev': ['uk', 'ru', 'en'], 'Barcelona': ['ca', 'es', 'en'],
                             'Brussels': ['fr', 'nl', 'en', 'es', 'it', 'de', 'tr', 'pt', 'el', 'da','ar'],
                             'Riga': ['lv', 'ru', 'en']}

    def __init__(self, data_file_name, city):
        if not os.path.exists(data_file_name):
            open(data_file_name, 'w+').close()
        self.data_file_name = data_file_name
        self.city = city
        self.key_words = {'Barcelona': {'country': ['Catalu'],
                                        'city': ['Barcel']},
                          'Kiev': {'country': ['Україна', 'Ukraine', 'Украина'],
                                   'city': ['Kiev', 'Kyiv', 'Київ', 'Киев']},
                          'Brussels': {'country': ['Belg'], 'city': ['Bruxel', 'Brussel']},
                          'Riga': {'country': ['Latvija', 'Латвия', 'Latvia'],
                                   'city': ['Rīg', 'Rig', 'Рига']}
                          }

        self.unique_flws = None
        self.tweets_from_followers = None
        self.av_nodes = None
        self.data_stats = pd.DataFrame()
        self.lang_settings_per_root_acc = defaultdict(dict)
        self.stats_per_root_acc = dict()

        
        #set_up API
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def get_available_nodes(self):
        with pd.HDFStore(self.data_file_name) as f:
            self.av_nodes = []
            pattern = r"".join([self.city, '/(\w+)', '/followers'])
            for n in f.keys():
                acc = re.findall(pattern, n)
                if acc:
                    self.av_nodes.append(acc[0])

    def get_account_network(self, account_name, rel_type='followers', max_num =100,
                            min_num_tweets=60, min_num_followers=50, only_city=False,
                            limited_search=False, avoid_repeat=None, cursor_on=False,
                            overwrite=False):
        """ Given an account by its account_name,
            find users that are linked to it via a specified relation type 'rel_type'.
            Args:
                * account_name: string. Twitter account name
                * rel_type: string. Specifies relation type (default is 'followers')
                * max_num: integer. Maximum number of 'related' users considered
                * key_words: list of strings. Used to filter retrieved users by location,
                    if specified
                * min_num_tweets: minimum number of tweets a related-user needs to have
                    in order to be included in the list
                * min_num_followers: minimum number of followers a related-user needs to have
                    in order to be included in the list
                * limited_search:
                * avoid_repeat: list of strings. List of accounts that do not have to be added
                * cursor_on: boolean. Set to True if computation needs restart from an
                    intermediate point ( to avoid starting from beginning again )
                * overwrite: boolean. If True, followers data is written on already existing node
            Returns:
                * list_users: list of account_names. Dta
        """
        pbar = pyprind.ProgBar(max_num)
        list_users = []
        # very important to set count=200 MAX VALUE -> Max 3000 accounts per 15 minutes interval
        if not cursor_on:
            cursor = tweepy.Cursor(getattr(self.api, rel_type, 0), 
                                   screen_name=account_name, count=200)
        else:
            node = '/'.join(['', self.city, account_name, rel_type])
            df_old = pd.read_hdf(self.data_file_name, node)
            cursor_id = df_old.cursor_id.values[-1]
            cursor = tweepy.Cursor(self.api.followers, screen_name=account_name,
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
            tmp_flws.to_pickle('_'.join(['tmp_flws', account_name]))
            pbar.update()
        if not limited_search:
            node = '/'.join(['', self.city, account_name, rel_type])
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
                            df_new.to_pickle('_'.join(['tmp_flws', account_name]))
            else:
                df_old = df_old.append(df_new, ignore_index=True)
                df_old.to_hdf(self.data_file_name, node)
                
        return list_users
    
    def select_root_accs_unique_followers(self, save=True, min_num_flws_per_acc=50,
                                          min_num_twts_per_acc=60):
        """
            Method to read all already-computed nodes with followers of root accounts and
            then compute a list of all unique followers for a given city
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
                            (df['statuses_count'] / df['followers_count'] < 30) &
                            (df['location'].str.contains(filter_words)) &
                            (df['statuses_count'] >= min_num_twts_per_acc) &
                            (~df['protected'])]
                    df_unique_flws = df_unique_flws.append(df)
            self.unique_flws = df_unique_flws.drop_duplicates('screen_name').reset_index()
            if save:
                self.unique_flws.to_hdf(self.data_file_name,
                                        '/'.join(['', self.city, 'unique_followers']))
                
    def load_root_accs_unique_followers(self):
        """ Load main followers from hdf file and assign them to class attribute
            as a pandas Dataframe object"""
        self.unique_flws = pd.read_hdf(self.data_file_name,
                                       '/'.join(['', self.city, 'unique_followers']))
           
    def get_account_tweets(self, id_account, max_num_twts=60):
        """ Given an account name,
            it retrieves a maximum number of tweets written or retweeted by account owner.
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
        """ Given a list of accounts by their ids, it gets
            tweets texts per each account and their corresponding lang,
            with a maximum number of tweets per account equal to param 'max_num_twts'.
            All URLs and tweet account names are removed from tweet
            texts since they are not relevant for language identification

            * list_accounts: list of ids (strings)
            * max_num_accounts:
            * max_num_tweets:
            * save:
            * random_walk:
        """
        pbar = pyprind.ProgBar(len(list_accounts))
        texts_tweets = []
        langs_tweets = []
        authors_tweets = []
        authors_id_tweets = []
        temp_df = pd.DataFrame()
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
        #df_tweets = df_tweets.drop_duplicates('tweets')
        df_tweets['tweets'] = df_tweets['tweets'].str.replace(r"RT ", "")
        df_tweets['tweets'] = df_tweets['tweets'].str.replace(r"[^\w\s'’,.!?]+", " ")
        df_tweets['lang_detected'] = df_tweets['tweets'].apply(self.detect_refined)
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
    
    def update_tweets_from_followers(self, download=False):
        """ Download tweets from newly detected followers and append them to saved data"""
        # get sets
        self.select_root_accs_unique_followers(save=True)
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
                # tff = tff[~((tff.lang == 'und') & (tff.lang_detected != langs[0]))]
            tff = tff[tff.lang.isin(langs)]
            tff = tff[tff.lang == tff.lang_detected]

            #     tff = tff[~((tff.lang == 'uk') & (tff.lang_detected != 'uk'))]
            # tff = tff[~((tff.lang == langs[1]) & (tff.lang_detected != langs[1]))]
            # tff = tff[~((tff.lang == langs[2]) & (tff.lang_detected != langs[2]))]
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
        """ Rewrite data file in order to optimize file size.
            It keeps original file name"""
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
        """ Get number of followers per each account of a given city
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
        """ Method to postprocess tweet data and create a pandas dataframe
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
        # rearrange dataframe
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
            mean = lang + '_mean'
            ESS = lang + '_ESS'
            SE = lang + '_SE'
            self.data_stats[mean] = self.data_stats[lang] / self.data_stats['tot_lang_counts']
            self.data_stats[ESS] = self.data_stats[mean] * (1 - self.data_stats[mean])
            self.data_stats[SE] = (np.sqrt(self.data_stats[mean] * (1 - self.data_stats[mean]) /
                                           self.data_stats['tot_lang_counts']))
        if save:
            self.data_stats.to_json(self.city + '_data_stats.json')

    def get_stats_per_root_acc(self, save=True, load_data=False):
        if load_data:
            try:
                self.stats_per_root_acc = pd.read_json(self.city + 'stats_per_root_acc.json')
            except ValueError:
                print('data file is not available in current directory')
        else:
            data_frames_per_lang = dict()
            if self.data_stats.empty:
                self.process_data()
            lang_1, lang_2 = self.langs_for_postprocess[self.city][:2]
            for lang in [lang_1, lang_2]:
                self.stats_per_root_acc = dict()
                mean = lang + '_mean'
                median = lang + '_median'
                std = lang + '_std'
                SE = lang + '_SE'
                for root_acc in self.av_nodes:
                    data = self.data_stats[self.data_stats[root_acc]]
                    if data.shape[0] > 100:
                            self.stats_per_root_acc[root_acc] = {mean: 100 * data[mean].mean(),
                                                                 median: 100 * data[mean].median(),
                                                                 std: 100 * data[mean].std(),
                                                                 SE: 100 * np.sqrt((data[SE] ** 2).sum()) / data.shape[0]
                                                                 }

                data_frames_per_lang[lang] = pd.DataFrame(self.stats_per_root_acc).transpose()

            self.stats_per_root_acc = data_frames_per_lang[lang_1].join(data_frames_per_lang[lang_2])
            if save:
                self.stats_per_root_acc.to_json(self.city + '_stats_per_root_acc.json')

    def get_lang_settings_stats_per_root_acc(self):
        """ Find distribution of lang settings in Twitter account
            for each account and for users residents in the city only"""
        if not self.av_nodes:
            self.get_available_nodes()
        for root_acc in self.av_nodes:
            node = "/".join([self.city, root_acc, 'followers'])
            df = pd.read_hdf('city_random_walks.h5', node)
            df = df[df.location.str.contains("|".join(self.key_words[self.city]['city']))]
            counts = df.lang.value_counts()

            sample_means = counts / counts.sum()
            self.lang_settings_per_root_acc[root_acc] = {'num_accs': df.shape[0]}

            for lang in self.langs_for_postprocess[self.city][:2]:
                mean_str = lang + '_mean'
                std_str = lang + '_SE'
                sample_mean = sample_means[lang]
                self.lang_settings_per_root_acc[root_acc][mean_str] = 100 * sample_mean
                self.lang_settings_per_root_acc[root_acc][std_str] = 100 * np.sqrt(sample_mean * (1 - sample_mean) /
                                                                                   df.shape[0])
        self.lang_settings_per_root_acc = pd.DataFrame(self.lang_settings_per_root_acc).transpose()

    def get_sample_size_per_root_acc(self, print_sizes=False):
        if not self.av_nodes:
            self.get_available_nodes()
        if self.data_stats.empty:
            try:
                self.data_stats = pd.read_json(self.city + 'data_stats.json')
            except:
                print('Requested file is not in current directory !')

        self.sample_size_per_root_acc = {acc: self.data_stats[self.data_stats[acc]].shape[0]
                                         for acc in self.av_nodes}
        if print_sizes:
            for acc in self.av_nodes:
                self.sample_size_per_root_acc
                print(acc, self.data_stats[self.data_stats[acc]].shape[0])

    def get_common_pct(self, ref_key, other_key):
        """ Get degree of similarity between follower sample of ref_key account
            as compared to other_acc : percntage of common followers relative to reference
            account
        """
        s_ref = set(self.data_stats[self.data_stats[ref_key]].id_str)
        s2 = set(self.data_stats[self.data_stats[other_key]].id_str)
        s_int = s_ref.intersection(s2)
        return len(s_int) / len(s_ref)

    def plot_stats(self, lang):
        if not isinstance(self.stats_per_root_acc, pd.DataFrame):
            try:
                self.stats_per_root_acc = pd.read_json(self.city + 'stats_per_root_acc.json')
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
                self.stats_per_root_acc = pd.read_json(self.city + 'stats_per_root_acc.json')
            except:
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


class StreamTweetData:
    pass


class PostProcessTweetData:
    def __init__(self, data_file_name, city):
        pass


    def get_num_relevant_flws_per_acc(self):
        self.relevant_flws_per_acc = dict()
        for acc in self.av_nodes:
            self.relevant_flws_per_acc[acc] = self.data_stats[self.data_stats[acc]].shape[0]



class PlotTweetData:
    def __init__(self):
        pass


