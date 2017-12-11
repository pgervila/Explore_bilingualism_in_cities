import os
import tweepy
from tweepy import Stream, OAuthHandler, StreamListener
import json
import time
from collections import Counter
import re
import pandas as pd
import numpy as np
from langdetect import detect
import pyprind
import deepdish as dd

#import secret codes
from twitter_pwd import access_token, access_token_secret, consumer_key, consumer_secret

class RandomWalkCityTweets:
    
    """ Get tweets from random relevant followers that live in a given city
        and return data on language use """
    
    Kiev_dict = {'KyivOperativ', 'kyivmetroalerts', 'nashkiev', 'auto_kiev', 
                 'Leshchenkos', 'poroshenko', 'Vitaliy_Klychko', 'kievtypical',
                 'ukrpravda_news', 'HromadskeUA','lb_ua', 'Korrespondent',
                 'LIGAnet', 'radiosvoboda', '5channel', 'tsnua', 'VWK668', 'Gordonuacom', 'zn_ua',
                 'patrolpoliceua', 'KievRestaurants'}
    
    Barcelona_dict = {'TMB_Barcelona', 'bcn_ajuntament', 'barcelona_cat', 'LaVanguardia', 'VilaWeb', 
                      'diariARA', 'elperiodico', 'elperiodico_cat', 'elpuntavui', 'meteocat', 'mossos',
                      'sport', 'VilaWeb'}

    def __init__(self, data_file_name, city):
        if not os.path.exists(data_file_name):
            open(data_file_name, 'w+').close()
        self.data_file_name = data_file_name
        self.city = city
        self.key_words = {'Barcelona':{'country': ['Catalu'], 
                                       'city': ['Barcel']}, 
                          'Kiev':{'country': ['Україна', 'Ukraine', 'Украина'], 
                                  'city': ['Kiev' ,'Kyiv' , 'Київ' , 'Киев']}}
        
        #set_up API
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def get_account_network(self, account_name, rel_type='followers', max_num =100,
                            min_num_tweets=0, min_num_followers=0, only_city=False, 
                            limited_search=False, avoid_repeat=None, cursor_on=False):
        """ Given an account by account_name, 
            find all users that are linked to it via a specified relation type 'rel_type'.
            Args:
                * account_name: string. Twitter account name
                * rel_type: string. Specifies relation type (default is 'followers')
                * max_num: integer. Maximum number of 'related' users considered
                * key_words: list of strings. Used to filter retrieved users by location,
                    if specified
                * min_num_tweets: minimum number of tweets a follower needs to have 
                    to be included in list
                * min_num_followers: minimum number of followers a follower needs to have 
                    to be included in list
                * limited_search:
                * avoid_repeat:
                * cursor_on:
            Returns:
                * list_people: list of account_names
        """
        pbar = pyprind.ProgBar(max_num)
        list_people = []
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
                if (found_loc and user.statuses_count > min_num_tweets and 
                    not user.protected):
                    user._json.update({'cursor_id': cursor.iterator.next_cursor})
                    if avoid_repeat:
                        if user._json['screen_name'] not in avoid_repeat:
                            list_people.append(user._json)
                    else:
                        list_people.append(user._json)
                    if len(list_people) > 2 and limited_search:
                        break
            except tweepy.TweepError as e:
                if 'Read timed out' in str(e):
                    print('fallen here')
                    print(e)
                    time.sleep(3)
                else:
                    time.sleep(60*16)
                    continue #user = next(users)
            except StopIteration:
                break            
            pbar.update()
        if not limited_search:
            node = '/'.join(['', self.city, account_name, rel_type])
            df_new = pd.DataFrame(list_people)
            if not cursor_on:
                with pd.HDFStore(self.data_file_name) as f:
                    if node not in f.keys():
                        df_new.to_hdf(self.data_file_name, node)
                        # update list unique followers
            else:
                df_old = df_old.append(df_new, ignore_index=True)
                df_old.to_hdf(self.data_file_name, node)
                
        return list_people
    
    def get_main_unique_followers(self, save=True, min_num_flws_per_acc=50): 
        """ Read all followers nodes and compute list of 
            all unique followers for a given city
        """
        key_words='|'.join(self.key_words[self.city]['city'])
        #initialize frame
        df_unique_flws = pd.DataFrame()
        with pd.HDFStore(self.data_file_name) as f:
            for n in f.keys():
                if re.findall(self.city, n) and not re.findall('_followers', n):
                    df = pd.read_hdf(f, n)
                    df = df[(df['followers_count'] > min_num_flws_per_acc) & 
                            (df['statuses_count'] / df['followers_count'] < 30) &
                            (df['location'].str.contains(key_words))]
                    df_unique_flws = df_unique_flws.append(df)
            self.df_unique_flws = df_unique_flws.drop_duplicates('screen_name')
            if save:
                self.df_unique_flws.to_hdf(self.data_file_name, 
                                           '/'.join(['', self.city, 'unique_followers']))
                
    def load_main_unique_followers(self):
        """ Load main followers from hdf file and assign them to class attribute
            as pandas Dataframe """
        self.df_unique_flws = pd.read_hdf(self.data_file_name, 
                                          '/'.join(['', self.city, 'unique_followers']))
           
    def get_account_tweets(self, account_name, max_num_twts=20):
        """ Given an account name,
            it retrieves a maximum number of tweets written or retweeted by account owner.
            It returns them in a list.
            Args:
                * account name: string. Screen_name that identifies the twitter account
                * max_num_twts: integer. Maximum number of tweets to be retrieved for each account
            Returns:
                * list_tweets: list including info of all retrieved tweets in JSON format"""
        list_tweets = []
        timeline = tweepy.Cursor(self.api.user_timeline, screen_name=account_name, 
                                 count=200, include_rts = True).items(max_num_twts)
        while True:
            try:
                tw = next(timeline)
                list_tweets.append(tw)
            except tweepy.TweepError as e:
                if '401' in str(e):    
                    print(e)
                    time.sleep(3)
                    break
                elif '404' in str(e):
                    print(e)
                    time.sleep(3)
                    break
                else:
                    time.sleep(60 * 15)
                    continue 
            except StopIteration:
                break
        return list_tweets

    def get_tweets_from_accounts(self, list_accounts, max_num_accounts=None, 
                                 max_num_twts=20, save=True, random_walk=False):
        """ Given a list of accounts, get its tweets texts, langs and authors
            All URLs and tweet account names are removed from tweet
            texts since they are not relevant for language identification
        """
        pbar = pyprind.ProgBar(len(list_accounts))
        texts_tweets = []
        langs_tweets = []
        authors_tweets = []
        if max_num_accounts:
            list_accounts = list_accounts[:max_num_accounts]
        for idx, acc in enumerate(list_accounts):
            twts = self.get_account_tweets(acc, max_num_twts=max_num_twts)
            texts_tweets.extend([re.sub(r"(@\s?[^\s]+|https?://?[^\s]+)", "", tw.text) 
                                 for tw in twts])
            langs_tweets.extend([tw.lang for tw in twts])
            authors_tweets.extend([acc for _ in twts])
            pbar.update()
        df_tweets = pd.DataFrame({'tweets':texts_tweets, 
                                  'lang':langs_tweets, 
                                  'screen_name':authors_tweets})
        df_tweets = df_tweets.drop_duplicates('tweets')
        if save:
            if not random_walk:
                df_tweets.to_hdf(self.data_file_name, 
                                 '/'.join(['', self.city, 'tweets_from_followers']))
            else:
                with pd.HDFStore('city_random_walks.h5') as f:
                    nodes = f.keys()
                digits = []
                pattern = r"".join([self.city, "/random_walk_", "(\d+)"])
                for e in nodes:
                    try:
                        digits.append(int(re.findall(pattern, e)[0]))
                    except:
                        continue
                if not digits:
                    df_tweets.to_hdf(self.data_file_name, 
                                     '/'.join(['', self.city, 'random_walk_1']))
                else:
                    i = max(digits) 
                    df_tweets.to_hdf(self.data_file_name, 
                                     '/'.join(['', self.city, 'random_walk_' + str(i + 1)]))               
        return df_tweets
    
    def update_tweets_from_main_followers(self, download=False):
        """ Download tweets from newly detected followers and append them to saved data"""
        # get sets
        self.get_main_unique_followers(save=True)
        all_flws = set(self.df_unique_flws.screen_name)
        availab_tweets = pd.read_hdf(self.data_file_name, 
                                     '/'.join(['', self.city, 'tweets_from_followers']))
        flws_with_twts = set(availab_tweets.screen_name)
        # compute set difference
        new_flws = all_flws.difference(flws_with_twts)
        # get tweets from new followers if any
        if new_flws:
            new_twts = self.get_account_tweets(new_flws, save=False)
            if self.city == 'Barcelona':
                new_twts['tweets'] = new_twts['tweets'].str.replace(r"RT ", "")
                new_twts['tweets'] = new_twts['tweets'].str.replace(r"[^\w\s'’,.!?]+", " ")
                new_twts['lang_detected'] = new_twts['tweets'].apply(self.detect_refined)
            # append new tweets
            availab_tweets = availab_tweets.append(new_twts, ignore_index=True)
            # save
            availab_tweets.to_hdf(self.data_file_name,
                                  '/'.join(['', self.city, 'tweets_from_followers']))
            
    def random_walk(self):
        """ 
            Select a list of accounts by randomly walking 
            all main followers' friends and followers
        """        
        
        # load main followers
        self.load_main_unique_followers()
        
        # get random sample from main followers
        sample = np.random.choice(self.df_unique_flws['screen_name'], 10, replace=False)
        
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
        self.random_walk_tweets = self.get_tweets_from_accounts(self.random_walk_accounts["screen_name"],  
                                                                max_num_twts=20, save=True, 
                                                                random_walk=True)

    def optimize_saving_space(self):
        with pd.HDFStore(self.data_file_name) as f:
            for n in f.keys():
                data = pd.read_hdf(f, n)
                data.to_hdf('new_city_random_walks.h5', n)
        os.remove(self.data_file_name)
        os.rename('new_city_random_walks.h5', self.data_file_name)

    def detect_refined(self, txt):
        if len(txt.split()) > 2 and len(txt) > 10:
            try:
                return detect(txt)
            except:
                return 'Undefined'
        else:
            return 'Undefined'




        #     def random_walk(self, regex_words = None):
#         """ 
#             Select a list of accounts by randomly walking 
#             all main followers' friends and followers
#         """
#         main_flws = []
#         with pd.HDFStore(self.data_file_name) as f:
#             for n in f.keys():
#                 if re.findall( r"followers", n):
#                     flws = pd.read_hdf(self.data_file_name, n)
#                     #####
#                     idx = [i for i, w in enumerate(key_words) 
#                            if re.findall(city, w)][0]
#                     regex_ words  ="|".join(key_words[idx:])
#                     try:
#                         min_people = 50
#                         flws = flws[
#                                     (flws['friends_count'] > 50) & 
#                                     (flws['followers_count'] > 50) &
#                                     (flws['location'].str.contains(regex_ words)) &
#                                     (flws['lang'].isin(['ru', 'uk']))
#                                    ]
#                     except 
#                     flws = flws['screen_name'].sample(50).unique().tolist()
#                     main_flws.extend(flws)
#         main_flws = set(main_flws)
        
#         sub_flws = []
#         for acc in main_flws:
#             for rel in ['friends', 'followers']:
#                 sub_flws.extend(self.get_account_network(
#                         acc, rel_type=rel, max_num=3000, 
#                         regex_ words=key_words)
#                                )
            