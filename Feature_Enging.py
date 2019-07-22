# -*- coding: UTF-8 -*-

import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
from sklearn import preprocessing
from Data_Handle import *

BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_PATH, 'jdata')
CACHE_PATH = os.path.join(BASE_PATH, 'cache')
SUB_PATH = os.path.join(BASE_PATH, 'submit')


def get_user_feature():
    '''
    获取用户特征
    :return:user_df
    '''
    user_df = Get_Data('user')
    for f in ['age', 'sex', 'user_reg_tm', 'user_lv_cd', 'city_level', 'province', 'city', 'county']:
        user_df[f] = user_df[f].fillna(user_df[f].mode()[0])

    sex_df = pd.get_dummies(user_df['sex'], prefix='sex')
    user_lv_df = pd.get_dummies(user_df['user_lv_cd'], prefix='user_lv_cd')
    city_level_df = pd.get_dummies(user_df['city_level'], prefix='city_level')
    province_df = pd.get_dummies(user_df['province'], prefix='province')
    city_df = pd.get_dummies(user_df['city'], prefix='city')

    user_df = pd.concat([user_df, sex_df, user_lv_df, city_level_df, province_df, city_df], axis=1)
    user_df['user_reg_tm'] = user_df['user_reg_tm'].apply(lambda x: x.split()[0].replace('-', '')).astype(int)

    return user_df


def get_product_feature():
    '''
    获取商品和商店的联合特征
    :return:product_df
    '''
    product_df = Get_Data('product')
    shop_df = Get_Data('shop')
    shop_df['shop_cate'] = shop_df['cate']
    del shop_df['cate']
    product_df = product_df.merge(shop_df, on=['shop_id'], how='left')

    for f in ['cate', 'shop_cate']:
        product_df[f] = product_df[f].fillna(product_df[f].mode()[0])

    for fea in ['fans_num', 'vip_num', 'shop_score']:
        product_df[fea] = product_df[fea].fillna(product_df[fea].mean())

    product_df['shop_reg_tm'] = product_df['shop_reg_tm'].fillna('0')
    product_df['shop_reg_tm'] = product_df['shop_reg_tm'].apply(lambda x: x.split()[0].replace('-', '')).astype(int)
    product_df['market_time'] = product_df['market_time'].apply(lambda x: x.split()[0].replace('-', '')).astype(int)
    product_df['is_cate'] = 0
    product_df['is_cate'][product_df['shop_cate'] == product_df['cate']] = 1
    del product_df['shop_id']
    del product_df['vender_id']

    return product_df


def get_actions(start_date, end_date):
    '''
    获取start_date到end_date日期内的action
    :param start_date:
    :param end_date:
    :return: action_df
    '''
    action_df = Get_Data('action')
    action_df['type'] = action_df['type'].fillna(action_df['type'].mode()[0])
    action_df = action_df[(action_df.action_time >= start_date) & (action_df.action_time < end_date)]

    return action_df


def get_action_feat(start_date, end_date):
    '''
    获取滑窗action特征
    :param start_date:
    :param end_date:
    :return:actions
    '''
    actions = get_actions(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type', 'action_time']]
    actions['action_time'] = actions['action_time'].apply(lambda x: x.split()[0].replace('-', '')).astype(int)
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    del actions['type']

    return actions


def get_accumulate_user_feat(start_date, end_date):
    feature = ['user_id', 'user_action_1_ratio',  'user_action_3_ratio', 'user_action_4_ratio', 'user_action_5_ratio']

    actions = get_actions(start_date, end_date)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions['user_id'], df], axis=1)
    actions = actions.groupby(['user_id'], as_index=False).sum()
    actions['user_action_1_ratio'] = actions['action_2'] / (actions['action_1'] + 1)
    actions['user_action_3_ratio'] = actions['action_2'] / (actions['action_3'] + 1)
    actions['user_action_4_ratio'] = actions['action_2'] / (actions['action_4'] + 1)
    actions['user_action_5_ratio'] = actions['action_2'] / (actions['action_5'] + 1)
    actions = actions[feature]

    return actions


def get_accumulate_product_feat(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_3_ratio', 'product_action_4_ratio', 'product_action_5_ratio']

    actions = get_actions(start_date, end_date)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions['sku_id'], df], axis=1)
    actions = actions.groupby(['sku_id'], as_index=False).sum()
    actions['product_action_1_ratio'] = actions['action_2'] / (actions['action_1'] + 1)
    actions['product_action_3_ratio'] = actions['action_2'] / (actions['action_3'] + 1)
    actions['product_action_4_ratio'] = actions['action_2'] / (actions['action_4'] + 1)
    actions['product_action_5_ratio'] = actions['action_2'] / (actions['action_5'] + 1)
    actions = actions[feature]

    return actions


def get_action_product_ratio(start_date, end_date):
    feature = ['sku_id', 'action_product_1_ratio', 'action_product_2_ratio', 'action_product_3_ratio', 'action_product_4_ratio',
               'action_product_5_ratio']

    actions = get_actions(start_date, end_date)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions['sku_id'], df], axis=1)
    actions = actions.groupby(['sku_id'], as_index=False).sum()
    actions['action_product_1_ratio'] = actions['action_1'] / (
            actions['action_2'] + actions['action_3'] + actions['action_4'] + actions['action_5'] + 1)
    actions['action_product_2_ratio'] = actions['action_2'] / (
            actions['action_1'] + actions['action_3'] + actions['action_4'] + actions['action_5'] + 1)
    actions['action_product_3_ratio'] = actions['action_3'] / (
            actions['action_1'] + actions['action_2'] + actions['action_4'] + actions['action_5'] + 1)
    actions['action_product_4_ratio'] = actions['action_4'] / (
            actions['action_1'] + actions['action_2'] + actions['action_3'] + actions['action_5'] + 1)
    actions['action_product_5_ratio'] = actions['action_5'] / (
            actions['action_1'] + actions['action_2'] + actions['action_3'] + actions['action_4'] + 1)
    actions = actions[feature]

    return actions


def get_action_user_ratio(start_date, end_date):
    feature = ['user_id', 'action_user_1_ratio', 'action_user_2_ratio', 'action_user_3_ratio', 'action_user_4_ratio',
               'action_user_5_ratio']

    actions = get_actions(start_date, end_date)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions['user_id'], df], axis=1)
    actions = actions.groupby(['user_id'], as_index=False).sum()

    actions['action_user_1_ratio'] = actions['action_1'] / (
            actions['action_2'] + actions['action_3'] + actions['action_4'] + actions['action_5'] + 1)
    actions['action_user_2_ratio'] = actions['action_2'] / (
            actions['action_1'] + actions['action_3'] + actions['action_4'] + actions['action_5'] + 1)
    actions['action_user_3_ratio'] = actions['action_3'] / (
            actions['action_1'] + actions['action_2'] + actions['action_4'] + actions['action_5'] + 1)
    actions['action_user_4_ratio'] = actions['action_4'] / (
            actions['action_1'] + actions['action_2'] + actions['action_3'] + actions['action_5'] + 1)
    actions['action_user_5_ratio'] = actions['action_5'] / (
            actions['action_1'] + actions['action_2'] + actions['action_3'] + actions['action_4'] + 1)
    actions = actions[feature]

    return actions


def get_labels(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[actions['type'] == 2]
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    actions['label'] = 1
    actions = actions[['user_id', 'sku_id', 'label']]

    return actions


def make_test_set(train_start_date, train_end_date):
    start_days = '2018-02-01'
    user = get_user_feature()
    product = get_product_feature()
    user_acc = get_accumulate_user_feat(start_days, train_end_date)
    product_acc = get_accumulate_product_feat(start_days, train_end_date)
    action_user_ratio = get_action_user_ratio(start_days, train_end_date)
    action_product_ratio = get_action_product_ratio(start_days, train_end_date)

    actions = None
    for i in (27, 1, 5, 10, 15, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                               on=['user_id', 'sku_id'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, product, how='left', on='sku_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, action_user_ratio, how='left', on='user_id')
    actions = pd.merge(actions, action_product_ratio, how='left', on='sku_id')
    actions = actions.fillna(0)

    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    return users, actions


def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date):
    start_days = "2018-02-01"
    user = get_user_feature()
    product = get_product_feature()
    user_acc = get_accumulate_user_feat(start_days, train_end_date)
    product_acc = get_accumulate_product_feat(start_days, train_end_date)
    action_user_ratio = get_action_user_ratio(start_days, train_end_date)
    action_product_ratio = get_action_product_ratio(start_days, train_end_date)
    labels = get_labels(test_start_date, test_end_date)

    actions = None
    for i in (27, 1, 5, 10, 15, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                               on=['user_id', 'sku_id'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, product, how='left', on='sku_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, action_user_ratio, how='left', on='user_id')
    actions = pd.merge(actions, action_product_ratio, how='left', on='sku_id')
    actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
    actions = actions.fillna(0)

    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['label']

    return users, actions, labels
