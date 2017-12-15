import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.metrics import fbeta_score
import xgboost as xgb


def find_related_logins_before(row,login_table,*args,**kw):
    related_logins = login_table[login_table['id'] == row.id]
    related_logins_before = related_logins[related_logins['time']<row.time]
    return related_logins_before    

def find_related_recent_logins_within_days(row,login_table,days,*args,**kw):    
    recent_logins = find_related_logins_before(row,login_table)
    if len(recent_logins)>0:
        recent_logins['from_now'] =  row.time - recent_logins['time']
        return recent_logins[recent_logins['from_now']<datetime.timedelta(days = days)]
    else:
        recent_logins['from_now'] = np.nan
        return recent_logins[recent_logins['from_now']<datetime.timedelta(days = days)]
        
def find_related_trades_before(row,trade_table,*args,**kw):
    related_trades = trade_table[trade_table['id'] == row.id]
    related_trades_before = related_trades[related_trades['time']<row.time]
    return related_trades_before    

def find_related_recent_trades_within_days(row,trade_table,days,*args,**kw):    
    recent_trades = find_related_trades_before(row,trade_table)
    if len(recent_trades)>0:
        recent_trades['from_now'] =  row.time - recent_trades['time']
        return recent_trades[recent_trades['from_now']<datetime.timedelta(days = days)]  
    else:
        recent_trades['from_now'] = np.nan
        return recent_trades[recent_trades['from_now']<datetime.timedelta(days = days)]

def get_multiple_feature_dicts_wihtin_days(id_row_tuple,login_table,trade_table,feature_generating_function,date_range_list):
    row = id_row_tuple[1]
    ori_id = id_row_tuple[0]
    
    result_dict = {}

    recent_trade_table = find_related_trades_before(row,trade_table)
    recent_login_table = find_related_logins_before(row,login_table)
    
    for date_range in date_range_list:
        recent_trade_table = find_related_recent_trades_within_days(row,recent_trade_table,date_range)
        recent_login_table = find_related_recent_logins_within_days(row,recent_login_table,date_range)
        
        result_dict[date_range] = feature_generating_function(recent_login_table,recent_trade_table)

    return ori_id,row.rowkey,result_dict

"""
transfer_function : returning a dict for each single column
"""
def get_transfer_result_wihtin_days(trade_tt,login_tt,date_range_list,spark_context,transfer_function):
#packing the rdd for spark
    trade_tt_rdd_buffer = []
    for (idx,row) in trade_tt.iterrows():
        trade_tt_rdd_buffer.append((idx,row))
    trade_tt_rdd = spark_context.parallelize(trade_tt_rdd_buffer)
    
    result_rdd = trade_tt_rdd.map(lambda x : get_multiple_feature_dicts_wihtin_days(x,login_tt,trade_tt,transfer_function,date_range_list))
    result_rdd_buffer = result_rdd.collect()
    
    #getting the new feature names
    #recent_trade_example=find_related_recent_trades_within_days(trade_tt.loc[0],trade_tt,30)
    #recent_login_example=find_related_recent_logins_within_days(trade_tt.loc[0],login_tt,30)
    
    #feature_list = list(transfer_function(recent_login_example,recent_trade_example).keys())
    
    #unstacking the result_rdd_dict
    result_rdd_to_df_buffer = []
    for ori_id,rowkey,result_dict in result_rdd_buffer:
        unit_dict= {}
        unit_dict['ori_id'] = ori_id
        unit_dict['rowkey'] = rowkey
        
        for date_range in date_range_list:
            for key in result_dict[date_range]:
                unit_dict[key+'_'+str(date_range)]= result_dict[date_range][key]
        
        result_rdd_to_df_buffer.append(unit_dict)

    result_df = pd.DataFrame(result_rdd_to_df_buffer)

    assert (np.sum(result_df['rowkey']!=trade_tt['rowkey']))==0
    assert (np.sum(result_df['ori_id']!=trade_tt.index))==0
    
    return result_df    

def check_nan_inf(DataFrame,feature_list):
    for feature in feature_list:
        try:
            if np.sum(np.isnan(DataFrame[feature]))>0:
                print("nan exist for %s",feature)
        except:
            print("failed to check nan for %s, type is %s" % (feature,str(DataFrame[feature].dtype)))
        try:
            if np.sum(DataFrame[feature]==np.inf)>0:
                print("inf exist for %s",feature)
        except:
            print("failed to check inf for %s, type is %s" % (feature,str(DataFrame[feature].dtype)))
        try:
            if np.sum(DataFrame[feature]==-np.inf)>0:
                print("-inf exist for %s",feature)
        except:
            print("failed to check -inf for %s, type is %s" % (feature,str(DataFrame[feature].dtype)))
            
def get_feature_emptyness(DataFrame,feature_list):
    feature_emptyness = {}
    feature_emptyness_list = []
    data_size = DataFrame.shape[0]
    for feature in feature_list:
        feature_emptyness[feature] = np.sum(DataFrame[feature] == -10)*1.0/data_size
        feature_emptyness_list.append((feature,feature_emptyness[feature]))
    return feature_emptyness_list

def f_beta_01(preds, train_data, threshold = 0.5):
    labels  = train_data.get_label()
    return 'fbeta_score_01',fbeta_score(labels, preds > threshold,0.1),True

def f_beta_01_xgb(preds, train_data, threshold = 0.5):
    labels  = train_data.get_label()
    return 'fbeta_score_01',fbeta_score(labels, preds > threshold,0.1)

#xgb for binary
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, 
     seed_val=0, early_stop = 20,num_rounds=10000, eta = 0.1,
     subsample = 0.75,colsample_bytree = 0.75,eval_metric = 'auc',feval = None,
     max_depth = 6,cv_dict = None,verbose_eval=True):
    
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['silent'] = 1
    #param['num_class'] = 3
    param['eval_metric'] = eval_metric
    param['min_child_weight'] = 1
    param['subsample'] = subsample
    param['colsample_bytree'] = colsample_bytree
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y,feature_names=feature_names)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y,feature_names=feature_names)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,
        early_stopping_rounds=early_stop,evals_result = cv_dict,
        verbose_eval = verbose_eval,feval = feval)
    else:
        xgtest = xgb.DMatrix(test_X,feature_names=feature_names)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

#for binary
def runLGBM(train_X, train_y, test_X, test_y=None, feature_names=None,
           seed_val=0, num_rounds=10000,watch_dict = None,max_bin=50000,
           num_leaves=16,early_stop=64,verbose=True,eta=0.1,
           bagging_fraction = 0.75 , feature_fraction = 0.75,feval = None,metric = 'binary_logloss',
           train_sample_weight = None,is_unbalance = False):
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': num_leaves,
        'learning_rate': eta,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': 5,
        'verbose': verbose,
        'is_unbalance':is_unbalance
    }
    
    num_rounds = num_rounds

    #plst = list(param.items())
    lgbtrain = lgb.Dataset(train_X, label=train_y,max_bin=max_bin,feature_name=feature_names,weight =train_sample_weight)

    if test_y is not None:
        lgbtest = lgb.Dataset(test_X, label=test_y,max_bin=max_bin,feature_name=feature_names)
        watchlist = [lgbtrain,lgbtest]
        watchlist_name=['train','test']
        model = lgb.train(params, lgbtrain, num_rounds, watchlist,watchlist_name, early_stopping_rounds=early_stop,\
                         evals_result = watch_dict,verbose_eval=verbose,feval = feval)
    else:
        #lgbtest = lgb.Dataset(test_X,feature_name=feature_names)
        model = lgb.train(params, lgbtrain, num_rounds)

    pred_test_y = model.predict(test_X)
    return pred_test_y, model

def check_feature_importance(models,features):
    feature_importance_total = np.zeros(len(features))
    for model in models:
        feature_importance_total+=model.feature_importance('gain')
    sorted_feature_importacne = sorted(zip(features,feature_importance_total),key = lambda x : x[1],reverse = True)
    return sorted_feature_importacne

def find_related_hcc_logins_before(row,login_table,hcc,*args,**kw):
    related_logins = login_table[login_table[hcc] == row[hcc]]
    related_logins_before = related_logins[related_logins['time']<row.time]
    return related_logins_before    

def find_related_hcc_logins_within_days(row,login_table,hcc,days,*args,**kw):    
    recent_logins = find_related_hcc_logins_before(row,login_table,hcc)
    if len(recent_logins)>0:
        recent_logins['from_now'] =  row.time - recent_logins['time']
        return recent_logins[recent_logins['from_now']<datetime.timedelta(days = days)]
    else:
        recent_logins['from_now'] = np.nan
        return recent_logins[recent_logins['from_now']<datetime.timedelta(days = days)]

#special modified version for this type of features
def get_multiple_hcc_feature_dicts_wihtin_days(id_row_tuple,login_table,hcc,feature_generating_function,date_range_list):
    row = id_row_tuple[1]
    ori_id = id_row_tuple[0]
    original_trade_id = row.id
    
    result_dict = {}

    #recent_trade_table = find_related_hcc_logins_before(row,trade_table)
    recent_login_table = find_related_hcc_logins_before(row,login_table,hcc)
    
    for date_range in date_range_list:
        #recent_trade_table = find_related_hcc_logins_within_days(row,recent_trade_table,date_range)
        recent_login_table = find_related_hcc_logins_within_days(row,recent_login_table,hcc,date_range)
        
        result_dict[date_range] = feature_generating_function(recent_login_table,hcc,original_trade_id)

    return ori_id,row.rowkey,result_dict

def mult_type_hcc_feature(id_row_tuple,login_table,feature_generating_function,date_range_list):
    hcc_feature_list = ['ip','device']
    result_dict = {}
    for hcc in hcc_feature_list:
        temp_result = get_multiple_hcc_feature_dicts_wihtin_days(id_row_tuple,login_table,hcc,feature_generating_function,date_range_list)
        if (len(result_dict.keys())==0):
            result_dict = temp_result[2]
        else:
            for date_range in date_range_list:
                result_dict[date_range] = combine_dicts(result_dict[date_range],temp_result[2][date_range])
    return temp_result[0],temp_result[1],result_dict

def combine_dicts(dict_1,dict_2):
    return dict(dict_1,**dict_2)

def resample(train_set,labels):
    false_set = train_set[labels==0]
    true_set = train_set[labels==1]
    true_set_size = true_set.shape[0]
    #print(len(false_set))
    new_false_set = false_set[np.random.choice(list(range(len(false_set))),size = 4*true_set_size)]
    
    new_train_set = np.vstack([true_set,true_set,new_false_set])
    new_train_label = np.hstack([np.ones(true_set_size,dtype=np.int),np.ones(true_set_size,dtype=np.int),np.zeros(4*true_set_size,dtype=np.int)])
    
    shuffle_index = list(range(new_train_set.shape[0]))
    np.random.shuffle(shuffle_index)
    
    new_train_set = new_train_set[shuffle_index]
    new_train_label = new_train_label[shuffle_index]

    return new_train_set,new_train_label