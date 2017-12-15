#-*-coding:utf8-*-
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 
import datetime

class Config:
    pass
config = pd.read_pickle('config.pkl')
print dir(config)

data_path = '../../kaggleData/JD_logging/'
feature_path = '../../kaggleData/JD_logging/features/'

login_tt = pd.read_csv(data_path+'login_tt.csv')
trade_tt = pd.read_csv(data_path+'trade_tt.csv')

login_tt['time'] = login_tt['time'].apply(lambda x : datetime.datetime.strptime(x , '%Y-%m-%d %H:%M:%S'))
trade_tt['time'] = trade_tt['time'].apply(lambda x : datetime.datetime.strptime(x , '%Y-%m-%d %H:%M:%S'))

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

def get_multiple_feature_dicts_wihtin_days(row,login_table,trade_table):
    date_range = [360,30,15,7,3,1]
    result_dict = {}
    for days in date_range:
        if days ==360:
            recent_trade_table = find_related_recent_trades_within_days(row,trade_table,days)
            recent_login_table = find_related_recent_logins_within_days(row,login_table,days)
        else:
            recent_trade_table = find_related_recent_trades_within_days(row,recent_trade_table,days)
            recent_login_table = find_related_recent_logins_within_days(row,recent_login_table,days)
        
        result_dict[days] = build_statistical_feature_dict(recent_login_table,recent_trade_table)    
    return result_dict

def build_statistical_feature_dict(recent_login_table,recent_trade_table,*args,**kw):
    """
    ID交易次数
    #最近的前一次交易时间 - 在顶层使用
    ID登录次数
    交易/登录次数比
    ID登录成功次数（大于零的项）
    ID登录失败次数（小于零的项）
    ID登录成功比率
    交易/成功登录次数比，交易/失败次数比
    是否有连续login失败
    login失败到下一次尝试的平均时间、最大时间、最小时间、时间中位数、方差
    timelong平均值，最大值，最小值，中位数，方差
    timelong方差（仅一个时为0或N/A）
    """
    result_dict = {}
    
    trade_times = len(recent_trade_table)
    login_times = len(recent_login_table)
    
    login_success_times = np.sum(recent_login_table['result']>0)
    login_fail_times = np.sum(recent_login_table['result']<0)
    
    result_dict['trade_times'] = trade_times
    result_dict['login_times'] = login_times
    result_dict['login_success_times'] = login_success_times
    result_dict['login_fail_times'] = login_fail_times
    
    if login_times ==0:
        result_dict['trade_login_rate'] = -10
    else:
        result_dict['trade_login_rate'] = trade_times*1.0/login_times
        
    if login_times ==0:
        result_dict['login_success_rate'] = -10
    else:
        result_dict['login_success_rate'] = login_success_times*1.0/login_times
    
    if login_success_times ==0:
        result_dict['trade_login_success_rate'] = -10
    else:
        result_dict['trade_login_success_rate'] = trade_times*1.0/login_success_times
    
    if login_fail_times ==0:
        result_dict['trade_login_fail_rate'] = -10
    else:
        result_dict['trade_login_fail_rate'] = trade_times*1.0/login_fail_times
    
    result_dict['multiple_fails'] = lower_than_zero_more_than_once(recent_login_table['result'])
    result_dict['after_fail_mean'],result_dict['after_fail_max'],result_dict['after_fail_min'],result_dict['after_fail_med']\
    ,result_dict['after_fail_std'] = get_averge_fail_to_success_time(recent_login_table)
    
    timelong_series =  np.log(recent_login_table['timelong']).dropna()
    if len(timelong_series) == 0:
        result_dict['timelong_mean'] = -10
        result_dict['timelong_max'] = -10
        result_dict['timelong_min'] = -10
        result_dict['timelong_med'] = -10
        result_dict['timelong_std'] = -10
    else:
        result_dict['timelong_mean'] = np.mean(timelong_series)
        result_dict['timelong_med'] = np.median(timelong_series)
        result_dict['timelong_min'] = np.min(timelong_series)
        result_dict['timelong_max'] = np.max(timelong_series)
        if len(timelong_series) > 1:
            result_dict['timelong_std'] =  np.std(timelong_series)
        else:
            result_dict['timelong_std'] = -10
    
    return result_dict

def lower_than_zero_more_than_once(sequence):
    if len(sequence)>2:
        sequence = list(sequence)
        for i in range(len(sequence)-1):
            if sequence[i] < 0:
                if sequence[i+1]<0:
                    return True
    return False

def get_averge_fail_to_success_time(recent_login_table):
    login_table_process = recent_login_table[['result','time']].sort_values(by = 'time')
    login_fail_times = np.sum(login_table_process['result'])
    
    if login_fail_times<1 or len(login_table_process)<2:
        return (-10,-10,-10,-10,-10)
    
    time_delta_list = []
    for i in range(len(login_table_process)-1):
        if login_table_process.iloc[i].result < 0:
            time_delta_list.append(login_table_process.iloc[i+1].time - login_table_process.iloc[i].time)
            
    time_delta_list = np.log(map(lambda x: x.total_seconds(),time_delta_list))
    
    if len(time_delta_list) < 2:
        std_return = -10
    else:
        std_return = np.std(time_delta_list)
        
    if len(time_delta_list) ==0:
        return -10, -10, -10, -10,-10
    
    return np.mean(time_delta_list),np.max(time_delta_list),np.min(time_delta_list),np.median(time_delta_list),std_return

#trade_tt['stat_result_dicts'] = trade_tt.apply(lambda row : get_multiple_feature_dicts_wihtin_days(row,login_tt,trade_tt),axis = 1)

#trade_tt.to_pickle(data_path+'trade_tt_stat_C_temp.pkl')

stat_result_dicts = []
i =0
for (idx, row) in trade_tt.iterrows():
	i+=1
	if i ==10:
		print "running for sample %d" % i

	if i%100 == 0:
		print "running for sample %d" % i
	stat_result_dicts.append(get_multiple_feature_dicts_wihtin_days(row,login_tt,trade_tt))

stat_result_dicts.to_pickle(data_path+'trade_tt_stat_C_result_dict.pkl')

