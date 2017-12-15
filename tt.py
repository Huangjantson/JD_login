# coding:utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
loginData=pd.read_csv("t_login.csv",dtype={'log_id':str,'id':str})
loginTestData=pd.read_csv("t_login_test.csv",dtype={'log_id':str,'id':str})
tradeData=pd.read_csv("trade_2_login.csv",dtype={'id':str,'login_id':str})
tradeTestData=pd.read_csv("t_trade_test_2_login.csv",dtype={'id':str,'login_id':str})

loginData = loginData[(loginData.time < '2015-05-01') & (loginData.time >= '2015-02-01')]
tradeData = tradeData[(tradeData.time >= '2015- -01') & (tradeData.time < '2015-06-01')]

loginTestData=loginTestData.append(loginData)

def getUserIDFromDataFrame(dataFrame):
    return pd.DataFrame({'id':dataFrame['id'].unique()})


#得到登录中所用时间为整秒的计算
def getTimelongScale(loginData,test):
    #计算是否为整秒
    loginData['timeScale']=(loginData['timelong']%1000==0)
    #分别为整数 ，不为整数的次数
    gd=loginData['timeScale'].groupby([loginData['id'],loginData['timeScale']])
    tmp=gd.count().unstack().reset_index()
    tmp=tmp.fillna(0)
    test=pd.merge(tmp,test,on='id',how='inner')
#     print(test.head(2))
    del loginData['timeScale'],tmp
    return test

#得到某列的分组不同的个数
def getCountsByColumnName(loginData,test,columnName):
    num=loginData[['id',columnName]].groupby(loginData['id'])[columnName].nunique()
    t=pd.DataFrame(num)
    t.insert(0, 'id', num.index)
    t.columns=['id',columnName+'_Counts']
    test=pd.merge(t,test,on='id',how='inner')
    return test

#得到某列的分组平均值
def getMeanByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.mean().reset_index()
    t.columns=['id',columnName+'_Mean']
    test=pd.merge(t,test,on='id',how='inner')
    return test
#得到某列的分组最小值
def getMinByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.min().reset_index()
    t.columns=['id',columnName+'_min']
    test=pd.merge(t,test,on='id',how='inner')
    return test
def getMaxByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.max().reset_index()
    t.columns=['id',columnName+'_max']
    test=pd.merge(t,test,on='id',how='inner')
    return test
def getVarByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.var().reset_index()
    t.columns=['id',columnName+'_var']
    test=pd.merge(t,test,on='id',how='inner')
    return test
def getStdByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.std().reset_index()
    t.columns=['id',columnName+'_std']
    test=pd.merge(t,test,on='id',how='inner')
    return test

def getLoginResultCount(loginData,test):
    r31=loginData[loginData['result']!=1]
    t=r31['result'].groupby(r31['id'])
    t=t.count().reset_index()
    t.columns=['id','result_no1_Counts']
    test=pd.merge(t,test,on='id',how='inner')
    test=test.fillna(0)
    return test


from sklearn.preprocessing import OneHotEncoder


# 处理交易数据
def tradeDataInit(tradeData, loginData):
    tradeData = tradeData.copy()
    print tradeData.columns
    # 构建一个小时的
    tradeData['hours'] = tradeData['time'].str.extract('\\s(\\d\\d):')
    tradeData['hours'] = tradeData['hours'].astype('int')
    tradeData = pd.merge(tradeData, loginData[['log_id', 'time']], left_on='login_id', right_on='log_id', how='inner')
    del tradeData['log_id']
    #     tradeData['hours']=
    #     print(OneHotEncoder(sparse = False).fit_transform( tradeData[['hours']]))
    # 交易时间
    tradeData['tx'] = pd.to_datetime(tradeData['time_x'])
    # 登录时间
    tradeData['ty'] = pd.to_datetime(tradeData['time_y'])
    # 交易时间和登录时间之间的差值
    tradeData['st'] = (tradeData['tx'] - tradeData['ty']).dt.seconds
    # 每次交易时间的差
    # tradeData['trade_time_sub'] = tradeData.sort_values(by='tx').groupby('id')['tx'].diff()
    # tradeData['trade_time_sub_day'] = tradeData['trade_time_sub'].dt.days
    # tradeData = getMaxByColumnName(tradeData, tradeData, 'trade_time_sub_day')
    # tradeData = getMeanByColumnName(tradeData, tradeData, 'trade_time_sub_day')
    # tradeData = getMinByColumnName(tradeData, tradeData, 'trade_time_sub_day')
    # tradeData = getStdByColumnName(tradeData, tradeData, 'trade_time_sub_day')
    # tradeData = getVarByColumnName(tradeData, tradeData, 'trade_time_sub_day')
    # 使用的比例
    tradeData = getTradeLoginColumScale(tradeData, loginData, 'device')
    tradeData = getTradeLoginColumScale(tradeData, loginData, 'city')

    del tradeData['tx'], tradeData['ty'], tradeData['time_y'], tradeData['login_id']
    tradeData.rename(columns={'time_x': 'time'}, inplace=True)
    return tradeData


def getTradeLoginColumScale(tradeData,loginData,cname):
    #计算cname列上分别使用的个数
    c_count=loginData[['id',cname,'log_id']].groupby(['id',cname]).count()
    c_count=c_count.reset_index()
    c_count.columns=['id',cname,cname+'_count_tmp']
    #使用的总个数
    c_sum=c_count[['id',cname+'_count_tmp']].groupby('id').sum().reset_index()
    c_sum.columns=['id',cname+'_count_sum']
    c=pd.merge(c_count,c_sum,on='id',how='inner')
    #使用的比例
    c[cname+'_scale']=c[cname+'_count_tmp']/c[cname+'_count_sum']
    del c[cname+'_count_tmp'],c[cname+'_count_sum']
    c=pd.merge(loginData[['log_id','id',cname]],c,on=['id',cname],how='inner')
    tradeData=pd.merge(tradeData,c[['log_id',cname+'_scale']],left_on='login_id',right_on='log_id',how='inner')
    del c,tradeData['log_id']
    return tradeData

def createAllData(test,tradeData):
    tmp=pd.merge(test,tradeData,on='id',how='inner')
#     tmp=tmp.fillna(0)
#     print(tmp.info())
    return tmp

from sklearn.metrics import fbeta_score
#评估函数
def rocJdScore(*args):
    from sklearn import metrics
    return metrics.make_scorer(fbeta_score,beta=0.1, greater_is_better=True)(*args)

# 得到训练用的测试集元组（x，y）
def getTrainData(isUndersample=False):
    allData = transferData(loginData, tradeData)
    if (isUndersample):
        # 进行undersample的处理
        number_records_fraud = len(allData[allData['is_risk'] == 1])  # 有风险的个数
        fraud_indices = np.array(allData[allData['is_risk'] == 1].index)  # 有风险的index
        normal_indices = allData[allData['is_risk'] == 0].index
        random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
        random_normal_indices = np.array(random_normal_indices)
        under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
        allData = allData.iloc[under_sample_indices, :]

    x = allData.iloc[:, 3:].values
    y = allData['is_risk'].values

    return x, y


# 生成学习算法
def jdPipeFited():
    x, y = getTrainData(isUndersample=False)
    from sklearn.cross_validation import train_test_split
    # 拆分成训练集(80%)和测试集(20%)
    train_X, test_X, train_y, test_y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.05
    param['max_depth'] = 5
    param['eval_metric'] = "auc"
    param['min_child_weight'] = 3
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = 321
    param['silent'] = 1
    param['nthread'] = 3
    num_rounds = 10000

    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgtest = xgb.DMatrix(test_X, label=test_y)

    watchlist = [(xgtrain, 'train'), (xgtest, 'eval')]
    xgbFier = xgb.train(param, xgtrain, num_rounds, watchlist, early_stopping_rounds=40)

    importance = xgbFier.get_fscore()
    print(sorted(importance.items()))
    xgbFier.dump_model("dump.raw.txt")

    return xgbFier

def ff(preds, dtrain):
    label = dtrain.get_label()
    preds = map(lambda x: 1 if x > 0.8 else 0, preds)
    return 'fbeta_score', fbeta_score(label, preds, beta=0.1)

def transferData(loginData, tradeData):
    lData = getUserIDFromDataFrame(loginData)
    # 登录耗费时间段是否为整数
    lData = getTimelongScale(loginData, lData)
    lData = getMinByColumnName(loginData, lData, 'timelong')
    lData = getMaxByColumnName(loginData, lData, 'timelong')
    lData = getMeanByColumnName(loginData, lData, 'timelong')
    # lData = getVarByColumnName(loginData, lData, 'timelong')
    lData = getStdByColumnName(loginData, lData, 'timelong')
    # #     #device的个数
    lData = getCountsByColumnName(loginData, lData, 'device')
    # #     #ip的个数
    lData = getCountsByColumnName(loginData, lData, 'ip')
    lData = getCountsByColumnName(loginData, lData, 'log_from')

    lData = getCountsByColumnName(loginData, lData, 'type')
    #     #城市的个数
    lData = getCountsByColumnName(loginData, lData, 'city')
    # 登录的次数
    lData = getCountsByColumnName(loginData, lData, 'log_id')
    # 平均值
    #     lData=getMeanByColumnName(loginData,lData,'device')
    #     lData=getMeanByColumnName(loginData,lData,'log_from')
    #     lData=getMeanByColumnName(loginData,lData,'ip')
    #     lData=getMeanByColumnName(loginData,lData,'city')
    #     lData=getMeanByColumnName(loginData,lData,'result')
    #     lData=getMeanByColumnName(loginData,lData,'type')
    # result的处理
    lData = getLoginResultCount(loginData, lData)

    tData = tradeData.copy()
    tData = tradeDataInit(tData, loginData)
    #     tData=getLastSubTime(loginData,tData)
    allData = createAllData(lData, tData)
    del allData['time']
    # get a list of columns
    cols = list(allData)
    if 'is_risk' in allData.columns:
        cols.insert(0, cols.pop(cols.index('is_risk')))
    cols.insert(0, cols.pop(cols.index('rowkey')))

    allData = allData.ix[:, cols]
    print(allData.head(2))
    return allData

if __name__ == '__main__':
    #k-fold交叉验证
    from sklearn.cross_validation import cross_val_score
    # pipe_lr=getPipe()
    # X_train,y_train=getTrainData(isUndersample=False)
    #记录程序运行时间
    import time
    start_time = time.time()
    # scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=5, n_jobs=2,scoring=rocJdScore)
    # print(scores)
    #整体预测
    X_train,y_train=getTrainData(isUndersample=False)

    #输出运行时长
    cost_time = time.time()-start_time
    print("交叉验证 success!",'\n',"cost time:",cost_time,"(s)")

    pipe=jdPipeFited()
    preData=transferData(loginTestData,tradeTestData)
    x_pred=preData.iloc[:,2:].values
    xgtest = xgb.DMatrix(x_pred)
    y_pred=pipe.predict(xgtest)

    p=pd.DataFrame(y_pred)
    p.columns=['is_risk']
    p.loc[p['is_risk'] > 0.8, 'is_risk'] = 1
    p.loc[p['is_risk'] <= 0.8, 'is_risk'] = 0
    print p.is_risk.value_counts()
    subData=pd.DataFrame(preData['rowkey'])
    subData['is_risk']=p
    #之前用很多inner join，很多数据没有，都默认处理为没有风险
    subData=pd.merge(tradeTestData[['rowkey']],subData,on='rowkey',how='left')
    subData=subData.fillna(0)
    subData['is_risk']=subData['is_risk'].astype('int')

    subData.to_csv('./sub.csv',header=False,index=False)