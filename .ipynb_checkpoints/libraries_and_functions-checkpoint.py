# Create a file allowing to import upper level(usefull throughout the whole solution) packages and functions with one line: %run libraries

import os #The functions that the OS module provides allows you to interface with the underlying operating system that Python is running on 

import pickle # Fast saving/loading data

import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Import visualizations
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,5) # Set standard output figure size

import seaborn as sns # sns visualization library

from IPython.display import display # Allows to nicely display/output several figures or dataframes in one cell

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

from itertools import product

import shap # Calculate feature importance for tree-based algorithms

######################################################################################################################################################################

def create_directory_structure():
    # Create an output' folder to save data from the notebook
    try: os.mkdir('output') # Try to create
    except FileExistsError: pass # if already exist pass

    # Create 'data' folder in the 'output' folder
    try: os.mkdir(r'output/data')
    except FileExistsError: pass
    # Create 'submissions' folder in the 'output' folder
    try: os.mkdir(r'output/submissions')
    except FileExistsError: pass
    # Create 'models' folder in the 'output' folder
    try: os.mkdir(r'output/models')
    except FileExistsError: pass

def create_journal():
    # Create Experimental Journal, for experiments tracking
    from os.path import exists
    if exists('Journal.csv'):
        pass
    else:
        row = 'Test name, RMSE_train, RMSE_val, RMSE_Public, Comments'
        with open('Journal.csv','a') as file:
            file.write(row)

######################################################################################################################################################################            
            
# 1.EDA functions

# Create functions.py file, which contains usefull functions for the notebook
# Define dataframe information function
def df_info(df):
    print('-------------------------------------------shape----------------------------------------------------------------')
    print(df.shape)
    print('-------------------------------------head() and tail(1)---------------------------------------------------------')
    display(df.head(), df.tail(1))
    print('------------------------------------------nunique()-------------------------------------------------------------')
    print(df.nunique())
    print('-------------------------------------describe().round()---------------------------------------------------------')
    print(df.describe().round())
    print('--------------------------------------------info()--------------------------------------------------------------')
    print(df.info())
    print('-------------------------------------------isnull()-------------------------------------------------------------')
    print(df.isnull().sum())
    print('--------------------------------------------isna()--------------------------------------------------------------')
    print(df.isna().sum())
    print('-----------------------------------------duplicated()-----------------------------------------------------------')
    print(len(df[df.duplicated()]))
    print('----------------------------------------------------------------------------------------------------------------')  
#-------------------------------------------------------------------------------------------


# Train set contains only those pairs, which were sold or returned in the past, - no zero values.
# In order to make train and test sets similar we need to modify the train set to a different view:
# i.e. calculate monthly sales for each unique pair (item_id, shop_id) within each month, 
# those wich had zero sales should be zeroed.
def TrainTestTransform(train, test):
    X = []
    cols = ['date_block_num','shop_id','item_id']
    for i in range(34):
        sales = train[train.date_block_num == i]
        X.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16')) 
    X = pd.DataFrame(np.vstack(X), columns=cols)
    X.sort_values(cols,inplace=True)
    
    # Aggregate train set by month-shop-item to calculate target aggreagates.
    # This way train target will be similar to the test predictions.
    group = train.groupby(cols).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month']
    group.reset_index(inplace=True)
    X = pd.merge(X, group, on=cols, how='left')
    
    #X['item_cnt_month'] = X['item_cnt_month'].clip(clip_min,clip_max) # Competition evaluation requirement: True target values are clipped into [0,20] range.
    X['item_cnt_month'] = X['item_cnt_month'].fillna(0)
    
    # Append test set pairs to X
    test['date_block_num'] = 34
    X = pd.concat([X, test], ignore_index=True, sort=False, keys=cols)
    
    return X
#-------------------------------------------------------------------------------------------


def groupby_and_plot(df, groupby_cols, target_col):
    fig, ax = plt.subplots(3,1, figsize=(15,12), sharex = True)
    plt.xticks(rotation=90)
    plt.subplots_adjust(hspace=0.0)
    
    ts=df.groupby(groupby_cols)[target_col].sum()
    ax[0].set_title('SUM', x=0.5, y=0.9)
    ax[0].bar(ts.index, ts.values)
    
    ts=df.groupby(groupby_cols)[target_col].mean()
    ax[1].set_title('MEAN', x=0.5, y=0.9)
    ax[1].bar(ts.index, ts.values)
    
    ts=df.groupby(groupby_cols)[target_col].std()
    ax[2].set_title('STD', x=0.5, y=0.9)
    ax[2].bar(ts.index, ts.values)
    
######################################################################################################################################################################
    
    


    
# 2.FE functions    

# LGBM expanding window validation scheme function
# Define downcast function allowing to save memory:
# downcasting to float first (NaN is supported by floats only), it downcasts to float32 only, so might need more downcasting if one has memory issue
# Further downcasting to int and uint (this one works up to int8 and uint8 )
def downcast(X):
    for column in X:
        X[column] = pd.to_numeric(X[column],downcast = 'float')
        X[column] = pd.to_numeric(X[column],downcast = 'integer')
        #X[column] = pd.to_numeric(X[column],downcast = 'unsigned', errors='ignore') # ignore, otherwize it will raise error for negative values
# There are certain downsides to downcast to float 16: min, max value: +/- 65504, arithmetic errors accumulate quite quickly with float16s: np.array([0.1,0.2], dtype='float16').sum() equals (approximately) 0.2998.
# Especially when computations require thousands of arithmetic operations, this can be an unacceptable amount of error for many applications.
# There were no noticable difference in performance when float 32 or float 16 were used in all calculations below. Neither in time, nor in accuracy. Since it is same, why risk and downcast to float16.
# The only reason to downcast to float16 is memory errror, which is not the case here. But if you have memory error - go ahead and downcast to float 16.
#-------------------------------------------------------------------------


def lag_feature(df, lags, col, drop = False):
    tmp = df[['date_block_num','shop_id','item_id', col]]                                  # 3 first rows are used as index (keys in merge) when merging this temporary df with the main df. Only when all three keys are the same will the value appear in the new merged df, otherwize NaN
    for i in lags:
        new_col_name = col+'_lag_'+str(i)
        shifted = tmp.copy()                                                              #create a temporary df
        shifted.columns = ['date_block_num','shop_id','item_id', new_col_name]            # rename last column in temporary df
        shifted['date_block_num'] += i                                                    #change date_block_num to shifted value
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left') #merge the shifted column to main df
    if drop == True:
        df.drop(col, axis = 1, inplace = True) 
    return df
#-------------------------------------------------------------------------


# Alhpa smoothing can be applied to both train and test simultaneously 
def alpha_smoothing(df, groupby_columns, target, new_col_name, alpha = 5):
    # calculate mean value for all
    target_mean = df[target].mean()
    # calculate group
    group =  df.groupby(groupby_columns)[target]
    # calculate mean value per each category in the group
    mean_by_category =group.transform('mean')
    # calculate number of samples per each category in the group
    n_samples_in_category = group.transform('count')
    # calculate smoothed value
    df[new_col_name]  = ((mean_by_category*n_samples_in_category) + (target_mean*alpha))/(alpha + n_samples_in_category)
    # fill NaN with global mean value
    df[new_col_name].fillna(target_mean, inplace=True)
    return df
#-------------------------------------------------------------------------


# Mean encodings, KFold regularization
def mean_kfold_feature(df, groupby_columns, target, new_col_name, n_splits = 5, tr_test_split = 33):
    # Check that groupby_columns is list
    if not isinstance(groupby_columns, list):  
        raise Exception('groupby_columns should be of list type')
    
    df_train = df[df.date_block_num < tr_test_split]
   
    #Calculate target mean
    target_mean = df_train[target].mean()
    # Initialize new column
    df[new_col_name] = np.NaN
   
    kf = KFold(n_splits, shuffle = True, random_state=42)
    for tr_ind, val_ind in kf.split(df):
        
        # find intersection between indices. get tr_ind but only those ones that are in train data, not in test
        indices_train = pd.Series(list(set(df_train.index.to_series()) & set(tr_ind)))
        
        # If only one column is used in groupby_columns use map method
        if len(groupby_columns) == 1:
            # Caluclate mapping from intersection indices
            means_dict = df_train.iloc[indices_train].groupby(groupby_columns)[target].mean()
            # map to val indices from both train and test
            df[new_col_name].iloc[val_ind] = df[groupby_columns].iloc[val_ind].squeeze().map(means_dict)
            
        # Elif several columns are used for groupby - use merge method 
        elif len(groupby_columns) > 1:
            group = df_train.iloc[indices_train].groupby(groupby_columns)[target].mean()
            tmp = df[groupby_columns].iloc[val_ind]
            indices = tmp.index
            merged = tmp.merge(group, on = groupby_columns, how = 'left')
            merged.index = indices
            df[new_col_name].iloc[val_ind] = merged.iloc[:,-1]
        else:
            raise Exception('wrong number of columns in groupby_columns')
     
    df[new_col_name].fillna(target_mean, inplace=True)        
    return df
#-------------------------------------------------------------------------


def kfmean_lag_feature(df, groupby_columns, new_col_name, lags, target = 'item_cnt_month', n_splits = 5, alpha = 5, tr_test_split = 33):
    if alpha ==0:
        pass
    else:
        df = alpha_smoothing(df, groupby_columns, target, new_col_name, alpha)        
    df = mean_kfold_feature(df, groupby_columns, target, new_col_name, n_splits, tr_test_split)
    df = lag_feature(df, lags, new_col_name, drop = True)
    return df
#----------------------------------------------------


# Define aggregate function, which groups values by groupby_columns according to aggregate_how parameter from df_from to df_to and create a new_column_name for this new column 
def aggregate(df_to, df_from, groupby_columns, aggregate_how, new_column_name):
    group = df_from.groupby(groupby_columns).agg(aggregate_how)
    group.columns = new_column_name
    group.reset_index(inplace=True)
    df_to = pd.merge(df_to, group, on=groupby_columns, how='left')
    return df_to

######################################################################################################################################################################



# 3.Validation functions    

def LGBM_EXPANDING_WINDOW(X, min_month, num_boost_round = 1, early_stopping_rounds = None, lambda_l2 = 0, learning_rate = 0.1):
    error = []
    for i in range(min_month,34):     
        X_train = X[X.date_block_num < i].drop(['item_cnt_month'], axis=1)
        y_train = X[X.date_block_num < i]['item_cnt_month']
        X_val   = X[X.date_block_num == i].drop(['item_cnt_month'], axis=1)
        y_val   = X[X.date_block_num == i]['item_cnt_month']

        train_dataset = lgb.Dataset(X_train, y_train)
        val_dataset = lgb.Dataset(X_val, y_val) 
    
        params = {'objective':        'regression',
                  'metric':           'rmse',
                  'num_leaves':       1417,
                  'max_depth':        10,
                  'min_data_in_leaf': 469,
                  'max_bin':          97,
                  'lambda_l2':        lambda_l2,
                  'learning_rate':    learning_rate,
                  'bagging_fraction': 0.5,
                  'feature_fraction': 0.5,
                  'bagging_freq':     1,
                  'bagging_seed':     42,
                  'verbosity':        -1,
                  'seed':             42}

        model = lgb.train(params,
                          train_dataset,
                          valid_sets=[train_dataset, val_dataset],
                          valid_names=['train','val'],
                          num_boost_round = num_boost_round, 
                          early_stopping_rounds=early_stopping_rounds,                  
                          verbose_eval=num_boost_round)

        rmse_train=model.best_score['train']['rmse']
        rmse_train_norm=model.best_score['train']['rmse']/y_train.mean()
        rmse_train_std=model.best_score['train']['rmse']/y_train.std()
        rmse_val=model.best_score['val']['rmse']
        rmse_val_norm=model.best_score['val']['rmse']/y_val.mean()
        rmse_val_std=model.best_score['val']['rmse']/y_val.std()
        error.append([i,rmse_train,rmse_val, rmse_train_norm, rmse_val_norm, rmse_train_std, rmse_val_std])
    
    error=np.array(error)

    #--Plot-----------------------------------------------------------    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(16,10), sharex=True)
    plt.xticks(range(7,34))
    plt.subplots_adjust(hspace=0.01)
    
    ax0.plot(error[:,0], error[:,1], 'go--', label = ['train_rmse'])
    ax0.plot(error[:,0], error[:,2], 'ro--', label = ['val_rmse'])    
    ax0.grid()
    ax0.legend(loc="upper right")
    
    ax1.plot(error[:,0], error[:,3], 'go--', label = ['train_rmse/mean'])
    ax1.plot(error[:,0], error[:,4], 'ro--', label = ['val_rmse/mean'])    
    ax1.grid()
    ax1.legend(loc="upper right")
    
    ax2.plot(error[:,0], error[:,5], 'go--', label = ['train_rmse/std'])
    ax2.plot(error[:,0], error[:,6], 'ro--', label = ['val_rmse/std'])
    ax2.grid()
    ax2.legend(loc="upper right")
    #------------------------------------------------------------------
    return error
#-------------------------------------------------------------------------




def LGBM(X,
         num_boost_round = 100,
         early_stopping_rounds = None,
         lambda_l2 = 10,
         learning_rate = 0.1,
         plot_error = True,
         plot_shap = True,
         save_to_journal = False,
         save_subm_preds = False,
         test_name = 'test'):
    
    """
    LGBM function

    Arguments:
    X -- data dataframe
    num_boost_round -- int, number of boosting rounds
    early_stopping_rounds -- int, if val error doesn't improve for this number of rounds - stop
    lambda_l2 -- float, lgbm l2 regularization parameter
    learning_rate -- float, lgbm learning_rate parameter
    plot_error -- bool, plot train/val error graph
    plot_shap -- bool for plotting and calculating shap featue importance
    save_subm_preds -- False/string - name for submission file

    Returns:
    y_pred_train -- predicts for train
    y_pred_val -- predicts for val (last month)
    feature_importance -- shap feature importance dataframe
    """

    # initialize outputs:
    y_pred_train = None
    y_pred_val = None
    feature_importance = None
    #------------------------------------------------------------------------------

    #Prepare Data
    last_month = max(X.date_block_num)
    X_train = X[X.date_block_num < last_month].drop(['item_cnt_month', 'date_block_num'], axis=1)
    y_train = X[X.date_block_num < last_month]['item_cnt_month']
    X_val   = X[X.date_block_num == last_month].drop(['item_cnt_month', 'date_block_num'], axis=1)
    y_val   = X[X.date_block_num == last_month]['item_cnt_month']

    train_dataset = lgb.Dataset(X_train, y_train)
    val_dataset = lgb.Dataset(X_val, y_val) 
    #------------------------------------------------------------------------------
    
    #Train model
    print('Training model...')
    params = {'objective':        'regression',
              'metric':           'rmse',
              'num_leaves':       1417,
              'max_depth':        10,
              'min_data_in_leaf': 469,
              'max_bin':          97,
              'lambda_l2':        lambda_l2,
              'learning_rate':    learning_rate,        
              'bagging_fraction': 0.5,
              'feature_fraction': 0.5,
              'bagging_freq':     1,
              'bagging_seed':     42,
              'verbosity':        -1,
              'seed':             42}

    evals_result = {}  # to record eval results for plotting
    
    valid_sets=[train_dataset, val_dataset]
    valid_names=['train','val']
    if np.isnan(y_val).any():
        valid_sets=[train_dataset]
        valid_names=['train']
    
    model = lgb.train(params,
                      train_dataset,
                      valid_sets=valid_sets,
                      valid_names=valid_names,
                      num_boost_round = num_boost_round, 
                      early_stopping_rounds=early_stopping_rounds,
                      evals_result=evals_result,                      
                      verbose_eval=500)

    y_pred_train = model.predict(X_train).clip(0,20) # Competition evaluation requirement: True target values are clipped into [0,20] range 
    y_pred_val = model.predict(X_val).clip(0,20)
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    if np.isnan(y_val).any(): RMSE_val = None 
    else: RMSE_val = np.sqrt(mean_squared_error(y_val, y_pred_val))    
    print(f'LGBM: RMSE train: {RMSE_train}  RMSE val: {RMSE_val}')
    #------------------------------------------------------------------------------

    # Plot train and test errors
    if plot_error == True:    
        plt.figure(figsize=(9,9))
        plt.plot(evals_result['train']['rmse'], 'g', label='train')
        if np.isnan(y_val).any() == False:
            plt.plot(evals_result['val']['rmse'], 'r', label='val')
        plt.legend(loc = 'upper right')
        plt.grid(True)
        plt.show()
    #------------------------------------------------------------------------------

    # Plot shap values and prepare feature_importance df
    if plot_shap == True:
        print('Plotting shap values...')
        shap_values = shap.TreeExplainer(model).shap_values(X_val)
        shap.summary_plot(shap_values, X_val, plot_type='bar', max_display=30) # plot_type = 'bar'/'dot'/'violin'
        plt.show()
        # Prepare feature_importance dataframe
        vals= np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    #------------------------------------------------------------------------------

    # Save test results to Journal
    if save_to_journal == True:
        row = '\n'+test_name+','+str(RMSE_train)+','+str(RMSE_val)
        with open('Journal.csv','a') as file:
            file.write(row)
    #------------------------------------------------------------------------------

    # Save the predictions to csv for submission:
    if save_subm_preds != False:
        df = pd.DataFrame(y_pred_val, columns = ['item_cnt_month']).reset_index()
        df.rename(columns = {'index':'ID'}, inplace=True)
        df.to_csv(r'output/submissions/'+test_name+'.csv', index=False)
    #------------------------------------------------------------------------------

    return y_pred_train, y_pred_val, feature_importance

######################################################################################################################################################################


print('Libraries and functions loaded')   



'''
# custome metric optimization function
def LGBM_custom_metric(X, Test_name = 'test', num_boost_round = 5000, early_stopping_rounds = 100, lambda_l2 = 0):
    
    X_train = X[X.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    y_train = X[X.date_block_num < 33]['item_cnt_month']
    X_val   = X[X.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    y_val   = X[X.date_block_num == 33]['item_cnt_month']
    X_test  = X[X.date_block_num == 34].drop(['item_cnt_month'], axis=1) 
    
    lgb_train_data = lgb.Dataset(X_train, label=y_train)
    lgb_val_data = lgb.Dataset(X_val, label=y_val)
#_________________________________________________________________________
    
    params = {'objective':        'regression',
              #'metric':           clip_rmse(y_true, y_pred),#'rmse',
              
              'num_leaves':       1417,
              'max_depth':        10,
              'min_data_in_leaf': 469,
              'max_bin':          97,
              
              'lambda_l2':        lambda_l2,
              
              'learning_rate':    0.01,
              'bagging_fraction': 0.5,
              'feature_fraction': 0.5,
              'bagging_freq':     1,
              'bagging_seed':     42,
              'verbosity':        -1,
              'seed':             42}
    
    def rmse_clip(y_hat, data): # define the optimization metric function
        y_train = data.get_label()
        return 'rmse_clip', np.sqrt(np.mean((y_train.clip(0,20)-y_hat.clip(0,20))**2)), False
    
    
    evals_result = {}  # to record eval results for plotting

    model = lgb.train(params,
                      lgb_train_data,
                      valid_sets=[lgb_train_data, lgb_val_data],
                      valid_names=['train','val'],
                      
                      num_boost_round = num_boost_round, 
                      early_stopping_rounds=early_stopping_rounds,
                      feval=rmse_clip,
                      evals_result=evals_result,  
                      verbose_eval=1000)
    
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration).clip(0,20) # Competition evaluation requirement: True target values are clipped into [0,20] range 
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration).clip(0,20)    
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration).clip(0,20)
    
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    RMSE_val = np.sqrt(mean_squared_error(y_val, y_pred_val)) 
   
    print(f'LGBM: RMSE train: {RMSE_train}  RMSE val: {RMSE_val}')
    
#______________________________________________________________________________________    
    # Save the predictions to csv for submission: 
    df = pd.DataFrame(y_pred_test, columns = ['item_cnt_month']).reset_index()
    df.rename(columns = {'index':'ID'}, inplace=True)
    df.to_csv(r'output/submissions/'+Test_name+'.csv', index=False)

#_______________________________________________________________________________________
    # Save test results to Journal
    row = '\n'+str(Test_name)+','+str(RMSE_train)+','+str(RMSE_val)
    with open('Journal.csv','a') as file:
        file.write(row)
        
#_____________________________________________________________________________________
#Feature importance in LGBM can be calculated as split or gain. Here I take harmonic average of the two as a ranking for feature importance
      
    gain  = model.feature_importance('gain')
    gain_fraction = np.round((100 * gain / gain.sum()),1)
    split = model.feature_importance('split')
    split_fraction = np.round((100 * split / split.sum()),1)
    harmonic_mean = np.round(2*split_fraction*gain_fraction/((split_fraction+gain_fraction)), 1)
        
    FI_df = pd.DataFrame({'feature': model.feature_name(),
                          'harmonic_mean': harmonic_mean,
                          'gain': gain_fraction,
                          'split': split_fraction}).sort_values('harmonic_mean', ascending=False)
    FI_df = FI_df.reset_index(drop=True) # set the index column in order with the harmonic_mean order
    
#________________________________________________________________________________________
# Plotting results: Feature Importance and Train/Val Errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,15))
    
    ax1.plot(evals_result['train']['rmse_clip'], 'g', label='train')
    ax1.plot(evals_result['val']['rmse_clip'], 'r', label='val')
    ax1.legend(loc = 'upper right')
    ax1.grid(True)
    
    ax2.barh(FI_df.index, FI_df.harmonic_mean.values, tick_label = FI_df.feature)
    ax2.invert_yaxis()
    ax2.yaxis.tick_right()
    
    plt.show()
    
    return y_pred_train, y_pred_val, y_pred_test 

'''

'''
# Generate polynomial features
def generate_poly_features(X, FI, num_features, n_poly=2):
    
    #Generates polynomial features
    
    #Arguments:
    #X - dataframe with data
    #FI - Feature importance dataframe in assending order. col_name column holds feature names 
    #num_features -- how many top features to select for polynomial features generation
    #n_poly -- order of polynomial to use
    
    #Outputs:
    #X_poly -- dataframe with old and new (polynomial) features
        
    
    # initialize 
    X_poly = None
    
    X = X.fillna(0) # NaN not allowed in polynomial features. fillna does not influence lgbm solution results
    
    # Feature Importance df, drop features with zero importance
    FI=FI[FI.feature_importance_vals>0]
    
    print('initial dataframe shape = ', X[FI['col_name']].shape)
    
    # using threshold value, select most important features for creating polynomial features
    FI_for_poly=FI.iloc[0:num_features]
    
    # select other features
    FI_other = FI.iloc[num_features:]
    
    # select columns from the dataframe with other features for later concatination
    X_other_features = X[FI_other['col_name']]
    # copy target to other features df
    X_other_features['item_cnt_month'] = X['item_cnt_month']       

    # Select only columns for polinomial features
    X = X[FI_for_poly['col_name']]
    print('dataframe for polyfeatures shape = ', X.shape)
        
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(n_poly, interaction_only=True, include_bias = False)
    
    X_poly = pd.DataFrame(poly.fit_transform(X), columns = poly.get_feature_names(X.columns))
    
    X_poly.index = X_other_features.index
    X_poly = pd.concat([X_poly, X_other_features], axis = 1)
    
    print('final dataframe shape = ', X_poly.shape)
      
    return X_poly
'''