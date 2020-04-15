from ivanocode.pipeline.wrmsse import wrmsse_total, with_aggregate_series
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib

raw = 'raw'
processed = 'processed'
submissions = 'submissions'
tmp = 'tmp'

n_total_series = 30490

from joblib import Memory
location = './tmp'
memory = Memory(location, verbose=5)

@memory.cache
def read_series_sample(n_sample_series = 10):
    sample_idx = set(np.random.choice(
        range(1, n_total_series + 1),
        n_sample_series
    ))

    # header
    sample_idx.add(0)

    return pd.read_csv(f'{raw}/sales_train_validation.csv', skiprows = lambda i: i not in sample_idx)

@memory.cache
def melt_sales_series(df_sales_train):
    id_columns = [col for col in df_sales_train.columns if 'id' in col]
    sales_columns = [col for col in df_sales_train.columns if 'd_' in col]
    cat_columns = [col for col in id_columns if col != 'id']

    df_sales_train_melt = df_sales_train.melt(
        id_vars=id_columns,
        var_name='day_id',
        value_name='sales'
    )

    for col in id_columns:
        df_sales_train_melt[col] = df_sales_train_melt[col].astype('category')

    df_sales_train_melt['sales'] = df_sales_train_melt['sales'].astype('int16')

    return df_sales_train_melt

@memory.cache
def extract_day_ids(df_sales_train_melt):
    sales_columns = [f'd_{col}' for col in range(1,1913+1)]
    mapping = {col: int(col.split('_')[1]) for col in sales_columns}
    df_sales_train_melt['day_id'] = df_sales_train_melt['day_id'].map(mapping)

    import datetime
    d_1_date = pd.to_datetime('2011-01-29')
    mapping = {day:(d_1_date + datetime.timedelta(days=day-1)) for day in range(1, 1913+1)}
    df_sales_train_melt['day_date'] = df_sales_train_melt['day_id'].map(mapping)

    mapping = {day:str((d_1_date + datetime.timedelta(days=day-1)).date()) for day in range(1, 1913+1)}
    # gonna need it for joining with calendars & stuff
    df_sales_train_melt['day_date_str'] = df_sales_train_melt['day_id'].map(mapping)

    df_sales_train_melt['day_date_str'] = df_sales_train_melt['day_date_str'].astype('category')
    df_sales_train_melt['day_id'] = df_sales_train_melt['day_id'].astype('int16')
    df_sales_train_melt['month_id'] = df_sales_train_melt['day_date'].dt.month.astype('uint8')

    return df_sales_train_melt

@memory.cache
def join_w_calendar(df_sales_train_melt):
    df_calendar = pd.read_csv(f'{raw}/calendar.csv')
    df_sales_train_melt =  df_sales_train_melt.merge(
        df_calendar[['date', 'wm_yr_wk']],
        left_on='day_date_str', right_on='date',
        validate='many_to_one')

    df_sales_train_melt['wm_yr_wk'] = df_sales_train_melt['wm_yr_wk'].astype('int16')
    return df_sales_train_melt

@memory.cache
def join_w_prices(partition):
    df_prices = pd.read_csv(f'{raw}/sell_prices.csv')
    partition = partition.merge(
        df_prices,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )
    partition['sell_price'] = partition['sell_price'].astype('float32')
    partition['sales_dollars'] = (partition['sales'] * partition['sell_price']).astype('float32')
    partition.fillna({'sales_dollars': 0}, inplace=True)
    return partition

from fastai.tabular import *

def model_as_tabular(df_sales_train_melt):
    df_sample = df_sales_train_melt.query('sales_dollars > 0')

    day_ids = list(sorted(df_sample['day_id'].unique()))
    valid_idx = np.flatnonzero(df_sample['day_id'] > 1000)

    procs = [FillMissing, Categorify, Normalize]
    dep_var = 'sales_dollars'
    cat_names = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'month_id', 'id']
    cols = cat_names + ['sell_price'] + [dep_var]

    path ='./tmp'
    data = TabularDataBunch.from_df(path, df_sample[cols], dep_var, valid_idx=valid_idx,
                                    procs=procs, cat_names=cat_names)

    sales_range = df_sales_train_melt.agg({dep_var: ['min', 'max']})
    learn = tabular_learner(data, layers=[1000,100], emb_szs=None, metrics=rmse, 
                        y_range=sales_range[dep_var].values)
    #learn.lr_find()
    #fig = learn.recorder.plot(return_fig=True)
    #fig.savefig('lr_find.png')
    learn.fit_one_cycle(1, 1e-1)
    fig = learn.recorder.plot_losses(return_fig=True)
    fig.savefig('loss_log.png')

    """
    the above is pretty unstable, still we sort of got it to overfit slightly
    epoch     train_loss  valid_loss  root_mean_squared_error  time    
    0         83.365913   53.201424   7.188121                 00:00                                                                                      
    1         56.507870   55.011505   7.313859                 00:00                                                                                      
    2         48.014706   55.011673   7.313872                 00:00                                                                                      
    3         44.242149   55.001785   7.313148                 00:00                                                                                      
    4         42.347946   57.570885   7.462979                 00:00 
    epoch     train_loss  valid_loss  root_mean_squared_error  time    
    0         99.312691   88.176826   9.289624                 00:00                                                                                      
    1         53.812771   45.480507   6.625162                 00:00                                                                                      
    2         35.418938   19.952007   4.353195                 00:00                                                                                      
    3         23.099392   17.516432   4.008787                 00:00                                                                                      
    4         17.082275   17.641117   4.019526                 00:00 
    """
    return learn, valid_idx

sales_series = read_series_sample()
sales_series = melt_sales_series(sales_series)
sales_series = extract_day_ids(sales_series)
sales_series = join_w_calendar(sales_series)
sales_series = join_w_prices(sales_series)
learn, valid_idx = model_as_tabular(sales_series)
val = sales_series[sales_series.index.isin(valid_idx)]
trn = sales_series[~sales_series.index.isin(valid_idx)]
preds, y = learn.get_preds(DatasetType.Valid)

pred = val.copy()
pred['sales_dollars'] = y
pred['sales'] = pred['sales_dollars'] / pred['sell_price']
pred['sales'].fillna(0, inplace=True)
val_w_aggs = with_aggregate_series(val.copy())
trn_w_aggs = with_aggregate_series(trn.copy())
pred_w_aggs = with_aggregate_series(pred.copy())
score = wrmsse_total(
    trn_w_aggs,
    val_w_aggs,
    pred_w_aggs
)