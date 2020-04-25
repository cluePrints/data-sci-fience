from ivanocode.pipeline.wrmsse import wrmsse_total, with_aggregate_series
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib
import os

raw = os.environ.get('DATA_RAW_DIR', 'raw')
processed = 'processed'
submissions = 'submissions'
tmp_dir = './tmp'

n_days_total = 1913
n_total_series = 30490
trn_days = int(os.environ.get('N_TRAIN_DAYS', '1900'))
n_sample_series = int(os.environ.get('N_TRAIN_SAMPLE_SERIES', '150'))

from joblib import Memory
joblib_location  = os.environ.get('JOBLIB_CACHE_DIR', './tmp')
joblib_verbosity = int(os.environ.get('JOBLIB_VERBOSITY', '1'))
memory = Memory(joblib_location, verbose=joblib_verbosity)

do_submit        = bool(os.environ.get('PREPARE_SUBMIT', '').lower() == 'true')

dataloader_num_workers = None

def reproducibility_mode():
    global dataloader_num_workers
    seed = 42

    print(f"Reproducibility mode ON (seed: {seed})")
    dataloader_num_workers = 0

    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)

@memory.cache
def read_series_sample(n = 10):
    indexes = np.arange(1, n_total_series + 1)
    np.random.shuffle(indexes)
    sample_idx = set(indexes[:n])

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
    sales_columns = [f'd_{col}' for col in range(1, n_days_total+1)]
    mapping = {col: int(col.split('_')[1]) for col in sales_columns}
    df_sales_train_melt['day_id'] = df_sales_train_melt['day_id'].map(mapping)

    import datetime
    d_1_date = pd.to_datetime('2011-01-29')
    mapping = {day:(d_1_date + datetime.timedelta(days=day-1)) for day in range(1, n_days_total+1)}
    df_sales_train_melt['day_date'] = df_sales_train_melt['day_id'].map(mapping)

    mapping = {day:str((d_1_date + datetime.timedelta(days=day-1)).date()) for day in range(1, n_days_total+1)}
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
from IPython.display import display

def _transform_target(preds, df):
    y_pred = preds.numpy().flatten()
    df['sales_dollars'] = y_pred
    df['sales'] = df['sales_dollars'] / df['sell_price']
    # TODO: rounding differently might be important for sporadic sales on lower-volume items
    df['sales'].fillna(0, inplace=True)
    df['sales'] = df['sales'].astype('int')

def _wrmsse(preds, val, trn):
    pred = val.copy()
    _transform_target(preds, df=pred)

    val_w_aggs = with_aggregate_series(val.copy())
    trn_w_aggs = with_aggregate_series(trn.copy())
    pred_w_aggs = with_aggregate_series(pred.copy())
    score = wrmsse_total(
        trn_w_aggs,
        val_w_aggs,
        pred_w_aggs
    )
    return score

from fastai.callbacks import *
# mega workaroundish way of plugging a metric func designed to work with all the data to run after each epoch 
# fast.ai is designed around per batch metrics + aggregating them
class MyMetrics(LearnerCallback):
    # should run before the recorder
    _order = -30
    def __init__(self, learn, trn, val):
        super().__init__(learn)
        self.trn = trn
        self.val = val
        self.learn = learn

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['wrmsse'])
        pass
    
    def on_epoch_end(self, last_metrics, **kwargs):
        # TODO: collect progress for a single val batch and make it available for display
        # display(pd.DataFrame({'y_pred': y, 'y_val': val['sales_dollars']}).transpose())
        rec = self.learn.recorder
        preds, y_val = self.learn.get_preds(DatasetType.Valid)
        self.learn.recorder = rec
        score = _wrmsse(preds, self.val, self.trn)
        return {'last_metrics': last_metrics + [score]}
    
class SingleBatchProgressTracker(LearnerCallback):
    # should run before the recorder
    _order = -31
    def __init__(self, learn, metric, metric_name):
        self.learn = learn
        self.batch = learn.data.one_batch(DatasetType.Train)
        self.x = self.batch[0]
        self.y_true = self.batch[1]
        self.metric = metric
        self.metric_name = metric_name
        
    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names([self.metric_name])
        pass
        
    def on_epoch_end(self, last_metrics, **kwargs):
        y_model = self.learn.model.forward(*self.x)
        score = self.metric(self.y_true, y_model)
        return {'last_metrics': last_metrics + [score]}

def model_as_tabular(df_sales_train_melt, lr_find=False):
    non_zero = df_sales_train_melt.query('sales_dollars > 0').reset_index(drop=True)
    valid_idx = np.flatnonzero(non_zero['day_id'] > trn_days)

    # TODO: this in fact is fragile and won't work without reset index above, can I do this differently?
    val_mask = non_zero.index.isin(valid_idx)
    val = non_zero[val_mask]
    trn = non_zero[~val_mask]
    print(f"sample: {len(df_sales_train_melt)} non_zero: {len(non_zero)} trn: {len(trn)} val: {len(val)}")

    my_metrics_cb = partial(MyMetrics, val=val, trn=trn)
    single_batch_metric = partial(SingleBatchProgressTracker, metric=rmse, metric_name='example_batch_rmse')

    procs = [FillMissing, Categorify, Normalize]
    dep_var = 'sales_dollars'
    cat_names = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'month_id', 'id']
    cols = cat_names + ['sell_price'] + [dep_var]

    path = tmp_dir
    data = TabularDataBunch.from_df(path, non_zero[cols], dep_var, valid_idx=valid_idx,
                                    bs=256,
                                    num_workers=dataloader_num_workers,
                                    procs=procs, cat_names=cat_names)

    sales_range = df_sales_train_melt.agg({dep_var: ['min', 'max']})
    learn = tabular_learner(data, layers=[200,200,20,20], emb_szs=None, metrics=[rmse], 
                        y_range=sales_range[dep_var].values, callback_fns=[
                            my_metrics_cb, 
                            single_batch_metric],
                        use_bn=True,
                        wd=0)
    # Note to self: default wd seem to big - results converged to basically nothing in the first ep
    if lr_find:
        learn.lr_find()
        fig = learn.recorder.plot(return_fig=True)
        fig.savefig('lr_find.png')
        # TODO: make this compatible with paperspace !open lr_find.png
    learn.fit_one_cycle(3, 1e-1)
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
    return learn, trn, val

def extract_id_columns(t):
    extracted = t['id'].str.extract('([A-Z]+)_(\\d)_(\\d{3})_([A-Z]{2})_(\d)')
    t['cat_id'] = extracted[0]
    t['dept_id'] = t['cat_id'] + '_' + extracted[1]
    t['item_id'] = t['cat_id'] + '_' + extracted[2]
    t['state_id'] = extracted[3]
    t['store_id'] = t['state_id'] + '_' + extracted[4]
    return t

@memory.cache
def get_submission_template_melt(df_sales_train_melt):
    df_sample_submission = pd.read_csv('raw/sample_submission.csv')
    df_sample_submission.head()

    from datetime import timedelta
    d_1_date = pd.to_datetime(df_sales_train_melt['day_date'].max())
    # TODO: for evaluation rows these dates should be one month later
    mapping = {f'F{day}':(d_1_date + timedelta(days=day)).date() for day in range(1,29)}
    mapping['id'] = 'id'
    df_sample_submission.columns = df_sample_submission.columns.map(mapping)
    df_sample_submission_melt = df_sample_submission.melt(id_vars='id', var_name='day', value_name='sales')

    last_prices = df_sales_train_melt[['id', 'sell_price']].groupby('id').tail(1)
    last_prices.head(1)

    df_sample_submission_melt = df_sample_submission_melt.merge(
        last_prices, on='id', how='left', validate='many_to_one')

    df_sample_submission_melt = extract_id_columns(df_sample_submission_melt)
    return df_sample_submission_melt

def to_submission(learn, df_sample_submission_melt):
    from datetime import timedelta
    submission = df_sample_submission_melt.pivot(index='id', columns='day', values='sales').reset_index()
    
    learn.data.add_test(TabularList.from_df(df=df_sample_submission_melt, 
                                        cat_names=learn.data.cat_names, 
                                        cont_names=learn.data.train_ds.x.cont_names,
                                        processor = learn.data.train_ds.x.processor))
    preds, _ = self.learn.get_preds(DatasetType.Valid)
    _transform_target(preds, df=submission)

    d_1_date = pd.to_datetime(submission.columns[1])
    mapping = {(d_1_date + timedelta(days=day-1)).date():f'F{day}' for day in range(1,29)}
    mapping['id'] = 'id'
    
    submission.columns = submission.columns.map(mapping)
    
    return submission

# TODO: all of this preprocessing ought to happen once and than sampling can use that final dataframe
reproducibility_mode()
sales_series = read_series_sample(n_sample_series)
sales_series = melt_sales_series(sales_series)
sales_series = extract_day_ids(sales_series)
sales_series = join_w_calendar(sales_series)
sales_series = join_w_prices(sales_series)
submission_template = get_submission_template_melt(
    sales_series[['id', 'sell_price', 'day_date']]
)
learn, trn, val = model_as_tabular(sales_series)
if do_submit:
    submission = to_submission(learn, submission_template)
    submission.to_csv(f'{tmp_dir}/0500-fastai-pipeline.csv', index=False)