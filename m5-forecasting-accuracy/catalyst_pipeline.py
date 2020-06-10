from ivanocode.pipeline.pipeline import timeit, timings, read_series_sample, melt_sales_series, extract_day_ids, join_w_calendar, join_w_prices, reproducibility_mode, out_dir, raw, LOG, n_sample_series, get_submission_template_melt, force_data_prep

import numpy as np
from petastorm import make_batch_reader, TransformSpec
from petastorm.pytorch import DataLoader as PetaDataLoader
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader, IterableDataset
from sklearn import preprocessing
from collections import OrderedDict
from torch import tensor
import math
import os
import re

import torch
from torch import nn
import torch.nn.functional as F

from catalyst.dl import SupervisedRunner
from catalyst.utils import set_global_seed

FILE_PREFIX = 'file:'
processed = f'{out_dir}/processed'

pre_open_fds = None
def patch_leaking_fd():
    global pre_open_fds
    from pyarrow.parquet import ParquetFile, ParquetReader
    def _patched_init(self, source, **kwargs):
        self.source = source
        return ParquetFile.__old_init__(self, source, **kwargs)

    def _exit(self, *args, **kwargs):
        if hasattr(self.source, 'close'):
            self.source.close()
            del self.source

    def _bopen(fn):    
        return open(fn, 'rb')

    pre_open_fds = _bopen
    if not hasattr(ParquetFile, '__old_init__'):
        print("Patching")
        ParquetFile.__old_init__ = ParquetFile.__init__

        ParquetFile.__init__ = _patched_init
        ParquetFile.__exit__ = _exit
        ParquetFile.__del__ = _exit

    else:
        print("Already patched")

patch_leaking_fd()



class MyIterableDataset(IterableDataset):
    def __init__(self, filename, rex=None):
        super(MyIterableDataset).__init__()
        self._filename_param = filename
        self.filename = self._init_filenames(filename, rex)

    def _init_filenames(self, filename, rex):
        if rex is None:
            return filename
        
        filename = filename[len(FILE_PREFIX):]
        if not os.path.isdir(filename):
            raise ValueError(f"Filtering only possible for dirs, {filename} is not a one")
        paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(filename) for f in fn]
        res = list(map(
            lambda f: FILE_PREFIX + f,
            filter(lambda f: re.match(rex, f) is not None, paths)
        ))
        if (len(res) == 0):
            raise ValueError(f"0 files remained out ot {len(paths)} - seems regex is too restrictive")

        return res;

    def _init_petaloader(self):
        def _transform_row(df_batch):
            return df_batch

        transform = TransformSpec(_transform_row, removed_fields=['cat_id', 'store_id', 'state_id'])
        reader = make_batch_reader(self.filename,
                 schema_fields=['id', 'item_id', 'dept_id', 'cat_id', 'day_id',
               'sales', 'day_date_str', 'month_id', 'date', 'wm_yr_wk',
               'snap_flag', 'sell_price', 'sales_dollars', 'store_id', 'state_id'],
                workers_count=1
                #,transform_spec = transform
        )
        return PetaDataLoader(reader=reader, batch_size=128, shuffling_queue_capacity=100000)
        
    def __len__(self):
        return 1913*30490 # can be arbitrary large value to prevent WARN logs, seem to be ignored anyway

    def __iter__(self):
        print(f"Iterator created on {self._filename_param}")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            count_cells = 0
            count_batches = 0
            with self._init_petaloader() as loader:
                if pre_open_fds:
                    loader.reader.dataset.fs.open = pre_open_fds
                for batch in loader:
                    count_batches += 1
                    # TODO: propagate petaloader's batches without breaking them into individual items
                    for idx in range(len(batch['sell_price'])):
                        price         = batch['sell_price'][idx]
                        sales_dollars = batch['sales_dollars'][idx] if ('sales_dollars' in batch) else -1.
                        price_is_nan = math.isnan(price)
                        # TODO: this starts to look like feature extraction, doesn't belong here
                        price_or_zero = 0. if price_is_nan else price
                        count_cells += 1
                        # float32 needed for pytorch downstream
                        yield {'features': tensor([price_or_zero, price_is_nan], dtype=torch.float32),
                               'targets': tensor([sales_dollars])}
                        
            print(f'Done iterating: {count_batches} batches / ({count_cells} cells) ')
        else:
            raise ValueError("Not implemented for multithreading")

@timeit(log_time = timings)
def to_parquet(sales_series, name):
    encoders = {}
    sales_series['parquet_partition'] = np.random.randint(1, 300, sales_series.shape[0]).astype('uint8')
    from sklearn.preprocessing import LabelEncoder
    if 'day_date' in sales_series.columns:
        sales_series.drop(['day_date'], axis=1, inplace=True)

    for col in sales_series.columns:
        if col in encoders:
            print(f'{col} already encoded - skipping')
            continue

    # petastorm can't read these
    if str(sales_series[col].dtype) == 'uint8':
        sales_series[col] = sales_series[col].astype('int')

    if str(sales_series[col].dtype) in ['category', 'object']:
        print(f'{col} - encoding')
        enc = LabelEncoder()
        sales_series[col] = enc.fit_transform(sales_series[col])
        encoders[col] = enc

    for name, enc in encoders.items():
        np.save(f'{processed}/{name}.npy', enc.classes_)

    # TODO: uint -> int, category/object -> int, day_date -> drop
    sales_series.to_parquet(
        f'{processed}/{name}',
        index=False,
        partition_cols=['parquet_partition'],
        row_group_size=1000
    )

@timeit(log_time = timings)
def load_encoders():
    processed = 'processed'
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    def _load(fn):
        l = LabelEncoder()
        l.classes_ = np.load(f'{processed}/{fn}', allow_pickle=True)
        return l

    encoders_paths = filter(lambda p: p.endswith('.npy'), os.listdir(processed))
    encoders = {fn[:-len('.npy')]:_load(fn) for fn in encoders_paths}

    return encoders

@timeit(log_time = timings)
def encode(me):
    encoders = load_encoders()
    continuous_cols = ['sell_price']

    for col in me.columns:
        dtype_str = str(me[col].dtype)
        if col in continuous_cols:
            LOG.debug(f"Encoding {col} ({dtype_str}) as float32 just in case for pytorch")
            me[col] = me[col].astype('float32')
            continue

        LOG.debug(f"Encoding {col} ({dtype_str}) as categorical ")

        unlabelable = ~me[col].isin(encoders[col].classes_)
        unlabelable_count = unlabelable.sum()
        if unlabelable_count > 0:
            default_label = encoders[col].classes_[0]
            LOG.warning(f"{unlabelable_count} entries for {col} can't be labeled. Defaulting to {default_label} e.g.\n {me[unlabelable][col][:3].values}")
            me.loc[unlabelable, col] = default_label

        me[col] = encoders[col].transform(me[col])

    return me

@timeit(log_time = timings)
def prepare_data():
    expected_path = f'{processed}/sales_series_melt.parquet'
    if os.path.exists(expected_path) and not force_data_prep:
        LOG.info(f'Found parquet file ({expected_path})- skipping the prep')
        return

    LOG.info(f'Not found parquet file ({expected_path}) - preparing the data')

    sales_series = read_series_sample(n_sample_series)
    sales_series = melt_sales_series(sales_series)
    sales_series = extract_day_ids(sales_series)
    sales_series = join_w_calendar(sales_series)
    sales_series = join_w_prices(sales_series)
    to_parquet(sales_series, 'sales_series_melt.parquet')

def prepare_test_data():
    expected_path = f'{processed}/test_series_melt.parquet'
    if os.path.exists(expected_path) and not force_data_prep:
        LOG.info(f'Found parquet file ({expected_path})- skipping the prep')
        return

    test_data = get_submission_template_melt()
    test_data = encode(test_data)
    to_parquet(test_data, 'test_series_melt.parquet')

class Net(nn.Sequential):
    def __init__(self, num_features):
        layers = []
        layer_dims = [num_features, 200,200,20,20,1]
        for in_features, out_features in zip(layer_dims[:-1], layer_dims[1:]):
            l = nn.Linear(in_features, out_features)
            # Note to self: loss @ init is quite important!
            torch.nn.init.xavier_uniform_(l.weight) 
            torch.nn.init.zeros_(l.bias)

            layers.append(l)
            layers.append(nn.ReLU())
        super(Net, self).__init__(*layers)

class MyLoss(nn.MSELoss):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, inp, target):
        return super().forward(inp, target)

def do_train():
    batch = 128

    train_ds = MyIterableDataset(f'file:{processed}/sales_series_melt.parquet/parquet_partition=2')
    valid_ds = MyIterableDataset(f'file:{processed}/sales_series_melt.parquet/parquet_partition=1')
    test_ds  = MyIterableDataset(f'file:{processed}/test_series_melt.parquet')

    train_dl = TorchDataLoader(train_ds, batch_size=batch, shuffle=False, num_workers=0, drop_last=False)
    valid_dl = TorchDataLoader(valid_ds, batch_size=batch, shuffle=False, num_workers=0, drop_last=False)
    test_dl  = TorchDataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=0, drop_last=False)

    data = OrderedDict()
    data["train"] = train_dl
    data["valid"] = valid_dl
    data["test"]  = test_dl

    model = Net(num_features = 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = MyLoss()
    runner = SupervisedRunner()

    for key, dl in data.items():
        batch = next(iter(dl))
        LOG.debug(f"{key} model out @ init {model.forward(batch['features']).transpose(1, 0)}")

    LOG.debug("Starting training")
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=data,
        logdir="run",
        load_best_on_end=True,
        num_epochs=1)

    for key, dl in data.items():
        batch = next(iter(dl))
        LOG.debug(f"{key} model out @ exit {model.forward(batch['features']).transpose(1, 0)}")

reproducibility_mode()
prepare_test_data()
prepare_data()
do_train()
