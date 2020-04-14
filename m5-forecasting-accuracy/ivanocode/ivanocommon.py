import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from IPython.display import display

def overlay_x_axis(fig, ax, data):
    from pandas.core.series import Series
    if (type(data) == Series):
        data = data.values

    newax = ax.twiny()
    fig.subplots_adjust(bottom=0.10)
    newax.set_frame_on(True)
    newax.patch.set_visible(False)
    newax.set_xlim(data[0], data[-1])
    newax.xaxis.set_ticks_position('bottom')
    newax.xaxis.set_label_position('bottom')
    newax.spines['bottom'].set_position(('outward', 20))

def plot_item_series(df, id):
    subset = df.query(f'id == "{id}"')
    sales = subset['sales'].values
    dates = [dt.date() for dt in subset['day_date'].dt.to_pydatetime()]
    xaxis_data = dates

    fig, ax = plt.subplots(figsize=(20,5))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    # main series
    ax.plot(xaxis_data, sales)
    overlay_x_axis(fig, ax, subset['day_id'])

    # avg
    daily_avg = subset['daily_avg_count'].unique()[0]
    # Note to self: mixing dates & ints as x for different series of the same plot cost me some debug
    ax.hlines(y=daily_avg, xmin=xaxis_data[0], xmax=xaxis_data[-1], colors='r', linestyles='--', lw=2)

    # prices
    prices = None
    prices_legend = []
    if 'sell_price' in subset:
        prices = subset['sell_price'].values
        prices_legend = ['prices']
        ax.plot(xaxis_data, prices, 'k-')

    # stats & legend
    stats = f"daily_avg: {subset[0:1]['daily_avg_count'].unique()}\n"+\
            f"month_avg: {subset[0:1]['monthly_avg_count'].unique()}\n"+\
            f"sum: {sales.sum()}"
    ax.legend(['series']+
              prices_legend+
               [stats])

def drop_level(df, level, axis, inplace=False):
    if axis == 1:
        assert hasattr(df.columns, 'levels') and len(df.columns.levels) > level
    if axis == 0:
        raise ValueError("Not supported")

    if not inplace:
        df = df.copy()

    def remove_empty(cols):
        return filter(lambda col: len(col) >0, cols)

    df.columns = ['_'.join(remove_empty(col)) for col in df.columns]
    return df

# Note to self: dense rank looks like an easier way to do this
def count_flips(col):
    """Counts flips between na==zero and regular numbers"""
    return (col
     .fillna(0)
     .clip(0,1)
     .diff()
     .fillna(0)
     .astype(int)
     .sum())

from IPython.display import display
def no_op(*args, **kwargs):
    pass

def display_log(name, val):
    display(val)

def capture_log_into(result_dict):
    def _capture(name, val):
        result_dict[name] = val.copy()
    return _capture

def jupyter_memory_report():
    import sys  

    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    report = sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)

    return report