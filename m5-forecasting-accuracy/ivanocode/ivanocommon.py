import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

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