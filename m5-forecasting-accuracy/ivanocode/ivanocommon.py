import matplotlib.pyplot as plt

def plot_item_series(df, itemid):
    subset = df.query(f'id == "{itemid}"')
    sales = subset['sales'].values
    days = subset['day_id'].values
    stats = f"daily_avg: {subset[0:1]['daily_avg_count'].unique()}\n"+\
            f"month_avg: {subset[0:1]['monthly_avg_count'].unique()}\n"+\
            f"sum: {subset['sales'].sum()}"

    daily_avg = subset['daily_avg_count'].unique()[0]
    fig, ax = plt.subplots(figsize=(20,5))
    ax.hlines(y=daily_avg, xmin=days[0], xmax=days[-1], colors='r', linestyles='--', lw=2)
    ax.plot(days, sales)
    ax.legend(['series', 
               stats])