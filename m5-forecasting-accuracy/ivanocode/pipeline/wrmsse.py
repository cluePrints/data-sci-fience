aggregation_levels = {
    11: ['item_id', 'state_id'],
    10: ['item_id'],
    9:  ['store_id', 'dept_id'],
    8:  ['store_id', 'cat_id'],
    7:  ['state_id', 'dept_id'],
    6:  ['state_id', 'cat_id'],
    5:  ['dept_id'],
    4:  ['cat_id'],
    3:  ['store_id'],
    2:  ['state_id'],
    1:  [],
}

def no_op(*args, **kwargs):
    pass

def display_log(name, val):
    display(val)

def capture_log_into(result_dict):
    def _capture(name, val):
        result_dict[name] = val.copy()
    return _capture

def add_group_id_col(df, grp_id, grp_fields):
    df['id'] = str(grp_id)
    if len(grp_fields) > 0:
        # Note to self: categoricals require cast
        for col in grp_fields:
            df['id'] = df['id'].str.cat(df[col], sep=':')

def with_aggregate_series(df, agg_levels=aggregation_levels):
    result = [df]
    df['agg_level'] = 12
    # Working around categories grouping: https://stackoverflow.com/questions/48471648/pandas-groupby-with-categories-with-redundant-nan
    OBSERVED = True
    for grp_id, grp_fields in agg_levels.items():
        grp_11 = (df
                  .groupby(grp_fields + ['day_id'], as_index=False, observed=OBSERVED)
                  .agg({'sales': 'sum', 'sales_dollars': 'sum'})
        )
        grp_11['agg_level'] = grp_id
        add_group_id_col(grp_11, grp_id, grp_fields)

        result.append(grp_11)

    df = pd.concat(result, sort=False)
    df['sales_delta_sum'] = (df
              .groupby(['id'], as_index=False)
              ['sales']
              .transform(lambda x: x.diff().abs().sum())
    )
    # Note to self: ordering dependencies are a beast! Wouldn't it be fun to prevent cell & statement reordering unless it's safe & consistent?
    df['day_id_rel'] = df['day_id'] - df['day_id'].min()

    return df

def wrmsse_total(df_train, df_valid_w_aggs, df_pred_w_aggs, display=no_op):
    cols = ['id', 'day_id_rel', 'sales', 'sales_$', 'agg_level']
    t = (df_valid_w_aggs[cols]
            .merge(df_pred_w_aggs[cols], 
                   on=['id', 'day_id_rel'], 
                   suffixes=('_valid', '_pred')))
    t['daily_sales_err^2'] = (t['sales_valid'] - t['sales_pred']).pow(2)
    display("t1", t)

    t = t.groupby(['id', 'agg_level_valid'], as_index=False).agg({
        'daily_sales_err^2': lambda x: x.sum(),
        'sales_$_valid': 'sum'
    }).rename({
        'daily_sales_err^2': 'sales_err^2_sum'
    }, axis=1)
    display("t2", t)
    t['agg_weight'] = (t.groupby('agg_level_valid')
        ['sales_$_valid']
       .transform(lambda x: x.sum())
    )
    t['series_weight'] = t['sales_$_valid']/t['agg_weight']

    diff_sum_squared = lambda x: x.diff().pow(2).sum()
    diff_sum_squared.__name__ = 'diff_sum_squared'
    sales_diff_summs = (df_train
        .groupby(['id'], as_index=False)
        .agg({
            'sales': [
                diff_sum_squared,
                'count'],
        })
        .pipe(
            lambda df: drop_level(df, level=1, axis=1)
        ).rename({
            'sales_count': 'trn_sales_count',
            'sales_diff_sum_squared': 'trn_diff_sum_squared'
        }, axis=1)
    )
    display("sales_diff_summs", sales_diff_summs)

    # TODO: this sort of ignores aggs unless they're there
    t = t.merge(sales_diff_summs, on='id', how='left', validate='one_to_one')
    display("t3", t)
    
    h = df_valid_w_aggs['day_id_rel'].nunique()
    n_aggs = t['agg_level_valid'].nunique()
    t['rmsse'] = ((
                    (t['trn_sales_count']-1.0) * t['sales_err^2_sum'])/
                    (h * t['trn_diff_sum_squared'])
                 ).pow(0.5)
    display("t4", t)

    # TODO: rmsse est for series with no sales - will not be a problem when coming from trn
    # TODO: failing fast might also be a better idea than hiding inconsistent test data 
    t.loc[t['sales_$_valid'] == 0, 'rmsse'] = 0
    t['wrmsse'] = t['rmsse']*t['series_weight']

    wrmsse_total = t['wrmsse'].sum()/n_aggs
    return wrmsse_total