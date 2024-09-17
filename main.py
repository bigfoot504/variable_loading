import numpy as np
import pandas as pd
import os
from config.config import config_reader_yaml
from pprint import pprint

rng = np.random.default_rng()


def round_retain_sum(x):
    shape = x.shape
    x = x.flatten()
    N = np.round(np.sum(x)).astype(int)
    y = x.astype(int)
    M = np.sum(y)
    K = N - M
    z = y - x
    if K != 0:
        idx = np.argpartition(z, K)[:K]
        y[idx] += 1
        
    return y.reshape(shape)


def gen_load(wt_rd_factor, size=None, one_rep_max=None):
    '''Generate load(s).

    Uses beta distribution so that random load(s) are between 50% and 100%.
    Beta parameters have been tuned so that 25% of the loads are between
    50-70%, 50% of the loads are between 70-85%, and 5% of the loads are
    greater than 85%.

    Parameters:
    -----------
    wt_rd_factor : int or float
        E.g., if 5, will round randomly generated load to nearest 5lbs.
    size : None, int, or iterable
        Dimensions of random loads to generate.
        If None, load is generate as a float.
        If int, load is generated as a radnom numpy array of that length.
        If iterable, load is generated as a random numpy array of that shape.
    one_rep_max : None, int, or float
        If None, load remains as a percent.
        If int or float, load is generated as a percentage of 1 rep max.

    Returns:
    --------

    '''
    assert type(wt_rd_factor) in [int, float] and (0<wt_rd_factor and wt_rd_factor<=10)
    assert size is None or type(size) in [int, tuple]
    assert one_rep_max is None or type(one_rep_max) in [int, float]

    # Generate load as percent
    load = 0.5 + 0.5 * rng.beta(a=7.09, b=7.39, size=size)
    
    # Case to change load from percent to weight
    if one_rep_max:
        load *= one_rep_max

    if isinstance(load, np.ndarray):
        load = (load / wt_rd_factor).round() * wt_rd_factor
    else:
        load = round(load / wt_rd_factor) * wt_rd_factor

    return load


def get_config_data(is_print=False):
    """
    Get configuration data from the yaml.
    """
    config_dict = config_reader_yaml()

    if is_print:
        for k,v in config_dict.items():
            print(k, '\n', v)
            
    return config_dict


def make_blocks_data(lifts_dict, vol_distr_dict):
    # Populate block-deep cycle data
    for lift, lift_dict in lifts_dict.items():
        # Array containing volume for each block of the cycle for the lift
        lift_cycle_blocks_vol = round_retain_sum(
            vol_distr_dict['Blocks']*lift_dict['Blocks']['Volume']
        )
        
        lifts_dict[lift]['Blocks'].update(
            {
                block_num+1: {
                    'Volume': vol,
                    'Percent': vol / lift_cycle_blocks_vol.sum()
                } for block_num, vol in enumerate(lift_cycle_blocks_vol)
            }
        )
    pprint(lifts_dict); input()


def make_weeks_data(lifts_dict, vol_distr_dict):
    # Populate week-deep block data for each lift for each block

    cycle_len = len(vol_distr_dict['Blocks'])

    for block_num in range(1, cycle_len+1):
        block_distr = rng.permuted(vol_distr_dict['Weeks'])
        # All lifts follow same volume distribution in splitting block volume over the weeks
        for lift, lift_dict in lifts_dict.items():
            # Array containing volume for each week of the block for the lift
            lift_block_weeks_vol = round_retain_sum(
                block_distr * lift_dict['Blocks'][block_num]['Volume']
            )
            lift_dict['Blocks'][block_num]["Weeks"] = {
                week_num + 1: {
                    "Volume": vol,
                    "Percent": vol / lift_block_weeks_vol.sum(),
                } for week_num, vol in enumerate(lift_block_weeks_vol)
            }
    pprint(lifts_dict); input()


def main():  

    config_dict = get_config_data()

    vol_distr_dict = config_dict['Volume_Distributions']
    lifts_dict     = config_dict['Lifts']

    training_days_per_week = max(
        [
            len(lift_dict['Weekly Distribution']) for lift_dict in lifts_dict.values()
        ]
    )

    schedule_df = pd.DataFrame(
        columns = ['Block', 'Week', 'Day', 'Lift', 'Weight', 'Volume'],
    )

    # Populate block-deep cycle data
    make_blocks_data(lifts_dict, vol_distr_dict)

    # Populate week-deep block data for each lift for each block
    make_weeks_data(lifts_dict, vol_distr_dict)

    # Populate day-deep week data for each week for each block for each lift
    for lift, lift_dict in lifts_dict.items():
        for block_num, block_dict in lift_dict['Blocks'].items():
            for week_num, week_dict in block_dict["Weeks"].items():
                week_distr = rng.permuted(lift_dict['Weekly Distribution'])
                lift_week_days_vol = round_retain_sum(week_distr * week_dict['Volume'])
                # In case the lift is not trained on every training day
                day_nums = rng.choice(
                    training_days_per_week,
                    len(lift_dict['Weekly Distribution']),
                    replace = False,
                ) + 1
                week_dict["Days"] = {
                    'Day ' + str(day_num): {
                        'Volume': vol,
                        'Percent': vol / lift_week_days_vol.sum(),
                    } for day_num,vol in zip(day_nums, lift_week_days_vol)
                }
                for day_num, day_dict in week_dict.items():
                    schedule_df = pd.concat((
                        schedule_df,
                        pd.DataFrame(
                            {
                                'Block': block_num,
                                'Week': week_num,
                                'Day': day_num,
                                'Lift': lift,
                                'Weight': gen_load(
                                    wt_rd_factor = config_dict['Weight_rounding_factor'],
                                    one_rep_max = lift_dict['Max']
                                ),
                                'Volume': day_dict['Volume'],
                            },
                            index = [0],
                        )
                    ))
    schedule_df.sort_values(
        ['Block', 'Week', 'Day', 'Lift'],
    ).reset_index(drop=True, inplace=True)

    for block_num in schedule_df['Block'].unique():
        for lift in schedule_df['Lift'].unique():
            block_lift_filter = (schedule_df['Block']==block_num) & (schedule_df['Lift']==lift)
            lift_filter = schedule_df['Lift']==lift

            schedule_df.loc[block_lift_filter, 'Block Pct Vol'] = \
                schedule_df['Volume'][block_lift_filter].sum() / schedule_df['Volume'][lift_filter].sum()

    for block_num in schedule_df['Block'].unique():
        for lift in schedule_df['Lift'].unique():
            block_lift_filter = (schedule_df['Block']==block_num) & (schedule_df['Lift']==lift)
            for week_num in schedule_df['Week'].unique():
                block_week_lift_filter = (schedule_df['Block']==block_num) & (schedule_df['Week']==week_num) & (schedule_df['Lift']==lift)
                schedule_df.loc[block_week_lift_filter, 'Week Pct Vol'] = \
                    schedule_df['Volume'][block_week_lift_filter].sum() / schedule_df['Volume'][block_lift_filter].sum()
                
    for block_num in schedule_df['Block'].unique():
        for week_num in schedule_df['Week'].unique():
            for lift in schedule_df['Lift'].unique():
                block_week_lift_filter = (schedule_df['Block']==block_num) & (schedule_df['Week']==week_num) & (schedule_df['Lift']==lift)
                for day_num in schedule_df['Day'][block_week_lift_filter].unique():
                    block_week_day_lift_filter = (schedule_df['Block']==block_num) & (schedule_df['Week']==week_num) & (schedule_df['Day']==day_num) & (schedule_df['Lift']==lift)
                    schedule_df.loc[block_week_day_lift_filter, 'Day Pct Vol'] = \
                        schedule_df['Volume'][block_week_day_lift_filter].sum() / schedule_df['Volume'][block_week_lift_filter].sum()

    for i, row in schedule_df[['Lift', 'Weight']].iterrows():
        schedule_df.loc[i, 'Pct 1RM'] = row['Weight'] / lifts_dict[row['Lift']]['Max']

    if not os.path.exists(config_dict['Results']['Directory']):
        os.makedirs(config_dict['Results']['Directory'])
    schedule_df.to_csv(os.path.join(config_dict['Results']['Directory'], config_dict['Results']['Filename']))
    


if __name__ == '__main__':
    main()
