import numpy as np
import pandas as pd
import os
from config.config import config_reader_yaml
from pprint import pprint

rng = np.random.default_rng(123)


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

    load = 0.5 + 0.5 * rng.beta(a=7.09, b=7.39, size=size)
    if one_rep_max:
        load *= one_rep_max

    if isinstance(load, np.ndarray):
        load = (load / wt_rd_factor).round() * wt_rd_factor
    else:
        load = round(load / wt_rd_factor) * wt_rd_factor

    return load


def get_config_data(is_print=False):

    config_dict = config_reader_yaml()

    if is_print:
        for k,v in config_dict.items():
            print(k, '\n', v)
            
    return config_dict


def main():  

    config_dict = get_config_data()

    vol_distr_dict = config_dict['Volume_Distributions']
    lifts_dict     = config_dict['Lifts']
    cycle_len = len(vol_distr_dict['Cycle'])

    training_days_per_week = max([len(lift_dict['Weekly Distribution']) for lift_dict in lifts_dict.values()])

    schedule_df = pd.DataFrame(columns=['Block', 'Week', 'Day', 'Lift', 'Weight', 'Volume'])

    # Populate block-deep cycle data
    for lift, lift_dict in lifts_dict.items():
        # Array containing volume for each block of the cycle for the lift
        lift_cycle_blocks_vol = round_retain_sum(vol_distr_dict['Cycle']*lift_dict['Cycle']['Volume'])
        
        lifts_dict[lift]['Cycle'].update({'Block '+str(block_num+1):{'Volume':vol,
                                                                     'Percent':vol/lift_cycle_blocks_vol.sum()} for block_num,vol in enumerate(lift_cycle_blocks_vol)})
    # pprint(lifts_dict); input()

    # Populate week-deep block data for each lift for each block
    for block_num in range(1, cycle_len+1):
        block_distr = rng.permuted(vol_distr_dict['Block'])
        # All lifts follow same volume distribution in splitting block volume over the weeks
        for lift, lift_dict in lifts_dict.items():
            # Array containing volume for each week of the block for the lift
            lift_block_weeks_vol = round_retain_sum(block_distr*lift_dict['Cycle']['Block '+str(block_num)]['Volume'])
            lifts_dict[lift]['Cycle']['Block '+str(block_num)].update({'Week '+str(week_num+1):{'Volume':vol,
                                                                                                'Percent':vol/lift_block_weeks_vol.sum()} for week_num,vol in enumerate(lift_block_weeks_vol)})
    # pprint(lifts_dict); input()

    # Populate day-deep week data for each week for each block for each lift
    for lift, lift_dict in lifts_dict.items():
        for block_name, block_dict in lift_dict['Cycle'].items():
            if not block_name.startswith('Block'): continue
            for week_name, week_dict in block_dict.items():
                if not week_name.startswith('Week'): continue
                week_distr = rng.permuted(lift_dict['Weekly Distribution'])
                lift_week_days_vol = round_retain_sum(week_distr*week_dict['Volume'])
                # In case the lift is not trained on every training day
                day_nums = rng.choice(training_days_per_week, len(lift_dict['Weekly Distribution']), replace=False)+1
                lifts_dict[lift]['Cycle'][block_name][week_name].update({'Day '+str(day_num):{'Volume':vol,
                                                                                                'Percent':vol/lift_week_days_vol.sum()} for day_num,vol in zip(day_nums, lift_week_days_vol)})
                for day_name, day_dict in lifts_dict[lift]['Cycle'][block_name][week_name].items():
                    if not day_name.startswith('Day'): continue
                    schedule_df = pd.concat((schedule_df,
                                             pd.DataFrame({'Block': int(block_name.split()[1]),
                                                           'Week': int(week_name.split()[1]),
                                                           'Day': int(day_name.split()[1]),
                                                           'Lift': lift,
                                                           'Weight': gen_load(wt_rd_factor=config_dict['Weight_rounding_factor'],
                                                                              one_rep_max=lift_dict['Max']),
                                                           'Volume': day_dict['Volume'],
                                                           }, index=[0])))
    schedule_df = schedule_df.sort_values(['Block', 'Week', 'Day', 'Lift']).reset_index(drop=True)

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