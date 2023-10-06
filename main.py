import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

rng = np.random.default_rng(1234)


def gen_volume_permutation(vol=None):
    # Error check
    if vol is not None:
        assert isinstance(vol, (int, float)) and vol > 0
        if isinstance(vol, float):
            vol = int(float)
    
    # Move to config later
    vol_distr = np.array([0.15, 0.20, 0.30, 0.35])

    rng.shuffle(vol_distr)

    if vol:
        # Convert distribution to integers summing to vol
        vol_distr = round_retain_sum(vol_distr * vol)
    
    return vol_distr


def round_retain_sum(x):
    N = np.round(np.sum(x)).astype(int)
    y = x.astype(int)
    M = np.sum(y)
    K = N - M
    z = y - x
    if K != 0:
        idx = np.argpartition(z, K)[:K]
        y[idx] += 1
        
    return y


def gen_load(num=None, one_rep_max=None):
    if num:
        assert isinstance(num, int)
        if num == 1: num = None
    if one_rep_max is not None:
        assert isinstance(one_rep_max, (int, float)) and one_rep_max > 0

    rd_wt_factor = 5
    load = 0.5 + 0.5 * rng.beta(a=7.09, b=7.39, size=num)
    if one_rep_max:
        load *= one_rep_max

    return (load / rd_wt_factor).round() * rd_wt_factor


def main():
    days_ls = ['1   Monday',
               '2  Tuesday',
               '3 Thursday',
               '4   Friday']
    lifts_ls = ['Squat', 'Bench', 'Deadlift', 'Press']
    lift_vol_dict = {
        'Squat': 100,
        'Bench': 150,
        'Deadlift': 50,
        'Press': 150,
    }
    lift_max_dict = {
            'Squat': 450,
            'Bench': 340,
            'Deadlift': 550,
            'Press': 180,
    }
    schedule_df = pd.DataFrame(columns=['Day', 'Lift', 'Weight', 'Volume'])
    for lift in lifts_ls:
        vol_distr = gen_volume_permutation(lift_vol_dict[lift])
        loads = gen_load(num=len(vol_distr), one_rep_max=lift_max_dict[lift])
        for day, load, vol in zip(days_ls, loads, vol_distr):
            schedule_df = pd.concat([schedule_df,
                                     pd.DataFrame({
                                         'Day': day,
                                         'Lift': lift,
                                         'Weight': load,
                                         'Volume': vol
                                     }, index=[0])])
    schedule_df.sort_values('Day', inplace=True)
    print(schedule_df)
    


if __name__ == '__main__':
    main()