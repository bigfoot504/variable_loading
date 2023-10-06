import numpy as np
import pandas as pd
from config.config import config_reader_yaml

rng = np.random.default_rng(12345)


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

    return (load / wt_rd_factor).round() * wt_rd_factor


def get_config_data(is_print=False):

    config_dict = config_reader_yaml()
    config_dict['lifts_df'] = pd.DataFrame.from_dict(config_dict['Lifts'], orient='index').rename_axis('Lifts')
    config_dict.pop('Lifts')

    if is_print:
        for k,v in config_dict.items():
            print(k, '\n', v)
            
    return config_dict


def main():  

    config_dict = get_config_data()

    vol_distr_dict = config_dict['Volume_Distributions']
    lifts_df       = config_dict['lifts_df']

    cycle_distr = vol_distr_dict['Cycle'].reshape((-1,  1,  1))
    block_distr = vol_distr_dict['Block'].reshape(( 1, -1,  1))
    week_distr  = vol_distr_dict['Week' ].reshape(( 1,  1, -1))

    schedule_df = pd.DataFrame(columns=['Block', 'Week', 'Day', 'Lift', 'Weight', 'Volume'])

    for lift, row in lifts_df.iterrows():
        lift_vol = round_retain_sum(cycle_distr * block_distr * week_distr * \
                                    row['Cycle Volume'])

        for b in range(lift_vol.shape[0]):
            rng.shuffle(lift_vol[b,:,:])
        lift_vol = rng.permuted(lift_vol, axis=2)

        loads = gen_load(config_dict['Weight_rounding_factor'], size=lift_vol.size, one_rep_max=row['Max'])

        bwd = []
        for i in range(lift_vol.shape[0]):
            for j in range(lift_vol.shape[1]):
                for k in range(lift_vol.shape[2]):
                    bwd.append([i+1,j+1,k+1])
        bwd = np.array(bwd)

        schedule_df = pd.concat([schedule_df, pd.DataFrame(data=np.hstack((bwd,
                                                                           np.tile(np.array(lift), (lift_vol.size,1)),
                                                                           loads.reshape(-1,1),
                                                                           lift_vol.reshape((-1,1)))),
                                                           columns = schedule_df.columns)])

    schedule_df = schedule_df.sort_values(['Block', 'Week', 'Day', 'Lift']).reset_index(drop=True)
    print(schedule_df.head(50))
    schedule_df.to_csv('data/schedule.csv')
    


if __name__ == '__main__':
    main()