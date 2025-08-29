"""
Created 5/12/2025.
This is a new module aimed at using the variable-loading principles to generate a training program. It hinges upon varying the volume by about 20% per day among days of the week, by about 20% per week for weeks of the block (month), and by about 10% per block for blocks of the macrocycle.
New in this version is that it is object-oriented and hence more flexible.

To-Do: A future version will use the VariableLoading class to generate one program for all of the lifts at one time and to print the comprehensive program.
"""

from datetime import datetime
from tabulate import tabulate
from pathlib import Path

import numpy as np


class Lift:
    def __init__(
        self,
        name: str,
        _max: float,
        weekly_max_incr: float,
        total_volume: int,
        num_days_per_week: int=4,
        num_weeks_per_block: int=4,
        num_blocks: int=3,
        round_weight_factor: float=5,
        num_loads: tuple=(1, 1),
    ):
        assert isinstance(name, str)
        assert isinstance(_max, (int, float)) and _max > 0
        assert isinstance(weekly_max_incr, (int, float)) and weekly_max_incr > 0
        total_volume = int(total_volume)
        assert total_volume > 0
        assert round_weight_factor > 0
        assert len(num_loads) == 2 and all(isinstance(i, int) and i >= 1 for i in num_loads)
        self.name = name
        self.max = _max
        self.weekly_max_incr = weekly_max_incr
        self.program = None
        self.total_volume = total_volume
        self.num_days_per_week = num_days_per_week
        self.num_weeks_per_block = num_weeks_per_block
        self.num_blocks = num_blocks
        self.round_weight_factor = round_weight_factor
        self.num_loads = num_loads
    
    def __str__(self):
        return (
            f"{self.name}: {self.max}\n" +
            f"{self.weekly_max_incr} increase per week.\n"
        )
    
    @staticmethod
    def vol_distr(n, p):
        return [p**i for i in range(n)]
    
    @staticmethod
    def get_shape(x):
        if type(x) == list:
            shape = len(x)
        elif type(x) == np.ndarray:
            shape = x.shape
        else:
            shape = None
        return shape
    
    @staticmethod
    def estimate_1RM(weight: float, reps: int) -> float:
        assert weight > 0
        assert isinstance(reps, int) and reps >= 1
        estimated_1RM = weight * (1 + reps / 30)
        return estimated_1RM
    
    def estimate_max_reps(self, weight: float | np.ndarray) -> int | np.ndarray:
        if isinstance(weight, np.ndarray):
            _int, _round = np.int64, np.round
            weight_good = all(weight > 0)
        else:
            _int, _round = int, round
            weight_good = weight > 0
        if not weight_good:
            raise ValueError(f"weight must be positive, but instead it's {weight}")
        max_reps = _int(_round((self.max / weight - 1) * 30))
        return max_reps
    
    def estimate_rep_max(self, reps):
        assert isinstance(reps, int) and reps >= 1
        weight = self.max / (1 + reps/30)
        return weight
    
    @staticmethod
    def __rand_partition(x, n):
        # Randomly partition number or array x into n segments.
        # If x is int, then break x into n uniformly sized paritions
        # If x is array, then add a new dimension of length n and partition all elements along that new dimension
        rng = np.random.default_rng()
        if type(x) == list:
            x = np.array(x)
        x_shape = Lift.get_shape(x)
        if type(x) == np.ndarray:
            bins = []
            for xi in x.flatten():
                partitions = rng.choice(range(1, xi), size=n-1, replace=False)
                partitions = np.concatenate(([0], sorted(partitions), [xi]))
                bins.append(partitions[1:] - partitions[:-1])
            bins = np.array(bins).reshape(x.shape + (n,))
        else: # x_shape is None
            partitions = rng.choice(range(1, x), size=n-1, replace=False)
            partitions = np.concatenate(([0], sorted(partitions), [x]))
            bins = np.array(partitions[1:] - partitions[:-1], dtype=int)
        return bins
    
    def gen_load(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # Generate random samples
        x = rng.beta(a=7.09, b=7.39)
        # Convert to load
        load = (0.5 + x/2) * self.max
        return load
    
    def gen_multi_wt_vol(self, vol: int) -> dict[str, np.ndarray]:
        """Generate multiple training loads across a day's volume.
        
        Independently generates additional training loads (one each representing
        an additional set) until all training volume has been allocated.
        Ensures that reps per set are appropriate for each load generated.

        Parameters
        ----------
        vol : int
            _description_

        Returns
        -------
        dict[str, np.ndarray]
            _description_
        """
        rng = np.random.default_rng()
        wts = []
        # loop gets number of and values of working weights

        # We will try to make at most this many different training loads.
        # Ex: self.num_loads=(2,4) -> We may draw num_loads_tgt = 2, 3, or 4.
        num_loads_tgt = rng.integers(self.num_loads[0], self.num_loads[1]+1)
        for _ in range(num_loads_tgt):
            wts.append(self.gen_load())
            # Check if enough volume available to cover at least 1/3 of rep-max
            # at each weight.
            if vol <= sum(self.estimate_max_reps(wt) / 3 for wt in wts) and len(wts) > 1:
                wts.pop(-1)
                break
        wts = np.array(wts)
        # assign num reps to each weight
        if len(wts) > 1:
            reps_per_set = self.estimate_max_reps(wts)
            reps_distr = reps_per_set / sum(reps_per_set)
            vols = round_retain_sum(reps_distr * vol)
        else:
            vols = np.array([vol])
        return {"weight": wts, "volume": vols}
    
    def gen_program(self, seed=None):
        rng = np.random.default_rng(seed=seed)
        self.program = []
        vol_distr_blocks = rng.permutation(
            self.vol_distr(self.num_blocks, 0.9)
        )
        vol_distr_blocks = vol_distr_blocks /    vol_distr_blocks.sum()
        volume_blocks = round_retain_sum(
            self.total_volume * vol_distr_blocks * 1.0
        )
        original_max = self.max
        vol_distr_weeks = self.vol_distr(self.num_weeks_per_block, 0.8)
        for b in range(self.num_blocks):
            self.program.append([])
            vol_distr_weeks = self.gen_permutation(
                vol_distr_weeks,
                new_only=b>0,
                rng=rng,
            )
            vol_distr_weeks = vol_distr_weeks / vol_distr_weeks.sum()
            volume_weeks = round_retain_sum(
                volume_blocks[b] * vol_distr_weeks
            )
            vol_distr_days = self.vol_distr(self.num_days_per_week, 0.8)
            for w in range(self.num_weeks_per_block):
                self.program[b].append([])
                vol_distr_days = self.gen_permutation(
                    vol_distr_days,
                    new_only=w>0,
                    rng=rng,
                )
                vol_distr_days = vol_distr_days / vol_distr_days.sum()
                volume_days = round_retain_sum(volume_weeks[w] * vol_distr_days)
                for d in range(self.num_days_per_week):
                    day_volume = self.gen_multi_wt_vol(volume_days[d])
                    self.program[b][w].append(day_volume)
                self.max += self.weekly_max_incr
        self.max = original_max
    
    @staticmethod
    def gen_permutation(
        x,
        new_only=True,
        rng=np.random.default_rng(),
    ):
        """
        Permute x to a new order.
        Uses x as the old permutation unless one is specified in x_old.
        """
        x_new = rng.permutation(x)
        while new_only and all(x_new == x):
            x_new = rng.permutation(x)
        
        return x_new
    
    def round_weight(self, weight: float | np.ndarray) -> float | np.ndarray:
        rounded_weight = round(
            weight / self.round_weight_factor
        ) * self.round_weight_factor
        return rounded_weight
    
    def print_program(self):
        if self.program is None:
            return
        for b, block in enumerate(self.program):
            print(f"\n**----------Block {b+1}:----------**")
            vol_block = 0
            for w, week in enumerate(block):
                print(f"\n*-----Week {w+1}:-----*\n")
                print(self.name)
                vol_week = 0
                for d, day in enumerate(week):
                    wt = round(day["weight"]/self.round_weight)*self.round_weight
                    vol = day["volume"]
                    print(f"Day {d+1}: {wt} lbs x {vol} reps")
                    vol_week += vol
                print(f"\nWeek {w+1} volume: {vol_week}")
                vol_block += vol_week
            print(f"Block {b+1} volume: {vol_block}")


class VariableLoadingProgram:
    def __init__(self):
        self.lifts: list[Lift] = []
    
    def add_lift(self, lift: Lift):
        self.lifts.append(lift)
    
    def add_lifts(self, lifts: list[Lift]):
        [self.add_lift(lift) for lift in lifts]
    
    def gen_program(self):
        for lift in self.lifts:
            lift.gen_program()
    
    def print_program(self, to_file:bool=False):
        if to_file:
            filename = (
                "program_" + datetime.now().replace(microsecond=0).strftime(
                    "%Y_%m_%d_at_%H_%M_%S"
                ) + ".txt"
            )
            dir_path = Path("./data")
            dir_path.mkdir(exist_ok=True)
            file_path = dir_path / filename

        def _print(*args, **kwargs):
            print(*args, **kwargs)
            if to_file:
                with open(file_path, "a") as f:
                    print(*args, **kwargs, file=f)
        
        max_num_blocks = max(lift.num_blocks for lift in self.lifts)
        for b in range(max_num_blocks):
            _print(f"\n**----------Block {b+1}:----------**")
            max_num_weeks = max(lift.num_weeks_per_block for lift in self.lifts)
            for w in range(max_num_weeks):
                _print(f"\n{' '*7}*-----Week {w+1}:-----*")
                max_num_days = max(lift.num_days_per_week for lift in self.lifts)
                for d in range(max_num_days):
                    _print(f"\n{' '*3}Day {d+1}:")
                    table_data = []
                    for lift in self.lifts:
                        if not (
                            b < len(lift.program)
                            and w < len(lift.program[b])
                            and d < len(lift.program[b][w])
                        ):
                            continue
                        day = lift.program[b][w][d]
                        wts = [lift.round_weight(wt) for wt in day["weight"]]
                        vols = day["volume"]
                        # Check if rounding causes duplicative weights &
                        # consolidate their volumes.
                        prev_wt = None
                        ids_to_rmv = []
                        for i, (wt, vol) in enumerate(zip(wts, vols)):
                            if wt == prev_wt:
                                vols[i-1] += vol
                                ids_to_rmv.append(i)
                            prev_wt = wt
                        # remove wts and vols that were already consolidated
                        wts = [wt for i, wt in enumerate(wts) if i not in ids_to_rmv]
                        vols = [vol for i, vol in enumerate(vols) if i not in ids_to_rmv]
                        # put into table
                        for wt, vol in zip(wts, vols):
                            # _print(f"{lift.name+':':<{max_lift_name_len+1}} {wt:>3} lbs x {vol:>2} reps")
                            table_data.append(
                                [
                                    lift.name if (table_data or ["None"])[-1][0]!=lift.name else "",
                                    str(wt) + " lbs",
                                    "x " + f"{vol:>3}" + " reps"
                                ]
                            )
                    print(tabulate(table_data, headers=["Lift", "Weight", "Volume"], tablefmt="psql"))
    
    def print_lifts(self):
        for lift in self.lifts:
            print()
            print(lift)


def round_retain_sum(x: np.ndarray) -> np.ndarray:
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


def main():
    deadlift = Lift(
        "Deadlift", 475, 1,
        total_volume=450,
        num_days_per_week=3,
        num_loads=(1,3),
    )
    bench = Lift(
        "Bench Press", 265, 0.5,
        total_volume=700,
        num_days_per_week=3,
        num_loads=(1, 3),
    )
    squat = Lift(
        "Squat", 375, 1,
        total_volume=450,
        num_days_per_week=3,
        num_loads=(1,3),
    )
    bb_row = Lift(
        "Barbell Row", 245, 0.5,
        total_volume=350,
        num_days_per_week=2,
    )
    pullup = Lift(
        "Pull-up", 1, 0.5,
        total_volume=800,
        num_days_per_week=5,
    )
    kb_swing = Lift(
        "KB Swing", 1, 0.1,
        total_volume=3*100*12,
        num_days_per_week=3,
    )
    kb_press = Lift(
        "KB Press", 80, 0.5,
        total_volume=3*30*12,
        num_days_per_week=3,
        num_loads=(1, 3),
        round_weight_factor=18,
    )
    
    vl_program = VariableLoadingProgram()
    vl_program.add_lifts([deadlift, bench, squat, bb_row, pullup, kb_swing, kb_press])
    vl_program.gen_program()
    vl_program.print_program(to_file=True)
    #deadlift.print_program()
    #bench.print_program()
    #pullup.print_program()


if __name__ == "__main__":
    main()
