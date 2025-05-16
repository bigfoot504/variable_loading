"""
Created 5/12/2025.
This is a new module aimed at using the variable-loading principles to generate a training program. It hinges upon varying the volume by about 20% per day among days of the week, by about 20% per week for weeks of the block (month), and by about 10% per block for blocks of the macrocycle.
New in this version is that it is object-oriented and hence more flexible.

To-Do: A future version will use the VariableLoading class to generate one program for all of the lifts at one time and to print the comprehensive program.
"""

from datetime import datetime
import os

import numpy as np


class Lift:
  instances = []
  
  def __init__(
    self,
    name: str,
    _max: float,
    weekly_max_incr: float,
    total_volume: int,
    num_days_per_week: int=4,
    num_weeks_per_block: int=4,
    num_blocks: int=3,
    round_weight_factor: float=5.0,
    num_sets: tuple=(1, 1),
  ):
    assert isinstance(name, str)
    assert isinstance(_max, (int, float)) and _max > 0
    assert isinstance(weekly_max_incr, (int, float)) and weekly_max_incr > 0
    total_volume = int(total_volume)
    assert total_volume > 0
    assert round_weight_factor > 0
    assert len(num_sets) == 2 and all(isinstance(i, int) and i >= 1 for i in num_sets)
    self.name = name
    self.max = _max
    self.weekly_max_incr = weekly_max_incr
    Lift.instances.append(self)
    self.program = None
    self.total_volume = total_volume
    self.num_days_per_week = num_days_per_week
    self.num_weeks_per_block = num_weeks_per_block
    self.num_blocks = num_blocks
    self.round_weight_factor = round_weight_factor
    self.num_sets = num_sets
  
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
  
  @staticmethod
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
  
  def gen_load(self, rng=None):
    if rng is None:
      rng = np.random.default_rng()
    # Generate random samples
    x = rng.beta(a=7.09, b=7.39)
    # Convert to load
    load = (0.5 + x/2) * self.max
    return load
  
  def gen_program(self, seed=None):
    rng = np.random.default_rng(seed=seed)
    self.program = []
    vol_distr_blocks = rng.permutation(
      self.vol_distr(self.num_blocks, 0.9)
    )
    vol_distr_blocks = vol_distr_blocks /  vol_distr_blocks.sum()
    volume_blocks = Lift.round_retain_sum(
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
      volume_weeks = Lift.round_retain_sum(
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
        volume_days = Lift.round_retain_sum(
          volume_weeks[w] * vol_distr_days
        )
        for d in range(self.num_days_per_week):
          num_sets = rng.integers(self.num_sets[0], self.num_sets[1]+1)
          self.program[b][w].append(
            {
              "weight": [self.gen_load() for _ in range(num_sets)],
              "volume": self.__rand_partition(volume_days[d], num_sets),
            }
          )
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
  
  def round_weight(self, weight):
    rounded_weight = float(
      round(
        weight / self.round_weight_factor
      ) * self.round_weight_factor
    )
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


class VariableLoading:
  @staticmethod
  def gen_program():
    for lift in Lift.instances:
      lift.gen_program()
  
  @staticmethod
  def print_program(
    to_file:bool=False
  ):
    if to_file:
      fname = (
        "data/program_" +
        str(datetime.now().replace(microsecond=0)).replace(" ", "_") +
        ".txt"
      )
      if not os.path.exists("data"):
        os.makedirs("data")
      

    def _print(*args, **kwargs):
      print(*args, **kwargs)
      if to_file:
        with open(fname, "a") as f:
          print(*args, **kwargs, file=f)
    max_lift_name_len = max(
      len(lift.name) for lift in Lift.instances
    )
    for b in range(max(lift.num_blocks for lift in Lift.instances)):
      _print(f"\n**----------Block {b+1}:----------**")
      for w in range(max(lift.num_weeks_per_block for lift in Lift.instances)):
        print(f"\n*-----Week {w+1}:-----*")
        for d in range(max(lift.num_days_per_week for lift in Lift.instances)):
          _print(f"\nDay {d+1}:")
          for lift in Lift.instances:
            if not (
              b < len(lift.program)
              and w < len(lift.program[b])
              and d < len(lift.program[b][w])
            ):
              continue
            day = lift.program[b][w][d]
            wts = [lift.round_weight(wt) for wt in day["weight"]]
            vols = day["volume"]
            for wt, vol in zip(wts, vols):
              _print(f"{lift.name+':':<{max_lift_name_len+1}} {wt:>3} lbs x {vol:>2} reps")
  
  @staticmethod
  def print_lifts():
    for lift in Lift.instances:
      print()
      print(lift)


def main():
  deadlift = Lift(
    "Deadlift", 475, 1,
    total_volume=450,
    num_days_per_week=3,
    num_sets=(1,3),
  )
  bench = Lift(
    "Bench Press", 265, 0.5,
    total_volume=700,
    num_days_per_week=3,
  )
  pullup = Lift(
    "Pull-up", 1, 0.5,
    total_volume=800,
    num_days_per_week=5,
  )
  
  VariableLoading.gen_program()
  VariableLoading.print_program(to_file=True)
  #deadlift.print_program()
  #bench.print_program()
  #pullup.print_program()


if __name__ == "__main__":
  main()
