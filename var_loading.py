"""
Created 5/12/2025.
This is a new module aimed at using the variable-loading principles to generate a training program. It hinges upon varying the volume by about 20% per day among days of the week, by about 20% per week for weeks of the block (month), and by about 10% per block for blocks of the macrocycle.
New in this version is that it is object-oriented and hence more flexible.

To-Do: A future version will use the VariableLoading class to generate one program for all of the lifts at one time and to print the comprehensive program.
"""

import numpy as np


class Lift:
  instances = []
  vol_distr80 = {
    2: [44, 56],
    3: [26, 33, 41],
    4: [17, 22, 28, 33]
  }
  vol_distr90 = {
    2: [47, 53],
    3: [30, 33, 37],
    4: [20, 23, 26, 31]
  }
  def __init__(
    self,
    name: str,
    _max: float,
    weekly_max_incr: float,
    total_volume: int,
    num_days_per_week: int=4,
    num_weeks_per_block: int=4,
    num_blocks: int=3,
    round_weight = 5,
  ):
    assert isinstance(name, str)
    assert isinstance(_max, (int, float)) and _max > 0
    assert isinstance(weekly_max_incr, (int, float)) and weekly_max_incr > 0
    assert isinstance(total_volume, int) and total_volume > 0
    self.name = name
    self.max = _max
    self.weekly_max_incr = weekly_max_incr
    Lift.instances.append(self)
    self.program = None
    self.total_volume = total_volume
    self.num_days_per_week = num_days_per_week
    self.num_weeks_per_block = num_weeks_per_block
    self.num_blocks = num_blocks
    self.round_weight = round_weight
  
  def __str__(self):
    return (
      f"{self.name}: {self.max}\n" +
      f"{self.weekly_max_incr} increase per week.\n"
    )
  
  @staticmethod
  def __rand_partition(x, n):
    # Randomly partition number or array x into n segments.
    # If x is int, then break x into n uniformly sized paritions
    # If x is array, then add a new dimension of length n and partition all elements along that new dimension
    if type(x) == list:
      x = np.array(x)
    x_shape = get_shape(x)
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
  
  def gen_program(self):
    rng = np.random.default_rng()
    self.program = []
    vol_distr_blocks = rng.permutation(
      self.vol_distr90[self.num_blocks]
    )
    vol_distr_blocks = vol_distr_blocks /  vol_distr_blocks.sum()
    volume_blocks = Lift.round_retain_sum(
      self.total_volume * vol_distr_blocks * 1.0
    )
    original_max = self.max
    vol_distr_weeks = self.vol_distr80[self.num_weeks_per_block]
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
      vol_distr_days = self.vol_distr80[self.num_days_per_week]
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
          self.program[b][w].append(
            {
              "weight": self.gen_load(),
              "volume": volume_days[d],
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
  
  def print_program(self):
    if self.program is None:
      return
    for b, block in enumerate(self.program):
      print(f"\n**----------Block {b}:----------**")
      vol_block = 0
      for w, week in enumerate(block):
        print(f"\n*-----Week {w}:-----*\n")
        print(self.name)
        vol_week = 0
        for d, day in enumerate(week):
          wt = round(day["weight"]/self.round_weight)*self.round_weight
          vol = day["volume"]
          print(f"Day {d}: {wt} lbs x {vol} reps")
          vol_week += vol
        print(f"\nWeek {w} volume: {vol_week}")
        vol_block += vol_week
      print(f"Block {b} volume: {vol_block}")


class VariableLoading:
  @classmethod
  def print_lifts(cls):
    for lift in Lift.instances:
      print()
      print(lift)


def main():
  squat = Lift(
    "Squat", 405, 1, 600
  )
  bench = Lift(
    "Bench Press", 315, 0.5, 900
  )
  VariableLoading.print_lifts()
  squat.gen_program()
  squat.print_program()
  bench.gen_program()
  bench.print_program()


if __name__ == "__main__":
  main()
