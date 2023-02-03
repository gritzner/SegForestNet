import torch
import numpy as np


class LearningRateScheduler():
    def __init__(self, optimizer, min_learning_rate, num_cycles, cycle_length_factor, num_iterations):
        self.index = 0
        self.optimizer = optimizer
        self.min_learning_rate = min_learning_rate
        
        self.cycles = LearningRateScheduler.get_cycle_breakpoints(
            num_cycles, cycle_length_factor, num_iterations
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cycles[0], eta_min=self.min_learning_rate 
        )
    
    @staticmethod
    def get_cycle_breakpoints(num_cycles, cycle_length_factor, num_iterations):
        b = num_iterations / np.sum([cycle_length_factor**i for i in range(num_cycles)])
        cycles = [round(b*cycle_length_factor**i) for i in range(num_cycles)]
        cycles = [np.sum(cycles[:i+1]) for i in range(num_cycles)]
        cycles[-1] = num_iterations
        return np.asarray(cycles) - 1
    
    def get_lr(self):
        return self.scheduler.get_last_lr()[0]
    
    def step(self, step_num):
        if step_num < self.cycles[self.index]:
            self.scheduler.step()
        elif self.index+1 < len(self.cycles):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, (self.cycles[self.index+1] - self.cycles[self.index]) - 1, eta_min=self.min_learning_Rate
            )
            self.index += 1
    