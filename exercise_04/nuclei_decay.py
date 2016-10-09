#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
created on Oct,9 2016
@author: Mageluer
"""

import pylab as pl
import math

class nuclei_decay:
    """
    a decay process with two types of nuclei A and B characterized
    with by the same time constant tau
    """
    def __init__(self):
        self.initial_number_A = int(raw_input("Enter initial number of nuclei_A: ") or 100)
        self.initial_number_B = int(raw_input("Enter initial number of nuclei_B: ") or 0)
        self.tau = float(raw_input("Enter time constant: ") or 1)
        self.time_duration = float(raw_input("Enter time of duration: ") or 10)
        self.time_step = float(raw_input("Enter time step: ") or 0.01)
        self.time = [self.time_step * i for i in range(int(self.time_duration // self.time_step + 1))]
        self.number_A = [self.initial_number_A] + (len(self.time) - 1)*[0]

    def calculate(self):
        for i in range(len(self.number_A)-1): self.number_A[i+1] = self.number_A[i] + ( self.initial_number_A + self.initial_number_B - 2 * self.number_A[i]) / self.tau * self.time_step
        self.number_B = [self.initial_number_A + self.initial_number_B - number_A for number_A in self.number_A]
        self.theory_A, self.theory_B = [(self.initial_number_A + self.initial_number_B)/2.0 + (self.initial_number_A - self.initial_number_B)/2.0 * math.exp(-2 * t / self.tau) for t in self.time], [(self.initial_number_A + self.initial_number_B)/2.0 - (self.initial_number_A - self.initial_number_B)/2.0 * math.exp(-2 * t / self.tau) for t in self.time] 

    def show_results(self):
        pl.plot(self.time, self.number_A, 'b', label = "$N_A$")
        pl.plot(self.time, self.number_B, 'g', label = "$N_B$")
        pl.plot(self.time, self.theory_A, 'bx', label = "$N_{A, theory}$")
        pl.plot(self.time, self.theory_B, 'g+', label = "$N_{B, theory}$")
        pl.xlabel('time ($s$)')
        pl.ylabel('Number of Nuclei')
        pl.xlim(0, self.time_duration)
        pl.legend()
        pl.show()

    def store_results(self):
        with open('decaydata.txt', 'w') as f:
            if raw_input("Save the data?(y/n): ") == "y":
                for decay in zip(self.time, self.number_A, self.number_B, self.theory_A, self.theory_B): f.write(str(decay))

decay = nuclei_decay()
decay.calculate()
decay.show_results()
decay.store_results()
