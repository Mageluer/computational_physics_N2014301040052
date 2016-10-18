#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
created on Oct,16 2016
@author: Mageluer
"""

import pylab as pl
import math

class cannon:
    def __init__(self):
        self.x, self.y, self.dt, self.g = [0], [0], 0.01, 9.8
        self.v_0 = float(raw_input("Enter initial velocity(m/s): ") or 300)
        self.theta = float(raw_input("Enter firing angle(degree): ") or 40)
	self.v_x, self.v_y = [self.v_0 * math.cos(self.theta * math.pi / 180)], [self.v_0 * math.sin(self.theta * math.pi / 180)] 

    def calculate(self):
    	while self.y[-1] >= 0:
	    self.x.append(self.x[-1] + self.v_x[-1] * self.dt)
	    self.y.append(self.y[-1] + self.v_y[-1] * self.dt)
	    self.v_x.append(self.v_x[-1])
	    self.v_y.append(self.v_y[-1] - self.g * self.dt)
	self.x[-1], self.y[-1] = (self.x[-1] * self.y[-2] - self.x[-2] * self.y[-1]) / (self.y[-2] - self.y[-1]), 0

    def show_results(self):
        pl.plot(self.x, self.y, 'b', label = '$v_0$= '+ str(self.v_0) + '$m/s$' + ', $\\theta$= ' + str(self.theta) + '$^{\\circ}$')
	pl.title('Trajectory of a Cannon Shell')
        pl.xlabel('x ($m$)')
        pl.ylabel('y ($m$)')
        pl.xlim(0, )
	pl.ylim(0, )
        pl.legend()
        pl.show()

    def store_results(self):
        if raw_input("Save the data?(y/n): ") == "y":
            with open('cannondata.txt', 'w') as f:
                for position in zip(self.x, self.y): f.write(str(position))

class cannon_drag(cannon):
    def calculate(self):
    	drag_constant = 4 * 10 ** (-2)
     	while self.y[-1] >= 0:
	    self.x.append(self.x[-1] + self.v_x[-1] * self.dt)
	    self.y.append(self.y[-1] + self.v_y[-1] * self.dt)
	    v = math.sqrt(self.v_x[-1] ** 2 + self.v_y[-1] ** 2)
	    self.v_x.append(self.v_x[-1] - drag_constant * v * self.v_x[-1] * self.dt)
	    self.v_y.append(self.v_y[-1] - self.g * self.dt - drag_constant * v * self.v_y[-1] * self.dt)
	self.x[-1], self.y[-1] = (self.x[-1] * self.y[-2] - self.x[-2] * self.y[-1]) / (self.y[-2] - self.y[-1]), 0

class cannon_isothermal(cannon):
    def calculate(self):
        self.y[0] = 1
    	drag_constant = 4 * 10 ** (-2)
	density_constant = 3 * 10 ** (-2)
     	while self.y[-1] >= 0:
	    self.x.append(self.x[-1] + self.v_x[-1] * self.dt)
	    self.y.append(self.y[-1] + self.v_y[-1] * self.dt)
	    v = math.sqrt(self.v_x[-1] ** 2 + self.v_y[-1] ** 2)
	    self.v_x.append(self.v_x[-1] - drag_constant * v * self.v_x[-1] * self.dt - math.exp(- self.y[-1] / self.y[0]) * density_constant * v * self.v_x[-1] * self.dt)
	    self.v_y.append(self.v_y[-1] - self.g * self.dt - drag_constant * v * self.v_y[-1] * self.dt - math.exp(- self.y[-1] / self.y[0]) * density_constant * v * self.v_x[-1] * self.dt)
	self.x[-1], self.y[-1] = (self.x[-1] * self.y[-2] - self.x[-2] * self.y[-1]) / (self.y[-2] - self.y[-1]), 0

class cannon_adiabatic(cannon):
    def calculate(self):
    	drag_constant = 4 * 10 ** (-2)
	density_constant = 3 * 10 ** (-2)
	a = 6.5
	alpha = 2.5
	temperature_0 = 300
     	while self.y[-1] >= 0:
	    self.x.append(self.x[-1] + self.v_x[-1] * self.dt)
	    self.y.append(self.y[-1] + self.v_y[-1] * self.dt)
	    v = math.sqrt(self.v_x[-1] ** 2 + self.v_y[-1] ** 2)
	    self.v_x.append(self.v_x[-1] - drag_constant * v * self.v_x[-1] * self.dt - (1 - a * self.y[-1] / temperature_0) ** alpha * density_constant * v * self.v_x[-1] * self.dt)
	    self.v_y.append(self.v_y[-1] - self.g * self.dt - drag_constant * v * self.v_y[-1] * self.dt -  (1 - a * self.y[-1] / temperature_0) ** alpha * density_constant * v * self.v_y[-1] * self.dt)
	self.x[-1], self.y[-1] = (self.x[-1] * self.y[-2] - self.x[-2] * self.y[-1]) / (self.y[-2] - self.y[-1]), 0

class main:
    def __init__(self):
        cannons = {'1': 'cannon', '2': 'cannon_drag', '3': 'cannon_isothermal', '4': 'cannon_adiabatic'}
        self.fire = globals()[cannons[raw_input("\n1. Movement without drag\n2. Movement with air drag\n3. Movement with isothermal effect and drag\n4. Movement with adbiatic effect and resistance\nYour choice(1/2/3/4): ") or '1']]()
    def run(self):
        self.fire.calculate()
        self.fire.show_results()
        self.fire.store_results()
main().run()
