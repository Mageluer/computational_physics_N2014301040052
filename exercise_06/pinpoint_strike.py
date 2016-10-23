#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
created on Oct,23 2016
@author: Mageluer
"""

from __future__ import division
import pylab as pl
import math

class cannon:
    def __init__(self):
        self.x, self.y, self.dt, self.g = [0], [0], 0.1, 9.8
        self.v_0 = 100
        self.theta = 45
        self.target_x, self.target_y = 10000, 0 
    def calculate(self, v_0, theta, v_wind):
        def find_distance(x, y):
            return math.sqrt((x - target_x)**2 + (y - target_y)**2)
        def find_tangency(x1, y1, x2, y2, x0, y0):
            return ((x1-x2)**2 * x0 + (x1-x2) * (y1-y2) * y0 - (x1 * y2 - x2 * y1) * (y1-y2)) / ((y1-y2)**2 + (x1-x2)**2), ((y1-y2)**2 * y0 + (y1-y2) * (x1-x2) * x0 - (y1 * x2 - y2 * x1) * (x1 - x2)) / ((x1-x2)**2 + (y1-y2)**2 )
        def improved_Euler(x, y):
            drag_constant, a, T0, alpha = 4e-5, 6.5e-3, 300, 2.5
            def accelerate(x, y, vx, vy):
                v = math.sqrt(vx**2 + vy**2)
                ax = - drag_constant * v * (vx + v_wind) * (1 - a*y / T0)**alpha
                ay = - drag_constant * v * vy * (1 - a*y / T0)**alpha - g
                return ax, ay
            def loop(x, y, vx, vy, h):
                acc = accelerate(x[-1], y[-1], vx, vy)
                x_estimate, y_estimate = x[-1] + vx * h, y[-1] + vy * h
                vx_estimate, vy_estimate = vx + acc[0] * h, vy + acc[1] * h
                acc_next = accelerate(x_estimate, y_estimate, vx_estimate, vy_estimate)
                vx_next, vy_next = vx + (acc[0] + acc_next[0]) * h/2, vy + (acc[1] + acc_next[1]) * h/2
                x.append(x[-1] + (vx + vx_next) * h/2)
                y.append(y[-1] + (vy + vy_next) * h/2)
                vx, vy = vx_next, vy_next
                return x, y, vx, vy
            return loop
        ###################################
        #                                 #
        #    引用变量可不喜欢跟你开玩笑   #
        #                                 #
        ###################################
        x, y, h, g, target_x, target_y = self.x[:], self.y[:], self.dt, self.g, self.target_x, self.target_y
	vx, vy = v_0 * math.cos(theta * math.pi / 180) - v_wind, v_0 * math.sin(theta * math.pi / 180)
        loop = improved_Euler(x, y)
    	while x[-1] <= target_x and y[-1] <= target_y:
            x, y, vx, vy = loop(x, y, vx, vy, h)
    	while x[-1] <= target_x or y[-1] <= target_y:
            x, y, vx, vy = loop(x, y, vx, vy, h/100)
            if find_distance(x[-1], y[-1]) > find_distance(x[-2], y[-2]): break
        x[-1], y[-1] = find_tangency(x[-1], y[-1], x[-2], y[-2], target_x, target_y)
        return x, y, find_distance(x[-1], y[-1])
    def strike(self, theta, v_wind=0):
        v_min = math.sqrt(self.g * self.target_y + math.sqrt(self.g**2 * (self.target_x**2 + self.target_y**2)))
        v_max = 3 * v_min
        epsilon = 1e-7
        while True:
            v_mid = (v_min + v_max) / 2
            x, y, distance = self.calculate(v_mid , theta, v_wind)
            print 'find v: ', v_min,v_max,distance
            if v_max - v_min < epsilon or distance < epsilon: return x, y, distance, v_mid
            if y[-1] > self.target_y: v_max = v_mid
            else: v_min = v_mid
    def find_min_speed(self, v_wind=0):
        theta_min, theta_max, epsilon = 30, 70, 1e-2
        while True:
            theta_mid = (theta_min + theta_max) / 2
            v_min, v_mim = self.strike(theta_mid, v_wind)[-1], self.strike(theta_mid + epsilon, v_wind)[-1]
            print 'find theta: ', theta_min, theta_max
            if theta_max - theta_min < epsilon: return theta_mid, v_min
            if v_mim > v_min: theta_max = theta_mid
            else: theta_min = theta_mid + epsilon
    def show_results(self):
        pl.plot(self.x, self.y, 'b', label = '$v_0$= '+ str(self.v_0) + '$m/s$' + ', $\\theta$= ' + str(self.theta) + '$^{\\circ}$')
	pl.title('Trajectory of a Cannon Shell')
        pl.xlabel('x ($m$)')
        pl.ylabel('y ($m$)')
        pl.plot([0, self.target_x],[self.target_y, self.target_y],'g--')
        pl.plot([self.target_x, self.target_x],[0, self.target_y],'g--')
        pl.scatter([self.target_x,],[self.target_y,],50,color='red')
        pl.legend(loc='best')
        pl.show()
        ###################################
        #                                 #
        #    引用变量可不喜欢跟你开玩笑   #
        #                                 #
        ###################################
strikeit = cannon()
ta = [2334,5434,7789,233,52,3455,9999,]
tb = [334,34,78,233,1252,355,9,]
ao = ['b','g','r','c','m','y','k']
wind = range(10, 78, 10)
for i in range(7):
    strikeit.target_x=ta[i]
    strikeit.target_y=tb[i]
    strikeit.xzz, strikeit.yzz,d,v = strikeit.strike(45,0)
    pl.text(23, 3400,'$\\theta$= ' + str(45) + '$^{\\circ}$')
    pl.plot(strikeit.xzz, strikeit.yzz, ao[i], label =  str(strikeit.target_x) +  '$m$, ' +str(strikeit.target_y)+ '$m$' )
    pl.xlim(0,10100)
    pl.ylim(0,3500)
    pl.xlabel('x $m$')
    pl.ylabel('y $m$')
    pl.scatter([strikeit.target_x,],[strikeit.target_y,],50,color=ao[i])
    pl.title('Trajectory of cannon shell attack at the target\n with different wind speed in adiabatic model')
    pl.legend(loc='best')
pl.grid()
pl.show()
    
