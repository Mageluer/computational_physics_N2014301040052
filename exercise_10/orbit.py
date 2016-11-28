#!usr/bin/env/ python
#-*- coding=utf-8 -*-

"""
created on Oct,29 2016
"""

from __future__ import division
import pylab as pl
import numpy as np

class orbit:
    def __init__(self, r_0=2, theta_0=0, r1_0=0, theta1_0=2, k=-32, a=-2, dt=0.001, span=60):
        self.r, self.theta, self.r1, self.theta1, self.t = [r_0], [theta_0], [r1_0],[theta1_0], [0]
        self.k, self.a, self.dt, self.span = k, a, dt, span
    def calculate(self):
        while self.t[-1] < self.span:
            self.r1.append(self.r1[-1]+(self.k*pow(self.r[-1],self.a)+self.r[-1]*self.theta1[-1]**2)*self.dt)
            self.theta1.append(self.theta1[-1]-2*self.r1[-1]*self.theta1[-1]/self.r[-1]*self.dt)
            self.r.append(self.r[-1]+self.r1[-1]*self.dt)
            self.theta.append(self.theta[-1]+self.theta1[-1]*self.dt)
            self.t.append(self.t[-1] + self.dt)
            print self.t[-1]

#######  不如来画图  #########
aa=orbit(k=-32,a=-2,span=100)
aa.calculate()
bb=orbit(k=-32,a=-2,theta1_0=10,span=100)
bb.calculate()
cc=orbit(k=-32,r1_0=10,a=-2,span=100)
cc.calculate()
dd=orbit(k=-32,r1_0=-10,a=-2,span=50)
dd.calculate()

pl.style.use('dark_background')
for num,orbit,color,rmax in zip([221,222,223,224],[aa,bb,cc,dd],['r','g','b','c'],[2.2,17,5,6]):
    ax=pl.subplot(num, projection='polar')
    ax.plot(orbit.theta, orbit.r, color=color, lw=3, label='a='+str(orbit.a))
    ax.set_rmax(rmax)
    ax.grid(True)
    ax.legend()
pl.suptitle('when exponent equals -2')
pl.show()
