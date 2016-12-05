#!usr/bin/env/ python
#-*- coding=utf-8 -*-

"""
created on Dec,05 2016
"""

from __future__ import division
import pylab as pl
import numpy as np

class hyperion:
    def __init__(self, r_0=2, theta_0=0, r1_0=0, theta1_0=2, k=-32, a=-2, dt=0.001, span=60, m1=0.3, d=1e-3, rtheta=0, romega=0.1):
        self.r, self.theta, self.r1, self.theta1, self.t = [r_0], [theta_0], [r1_0],[theta1_0], [0]
        self.k, self.a, self.dt, self.span, self.m1, self.m2, self.d1, self.d2 = k, a, dt, span, m1, 1-m1, (1-m1)*d, m1*d
        self.I, self.rtheta, self.romega = m1*self.d1**2+(1-m1)*self.d2**2, [rtheta], [romega]
    def calculate(self):
        while self.t[-1] < self.span:
            self.r1.append(self.r1[-1]+(self.k*pow(self.r[-1],self.a)+self.r[-1]*self.theta1[-1]**2)*self.dt)
            self.theta1.append(self.theta1[-1]-2*self.r1[-1]*self.theta1[-1]/self.r[-1]*self.dt)
            self.romega.append(self.romega[-1]-self.k*self.r[-1]*np.sin(self.rtheta[-1]-self.theta[-1])/self.I*(self.m1*self.d1/(self.r[-1]**2+self.d1**2+2*self.r[-1]*self.d1*np.cos(self.rtheta[-1]-self.theta[-1]))**1.5-self.m2*self.d2/(self.r[-1]**2+self.d1**2-2*self.r[-1]*self.d1*np.cos(self.rtheta[-1]-self.theta[-1]))**1.5)*self.dt)
            self.r.append(self.r[-1]+self.r1[-1]*self.dt)
            self.theta.append(self.theta[-1]+self.theta1[-1]*self.dt)
            self.rtheta.append(self.rtheta[-1]+self.romega[-1]*self.dt)
            self.t.append(self.t[-1] + self.dt)
            print self.t[-1]

#######  不如来画图  #########
aa=hyperion(k=-32,a=-2,theta1_0=1.414,span=30,m1=0.3)
aa.calculate()
tta = [(i+np.pi)%(2*np.pi)-np.pi for i in aa.rtheta]
"""
pl.subplot(121)
pl.plot(aa.t,tta,lw=2,color='g',label=r"$\theta$-$t$")
pl.xlabel("time($s$)")
pl.ylabel(r"$\theta$($Rad$)")
pl.legend(loc='upper right',frameon=False,fontsize='small')
pl.subplot(122)
pl.plot(aa.t,aa.romega,lw=2,color='m',label=r"$\omega$-$t$")
pl.xlabel("time($s$)")
pl.ylabel(r"$\omega$($Rad/s$)")
pl.legend(loc='upper right',frameon=False,fontsize='small')
pl.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
pl.suptitle("elleptical orbit, $m1/M=$"+str(aa.m1))
"""
"""
pl.plot(tta,aa.romega,'o-',lw=1,color='y',label=r"$\omega$-$\theta$")
pl.xlabel(r"$\theta$($Rad$)")
pl.ylabel(r"$\omega$($Rad/s$)")
pl.legend(loc='upper right',frameon=True)
pl.title("elliptical orbit, $m1/M=$"+str(aa.m1))
"""
bb=hyperion(k=-32,a=-2,theta1_0=1.414,rtheta=1e-7,span=30,m1=0.3)
bb.calculate()
dtta=[i-j for i,j in zip(aa.rtheta, bb.rtheta)]
pl.semilogy(aa.t,dtta,'x-',lw=1,color='g',label=r"$\Delta\theta$-$t$")
pl.xlabel(r"$time$($s$)")
pl.ylabel(r"$\Delta\theta$($Rad$)")
pl.legend(loc='upper right',frameon=True)
pl.title("elliptical orbit, $m1/M=$"+str(aa.m1))
pl.show()
