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
"""
aa=orbit(k=-32,a=-2,theta1_0=2.1,span=10)
aa.calculate()
"""
"""
bb=orbit(k=-1,a=3,theta1_0=4,span=20)
bb.calculate()
cc=orbit(k=-1,r1_0=0,theta1_0=1,a=3,span=30)
cc.calculate()
dd=orbit(k=-1,r1_0=-2.5,theta1_0=2,a=3,span=60)
dd.calculate()

pl.style.use('dark_background')
for num,orbit,color,rmax in zip([221,222,223,224],[aa,bb,cc,dd],['r','g','b','c'],[2.2,4,2.1,3]):
    ax=pl.subplot(num, projection='polar')
    ax.plot(orbit.theta, orbit.r, color=color, lw=3, label='a='+str(orbit.a))
    ax.set_rmax(rmax)
    ax.grid(True)
    ax.legend()
pl.suptitle('when exponent equals -2')
"""
"""
pl.style.use('dark_background')
a,c=(max(aa.r)+min(aa.r))/2,(max(aa.r)-min(aa.r))/2
print a,c
r_elp=[a/np.sqrt(1-(c/a)**2*np.cos(i)**2) for i in aa.theta]
"""
"""
ax=pl.subplot(111, projection='polar')
ax.plot(aa.theta, aa.r, color='g', lw=3, label='numerical elliptical orbit')
ax.plot(aa.theta,r_elp,lw=3,color='y',label='the real elliptical orbit')
ax.grid(True)
ax.legend()
area=[0]
dtheta=[aa.theta[i+1]-aa.theta[i] for i in range(len(aa.theta)-1)]
for i,j in enumerate(dtheta):
    area.append(area[-1]+0.5*aa.r[i]**2*j)
"""
"""
pl.plot(aa.theta, aa.r, lw=3, color='c',label='numerical elliptical orbit')
pl.plot(aa.theta, r_elp, lw=3, color='r',label='real elliptical orbit')
pl.xlabel(r'$\theta$($Rad$)')
pl.ylabel(r'r($m$)')
pl.legend()
pl.title('radius versus angle of elliptical orbit')
"""
def getT(t,theta):
    for i,j in zip(t,theta):
        if j-2*np.pi>0:
            return i
def getA(r): return (max(r)+min(r))/2
A,T=[0],[0]
for r_0 in np.linspace(1.5,2.1,3):
    aa=orbit(r_0=r_0)
    aa.calculate()
    T.append(getT(aa.t,aa.theta))
    A.append(getA(aa.r))
A3=[i**3 for i in A]
T2=[i**2 for i in T]
pl.style.use('dark_background')
pl.plot(A3,T2,lw=3,color='y')
pl.xlabel(r'$A^3$')
pl.ylabel(r'$T^2$')
pl.title("Kepler's third law")
pl.show()
