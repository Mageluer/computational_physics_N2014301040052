#!usr/bin/env/ python
#-*- coding=utf-8 -*-

"""
created on Nov,13 2016
"""

from __future__ import division
import pylab as pl
import numpy as np

class pendulum:
    def __init__(self, theta_0=0.2, omega_0=0, q=1/2, l=9.8, g=9.8, F_D=1.2, omega_D=2/3, dt=np.pi/100, span=60):
        self.theta, self.omega, self.t = [theta_0], [omega_0], [0]
        self.q, self.l, self.g, self.F_D, self.omega_D, self.dt, self.span = q, l, g, F_D, omega_D, dt, span
    def calculate(self):
        while self.t[-1] < self.span:
            self.omega.append(self.omega[-1] - (self.g/self.l*np.sin(self.theta[-1]) + \
                    self.q*self.omega[-1] - self.F_D*np.sin(self.omega_D*self.t[-1]))*self.dt)
            self.theta.append(self.theta[-1] + self.omega[-1]*self.dt)
            self.t.append(self.t[-1] + self.dt)

#######  不如来画图  #########
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
"""
aa = pendulum(F_D=1.35, span=100, q=0.5)
aa.calculate()
bb = pendulum(F_D=1.44, span=100, q=0.5)
bb.calculate()
cc = pendulum(F_D=1.465, span=100, q=0.5)
cc.calculate()
###### 画指数图 ######
ateta = [(i+np.pi)%(2*np.pi)-np.pi for i in aa.theta]
bteta = [(i+np.pi)%(2*np.pi)-np.pi for i in bb.theta]
cteta = [(i+np.pi)%(2*np.pi)-np.pi for i in cc.theta]
pl.subplot(131)
pl.plot(aa.t, ateta, label=r'$F_D=1.35$')
pl.xlabel(r'time ($s$)', fontdict=font)
pl.ylabel(r'$\theta (radians)$', fontdict=font)
pl.xlim([0,100])
pl.legend(loc='best')
pl.subplot(132)
pl.plot(aa.t, bteta, label=r'$F_D=1.44$')
pl.xlabel(r'time ($s$)', fontdict=font)
pl.ylabel(r'$\theta (radians)$', fontdict=font)
pl.xlim([0,100])
pl.legend(loc='best')
pl.subplot(133)
pl.plot(aa.t, cteta, label=r'$F_D=1.465$')
pl.xlabel(r'time ($s$)', fontdict=font)
pl.ylabel(r'$\theta (radians)$', fontdict=font)
pl.xlim([0,100])
pl.legend(loc='best')
pl.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
pl.suptitle(r'$\theta$ versus time', fontdict=font)
"""
"""
###### 画吸引子 ######
hhha, hhhb, hhhc, hhhd, hhhe, hhhf= [1, 1, 1], [aa.omega, bb.omega, cc.omega], ['b^', 'ms', 'go'], [r'$0$', r'$\pi$', r'$\pi$/2'],  [aa.theta, bb.theta, cc.theta], ['1.350', '1.440', '1.465']
for ha, hb, hc, hd, he, hf in zip(hhha, hhhb, hhhc, hhhd, hhhe, hhhf):
    tm = [i for i in aa.t if ((i-2*np.pi/aa.omega_D/ha)%(2*np.pi/aa.omega_D))  < aa.dt]
    teta = [he[aa.t.index(i)] for i in tm]
    omg = [hb[aa.t.index(i)] for i in tm]
    tta = [(i+np.pi)%(2*np.pi)-np.pi for i in teta]
#    pl.subplot(2,2,hb)
    pl.plot(tta,omg,hc, label= r'$F_D=$'+hf)
pl.xlabel(r'$\theta$ (radians)', fontdict=font)
pl.ylabel(r'$\omega$ (radians/s)', fontdict=font)
pl.legend(loc='best')
#pl.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
pl.suptitle(r'$\omega$ versus $\theta$',fontdict=font)
pl.show()
"""
###################
def bifurcate(F_Dd):
    aa = pendulum(F_D=F_Dd, omega_D= 4/3, span=4000, q=0.5)
    aa.calculate()
    teta = [aa.theta[150*(300+i)] for i in range(100)]
    theta = [(i+np.pi)%(2*np.pi)-np.pi for i in teta]
#    pl.subplot(2,2,hb)
    return 100*[F_Dd], theta
fdd, ttta=[],[]
for i in np.arange(1.4,1.5,0.001):
    bb = bifurcate(i)
    fdd.extend(bb[0])
    ttta.extend(bb[1])
pl.scatter(fdd, ttta, s=0.1, label=r'$\theta$ versus $F_D$')
pl.xlabel(r'$F_D$', fontdict=font)
pl.ylabel(r'$\theta$ (radians)', fontdict=font)
#pl.xlim([0,100])
#pl.ylim([1, 1.30])
pl.title('Bifurcation diagram',fontdict=font)
pl.legend(loc='best')
pl.show()

