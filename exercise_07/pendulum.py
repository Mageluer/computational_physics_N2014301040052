#!usr/bin/env/ python
#-*- coding=utf-8 -*-

"""
created on Oct,29 2016
"""

from __future__ import division
import pylab as pl
import numpy as np

class pendulum:
    def __init__(self, theta_0=0.2, omega_0=0, q=1/2, l=9.8, g=9.8, F_D=1.2, omega_D=2/3, dt=0.04, span=60):
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

aa = pendulum(F_D=2.5, span=75, q=0.5)
aa.calculate()
###### 画指数图 ######
bb = pendulum(F_D=2.5, span=75, q=0.5+1e-4)
bb.calculate()
tta = [np.abs(i-j) for i,j in zip(aa.theta,bb.theta)]
ttam = [j for i,j in enumerate(tta) if 0<i<len(tta)-1 and tta[i-1]<=tta[i]>=tta[i+1]]
ttamt = [aa.t[tta.index(i)] for i in ttam]
ttamlg = [np.log(i) for i in ttam]
lnn = np.polyfit(ttamt,ttamlg,1)
llnn = [np.exp(i*lnn[0]+lnn[1]) for i in ttamt]
pl.semilogy(aa.t, tta,'g-', label=r'$\Delta\theta$')
pl.plot(ttamt, ttam, 'sc--',label='extrem point')
pl.plot(ttamt, llnn, 'm-', label=r'first order fit')
pl.xlabel(r'time ($s$)', fontdict=font)
pl.ylabel(r'$\Delta\theta (radians)$', fontdict=font)
pl.title(r'$\Delta\theta$ versus time with $F_D=$'+str(aa.F_D), fontdict=font)
pl.text(20,1e-7,'y = %+fx %+f\n$\lambda$ = %+f' % (lnn[0], lnn[1], lnn[0]), fontdict=font)
pl.legend(loc='best')
"""
###### 画吸引子 ######
hhha, hhhb, hhhc, hhhd = [1, 2, 4, 8], [1, 2, 3, 4], ['b^', 'ms', 'go', 'cd'], [r'$0$', r'$\pi$', r'$\pi$/2', r'$\pi$/4']
for ha, hb, hc, hd in zip(hhha, hhhb, hhhc, hhhd):
    tm = [i for i in aa.t if ((i-2*np.pi/aa.omega_D/ha)%(2*np.pi/aa.omega_D))  < aa.dt]
    teta = [aa.theta[aa.t.index(i)] for i in tm]
    omg = [aa.omega[aa.t.index(i)] for i in tm]
    tta = [(i+np.pi)%(2*np.pi)-np.pi for i in teta]
    pl.subplot(2,2,hb)
    pl.plot(tta,omg,hc, label= hd +" out of phase")
    pl.xlabel(r'$\theta$ (radians)', fontdict=font)
    pl.ylabel(r'$\omega$ (radians/s)', fontdict=font)
    pl.legend(loc='best')
pl.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
pl.suptitle(r'$\omega$ versus $\theta$, $F_D=$'+str(aa.F_D), fontdict=font)
"""
pl.show()
