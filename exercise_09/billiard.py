#!usr/bin/env/ python
#-*- coding=utf-8 -*-

"""
created on Nov,20 2016
"""

from __future__ import division
import pylab as pl
import numpy as np

class billibiard:
    def __init__(self, r=10, alpha=0.1, x_0=0, y_0=8, vx_0=0.1, vy_0=0, dt=1, span=600):
        self.r, self.alpha, self.x, self.y = r, alpha, [x_0], [y_0] 
        self.vx, self.vy, self.dt, self.t, self.span, self.v = [vx_0], [vy_0], dt, [0], span, np.sqrt(vx_0**2+vy_0**2)
    def calculate(self):
        def predict(x, y, vx, vy):
            k, nxt_x = vy/vx, (vx>0 and self.r or -self.r)
            nxt_y = k*(nxt_x-x)+y
            if np.abs(nxt_y) < self.r*self.alpha:
                return nxt_x, nxt_y, -vx, vy
            else:
                nxt_x = (k*(k*x-y-self.r*self.alpha*np.abs(y)/y)+np.abs(nxt_x)/nxt_x*np.sqrt(k**2*(k*x-y-self.r*self.alpha*np.abs(y)/y)**2 -(1+k**2)*((k*x-y-self.r*self.alpha*np.abs(y)/y)**2-self.r**2))) / (1+k**2)
                nxt_y = k*(nxt_x-x)+y
                nxt_vx, nxt_vy = vx-2*(vx*nxt_x+vy*nxt_y)/self.v/self.r*vx, vy-2*(vx*nxt_x+vy*nxt_y)/self.v/self.r*vy
                return nxt_x, nxt_y+self.r*self.alpha*np.abs(nxt_y)/nxt_y, nxt_vx, nxt_vy
        while self.t[-1] < self.span:
            if abs(self.x[-1])<=np.abs(predict(self.x[-1], self.y[-1], self.vx[-1], self.vy[-1])[0]):
                self.x.append(self.x[-1]+self.vx[-1]*self.dt)
                self.y.append(self.y[-1]+self.vy[-1]*self.dt)
                self.vx.append(self.vx[-1])
                self.vy.append(self.vy[-1])
                self.t.append(self.t[-1] + self.dt)
                print self.t[-1]
            else:
                self.x[-1]=predict(self.x[-2], self.y[-2], self.vx[-2], self.vy[-2])[0]
                self.y[-1]=predict(self.x[-2], self.y[-2], self.vx[-2], self.vy[-2])[1]
                self.vx[-1]=predict(self.x[-2], self.y[-2], self.vx[-2], self.vy[-2])[2]
                self.vy[-1]=predict(self.x[-2], self.y[-2], self.vx[-2], self.vy[-2])[3]

#######  不如来画图  #########
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
zz = billibiard()
zz.calculate()
pl.plot(zz.x, zz.y)
pl.show()
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
