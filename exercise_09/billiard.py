#!usr/bin/env/ python
#-*- coding=utf-8 -*-

"""
created on Nov,20 2016
"""

from __future__ import division
import pylab as pl
import numpy as np

class billibiard:
    def __init__(self, r=10, alpha=0., x_0=0, y_0=6, vx_0=1, vy_0=0, dt=0.1, span=500):
        self.r, self.alpha, self.x, self.y = r, alpha, [x_0], [y_0] 
        self.vx, self.vy, self.dt, self.t, self.span, self.v = [vx_0], [vy_0], dt, [0], span, np.sqrt(vx_0**2+vy_0**2)
    def calculate(self):
        def predict(x, y, vx, vy):
            k, nxt_x = vy/vx, vx>0 and self.r or -self.r
            nxt_y = k*(nxt_x-x)+y
            if np.abs(nxt_y) < self.r*self.alpha:
                return nxt_x, nxt_y, -vx, vy
            else:
                nxt_x = (k*(k*x-y+abs(y)/y*self.r*self.alpha)+np.abs(vx)/vx*np.sqrt(k**2*(k*x-y+abs(y)/y*self.r*self.alpha)**2 -(1+k**2)*((k*x-y+abs(y)/y*self.r*self.alpha)**2-self.r**2))) / (1+k**2)
                nxt_y = k*(nxt_x-x)+y-abs(y)/y*self.r*self.alpha
                nxt_vx, nxt_vy = vx-2*(vx*nxt_x+vy*nxt_y)/self.r**2*nxt_x, vy-2*(vx*nxt_x+vy*nxt_y)/self.r**2*nxt_y
                return nxt_x, nxt_y+abs(nxt_y)/nxt_y*self.r*self.alpha, nxt_vx, nxt_vy
        while self.t[-1] < self.span:
            if np.abs(self.x[-1])>self.r or np.abs(self.y[-1])>self.r*self.alpha and self.x[-1]**2+(self.y[-1]-abs(self.y[-1])/self.y[-1]*self.r*self.alpha)**2 > self.r**2:
                print np.abs(self.x[-1]), self.x[-1]**2+(self.y[-1]-abs(self.y[-1])/self.y[-1]*self.r*self.alpha)**2 - self.r**2
                self.x[-1]=predict(self.x[-2], self.y[-2], self.vx[-2], self.vy[-2])[0]
                self.y[-1]=predict(self.x[-2], self.y[-2], self.vx[-2], self.vy[-2])[1]
                self.vx[-1]=predict(self.x[-2], self.y[-2], self.vx[-2], self.vy[-2])[2]
                self.vy[-1]=predict(self.x[-2], self.y[-2], self.vx[-2], self.vy[-2])[3]
                self.t[-1] = self.t[-2]+np.sqrt((self.x[-1]-self.x[-2])**2+(self.y[-1]-self.y[-2])**2)/self.v
                print "adjust  ", self.t[-1], self.x[-1],self.y[-1],self.vx[-1],self.vy[-1]
            self.x.append(self.x[-1]+self.vx[-1]*self.dt)
            self.y.append(self.y[-1]+self.vy[-1]*self.dt)
            self.vx.append(self.vx[-1])
            self.vy.append(self.vy[-1])
            self.t.append(self.t[-1] + self.dt)
            print self.t[-1], self.x[-1],self.y[-1],self.vx[-1],self.vy[-1]

#######  不如来画图  #########
zz = billibiard(alpha=0.1, y_0=3, span=5000)
zz.calculate()
"""
bdx = [zz.r*np.cos(np.pi*i/100000) for i in range(200002)]
bdy = [zz.r*np.sin(np.pi*i/100000)-abs(i%200000-100000.5)/(i%200000-100000.5)*zz.r*zz.alpha for i in range(200002)]
pl.plot(bdx,bdy,'b-')
pl.plot(zz.x, zz.y, 'g-',label=r'$\alpha$='+str(zz.alpha))
pl.gca().set_aspect(1)
pl.xlim([-zz.r*1.1,zz.r*1.1])
pl.ylim([-(zz.r+zz.r*zz.alpha)*1.1,(zz.r+zz.r*zz.alpha)*1.1])
pl.xlabel(r'x($m$)')
pl.ylabel(r'y($m$)')
pl.legend(loc='upper right',frameon = True,fontsize='small')
pl.title(r'Trajectory of the billiard ball')
pl.show()
"""
"""
zzx = [i for i,j in zip(zz.x,zz.y) if abs(j)<0.05]
zzvx = [i for i,j in zip(zz.vx,zz.y) if abs(j)<0.05]
pl.plot(zzx, zzvx, 'ok',label=r'$\alpha$='+str(zz.alpha))
pl.xlabel(r'x($m$)')
pl.ylabel(r'$v_x$($m/s$)')
pl.legend(loc='upper right',frameon = True,fontsize='small')
pl.title(r'Phase Plot')
pl.show()
"""
yy = billibiard(alpha=0.1, y_0=3+1e-4, span=5000)
yy.calculate()
distance = [np.sqrt((i-j)**2+(m-n)**2) for i,j,m,n in zip(zz.x[0:50000],yy.x[0:50000],zz.y[0:50000],yy.y[0:50000])]
pl.semilogy(zz.t[0:50000], distance, 'g-',label=r'$\alpha$='+str(zz.alpha))
pl.legend(loc='upper right',frameon = True)
pl.show()
