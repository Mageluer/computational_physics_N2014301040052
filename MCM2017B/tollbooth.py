import numpy as np
import math
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import json

class TollBooth(object):
    def __init__(self, L_lane=6, L_booth=12, L_queue_in=20, L_queue_out=5, angle_in=1/8, angle_out=1/6, dt=0.45, time_span=600, tau=60, mu=3.5, sigma=1.08):
        self.L_lane, self.L_booth = L_lane, L_booth
        self.L_queue_in, self.L_queue_out = L_queue_in, L_queue_out
        self.angle_in, self.angle_out = angle_in, angle_out
        self.dt, self.time_span = dt, time_span

        self.offset1 = np.floor((L_booth-L_lane) / 2)
        self.offset2 = L_booth - L_lane - self.offset1
        self.L_expansion, self.L_merge = np.ceil(self.offset1 / angle_in), np.ceil(self.offset2 / angle_out)
        self.A_toll = self.L_expansion + self.L_queue_in
        self.L_total, self.W_total = self.L_expansion + self.L_queue_in + 1 + self.L_queue_out + self.L_merge,\
                self.L_booth + 2
        self.Z = np.zeros((self.W_total, self.L_total))
        for i in range(int(self.W_total)):
            for j in range(int(self.L_total)):
                if (i in (0, self.W_total-1)) or \
                        (i <= - self.offset1 / (self.L_expansion-1) * j + self.offset1) or \
                        (i <= self.offset1 / (self.L_merge-1) * (j-(self.L_total-self.L_merge))) or \
                        (i >= self.offset2 / (self.L_expansion-1) * j + self.W_total-1-self.offset2) or \
                        (i >= - self.offset2 / (self.L_merge -1) * (j-(self.L_total-self.L_merge)) + self.W_total-1):
                            self.Z[i,j] = 1.5
        self.BC = self.Z.copy()
        self.plaza_area = len(np.where(self.Z==0)[0])

        self.tau = tau
        self.mu = mu
        self.sigma = sigma

        self.in_count = [0]
        self.service_count = [0]
        self.out_count = [0]
        
        self.in_condition = [[0] * self.L_lane]
        self.service_condition = [[1] * self.L_booth]

        self.conflict_count = [0]
        self.delay_count = [0]

        self.t = [0]

    def update(self):
        in_count = self.in_count[-1]
        service_count = self.service_count[-1]
        out_count = self.out_count[-1]
        in_condition = self.in_condition[-1]
        service_condition = self.service_condition[-1]
        conflict_count = self.conflict_count[-1]
        delay_count = self.delay_count[-1]

        p_in = np.exp(-self.dt/self.tau) * self.dt / self.tau
        #p_leave_service = lambda n : np.sqrt(np.pi/2) * self.sigma * math.erf((self.dt * n - self.mu)/(np.sqrt(2)*self.sigma))
        p_leave_service = lambda n : math.erf((self.dt * n)/(np.sqrt(2)*self.sigma))

        with open("transform_car.json") as f:
            transform = json.load(f)

        N = self.Z.copy()

        for i,j in zip(np.where(self.BC==0)[0], np.where(self.BC==0)[1]):
            """only consider the driver's motion rather than the road
            """
            if self.Z[i, j] >= 1:
                if j == self.L_total - 1:
                    """out
                    """
                    out_count += 1
                    N[i, j] = 0
                elif j == self.A_toll:
                    """service
                    """
                    p_leave = p_leave_service(service_condition[i-1])
                    if (np.random.choice([1,0], p=[p_leave, 1-p_leave]) == 1) and (self.Z[i, j+1] == 0):
                        N[i, j], N[i, j+1] = 0, 1
                        service_count += 1
                        service_condition[i-1] = 1
                    else:
                        service_condition[i-1] += 1
                        delay_count += 1
                elif j == 0:
                    """in
                    """
                    """only one car at the entrance moves forward in dt
                    """
                    if self.Z[i, j+1] == 0:
                        N[i, j]=in_condition[i-self.offset1-1] = in_condition[i-self.offset1-1] - 1
                        N[i, j+1] = 1
                    else:
                        delay_count += 1
                else:
                    """other position
                    """
                    if self.Z[i,j+1]==1 and self.Z[i+1,j+1]==self.Z[i-1,j+1]==0 and self.Z[i+1,j]!=1!=self.Z[i-1,j]:
                        i_change = np.random.choice([1, -1])
                        N[i, j], N[i+i_change,j+1] = 0, 1
                    else:
                        change = transform[str(self.Z[i+1,j])+str(self.Z[i+1,j+1])+str(self.Z[i,j])+str(self.Z[i,j+1])+str(self.Z[i-1,j])+str(self.Z[i-1,j+1])] 
                        N[i+1,j] += change[0][0]
                        N[i+1,j+1] += change[0][1]
                        N[i,j] += change[0][2]
                        N[i,j+1] += change[0][3]
                        N[i-1,j] += change[0][4]
                        N[i-1,j+1] += change[0][5]
                        conflict_count += change[1]
                        delay_count += change[2]
        """cars flux in from the lanes under Possion Distribution
        """
        flux_in = np.random.choice([1,0], self.L_lane, p=[p_in, 1-p_in])
        in_count += sum(flux_in)
        in_condition += flux_in
        N[self.offset1+1:self.offset1+1+self.L_lane, 0] = in_condition
        for i in range(2, int(self.W_total-2)):
            for j in range(2, int(self.L_total)):
                if N[i, j]==2:
                    i_change = np.random.choice([1, -1])
                    N[i, j], N[i+i_change, j-1] = 1, 1
        self.Z = N.copy()
        self.in_count.append(in_count)
        self.service_count.append(service_count)
        self.out_count.append(out_count)
        self.in_condition.append(in_condition)
        self.service_condition.append(service_condition)
        self.conflict_count.append(conflict_count)
        self.delay_count.append(delay_count)
        self.t.append(self.t[-1]+self.dt)

        return self

    def draw(self):
        size = np.array(self.Z.shape)
        dpi = 72.0
        figsize = size[1], size[0]
        fig = pl.figure(figsize=figsize, dpi=dpi, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xticks([])
        ax.set_yticks([])
        note = ax.text(0.05,1.4,'',fontsize=12)
        img = pl.imshow(self.Z, interpolation='nearest', cmap=pl.cm.gray_r)
        def init():
            I = self.BC.copy()
            I[1:-1, self.A_toll] = 0.2
            img.set_data(I)
            return img, note
        def animate(frame):
            self.update()
            img.set_data(self.Z)
            note.set_text('step: %s' % frame)
            return img, note
        ani = animation.FuncAnimation(fig, animate, frames=500, interval=200, init_func=init, blit=True)
        pl.show()

def test():
    tb_ = []
    B_ = []
    for B in range(6,20):
        tb = TollBooth(tau=0.05, sigma=40, L_booth=B)
        tb_.append(tb.plaza_area)
        B_.append(tb.L_booth)
    pl.plot(B_,tb_,label=r'B = %d' % B)
    pl.legend(loc="best")
    pl.title(r'price vs B')
    pl.show()

def main():
    #tb = TollBooth(tau=0.1, sigma=10, L_booth=6)
    #tb.draw()
    test()

if __name__ == '__main__':
    main()

