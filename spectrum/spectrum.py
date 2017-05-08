# 绘制两个能级之间越迁（塞曼分裂）
import numpy as np
import matplotlib.pyplot as pl

def dec2frac(dec, signed=True):
    if abs( round(dec) - dec ) < 1e-6:
        return "%+d" % round(dec) if signed else "%d" % round(dec)
    for i in range(2, int(1e5)):
        if abs( round(i*dec) - i*dec ) < 1e-6:
            return "%+d/%d" % (round(i*dec), i) if signed else "%d/%d" % (round(i*dec), i)
    return "%+.4f" % dec if signed else "%.4f" % dec

l_sym_ = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J']
while True:
    n2 = int(input("n of the upper level(1,2,3...): "))
    l2 = int(input("l of the upper level(0,1,2...): "))
    n1 = int(input("n of the lower level(1,2,3...): "))
    l1 = int(input("l of the lower level(0,1,2...): "))
    if not abs(l2-l1) == 1:
    # check if l2-l1 == -1 or +1
        print("You should remember |l2-l1|=1!\nInput them all again!")
    elif l1 > 7 or l2 > 7:
        print("Are you kidding? l > 7?\nInput them all again!")
    else:
        break
n_ = [n1, n2]
l_ = [l1, l2]
s = 1/2
j_ = []
m_ = []
g_ = []
mg_ = []
delta_m_ = []
delta_mg_ = []
mg_jump_ = []
j_base_ = []
j_base_sym_ = []
j_base_expansion = np.sqrt(2)

# get j_, j_base_sym_
for l in l_:
    j = []
    j_base_sym = []
    if l == 0:
        j.append(s)
        j_base_sym.append(r'$^2$%c$_{%s}$' % (l_sym_[l], dec2frac(s, False)))
    else:
        j = [l-s, l+s]
        j_base_sym = [r'$^2$%c$_{%s}$' % (l_sym_[l], dec2frac(l-s, False)), r'$^2$%c$_{%s}$' % (l_sym_[l], dec2frac(l+s ,False))]
    j_.append(j)
    j_base_sym_.append(j_base_sym)

# get m_
for j in j_:
    jm_ = []
    for i in j:
        m = list(np.linspace(-i, i, 2*i+1))
        jm_.append(m)
    m_.append(jm_)

# get g_
for l, j in zip(l_, j_):
    g = [3/2 + 1/2 * (s*(s+1) - l*(l+1)) / (i*(i+1)) for i in j]
    g_.append(g)

# get mg_
for m, g in zip(m_, g_):
    mg = [[i*gk for i in mk] for mk, gk in zip(m, g)]
    mg_.append(mg)

# get j_base_
j_base_tmp = 0
tmp = 0
for mg in mg_:
    j_base = []
    for i in mg:
        tmp = (max(i) - min(i)) * j_base_expansion / 2
        j_base_tmp += tmp
        j_base.append(j_base_tmp)
        j_base_tmp += tmp
    j_base_.append(j_base)

# get delta_m_, delta_mg_ and mg_jump_
mg1_, mg2_ = mg_[0], mg_[1]
m1_, m2_ = m_[0], m_[1]
j1_, j2_ = j_[0], j_[1]
j_base1_, j_base2_ = j_base_[0], j_base_[1]
for mg1, m1, j1, j_base1 in zip(mg1_, m1_, j1_, j_base1_):
    for mg2, m2, j2, j_base2 in zip(mg2_, m2_, j2_, j_base2_):
        mg_jump = []
        delta_m = []
        delta_mg = []
        for i, j in zip(mg1, m1):
            for k, l in zip(mg2, m2):
                if abs(j-l) <= 1 and abs(j1-j2) <= 1:
                    mg_jump.append([i+j_base1, k+j_base2])
                    delta_m.append((l - j))
                    delta_mg.append(k - i)
        # rearrange
        for i in range(len(delta_mg) - 1):
            for j in range(i, len(delta_mg)):
                if delta_mg[i] > delta_mg[j]:
                    delta_mg[i], delta_mg[j] = delta_mg[j], delta_mg[i]
                    delta_m[i], delta_m[j] = delta_m[j], delta_m[i]
                    mg_jump[i], mg_jump[j] = mg_jump[j], mg_jump[i]
        if delta_m :
            mg_jump_.append(mg_jump)
            delta_m_.append(delta_m)
            delta_mg_.append(delta_mg)

# set horizon coordinate
horizon_ = [0, j_base_tmp/4, j_base_tmp/3, j_base_tmp]
horizon_nonmagnet_d = j_base_tmp / 4 / (len(delta_mg_)+1)
horizon_nonmagnet_ = []
horizon_nonmagnet_tmp = 0
horizon_sep = j_base_tmp / 9 / (len(delta_mg_) )
horizon_d = j_base_tmp * 5/9 / (sum([len(i) for i in delta_mg_]))
delta_mg_horizon_ = []
horizon_tmp = horizon_[2]

for i, j1 in zip(j_base_[0], j_[0]):
    for j ,j2 in zip(j_base_[1], j_[1]):
        if abs(j1 - j2) <= 1:
            horizon_nonmagnet_tmp += horizon_nonmagnet_d
            horizon_nonmagnet_.append(horizon_nonmagnet_tmp)
for i in delta_mg_:
    delta_mg_horizon = []
    horizon_tmp += horizon_sep
    for j in i:
        horizon_tmp += horizon_d
        delta_mg_horizon.append(horizon_tmp)
    delta_mg_horizon_.append(delta_mg_horizon)

# just draw it
fig = pl.figure(figsize=(8,6), dpi=72,facecolor="white")
axes = pl.subplot(111)
for j_base, j_base_sym, m, mg in zip(j_base_, j_base_sym_, m_, mg_):
    for i, jbs, l, j in zip(j_base, j_base_sym, m, mg):
        pl.plot(horizon_[0:2], [i, i])
        pl.text(horizon_[0], i, jbs, size='large')
        for p, k in zip(l, j):
            pl.plot(horizon_[1:3], [i,i+k],'--')
            pl.plot(horizon_[2:], [i+k,i+k])
            pl.text(horizon_tmp * 21 / 20, i+k, dec2frac(p))
            pl.text(horizon_tmp * 22 / 20, i+k, dec2frac(k))
pl.text(horizon_tmp * 21 / 20, j_base_[-1][-1] + 3/2*mg_[-1][-1][-1] - 1/2*mg_[-1][-1][-2], 'm')
pl.text(horizon_tmp * 22 / 20, j_base_[-1][-1] + 3/2*mg_[-1][-1][-1] - 1/2*mg_[-1][-1][-2], 'mg')

for i, j1 in zip(j_base_[0], j_[0]):
    for j, j2 in zip(j_base_[1], j_[1]):
        if abs(j1 - j2) <= 1:
            k = horizon_nonmagnet_.pop(0)
            pl.plot([k, k], [i, j])

for delta_mg_horizon, mg_jump, delta_m, delta_mg in zip(delta_mg_horizon_, mg_jump_, delta_m_, delta_mg_):
    for i, j, k, l in zip(delta_mg_horizon, mg_jump, delta_m, delta_mg):
        pl.plot([i,i], j)
        if k > 0:
            pl.text(i, - horizon_tmp / 20, '$\sigma^+$')
        elif k < 0:
            pl.text(i, - horizon_tmp / 20, '$\sigma^-$')
        else:
            pl.text(i, - horizon_tmp / 20, '$\pi$')
        pl.plot([i,i], [- horizon_tmp / 15, - horizon_tmp / 5])
        pl.text(i, - horizon_tmp / 4, dec2frac(l), rotation='vertical')
pl.text(2*delta_mg_horizon_[0][0] - delta_mg_horizon_[0][1], - horizon_tmp / 4, r'$mg_2-mg_1$', ha='right')

axes.set_xlim(0, horizon_tmp * 24 / 20)
axes.set_ylim(- horizon_tmp / 2.8, horizon_tmp * 22 / 20)
axes.set_xticks([])
axes.set_yticks([])
pl.title(r"%d%c$\to$%d%c" % (n_[1], l_sym_[l_[1]], n_[0], l_sym_[l_[0]]))
pl.show()
