# Chaos in the Driven Nonlinear Pendulum

## Abstract
With **Euler-Cromer Method** and **bisection method**, it is easy to achieve a better solution to the cannon strike problem. However **algorithm complexity analysis** and **time complicity analysis** are very necessary to improve efficiency.

## Background

### Euler-Cromer method
> From [Wikipedia](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method), the free encyclopedia

In mathematics, the semi-implicit Euler method, also called symplectic Euler, semi-explicit Euler, Euler–Cromer, and Newton–Størmer–Verlet (NSV), is a modification of the Euler method for solving Hamilton's equations, a system of ordinary differential equations that arises in classical mechanics. It is a symplectic integrator and hence it yields better results than the standard Euler method.

#### Setting
The semi-implicit Euler method can be applied to a pair of differential equations of the form

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/42db9fc0c412f7eb6cb328dbab614907f839c4d5)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/38faa28a39c42d118cf177c5dc567852d7457cf8)

where f and g are given functions. Here, x and v may be either scalars or vectors. The equations of motion in Hamiltonian mechanics take this form if the Hamiltonian is of the form

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/bbf7bda8bb37b7e8b54c00ed8121bb740a1a13d7)

The differential equations are to be solved with the initial condition

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/0ae7811a6d4e435f51969df2000ab5e7a1e44e75)

#### The method
The semi-implicit Euler method produces an approximate discrete solution by iterating

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/96b091ba914ac916145d3b4d9361fd16d0842531)

where Δt is the time step and tn = t0 + nΔt is the time after n steps.

The difference with the standard Euler method is that the semi-implicit Euler method uses vn+1 in the equation for xn+1, while the Euler method uses vn.

Applying the method with negative time step to the computation of {\displaystyle (x_{n},v_{n})} (x_n,v_n) from {\displaystyle (x_{n+1},v_{n+1})} (x_{n+1},v_{n+1}) and rearranging leads to the second variant of the semi-implicit Euler method

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/5345b72f06cf73c459fa753eb60f46b693f1dfbf)

which has similar properties.

The semi-implicit Euler is a first-order integrator, just as the standard Euler method. This means that it commits a global error of the order of Δt. However, the semi-implicit Euler method is a symplectic integrator, unlike the standard method. As a consequence, the semi-implicit Euler method almost conserves the energy (when the Hamiltonian is time-independent). Often, the energy increases steadily when the standard Euler method is applied, making it far less accurate.

![](https://upload.wikimedia.org/wikipedia/commons/f/f9/Symplectic_Euler_stability_region.jpeg)

### Chaos theory
> From [Wikipedia](https://en.wikipedia.org/wiki/Chaos_theory), the free encyclopedia

Chaos theory is the field of study in mathematics that studies the behavior of dynamical systems that are highly sensitive to initial conditions—a response popularly referred to as the butterfly effect. Small differences in initial conditions (such as those due to rounding errors in numerical computation) yield widely diverging outcomes for such dynamical systems, rendering long-term prediction impossible in general. This happens even though these systems are deterministic, meaning that their future behavior is fully determined by their initial conditions, with no random elements involved. In other words, the deterministic nature of these systems does not make them predictable. This behavior is known as deterministic chaos, or simply chaos. The theory was summarized by Edward Lorenz as:

> Chaos: When the present determines the future, but the approximate present does not approximately determine the future.

Chaotic behavior exists in many natural systems, such as weather and climate. It also occurs spontaneously in some systems with artificial components, such as road traffic. This behavior can be studied through analysis of a chaotic mathematical model, or through analytical techniques such as recurrence plots and Poincaré maps. Chaos theory has applications in several disciplines, including meteorology, sociology, physics, environmental science, computer science, engineering, economics, biology, ecology, and philosophy.

![](https://upload.wikimedia.org/wikipedia/commons/4/45/Double-compound-pendulum.gif)

#### Strange attractors
Some dynamical systems, like the one-dimensional logistic map defined by x → 4 x (1 – x), are chaotic everywhere, but in many cases chaotic behavior is found only in a subset of phase space. The cases of most interest arise when the chaotic behavior takes place on an attractor, since then a large set of initial conditions leads to orbits that converge to this chaotic region.

An easy way to visualize a chaotic attractor is to start with a point in the basin of attraction of the attractor, and then simply plot its subsequent orbit. Because of the topological transitivity condition, this is likely to produce a picture of the entire final attractor, and indeed both orbits shown in the figure on the right give a picture of the general shape of the Lorenz attractor. This attractor results from a simple three-dimensional model of the Lorenz weather system. The Lorenz attractor is perhaps one of the best-known chaotic system diagrams, probably because it was not only one of the first, but it is also one of the most complex and as such gives rise to a very interesting pattern, that with a little imagination, looks like the wings of a butterfly.

Unlike fixed-point attractors and limit cycles, the attractors that arise from chaotic systems, known as strange attractors, have great detail and complexity. Strange attractors occur in both continuous dynamical systems (such as the Lorenz system) and in some discrete systems (such as the Hénon map). Other discrete dynamical systems have a repelling structure called a Julia set, which forms at the boundary between basins of attraction of fixed points. Julia sets can be thought of as strange repellers. Both strange attractors and Julia sets typically have a fractal structure, and the fractal dimension can be calculated for them.

![](https://en.wikipedia.org/wiki/File:TwoLorenzOrbits.jpg)

#### Minimum complexity of a chaotic system
Discrete chaotic systems, such as the logistic map, can exhibit strange attractors whatever their dimensionality. In contrast, for continuous dynamical systems, the Poincaré–Bendixson theorem shows that a strange attractor can only arise in three or more dimensions. Finite-dimensional linear systems are never chaotic; for a dynamical system to display chaotic behavior, it must be either nonlinear or infinite-dimensional.

The Poincaré–Bendixson theorem states that a two-dimensional differential equation has very regular behavior. The Lorenz attractor discussed above is generated by a system of three differential equations such as:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2c45f902abf0c197d306843bbd84e9aeffae4adb)

## Problem Description
Here is the problem in the book.

> 2.10. Generalize the program developed for the previous problem so that it can deal with situations in which the target is at a different altitude than the cannon. Consider cases in which the target is higher and lower thatn the cannon. Also investgate how the minimum firing velocity required to hit the target varies as the altitude of the target is varied.

What's more, we should also find the velocity and the firing angle to strike the target precisely and search for the minimum velocity the hit the aim.

## Main
### Problem Analysis
Apart from finding velocity and angle, there is nothing else to add to the previous problem.  

Here I use the **Midpoint method** to get a more accurate solution and in fact it works as sopposed. I have introduced this method above but the difficulty is that there are more than two variables(read my code to get my idea).  
 
As for pinpoint strike and minimun speed, **bisection method** is not a bad choice. We must take efficiency into considerarion because  the models are not easy to maintain as they grow in size and complexity over many iterations.

![](http://images2015.cnblogs.com/blog/828056/201601/828056-20160110145045871-2120864610.png)
I use the smallest tangent distance to determinate the falling point.

## Results
### Level 1: Add the wind drag
trojectories under different wind speed:
![](./windspeed.png)

### Level 2: Strike the target
As you can see, each target is hit accurately(the precision up to 1e-8 m!) but it calculates quickly.

![](./strike.png)

### Strike the target from different firing angles
![](./strikes.png)

### Firing velocity versus firing angle
![](./vtheta.png)

## Discussion
There is no doubt that we prefer a analytic solution to a numerical solution. Why not try it?

## Acknowledgement
1. Thanks to **_John Hunter_**!  
If you have benefited from John's many contributions, please say thanks in the way that would matter most to him. Please consider making a donation to the <a href="http://numfocus.org/johnhunter/">John Hunter Technology Fellowship</a>
2. Thanks to **_Wikipedia_**! I copy too much from you.  
[This year, please consider making a donation of 50, 75, 100 yuan or whatever you can to protect and sustain Wikipedia.](https://donate.wikimedia.org/w/index.php?title=Special:FundraiserLandingPage&country=CN&uselang=en&utm_medium=sidebar&utm_source=donate&utm_campaign=C13_en.wikipedia.org)

