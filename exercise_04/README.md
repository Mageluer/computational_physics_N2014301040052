# Solutions to A Nuclei Decay Problem with Eluer Method

## Abstract
Euler method is a useful general algorithm for solving ordinary differential equations. Here we apply this method to abtain a numerical solution to the radioactive decay problem.

## Background
### Eluer Method
In mathematics and computational science, the Euler method is a first-order numerical procedure for solving ordinary differential equations (ODEs) with a given initial value. It is the most basic explicit method for numerical integration of ordinary differential equations and is the simplest Runge–Kutta method. The Euler method is named after Leonhard Euler, who treated it in his book Institutionum calculi integralis (published 1768–70).

The Euler method is a first-order method, which means that the local error (error per step) is proportional to the square of the step size, and the global error (error at a given time) is proportional to the step size. The Euler method often serves as the basis to construct more complex methods, e.g., Predictor–corrector method.

### Problem Description
Consider again a decay problem with two types of nuclei A and B, but now suppose that nuclei of type A decay into ones of type B, while nuclei of type B decay into type A. Strictly speaking, this is not a "decay" process, since it is possible for the type B nuclei to turn back into type A nuclei. A better analogy would be a resonance in which a system can tunnel or move back and forth between two states A and B which have equal energies. The corresponding rate equations are

![](http://latex.codecogs.com/gif.latex?%5Cfrac%7BdN_A%7D%7Bdt%7D%20%3D%20%5Cfrac%7BN_B%7D%7B%5Ctau%7D%20-%20%5Cfrac%7BN_A%7D%7B%5Ctau%7D%2C)

![](http://latex.codecogs.com/gif.latex?%5Cfrac%7BdN_B%7D%7Bdt%7D%20%3D%20%5Cfrac%7BN_A%7D%7B%5Ctau%7D%20-%20%5Cfrac%7BN_B%7D%7B%5Ctau%7D%2C)

where for simplicity we have assumed that the two types of decay are characterized by the same time constant, <img src="http://latex.codecogs.com/gif.latex?\tau." alt="" title="" /> Solve this system of equations for the numbers of nuclei, <img src="http://latex.codecogs.com/gif.latex?N_A" alt="" title="" /> and <img src="http://latex.codecogs.com/gif.latex?N_B" alt="" title="" />, as functions of time. Consider different initial conditions, such as ![](http://latex.codecogs.com/gif.latex?N_A%20%3D%20100%2C) ![](http://latex.codecogs.com/gif.latex?N_B%20%3D%200%2C) etc., and take ![](http://latex.codecogs.com/gif.latex?%5Ctau%20%3D%201) s. Show that your numerical results are consistent with the idea that the system reaches a stteady state in which <img src="http://latex.codecogs.com/gif.latex?N_A" alt="" title="" /> and <img src="http://latex.codecogs.com/gif.latex?N_B" alt="" title="" /> are constant. In such a steady state, the time derivatives <img src="http://latex.codecogs.com/gif.latex?dN_A/dt" alt="" title="" /> and <img src="http://latex.codecogs.com/gif.latex?dN_B/dt" alt="" title="" /> should vanish.

## Main
### Problem Analysis
It is easy to see 

<img src="http://latex.codecogs.com/gif.latex?\frac{d(N_A+N_B)}{dt}=0" />.

Thus, we get <img src="http://latex.codecogs.com/gif.latex?N_A+N_B{\equiv}N_{A0}+N_{B0}" /> and <img src="http://latex.codecogs.com/gif.latex?\frac{dN_A}{dt}=\frac{N-2N_A}{\tau}" /> by apply <img src="http://latex.codecogs.com/gif.latex?N_B=N-N_A" /> where <img src="http://latex.codecogs.com/gif.latex?N" /> is the sum of nuclei number. Then, we get

<img src="http://latex.codecogs.com/gif.latex?N_A(t+\delta{t})=N_A(t)+\frac{N-2N_A(t)}{\tau}\delta{t}" />

and the solution is around the corner.

### Theoretical Solution
Solve the simple ordinary equations we can get analytic solutions

<img src="http://latex.codecogs.com/gif.latex?N_A(t)=\frac{1}{2}(N_{A0}+N_{B0})+\frac{1}{2}(N_{A0}-N_{B0})e^{\frac{-2t}{\tau}}" alt="" title="" />   
<img src="http://latex.codecogs.com/gif.latex?N_A(t)=\frac{1}{2}(N_{A0}+N_{B0})-\frac{1}{2}(N_{B0}-N_{A0})e^{\frac{-2t}{\tau}}" alt="" title="" />

which provides a good comparison to numerical solutions.

## Results
<img src="http://latex.codecogs.com/gif.latex?N_A=100" alt="" title="" />;<img src="http://latex.codecogs.com/gif.latex?N_B=0" alt="" title="" />;<img src="http://latex.codecogs.com/gif.latex?\tau=1(s)" alt="" title="" /> 

![](figure_1.png)

We can try different values of <img src="http://latex.codecogs.com/gif.latex?N_{A0},N_{B0},\tau,\delta{t}" alt="" title="" />. Obviously the smaller <img src="http://latex.codecogs.com/gif.latex?\delta{t}" alt="" title="" /> is and the more precise the result will be.

## Discussion
There is no doubt that we preffer a analytic solution to a numerical solution. Why not try it?

## Acknowledgement
Thanks to **_John Hunter_**!  
If you have benefited from John's many contributions, please say thanks in the way that would matter most to him. Please consider making a donation to the <a href="http://numfocus.org/johnhunter/">John Hunter Technology Fellowship</a>
