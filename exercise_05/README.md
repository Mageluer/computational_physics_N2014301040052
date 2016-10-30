# Solutions to the Trajectory of a Cannon Shell with Eluer Method

## Abstract
Euler method is a useful general algorithm for solving ordinary differential equations. Here we apply this method to abtain a numerical solution to the trajectory of a flying cannon shell.

## Background
### Eluer Method
In mathematics and computational science, the Euler method is a first-order numerical procedure for solving ordinary differential equations (ODEs) with a given initial value. It is the most basic explicit method for numerical integration of ordinary differential equations and is the simplest Runge???Kutta method. The Euler method is named after Leonhard Euler, who treated it in his book Institutionum calculi integralis (published 1768???70).

The Euler method is a first-order method, which means that the local error (error per step) is proportional to the square of the step size, and the global error (error at a given time) is proportional to the step size. The Euler method often serves as the basis to construct more complex methods, e.g., Predictor???corrector method.

### Projectile Motion
Projectile motion is a form of motion in which an object or particle (called a projectile) is thrown near the earth's surface, and it moves along a curved path under the action of gravity only. The implication here is that air resistance is negligible, or in any case is being neglected in all of these equations. The only force of significance that acts on the object is gravity, which acts downward to cause a downward acceleration. Because of the object's inertia, no external horizontal force is needed to maintain the horizontal velocity of the object.

### Drag
In fluid dynamics, drag (sometimes called air resistance, a type of friction, or fluid resistance, another type of friction or fluid friction) is a force acting opposite to the relative motion of any object moving with respect to a surrounding fluid. This can exist between two fluid layers (or surfaces) or a fluid and a solid surface. Unlike other resistive forces, such as dry friction, which are nearly independent of velocity, drag forces depend on velocity. Drag force is proportional to the velocity for a laminar flow and the squared velocity for a turbulent flow. Even though the ultimate cause of a drag is viscous friction, the turbulent drag is independent of viscosity.

Drag forces always decrease fluid velocity relative to the solid object in the fluid's path.

### Atmospheric models
Static atmospheric models describe how the ideal gas properties (namely: pressure, temperature, density, and molecular weight) of an atmosphere change, primarily as a function of altitude. The World Meteorological Organization defines a standard atmosphere as "a hypothetical vertical distribution of atmospheric temperature, pressure and density which, by international agreement, is roughly representative of year-round, midlatitude conditions. Typical usages are as a basis for pressure altimeter calibrations, aircraft performance calculations, aircraft and rocket design, ballistic tables, and meteorological diagrams."

For example, the US Standard Atmosphere derives the values for air temperature, pressure, and mass density, as a function of altitude above sea level.

Other static atmospheric models may have other outputs, or depend on inputs besides altitude.

### Coriolis Force
In physics, the Coriolis force is an inertial force (also called a fictitious force) that acts on objects that are in motion relative to a rotating reference frame. In a reference frame with clockwise rotation, the force acts to the left of the motion of the object. In one with anticlockwise rotation, the force acts to the right. Though recognized previously by others, the mathematical expression for the Coriolis force appeared in an 1835 paper by French scientist Gaspard-Gustave de Coriolis, in connection with the theory of water wheels. Early in the 20th century, the term Coriolis force began to be used in connection with meteorology. Deflection of an object due to the Coriolis force is called the 'Coriolis effect'.

Newton's laws of motion describe the motion of an object in an inertial (non-accelerating) frame of reference. When Newton's laws are transformed to a rotating frame of reference, the Coriolis force and centrifugal force appear. Both forces are proportional to the mass of the object. The Coriolis force is proportional to the rotation rate and the centrifugal force is proportional to its square. The Coriolis force acts in a direction perpendicular to the rotation axis and to the velocity of the body in the rotating frame and is proportional to the object's speed in the rotating frame. The centrifugal force acts outwards in the radial direction and is proportional to the distance of the body from the axis of the rotating frame. These additional forces are termed inertial forces, fictitious forces or pseudo forces. They allow the application of Newton's laws to a rotating system. They are correction factors that do not exist in a non-accelerating or inertial reference frame.

### Problem Description
This problem is very simple. Maybe baseball is more interesting.

## Main
### Problem Analysis
In mathematics, an ordinary differential equation (ODE) is a differential equation containing one or more functions of one independent variable and its derivatives. The term ordinary is used in contrast with the term partial differential equation which may be with respect to more than one independent variable.

ODEs that are linear differential equations have exact closed-form solutions that can be added and multiplied by coefficients. By contrast, ODEs that lack additive solutions are nonlinear, and solving them is far more intricate, as one can rarely represent them by elementary functions in closed form: Instead, exact and analytic solutions of ODEs are in series or integral form. Graphical and numerical methods, applied by hand or by computer, may approximate solutions of ODEs and perhaps yield useful information, often sufficing in the absence of exact, analytic solutions.

## Results
You'd better try my program.

![](https://upload.wikimedia.org/wikipedia/commons/6/63/Inclinedthrow.gif)

## Discussion
There is no doubt that we prefer a analytic solution to a numerical solution. Why not try it?

## Acknowledgement
1. Thanks to **_John Hunter_**!  
If you have benefited from John's many contributions, please say thanks in the way that would matter most to him. Please consider making a donation to the <a href="http://numfocus.org/johnhunter/">John Hunter Technology Fellowship</a>
2. Thanks to **_Wikipedia_**! I copy too much from you.  
[This year, please consider making a donation of 50, 75, 100 yuan or whatever you can to protect and sustain Wikipedia.](https://donate.wikimedia.org/w/index.php?title=Special:FundraiserLandingPage&country=CN&uselang=en&utm_medium=sidebar&utm_source=donate&utm_campaign=C13_en.wikipedia.org)
