# Waves on the String and Frequency Spectrum

# 1. Abstract
Here the particular case of waves on a string is considered. At the beginning, a solution for the wave equation in the ideal case is introduced and developed, that is, for a perfectly flexible and frictionless string. Only one initial Gaussian wave packet and two initial Gaussian wave packets are considered to show that the wave packets are unaffected by the collisions. Besides, Fourier analysis is applied in the spectral analysis to exam the waves on a string.  

# 2. Background
Suppose the total length of the string is 1 unit.  
For Gaussian Wavepackte, the displacement of the string can be written as  
![](http://latex.codecogs.com/gif.latex?y_o%28x%29%3Dy_0%28i%5CDelta%20x%29%3DA%5Ctimes%20exp%5B-k%5Ctimes%28x-x_%7Bexcite%7D%29%5E2%5D)  
where the parameters ![](http://latex.codecogs.com/gif.latex?k%2Cx_%7Bobserve%7D) influence the width and center of the wavepacket respectively.  
A more realistic initial wavepacket is composed of two straight lines, that is:  
![](http://latex.codecogs.com/gif.latex?y_0%28x%29%3Dy_0%28i%5CDelta%20x%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x/x_%7Bexcite%7D%260%3Cx%3C%20x_%7Bexcite%7D%20%5C%5C%20%28x-1%29/%28x_%7Bexcite%7D-1%29%20%26%20x_%7Bexcite%7D%3Cx%3C1%20%5Cend%7Bmatrix%7D%5Cright.).  



# 3. Methodology and Solutions
## 3.1 Numerical Solution to the Waves in the Ideal Case
The central equation of wave motion is  
![]( http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%5E2%20y%7D%7B%5Cpartial%20t%5E2%7D%3Dc%5E2%5Cfrac%7B%5Cpartial%5E2%20y%7D%7B%5Cpartial%20x%5E2%7D)  
where ![]( http://latex.codecogs.com/gif.latex?c%5E2%3D%5Cfrac%7BT%7D%7B%5Crho%20%7D) is the ratio of the tension in the string to the density per length.  
To solve the time-dependent solution ![]( http://latex.codecogs.com/gif.latex?y%28x%2Ct%29), the wave equation should be attacked with rather different numerical treatments than those employed in the work with Laplace’s equation. The numerical approach can be written as follows.  
The variables are treated as discrete ones as ![]( http://latex.codecogs.com/gif.latex?x%3Di%5CDelta%20x%2Ct%3Dn%5CDelta%20t). The displacement of the string is a function of i and n, that is, ![]( http://latex.codecogs.com/gif.latex?y%28i%2Cn%29%5Cequiv%20y%28x%3Di%5CDelta%20x%2Ct%3Dn%5CDelta%20t%29). Inserting the expression for the second partial derivative, the wave equation can be rewritten as  
![]( http://latex.codecogs.com/gif.latex?%5Cfrac%7By%28i%2Cn&plus;1%29&plus;y%28i%2Cn-1%29-2y%28i%2Cn%29%7D%7B%28%5CDelta%20t%29%5E2%7D%5Capprox%20c%5E2%5B%5Cfrac%7By%28i&plus;1%2Cn%29&plus;y%28i-1%2Cn%29-2y%28i%2Cn%29%7D%7B%28%5CDelta%20x%29%5E2%7D%5D)  
Rearranging the above equation, we have  
![]( http://latex.codecogs.com/gif.latex?y%28i%2Cn&plus;1%29%3D2%281-r%5E2%29y%28i%2Cn%29-y%28i%2Cn-1%29&plus;r%5E2%5By%28i&plus;1%2Cn%29&plus;y%28i-1%2Cn%29%5D)  
Where ![]( http://latex.codecogs.com/gif.latex?r%5Cequiv%20c%5CDelta%20t/%5CDelta%20x). Thus if we know the string configuration as time steps n and n-1, the configuration at step n+1 can be calculated.
The boundary condition is ![]( http://latex.codecogs.com/gif.latex?y%280%2Cn%29%3Dy%28M%2Cn%29%5Cequiv%200) and the initial condition is ![]( http://latex.codecogs.com/gif.latex?y%28i%2C0%29%3Dy_o%28i%29).  

## 3.2 Routine Propagate for Simulating Waves on a String
+ Set the parameter combination ![]( http://latex.codecogs.com/gif.latex?r%5Cequiv%20c%5CDelta%20t/%5CDelta%20x).  
+ Loop through the interior points ![]( http://latex.codecogs.com/gif.latex?i%3D1) through ![]( http://latex.codecogs.com/gif.latex?i%3DM-1). Update according to  
>  ![]( http://latex.codecogs.com/gif.latex?y%28i%2Cn&plus;1%29%3D2%281-r%5E2%29y%28i%2Cn%29-y%28i%2Cn-1%29&plus;r%5E2%5By%28i&plus;1%2Cn%29&plus;y%28i-1%2Cn%29%5D)  
+ The ends at ![]( http://latex.codecogs.com/gif.latex?i%3D0) and ![]( http://latex.codecogs.com/gif.latex?i%3DM) are fixed as zero for all time steps.  

## 3.3 Power Spectrum
If we are interested in the frequencies ad relative amplitudes of the Fourier components but do not care about their phases, power spectrum is a useful way to display the results of an FFT.  
Formally, the power spectrum of a function ![]( http://latex.codecogs.com/gif.latex?y%28t%29) can be defined as the Fourier transformation of its autocorrelation  
![]( http://latex.codecogs.com/gif.latex?Corr%5By%5D%28t%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7Dy%28t%29%5E%5Cast%20y%28t&plus;%5Ctau%29d%5Ctau)  
And the power spectrum is given by  
![]( http://latex.codecogs.com/gif.latex?PS%5By%5D%28f%29%3D%5Cint_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7Dy%28t%29%5E%5Cast%20y%28t&plus;%5Ctau%29e%5E%7B2%5Cpi%20if%5Ctau%7Dd%5Ctau%3D%5Cleft%20%7C%20Y%28f%29%20%5Cright%20%7C%5E2)  


# 4. Code
Code for [Displacement of Waves on An Ideal String](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/14.1.py)&[Animation of Displacements of two different Initial Wavepackets and their combination](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/14.2.py)  
Code for [Frequency Spectrum of Waves on a String](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/14.3.py)  

# 5. Running and Analysis
## 5.1  Displacement of Waves on An Ideal String
If the initial wave packet is ![](http://latex.codecogs.com/gif.latex?y_0%28x%29%3De%5E%7B-1000%5Ctimes%28x-0.3%29%5E2%7D), the result is:  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_1.png)  
The motion that changes with time step can be shown below:  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/gif/anim1.gif)  

If the initial wave packet is ![](http://latex.codecogs.com/gif.latex?y_0%28x%29%3D-2e%5E%7B-300%5Ctimes%28x-0.6%29%5E2%7D), the result is:  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_2.png)  
The motion that changes with time step can be shown below:  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/gif/anim2.gif)  

If the initial wave packet is ![](http://latex.codecogs.com/gif.latex?y_0%28x%29%3De%5E%7B-1000%5Ctimes%28x-0.3%29%5E2%7D-2e%5E%7B-300%5Ctimes%28x-0.6%29%5E2%7D), the result is:  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_3.png)  
The motion that changes with time step can be shown below:  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/gif/anim3.gif)  

Besides, we can draw them together, here we have:  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/gif/anim4.gif)  

From the results above, we can draw the conclusion that when there are two Gaussian wave packets located at different places on the string, the wave packets may then propagate and collide but the wave packets are unaffected by the collisions.  

## 5.2Frequency Spectrum of Waves on a Gaussian-excited String
Suppose the initial wavepacket is ![](http://latex.codecogs.com/gif.latex?y_o%28x%29%3Dexp%5B-1000%5Ctimes%28x-x_%7Bexcite%7D%29%5E2%5D). And the total length of the string is 1 unit.  
When ![](http://latex.codecogs.com/gif.latex?x_%7Bexcite%7D%3D0.5) and ![](http://latex.codecogs.com/gif.latex?x_%7Bobserve%7D%3D0.05)  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_4.png)  

When ![](http://latex.codecogs.com/gif.latex?x_%7Bexcite%7D%3D0.5) and ![](http://latex.codecogs.com/gif.latex?x_%7Bobserve%7D%3D0.1)  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_5.png)  

When ![](http://latex.codecogs.com/gif.latex?x_%7Bexcite%7D%3D0.5) and ![](http://latex.codecogs.com/gif.latex?x_%7Bobserve%7D%3D0.05%2C0.1%2C0.2%2C0.3), their corresponding power spectrum is  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_6.png)  

When ![](http://latex.codecogs.com/gif.latex?x_%7Bexcite%7D%3D0.45) and ![](http://latex.codecogs.com/gif.latex?x_%7Bobserve%7D%3D0.05%2C0.1%2C0.2%2C0.3), their corresponding power spectrum is  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_7.png)  

because ![](http://latex.codecogs.com/gif.latex?%5Clambda%20f%3Dc%2C%5Clambda%20%3D2L/m%2Cm%5Cin%20N), we have the possible frequencies as ![](http://latex.codecogs.com/gif.latex?f%3Dmc/%282L%29%3D150mHz). This explains why the peaks in the spectral analysis in the above figures occur at regularly spaced frequencies. Each of the peaks correspond to one value of interger m. But some frequancies are missing and this can be traced to the operation of Fourier Transformation.  
Besides, the symmetry of the initial wavepacket can cause certain frequencies to be supressed. The power spectrum of ![](http://latex.codecogs.com/gif.latex?x_%7Bexcite%7D%3D0.45) can illustrate this clearly.  


## 5.3Frequency Spectrum of Waves on a String with Two Straight Lines At Start
The initial wavepacket is  
![](http://latex.codecogs.com/gif.latex?y_0%28x%29%3Dy_0%28i%5CDelta%20x%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x/x_%7Bexcite%7D%260%3Cx%3C%20x_%7Bexcite%7D%20%5C%5C%20%28x-1%29/%28x_%7Bexcite%7D-1%29%20%26%20x_%7Bexcite%7D%3Cx%3C1%20%5Cend%7Bmatrix%7D%5Cright.).  
  

When ![](http://latex.codecogs.com/gif.latex?x_%7Bexcite%7D%3D0.5) and ![](http://latex.codecogs.com/gif.latex?x_%7Bobserve%7D%3D0.05)  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_8.png)  

When ![](http://latex.codecogs.com/gif.latex?x_%7Bexcite%7D%3D0.5) and ![](http://latex.codecogs.com/gif.latex?x_%7Bobserve%7D%3D0.05%2C0.1%2C0.2%2C0.3), their corresponding power spectrum is  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_9.png)  

When ![](http://latex.codecogs.com/gif.latex?x_%7Bexcite%7D%3D0.45) and ![](http://latex.codecogs.com/gif.latex?x_%7Bobserve%7D%3D0.05%2C0.1%2C0.2%2C0.3), their corresponding power spectrum is  
![](https://github.com/JunyiShangguan/computationalphysics_N2013301020076/blob/master/ex14/figure_10.png)  

There exists one frequency with which the power is greatly excessive of others.  

## Discussion
1. Numberical solution makes approximation every step, but in chaos an arbitrarily small change, or perturbation, of the current trajectory may lead to significantly different future behavior. Any other approaches to this problem other than numerical method?
2. If we pick arbitrarily two mechanical quantities of a chaos, the patterns are different. Any better quantity or worse quantity?

## Acknowledgement
1. Thanks to **_John Hunter_**!  
If you have benefited from John's many contributions, please say thanks in the way that would matter most to him. Please consider making a donation to the <a href="http://numfocus.org/johnhunter/">John Hunter Technology Fellowship</a>
2. Thanks to **_Wikipedia_**! I copy too much from you.  
[This year, please consider making a donation of 50, 75, 100 yuan or whatever you can to protect and sustain Wikipedia.](https://donate.wikimedia.org/w/index.php?title=Special:FundraiserLandingPage&country=CN&uselang=en&utm_medium=sidebar&utm_source=donate&utm_campaign=C13_en.wikipedia.org)

