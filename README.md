# Overview
Module for Kernel-Density Estimation (KDE) on unit sphere. Created for [^1].

Use a Fisher distribution (i.e. Fisher-Von-Mises with p=3) as kernel function. No bandwidth selection. Calculation can be performed
in chunks to reduce max. memory requirements. 

# Coordinates

Coordinates on the sphere must be specified in spherical polar coordinates. The polar angle (theta) must be in the range [0,PI], with
0 at the north pole, and the azimuthal angle (phi) should be in range [0,2*PI]. Obviously both should be in units of radians. 

# Unit test

For a uniform distribution on the sphere, 

$f(\Omega)d\Omega = \frac{1}{4\pi} d\Omega $, 

an analytic expression for the Mean Integrated Squared Error (MISE) can be found. 


[^1]: Russell M.B., Johnson, C.L., Gilchrist, S.A.: 2024,"Investigating the Spatial Relationship of Shield Volcanism with Coronae on Venus",55th Lunar and Planetary Science Conference,3040,
      URL: https://ui.adsabs.harvard.edu/abs/2024LPICo3040.2582R
