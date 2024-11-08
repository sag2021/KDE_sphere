# Overview
Module for Kernel-Density Estimation (KDE) on unit sphere. Created for [^1].

Uses a Fisher distribution (i.e. Fisher-Von-Mises with p=3) as kernel function. No bandwidth selection. Calculation can be performed
in chunks to reduce max. memory requirements. 

# Coordinates

Coordinates on the sphere must be specified in spherical polar coordinates. The polar angle (theta) must be in the range [0,PI], with
0 at the north pole, and the azimuthal angle (phi) should be in range [0,2*PI]. Both should be in radians. 

# Requirements 

The base module only requires numpy. To run the unit tests, matplotlib and scipy are required. 

# Unit tests

There are two unit tests. The first tests that the Fisher kernel is correctly normalized. The second computes the KDE estimate
for a uniform distribution and compares the MISE to the known MISE: for the uniform distribution on the sphere, the MISE can be computed analytically.
The parameters for the unit tests are set directly in the scripts. 

[^1]: Russell M.B., Johnson, C.L., Gilchrist, S.A.: 2024,"Investigating the Spatial Relationship of Shield Volcanism with Coronae on Venus",55th Lunar and Planetary Science Conference,3040,
      URL: https://ui.adsabs.harvard.edu/abs/2024LPICo3040.2582R
