# BSD 3-Clause License
#
# Copyright (c) 2024, S.A. Gilchrist
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# Compute KDE estimate for uniform distribution of points on the
# sphere and compute MISE. Compare the numerical MISE with the 
# analytic value (see notes). 
# 
# The numerical value is computed with finite sample of random points 
# draw from a uniform distribution so it will differ from the analytic 
# value due to statistical noise. For enough samples, the error should
# fall below the scatter due to the finite sample. This is how
# success is defined for this test.
#
# If KAPPA is made too large, the number of numerical quad. points
# needs to be increased too. 
# 
#

# Numpy 
import numpy as np

# Local
from kde_sphere import  kde_fisher,compute_log_norm_constant
from kde_sphere_utests import trapz_2D

# Scipy 
from scipy.special import roots_legendre
  
# Parameters
N_DATA = 32    # Number of data points
N_QUAD = 128   # Number of Quad. points
N_SAMP = 128   # Number of samples for going from ISE to MISE
SEED   = 1331  # RNG seed
KAPPA  = 1.    # Kappa value to use. Too big and numerical quad. will be very wrong

# Setup RNG
rng = np.random.default_rng(SEED)

# Theta mesh: Gauss-Legendre 
xc,wc = roots_legendre(N_QUAD)
theta = np.arccos(xc[::-1])

# Phi mesh. Trap. rule weights
phi    = np.linspace(0,2*np.pi,2*N_QUAD)
dp     = phi[1]-phi[0]
wp     = np.full(phi.shape,dp)
wp[0]  = .5*dp
wp[-1] = .5*dp

# Full mesh and weights
THETA,PHI = np.meshgrid(theta,phi)
mesh      = (THETA,PHI)
W         = np.outer(wp,wc)

#
# Compute ISE with different samples from uniform distro
#
ISE = np.zeros(N_SAMP)
for i in range(ISE.size):

  # Random data
  phi   = rng.uniform(0,2*np.pi,N_DATA)
  u     = rng.uniform(0,1.,phi.size)
  theta = np.arccos(2*u-1.)
  data  = (theta,phi)

  log_fbar = kde_fisher(mesh,data,kappa=KAPPA,nchunks=4)
  fbar     = np.exp(log_fbar)
  
  # SE: Squared error (numerical)
  SE = (.25/np.pi - fbar)**2

  # Save value
  ISE[i] = np.sum(SE*W) 
 
# log of normalization constants 
C0 = compute_log_norm_constant(np.array([KAPPA]))[0]
L2 = compute_log_norm_constant(np.array([2*KAPPA]))[0]
  
# MISE (analytic)
gamma      = np.exp(2*C0-L2)
MISE_exact = (gamma - (.25/np.pi))/float(N_DATA)

# Compute numerical MISE. Use standard error as a basic
# measure of the Monte-Carlo error
MISE     = ISE.mean()
MISE_err = ISE.std()/np.sqrt(float(N_DATA))

# Compute difference 
error = np.abs(MISE-MISE_exact)

# Check if within scatter
if(error < MISE_err):
  print("PASS: Error smaller than scatter")
else:
  print("FAIL: Error larger than scatter")

# Print results
print("MISE,MISE_exct,error,MISE(numerical) standed error:",MISE,MISE_exact,error,MISE_err)


