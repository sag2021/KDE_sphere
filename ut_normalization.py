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
# Check the normalization of the Fisher kernel. The log 
# test only works for one kernel. 
# 

# Standard
import numpy as np
import matplotlib.pyplot as plt

# Local
from kde_sphere import  kde_fisher,compute_log_norm_constant
from kde_sphere_utests import trapz_2D

# Scipy 
from scipy.special import roots_legendre


# Parameters 
N_QUAD = 128   # Number of quadrature points
SEED   = 1221  # RNG seed

# Output names
oname1 = "ut_norm_plot1.pdf"
oname2 = "ut_norm_plot2.pdf"

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

# Random single data point
phi   = rng.uniform(0,2*np.pi,1)
u     = rng.uniform(0,1.,phi.size)
theta = np.arccos(2*u-1.)
data  = (theta,phi)

# Get range of kappa values
kappa_values = np.logspace(-4,6,32)
E1           = np.zeros(kappa_values.size)
E2           = np.zeros(kappa_values.size)

# Loop over kappa
for i,kappa in enumerate(kappa_values):

  # Compute KDE
  log_fbar = kde_fisher(mesh,data,kappa=kappa,nchunks=3)
  fbar     = np.exp(log_fbar)

  # Analytic value of log integral
  L_C0 = compute_log_norm_constant(np.array([kappa]))[0]
  Lc   = 4*np.pi*L_C0

  # Perform quadrature
  Q1 = np.sum(fbar*W)
  Q2 = np.sum(log_fbar*W)

  # Errors 
  E1[i] = np.abs(Q1-1.)
  E2[i] = np.abs(Q2-Lc)  

# Plot normalization error. Direct method 
plt.figure()
plt.loglog(kappa_values,E1,'r.')
plt.title("Direct method")
plt.xlabel("Concentration parameter")
plt.ylabel("Absolute error")
plt.grid('on',which='major',color='.85')
#plt.minorticks_on()

# Write
print("Writing: "+oname1)
plt.savefig(oname1)

# Plot normalization error. Log method
plt.figure()
plt.loglog(kappa_values,E2,'r.')
plt.title("Log method")
plt.xlabel("Concentration parameter")
plt.ylabel("Absolute error")
plt.grid('on',which='major',color='.85')
#plt.minorticks_on()

# Write
print("Writing: "+oname2)
plt.savefig(oname2)


plt.show()
