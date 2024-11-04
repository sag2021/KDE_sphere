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

# Numpy import
import numpy as np

def kde_fisher(mesh,data,kappa,nchunks=1):
  """
    Perform Kernel density estimation for data. This function
    returns the log of the probability density constructed as a 
    sum of Fisher distributions over the data. This KDE estimation
    is evaluated at a finite number of mesh points and returned as an
    array.

    Parameters:
    -----------
      mesh: (theta,phi)
        Tuple holding the (theta,phi) coordinates of the mesh
        on which the density is evacuated
      data: (theta,phi) 
        Tuple holding the (theta,phi) coordinates of the observations
      kappa: float 
        Value of the concentration parameter. When kappa is large, the 
        Kernel distributions are narrow, and when kappa is small they are wide. 
       nchunks: int (optional)
         The number of chunks to break the calculation into. 
         A larger value reduces peak memory usage at the cost of 
         speed.

     Returns:
     --------
       L: Array with shape mesh[0].shape
         Log of the probability distribution estimated based on KDE 

  """

  # Basic checks
  if(np.any(mesh[0].shape != mesh[1].shape)):
    raise ValueError("mesh[0] and mesh[1] must have the same shape")
  if(np.any(data[0].shape != data[1].shape)):
    raise ValueError("data[0] and data[1] must have the same shape")

  # Get total number of data points
  N = data[0].size

  # Construct unit vectors
  mu = spherical_to_cart(data[0],data[1])

  # Get size and shape of mesh
  mesh_shape = mesh[0].shape
  mesh_size  = mesh[1].size
 
  # (theta,phi): Convert mesh points to unit vectors 
  x = spherical_to_cart(mesh[0],mesh[1]).reshape(3,mesh_size)

  # Check if kappa is a scalar. If so, make it a one element array.
  if(np.isscalar(kappa)):
    kappa = np.array([kappa])

  # Precompute normalization
  log_Ak = compute_log_norm_constant(kappa)

  # Resultant 
  L_res = np.zeros(mesh_size)
  Lc    = np.zeros((2,) + L_res.shape)

  # Partition data into chunks
  chunks = build_chunk_list(N,nchunks)

  # 
  # Summation over chunks of data
  #
  for i,chunk in enumerate(chunks):
     
    # Extract chunk
    mu_chunk = mu[:,chunk[0]:chunk[1]]

    # Compute Fisher distro. 
    logp = log_fisher(x,mu_chunk,kappa,log_Ak=log_Ak)
 
    # Perform summation over kernel nodes
    L_chunk = log_sum_exp(logp.T,axis=0)

    # Clunky way to reuse log_sum_exp to add  the chunk to the 
    # resultant
    if(i==0):
      L_res = L_chunk
    else:
      Lc[0,:] = L_chunk
      Lc[1,:] = L_res
      L_res   = log_sum_exp(Lc,axis=0)

  # Reshape output
  L_res = L_res.reshape(mesh_shape)

  # Normalization factor 
  L_res = L_res - np.log(float(N))

  return L_res

# ---------------------------------------------------------------------

def log_fisher(x,mu,kappa,log_Ak=None):
  """

    Evaluate the log of the Fisher distribution. This is a 
    special case of the Fisher-Von-Mises distribution with p=3.

    Returns a (N,M) array:

    L_ij = log(P(x_i | mu_j,kappa_j))

    Parameters:
    -----------
      x: (3,N)
        Data points 
      mu: (3,M)
        Vector of means 
      kappa: (M,)
        Vector of concentration parameters
      log_Ak: (M,) optional
        Use pre-computed normalization constant. Useful when
        function is in a loop and kappa doesn't change.
 
     Returns:
     --------
       logP: (N,M)
       log of condition prob. L_ij = log(P(x_i | mu_j,kappa_j))

  """

  # Construct the log of the normalization constant
  if(log_Ak is None):
    log_Ak = compute_log_norm_constant(kappa)
 
  # [3,M] [3,N] -> [N,M]
  mu_dot_x = np.einsum('ij,ik->jk', x, mu)
  
  # Log log(P)
  logP = log_Ak + kappa*mu_dot_x

  return logP

# ---------------------------------------------------------------------

def compute_log_norm_constant(kappa):
  """
    Compute the log of the normalization constant for Fisher distribution.
    Coefficient can be written succinctly as 

    A(k) = a0*k/sinh(k),

    with a0 = 1/(4*pi). However, sinh(k) tends to overflow 
    (Here kappa and k mean the same thing.). An alternative form is 

    A(k) = 2*a0*k*exp(-k)/(1-exp(-2k)).

    The exponential form makes it easier to take the log. 
    The log of the denominator has the form log(1-eps) when
    k is large. This can be treated with log1p to reduce 
    cancellation.

    The case k = 0 should return a0 = 1/(4*pi). This special
    case is correctly handled.

    Parameters: 
    -----------
      kappa: (M,)
        Vector of kappa values (concentration parameters)

    Returns: 
    --------
      log_Ak: (M,)
        The natural log of the normalization constant for the 
        Fisher distribution 
  """

  # Terms for normalization constant
  a0    = -np.log(4*np.pi)
  exp2k = -np.exp(-2*kappa)

  # Initial set to a0
  #kappa*0. + a0
  log_Ak = np.full(kappa.shape,a0)

  # Add other log terms under a mask
  ii          = (kappa != 0.)
  log_Ak[ii] += np.log(kappa[ii]) - kappa[ii] - np.log1p(exp2k[ii]) + np.log(2.)

  return log_Ak

# ---------------------------------------------------------------------

def build_chunk_list(N,nchunks):
  """
    Build list of indices for breaking calculation into chunks
    for calculation
  """
   
  # Special case N = 1
  if(N==1): 
    chunks = [[0,1]]
    return chunks

  # Basic chunk size, the actual size may be smaller especially 
  # at the end of the chunk list
  chunk_size = int(np.ceil(float(N)/nchunks))

  # Construct chunks
  chunks     = [] 
  for i in range(nchunks):

    # Construct chunk and add to list
    c1 = i*chunk_size
    c2 = np.min([c1+chunk_size,N])
    chunks.append([c1,c2])

    # Break out of loop early if end is reached
    if(c2 == N):
      break 

  return chunks
  
# ---------------------------------------------------------------------

def spherical_to_cart(theta,phi):
  """
    Convert from spherical to Cartesian (theta,phi) -> (x,y,z)  
    Here theta is the polar angle (colatitude)

    Parameters:
    -----------
      theta: (N,)
        Polar angle/colatitude in units of radians
      phi: (N,)
        Azimuthal angle in units of radians

    Returns:
    --------
      cpts: (3,N)
        Cartesian coordinates 
        cpts[0,N] = x
        cpts[1,N] = y
        cpts[2,N] = z

  """

  # Trig. functions
  cos_t = np.cos(theta)
  sin_t = np.sin(theta)
  cos_p = np.cos(phi)
  sin_p = np.sin(phi)

  # Output
  cpts      = np.zeros((3,)+phi.shape)
  cpts[0,:] = cos_p*sin_t
  cpts[1,:] = sin_p*sin_t
  cpts[2,:] = cos_t

  return cpts

# ---------------------------------------------------------------------

def log_sum_exp(z,axis=0):
  """
    Computes the log of the sum of exps

               __
               \
    res = log( /_ exp(zi) ) 

    Parameters:
    -----------
      z: (N,)
        Log(x)

  """

  # Get max
  zmax = z.max(axis=axis)

  # Argument to exp 
  arg     = z - zmax
  exp_arg = np.exp(arg)

  # Compute summation
  res = zmax + np.log(exp_arg.sum(axis=axis))

  return res

