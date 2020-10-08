#include "cuda.h"

extern "C" {
#include <stdio.h>
#include "lb.h"
}

#include "cuda_utils.hpp"

//static const float c_sound_sq = 1.0f/3.0f;
static __constant__ float c_sound_sq = 1.0f/3.0f;

static size_t size_of_rho_v;
static size_t size_of_rho_v_pi;
static float *devNodesA = NULL;
static float *devNodesB = NULL;
static float *devBoundaryForces = NULL;
static LB_rho_v *devRhoV = NULL;
static LB_rho_v_pi *devRhoVpi = NULL;
static LB_rho_v_pi *print_rho_v_pi = NULL;
static unsigned int intflag = 1;
unsigned int *devBoundaryMap = NULL;
float *devExtForces = NULL;
float *devBoundaryVelocities = NULL;
float *devCurrentNodes = NULL;

float *hostExtForces = NULL;

//static __device__ __constant__ LBparameters d_LBparams;
__constant__ LBparameters d_LBparams;

LBparameters h_LBparams = {
// agrid  tau    rho
   1.0f,  1.0f,  1.0f,
// gammaShear  gammaBulk  gammaOdd  gammaEven
   1.0f,        1.0f,       0.0f,      0.0f,
// dimX  dimY  dimZ  numNodes
   0u,    0u,    0u,    0u,
// numBoundaries  boundaryVelocity
   0u,             {0.0f, 0.0f, 0.0f},
// extForceFlag  extForceDensity
   0u,            {0.0f, 0.0f, 0.0f}
};

__device__ void index_to_xyz (unsigned int index, unsigned int *xyz) {

  xyz[0] = index % d_LBparams.dimX;
  index /= d_LBparams.dimX;
  xyz[1] = index % d_LBparams.dimY;
  index /= d_LBparams.dimY;
  xyz[2] = index;
}

__device__ void calc_m_from_n (unsigned int index, float *n_a, float *mode) {

    // The following convention is used:
    // The $\hat{c}_i$ form B. Duenweg's paper are given by:

    /* c_0  = { 0, 0, 0}
       c_1  = { 1, 0, 0}
       c_2  = {-1, 0, 0}
       c_3  = { 0, 1, 0}
       c_4  = { 0,-1, 0}
       c_5  = { 0, 0, 1}
       c_6  = { 0, 0,-1}
       c_7  = { 1, 1, 0}
       c_8  = {-1,-1, 0}
       c_9  = { 1,-1, 0}
       c_10 = {-1, 1, 0}
       c_11 = { 1, 0, 1}
       c_12 = {-1, 0,-1}
       c_13 = { 1, 0,-1}
       c_14 = {-1, 0, 1}
       c_15 = { 0, 1, 1}
       c_16 = { 0,-1,-1}
       c_17 = { 0, 1,-1}
       c_18 = { 0,-1, 1} */

    // The basis vectors (modes) are constructed as follows
    // $m_k = \sum_{i} e_{ki} n_{i}$, where the $e_{ki}$ form a 
    // linear transformation (matrix) that is given by

    /* $e{ 0,i} = 1$
       $e{ 1,i} = c_{i,x}$
       $e{ 2,i} = c_{i,y}$
       $e{ 3,i} = c_{i,z}$
       $e{ 4,i} = c_{i}^2 - 1$
       $e{ 5,i} = c_{i,x}^2 - c_{i,y}^2$
       $e{ 6,i} = c_{i}^2 - 3*c_{i,z}^2$
       $e{ 7,i} = c_{i,x}*c_{i,y}$
       $e{ 8,i} = c_{i,x}*c_{i,z}$
       $e{ 9,i} = c_{i,y}*c_{i,z}$
       $e{10,i} = (3*c_{i}^2 - 5)*c_{i,x}$
       $e{11,i} = (3*c_{i}^2 - 5)*c_{i,y}$
       $e{12,i} = (3*c_{i}^2 - 5)*c_{i,z}$
       $e{13,i} = (c_{i,y}^2 - c_{i,z}^2)*c_{i,x}$
       $e{14,i} = (c_{i,x}^2 - c_{i,z}^2)*c_{i,y}$
       $e{15,i} = (c_{i,x}^2 - c_{i,y}^2)*c_{i,z}$
       $e{16,i} = 3*c_{i}^2^2 - 6*c_{i}^2 + 1$
       $e{17,i} = (2*c_{i}^2 - 3)*(c_{i,x}^2 - c_{i,y}^2)$
       $e{18,i} = (2*c_{i}^2 - 3)*(c_{i}^2 - 3*c_{i,z}^2)$ */

    // Such that the transformation matrix is given by

    /* {{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
        { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0}, 
        { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1}, 
        { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1}, 
        {-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
        { 0, 1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,-1,-1,-1,-1}, 
        { 0, 1, 1, 1, 1,-2,-2, 2, 2, 2, 2,-1,-1,-1,-1,-1,-1,-1,-1}, 
        { 0, 0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0}, 
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0}, 
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,-1,-1}, 
        { 0,-2, 2, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0}, 
        { 0, 0, 0,-2, 2, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1}, 
        { 0, 0, 0, 0, 0,-2, 2, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1}, 
        { 0, 0, 0, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1,-1, 1, 0, 0, 0, 0}, 
        { 0, 0, 0, 0, 0, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0,-1, 1,-1, 1}, 
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1,-1, 1,-1, 1, 1,-1}, 
        { 1,-2,-2,-2,-2,-2,-2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
        { 0,-1,-1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,-1,-1,-1,-1}, 
        { 0,-1,-1,-1,-1, 2, 2, 2, 2, 2, 2,-1,-1,-1,-1,-1,-1,-1,-1}} */

    // With weights 

    /* q^{c_{i}} = { 1/3, 1/18, 1/18, 1/18,
                    1/18, 1/18, 1/18, 1/36,
                    1/36, 1/36, 1/36, 1/36, 
                    1/36, 1/36, 1/36, 1/36, 
                    1/36, 1/36, 1/36 } */

    // Which makes the transformation satisfy the following
    // orthogonality condition:
    // \sum_{i} q^{c_{i}} e_{ki} e_{li} = w_{k} \delta_{kl},
    // where the weights are:

    /* w_{i} = {  1, 1/3, 1/3, 1/3,
                2/3, 4/9, 4/3, 1/9,
                1/9, 1/9, 2/3, 2/3,
                2/3, 2/9, 2/9, 2/9, 
                  2, 4/9, 4/3 } */

  // mass mode
  mode[0] = n_a[ 0 * d_LBparams.numNodes + index]
          + n_a[ 1 * d_LBparams.numNodes + index] + n_a[ 2 * d_LBparams.numNodes + index]
          + n_a[ 3 * d_LBparams.numNodes + index] + n_a[ 4 * d_LBparams.numNodes + index]
          + n_a[ 5 * d_LBparams.numNodes + index] + n_a[ 6 * d_LBparams.numNodes + index]
          + n_a[ 7 * d_LBparams.numNodes + index] + n_a[ 8 * d_LBparams.numNodes + index]
          + n_a[ 9 * d_LBparams.numNodes + index] + n_a[10 * d_LBparams.numNodes + index]
          + n_a[11 * d_LBparams.numNodes + index] + n_a[12 * d_LBparams.numNodes + index]
          + n_a[13 * d_LBparams.numNodes + index] + n_a[14 * d_LBparams.numNodes + index]
          + n_a[15 * d_LBparams.numNodes + index] + n_a[16 * d_LBparams.numNodes + index]
          + n_a[17 * d_LBparams.numNodes + index] + n_a[18 * d_LBparams.numNodes + index];

  // momentum modes
  mode[1] = (n_a[ 1 * d_LBparams.numNodes + index] - n_a[ 2 * d_LBparams.numNodes + index])
          + (n_a[ 7 * d_LBparams.numNodes + index] - n_a[ 8 * d_LBparams.numNodes + index])
          + (n_a[ 9 * d_LBparams.numNodes + index] - n_a[10 * d_LBparams.numNodes + index])
          + (n_a[11 * d_LBparams.numNodes + index] - n_a[12 * d_LBparams.numNodes + index])
          + (n_a[13 * d_LBparams.numNodes + index] - n_a[14 * d_LBparams.numNodes + index]);

  mode[2] = (n_a[ 3 * d_LBparams.numNodes + index] - n_a[ 4 * d_LBparams.numNodes + index])
          + (n_a[ 7 * d_LBparams.numNodes + index] - n_a[ 8 * d_LBparams.numNodes + index])
          - (n_a[ 9 * d_LBparams.numNodes + index] - n_a[10 * d_LBparams.numNodes + index])
          + (n_a[15 * d_LBparams.numNodes + index] - n_a[16 * d_LBparams.numNodes + index])
          + (n_a[17 * d_LBparams.numNodes + index] - n_a[18 * d_LBparams.numNodes + index]);

  mode[3] = (n_a[ 5 * d_LBparams.numNodes + index] - n_a[ 6 * d_LBparams.numNodes + index])
          + (n_a[11 * d_LBparams.numNodes + index] - n_a[12 * d_LBparams.numNodes + index])
          - (n_a[13 * d_LBparams.numNodes + index] - n_a[14 * d_LBparams.numNodes + index])
          + (n_a[15 * d_LBparams.numNodes + index] - n_a[16 * d_LBparams.numNodes + index])
          - (n_a[17 * d_LBparams.numNodes + index] - n_a[18 * d_LBparams.numNodes + index]);

  // stress modes
  mode[4] = - n_a[ 0 * d_LBparams.numNodes + index]
            + n_a[ 7 * d_LBparams.numNodes + index] + n_a[ 8 * d_LBparams.numNodes + index]
            + n_a[ 9 * d_LBparams.numNodes + index] + n_a[10 * d_LBparams.numNodes + index]
            + n_a[11 * d_LBparams.numNodes + index] + n_a[12 * d_LBparams.numNodes + index]
            + n_a[13 * d_LBparams.numNodes + index] + n_a[14 * d_LBparams.numNodes + index]
            + n_a[15 * d_LBparams.numNodes + index] + n_a[16 * d_LBparams.numNodes + index]
            + n_a[17 * d_LBparams.numNodes + index] + n_a[18 * d_LBparams.numNodes + index];

  mode[5] = (n_a[ 1 * d_LBparams.numNodes + index] + n_a[ 2 * d_LBparams.numNodes + index])
          - (n_a[ 3 * d_LBparams.numNodes + index] + n_a[ 4 * d_LBparams.numNodes + index])
          + (n_a[11 * d_LBparams.numNodes + index] + n_a[12 * d_LBparams.numNodes + index])
          + (n_a[13 * d_LBparams.numNodes + index] + n_a[14 * d_LBparams.numNodes + index])
          - (n_a[15 * d_LBparams.numNodes + index] + n_a[16 * d_LBparams.numNodes + index])
          - (n_a[17 * d_LBparams.numNodes + index] + n_a[18 * d_LBparams.numNodes + index]);

  mode[6] = (n_a[ 1 * d_LBparams.numNodes + index] + n_a[ 2 * d_LBparams.numNodes + index])
          + (n_a[ 3 * d_LBparams.numNodes + index] + n_a[ 4 * d_LBparams.numNodes + index])
          - (n_a[11 * d_LBparams.numNodes + index] + n_a[12 * d_LBparams.numNodes + index])
          - (n_a[13 * d_LBparams.numNodes + index] + n_a[14 * d_LBparams.numNodes + index])
          - (n_a[15 * d_LBparams.numNodes + index] + n_a[16 * d_LBparams.numNodes + index])
          - (n_a[17 * d_LBparams.numNodes + index] + n_a[18 * d_LBparams.numNodes + index])
          - 2.0f*(  (n_a[5 * d_LBparams.numNodes + index] + n_a[ 6 * d_LBparams.numNodes + index])
                  - (n_a[7 * d_LBparams.numNodes + index] + n_a[ 8 * d_LBparams.numNodes + index])
                  - (n_a[9 * d_LBparams.numNodes + index] + n_a[10 * d_LBparams.numNodes + index]));

  mode[7] = (n_a[7 * d_LBparams.numNodes + index] + n_a[ 8 * d_LBparams.numNodes + index])
          - (n_a[9 * d_LBparams.numNodes + index] + n_a[10 * d_LBparams.numNodes + index]);

  mode[8] = (n_a[11 * d_LBparams.numNodes + index] + n_a[12 * d_LBparams.numNodes + index])
          - (n_a[13 * d_LBparams.numNodes + index] + n_a[14 * d_LBparams.numNodes + index]);

  mode[9] = (n_a[15 * d_LBparams.numNodes + index] + n_a[16 * d_LBparams.numNodes + index])
          - (n_a[17 * d_LBparams.numNodes + index] + n_a[18 * d_LBparams.numNodes + index]);

  // kinetic modes
  mode[10] = - 2.0f*(n_a[ 1 * d_LBparams.numNodes + index] - n_a[ 2 * d_LBparams.numNodes + index])
                  + (n_a[ 7 * d_LBparams.numNodes + index] - n_a[ 8 * d_LBparams.numNodes + index])
                  + (n_a[ 9 * d_LBparams.numNodes + index] - n_a[10 * d_LBparams.numNodes + index])
                  + (n_a[11 * d_LBparams.numNodes + index] - n_a[12 * d_LBparams.numNodes + index])
                  + (n_a[13 * d_LBparams.numNodes + index] - n_a[14 * d_LBparams.numNodes + index]);

  mode[11] = - 2.0f*(n_a[ 3 * d_LBparams.numNodes + index] - n_a[ 4 * d_LBparams.numNodes + index])
                  + (n_a[ 7 * d_LBparams.numNodes + index] - n_a[ 8 * d_LBparams.numNodes + index])
                  - (n_a[ 9 * d_LBparams.numNodes + index] - n_a[10 * d_LBparams.numNodes + index])
                  + (n_a[15 * d_LBparams.numNodes + index] - n_a[16 * d_LBparams.numNodes + index])
                  + (n_a[17 * d_LBparams.numNodes + index] - n_a[18 * d_LBparams.numNodes + index]);

  mode[12] = - 2.0f*(n_a[ 5 * d_LBparams.numNodes + index] - n_a[ 6 * d_LBparams.numNodes + index])
                  + (n_a[11 * d_LBparams.numNodes + index] - n_a[12 * d_LBparams.numNodes + index])
                  - (n_a[13 * d_LBparams.numNodes + index] - n_a[14 * d_LBparams.numNodes + index])
                  + (n_a[15 * d_LBparams.numNodes + index] - n_a[16 * d_LBparams.numNodes + index])
                  - (n_a[17 * d_LBparams.numNodes + index] - n_a[18 * d_LBparams.numNodes + index]);

  mode[13] = (n_a[ 7 * d_LBparams.numNodes + index] - n_a[ 8 * d_LBparams.numNodes + index])
           + (n_a[ 9 * d_LBparams.numNodes + index] - n_a[10 * d_LBparams.numNodes + index])
           - (n_a[11 * d_LBparams.numNodes + index] - n_a[12 * d_LBparams.numNodes + index])
           - (n_a[13 * d_LBparams.numNodes + index] - n_a[14 * d_LBparams.numNodes + index]);

  mode[14] = (n_a[ 7 * d_LBparams.numNodes + index] - n_a[ 8 * d_LBparams.numNodes + index])
           - (n_a[ 9 * d_LBparams.numNodes + index] - n_a[10 * d_LBparams.numNodes + index])
           - (n_a[15 * d_LBparams.numNodes + index] - n_a[16 * d_LBparams.numNodes + index])
           - (n_a[17 * d_LBparams.numNodes + index] - n_a[18 * d_LBparams.numNodes + index]);

  mode[15] = (n_a[11 * d_LBparams.numNodes + index] - n_a[12 * d_LBparams.numNodes + index])
           - (n_a[13 * d_LBparams.numNodes + index] - n_a[14 * d_LBparams.numNodes + index])
           - (n_a[15 * d_LBparams.numNodes + index] - n_a[16 * d_LBparams.numNodes + index])
           + (n_a[17 * d_LBparams.numNodes + index] - n_a[18 * d_LBparams.numNodes + index]);

  mode[16] = n_a[ 0 * d_LBparams.numNodes + index]
           + n_a[ 7 * d_LBparams.numNodes + index] + n_a[ 8 * d_LBparams.numNodes + index]
           + n_a[ 9 * d_LBparams.numNodes + index] + n_a[10 * d_LBparams.numNodes + index]
           + n_a[11 * d_LBparams.numNodes + index] + n_a[12 * d_LBparams.numNodes + index]
           + n_a[13 * d_LBparams.numNodes + index] + n_a[14 * d_LBparams.numNodes + index]
           + n_a[15 * d_LBparams.numNodes + index] + n_a[16 * d_LBparams.numNodes + index]
           + n_a[17 * d_LBparams.numNodes + index] + n_a[18 * d_LBparams.numNodes + index]
           - 2.0f*(  (n_a[1 * d_LBparams.numNodes + index] + n_a[2 * d_LBparams.numNodes + index])
                   + (n_a[3 * d_LBparams.numNodes + index] + n_a[4 * d_LBparams.numNodes + index])
                   + (n_a[5 * d_LBparams.numNodes + index] + n_a[6 * d_LBparams.numNodes + index]));

  mode[17] = - (n_a[ 1 * d_LBparams.numNodes + index] + n_a[ 2 * d_LBparams.numNodes + index])
             + (n_a[ 3 * d_LBparams.numNodes + index] + n_a[ 4 * d_LBparams.numNodes + index])
             + (n_a[11 * d_LBparams.numNodes + index] + n_a[12 * d_LBparams.numNodes + index])
             + (n_a[13 * d_LBparams.numNodes + index] + n_a[14 * d_LBparams.numNodes + index])
             - (n_a[15 * d_LBparams.numNodes + index] + n_a[16 * d_LBparams.numNodes + index])
             - (n_a[17 * d_LBparams.numNodes + index] + n_a[18 * d_LBparams.numNodes + index]);

  mode[18] = - (n_a[ 1 * d_LBparams.numNodes + index] + n_a[ 2 * d_LBparams.numNodes + index])
             - (n_a[ 3 * d_LBparams.numNodes + index] + n_a[ 4 * d_LBparams.numNodes + index])
             - (n_a[11 * d_LBparams.numNodes + index] + n_a[12 * d_LBparams.numNodes + index])
             - (n_a[13 * d_LBparams.numNodes + index] + n_a[14 * d_LBparams.numNodes + index])
             - (n_a[15 * d_LBparams.numNodes + index] + n_a[16 * d_LBparams.numNodes + index])
             - (n_a[17 * d_LBparams.numNodes + index] + n_a[18 * d_LBparams.numNodes + index])
             + 2.0f*(  (n_a[5 * d_LBparams.numNodes + index] + n_a[ 6 * d_LBparams.numNodes + index])
                     + (n_a[7 * d_LBparams.numNodes + index] + n_a[ 8 * d_LBparams.numNodes + index])
                     + (n_a[9 * d_LBparams.numNodes + index] + n_a[10 * d_LBparams.numNodes + index]));
}

__device__ void update_rho_v (unsigned int index, float *mode, float *ext_forces, LB_rho_v *d_v) {

  float Rho_tot  = 0.0f;
  float u_tot[3] = {0.0f,0.0f,0.0f};
   
  // Note:
  // Remember that the populations are stored as differences to their equilibrium values.
  // Quantities are calculated in LB units rather than MD units (cf. ESPResSo)
  //d_v[index].rho[ii] = mode[0 +ii*LBQ] + para.rho[ii]*para.agrid*para.agrid*para.agrid;
  //Rho_tot  += mode[0+ii*LBQ]           + para.rho[ii]*para.agrid*para.agrid*para.agrid;
  d_v[index].rho = mode[0] + d_LBparams.rho;
  Rho_tot       += mode[0] + d_LBparams.rho;
  u_tot[0]      += mode[1];
  u_tot[1]      += mode[2];
  u_tot[2]      += mode[3];

  /** if forces are present, the momentum density is redefined to
    * inlcude one half-step of the force action.  See the
    * Chapman-Enskog expansion in [Ladd & Verberg]. */
  u_tot[0] += 0.5f*ext_forces[0*d_LBparams.numNodes + index];
  u_tot[1] += 0.5f*ext_forces[1*d_LBparams.numNodes + index];
  u_tot[2] += 0.5f*ext_forces[2*d_LBparams.numNodes + index];
  
  u_tot[0] /= Rho_tot;
  u_tot[1] /= Rho_tot;
  u_tot[2] /= Rho_tot;

  d_v[index].v[0] = u_tot[0]; 
  d_v[index].v[1] = u_tot[1]; 
  d_v[index].v[2] = u_tot[2]; 
}

__device__ void relax_modes (unsigned int index, float *ext_forces, LB_rho_v *d_v, float *mode) {

  float Rho;
  float j[3];
  float modes_from_pi_eq[6];
  float u_tot[3] = {0.0f,0.0f,0.0f};

  update_rho_v (index, mode, ext_forces, d_v);
  Rho = mode[0] + d_LBparams.rho;
  float inv_Rho = 1.0 / Rho;
  u_tot[0] = d_v[index].v[0];  
  u_tot[1] = d_v[index].v[1];  
  u_tot[2] = d_v[index].v[2];  
  j[0] = Rho * u_tot[0];
  j[1] = Rho * u_tot[1];
  j[2] = Rho * u_tot[2];

  /** equilibrium part of the stress modes (eq13 schiller)*/
  modes_from_pi_eq[0] = ((j[0]*j[0])+(j[1]*j[1])+(j[2]*j[2])) * inv_Rho;
  modes_from_pi_eq[1] = ((j[0]*j[0])-(j[1]*j[1])) * inv_Rho;
  modes_from_pi_eq[2] = (((j[0]*j[0])+(j[1]*j[1])+(j[2]*j[2])) - 3.0f*(j[2]*j[2])) * inv_Rho;
  modes_from_pi_eq[3] = j[0]*j[1] * inv_Rho;
  modes_from_pi_eq[4] = j[0]*j[2] * inv_Rho;
  modes_from_pi_eq[5] = j[1]*j[2] * inv_Rho;

  /** relax the stress modes (eq14 schiller)*/
  mode[4] = modes_from_pi_eq[0] + d_LBparams.gammaBulk  * (mode[4] - modes_from_pi_eq[0]);
  mode[5] = modes_from_pi_eq[1] + d_LBparams.gammaShear * (mode[5] - modes_from_pi_eq[1]);
  mode[6] = modes_from_pi_eq[2] + d_LBparams.gammaShear * (mode[6] - modes_from_pi_eq[2]);
  mode[7] = modes_from_pi_eq[3] + d_LBparams.gammaShear * (mode[7] - modes_from_pi_eq[3]);
  mode[8] = modes_from_pi_eq[4] + d_LBparams.gammaShear * (mode[8] - modes_from_pi_eq[4]);
  mode[9] = modes_from_pi_eq[5] + d_LBparams.gammaShear * (mode[9] - modes_from_pi_eq[5]);

  /** relax the ghost modes (project them out) */
  /** ghost modes have no equilibrium part due to orthogonality */
  mode[10] =  d_LBparams.gammaOdd*mode[10];
  mode[11] =  d_LBparams.gammaOdd*mode[11];
  mode[12] =  d_LBparams.gammaOdd*mode[12];
  mode[13] =  d_LBparams.gammaOdd*mode[13];
  mode[14] =  d_LBparams.gammaOdd*mode[14];
  mode[15] =  d_LBparams.gammaOdd*mode[15];
  mode[16] = d_LBparams.gammaEven*mode[16];
  mode[17] = d_LBparams.gammaEven*mode[17];
  mode[18] = d_LBparams.gammaEven*mode[18];
}

__device__ void reset_LB_forces (unsigned int index, float *ext_forces) {

  ext_forces[                        index] = d_LBparams.extForceDensity[0];
  ext_forces[  d_LBparams.numNodes + index] = d_LBparams.extForceDensity[1];
  ext_forces[2*d_LBparams.numNodes + index] = d_LBparams.extForceDensity[2];
}

__device__ void apply_forces (unsigned int index, float *ext_forces, LB_rho_v *d_v, float *mode) {
  
  float u[3] = {0.0f, 0.0f, 0.0f},
        C[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // Note: the values d_v were calculated in relax_modes()
  u[0] = d_v[index].v[0]; 
  u[1] = d_v[index].v[1]; 
  u[2] = d_v[index].v[2]; 
    
  C[0] += (1.0f + d_LBparams.gammaBulk) * u[0]*ext_forces[0*d_LBparams.numNodes + index] + 
          1.0f/3.0f * (d_LBparams.gammaBulk-d_LBparams.gammaShear) * (u[0]*ext_forces[0*d_LBparams.numNodes + index] + 
          u[1]*ext_forces[1*d_LBparams.numNodes + index] + u[2]*ext_forces[2*d_LBparams.numNodes + index]);

  C[2] += (1.0f + d_LBparams.gammaBulk) * u[1]*ext_forces[1*d_LBparams.numNodes + index] + 
          1.0f/3.0f * (d_LBparams.gammaBulk-d_LBparams.gammaShear) * (u[0]*ext_forces[0*d_LBparams.numNodes + index] + 
          u[1]*ext_forces[1*d_LBparams.numNodes + index] + u[2]*ext_forces[2*d_LBparams.numNodes + index]);

  C[5] += (1.0f + d_LBparams.gammaBulk) * u[2]*ext_forces[2*d_LBparams.numNodes + index] + 
          1.0f/3.0f * (d_LBparams.gammaBulk-d_LBparams.gammaShear) * (u[0]*ext_forces[0*d_LBparams.numNodes + index] + 
          u[1]*ext_forces[1*d_LBparams.numNodes + index] + u[2]*ext_forces[2*d_LBparams.numNodes + index]);

  C[1] += 1.0f/2.0f * (1.0f+d_LBparams.gammaShear) * (u[0]*ext_forces[1*d_LBparams.numNodes + index] + 
          u[1]*ext_forces[0*d_LBparams.numNodes + index]);

  C[3] += 1.0f/2.0f * (1.0f+d_LBparams.gammaShear) * (u[0]*ext_forces[2*d_LBparams.numNodes + index] + 
          u[2]*ext_forces[0*d_LBparams.numNodes + index]);

  C[4] += 1.0f/2.0f * (1.0f+d_LBparams.gammaShear) * (u[1]*ext_forces[2*d_LBparams.numNodes + index] + 
          u[2]*ext_forces[1*d_LBparams.numNodes + index]);
    
  /** update momentum modes */
  mode[1] += ext_forces[0*d_LBparams.numNodes + index];
  mode[2] += ext_forces[1*d_LBparams.numNodes + index];
  mode[3] += ext_forces[2*d_LBparams.numNodes + index];

  /** update stress modes */
  mode[4] += C[0] + C[2] + C[5];
  mode[5] += C[0] - C[2];
  mode[6] += C[0] + C[2] - 2.0f*C[5];
  mode[7] += C[1];
  mode[8] += C[3];
  mode[9] += C[4];

  // Note: Body forces are reset in coupling.cu
  //reset_LB_forces (index, ext_forces);
}

__device__ void normalize_modes (float* mode) {

  /** normalization factors enter in the back transformation */
  mode[ 0] *= 1.0f;
  mode[ 1] *= 3.0f;
  mode[ 2] *= 3.0f;
  mode[ 3] *= 3.0f;
  mode[ 4] *= 3.0f/2.0f;
  mode[ 5] *= 9.0f/4.0f;
  mode[ 6] *= 3.0f/4.0f;
  mode[ 7] *= 9.0f;
  mode[ 8] *= 9.0f;
  mode[ 9] *= 9.0f;
  mode[10] *= 3.0f/2.0f;
  mode[11] *= 3.0f/2.0f;
  mode[12] *= 3.0f/2.0f;
  mode[13] *= 9.0f/2.0f;
  mode[14] *= 9.0f/2.0f;
  mode[15] *= 9.0f/2.0f;
  mode[16] *= 1.0f/2.0f;
  mode[17] *= 9.0f/4.0f;
  mode[18] *= 3.0f/4.0f;
}

__device__ void calc_n_from_modes_push (unsigned int index, float *mode, float *n_b) {

  unsigned int xyz[3];
  index_to_xyz (index, xyz);
  unsigned int x = xyz[0];
  unsigned int y = xyz[1];
  unsigned int z = xyz[2];

  n_b[0*d_LBparams.numNodes + x + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*z] = 
  1.0f/3.0f * (mode[0] - mode[4] + mode[16]);

  n_b[1*d_LBparams.numNodes + (x+1)%d_LBparams.dimX + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*z] = 
  1.0f/18.0f * (mode[0] + mode[1] + mode[5] + mode[6] - mode[17] - mode[18] -2.0f*(mode[10] + mode[16]));

  n_b[2*d_LBparams.numNodes + (d_LBparams.dimX + x-1)%d_LBparams.dimX + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*z] =
  1.0f/18.0f * (mode[0] - mode[1] + mode[5] + mode[6] - mode[17] - mode[18] + 2.0f*(mode[10] - mode[16]));

  n_b[3*d_LBparams.numNodes + x + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*z] =
  1.0f/18.0f * (mode[0] + mode[2] - mode[5] + mode[6] + mode[17] - mode[18]- 2.0f*(mode[11] + mode[16]));

  n_b[4*d_LBparams.numNodes + x + d_LBparams.dimX*((d_LBparams.dimY+y-1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*z] =
  1.0f/18.0f * (mode[0] - mode[2] - mode[5] + mode[6] + mode[17] - mode[18] + 2.0f*(mode[11] - mode[16]));

  n_b[5*d_LBparams.numNodes + x + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ)] =
  1.0f/18.0f * (mode[0] + mode[3] - 2.0f*(mode[6] + mode[12] + mode[16] - mode[18]));

  n_b[6*d_LBparams.numNodes + x + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*((d_LBparams.dimZ+z-1)%d_LBparams.dimZ)] =
  1.0f/18.0f * (mode[0] - mode[3] - 2.0f*(mode[6] - mode[12] + mode[16] - mode[18]));

  n_b[7*d_LBparams.numNodes + (x+1)%d_LBparams.dimX + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*z] =
  1.0f/36.0f * (mode[0] + mode[1] + mode[2] + mode[4] + 2.0f*mode[6] + mode[7] + mode[10] + mode[11] + mode[13] + mode[14] + mode[16] + 2.0f*mode[18]);

  n_b[8*d_LBparams.numNodes + (d_LBparams.dimX+x-1)%d_LBparams.dimX + d_LBparams.dimX*((d_LBparams.dimY+y-1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*z] =
  1.0f/36.0f * (mode[0] - mode[1] - mode[2] + mode[4] + 2.0f*mode[6] + mode[7] - mode[10] - mode[11] - mode[13] - mode[14] + mode[16] + 
  2.0f*mode[18]);

  n_b[9*d_LBparams.numNodes + (x+1)%d_LBparams.dimX + d_LBparams.dimX*((d_LBparams.dimY+y-1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*z] =
  1.0f/36.0f * (mode[0] + mode[1] - mode[2] + mode[4] + 2.0f*mode[6] - mode[7] + mode[10] - mode[11] + mode[13] - mode[14] + mode[16] + 2.0f*mode[18]);

  n_b[10*d_LBparams.numNodes + (d_LBparams.dimX+x-1)%d_LBparams.dimX + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*z] = 
  1.0f/36.0f * (mode[0] - mode[1] + mode[2] + mode[4] + 2.0f*mode[6] - mode[7] - mode[10] + mode[11] - mode[13] + mode[14] + mode[16] + 2.0f*mode[18]);

  n_b[11*d_LBparams.numNodes + (x+1)%d_LBparams.dimX + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ)] =
  1.0f/36.0f * (mode[0] + mode[1] + mode[3] + mode[4] + mode[5] - mode[6] + mode[8] + mode[10] + mode[12] - mode[13] + mode[15] + mode[16] + mode[17] - mode[18]);

  n_b[12*d_LBparams.numNodes + (d_LBparams.dimX+x-1)%d_LBparams.dimX + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*((d_LBparams.dimZ+z-1)%d_LBparams.dimZ)] =
  1.0f/36.0f * (mode[0] - mode[1] - mode[3] + mode[4] + mode[5] - mode[6] + mode[8] - mode[10] - mode[12] + mode[13] - mode[15] + mode[16] + mode[17] - mode[18]);

  n_b[13*d_LBparams.numNodes + (x+1)%d_LBparams.dimX + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*((d_LBparams.dimZ+z-1)%d_LBparams.dimZ)] =
  1.0f/36.0f * (mode[0] + mode[1] - mode[3] + mode[4] + mode[5] - mode[6] - mode[8] + mode[10] - mode[12] - mode[13] - mode[15] + mode[16] + mode[17] - mode[18]);

  n_b[14*d_LBparams.numNodes + (d_LBparams.dimX+x-1)%d_LBparams.dimX + d_LBparams.dimX*y + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ)] =
  1.0f/36.0f * (mode[0] - mode[1] + mode[3] + mode[4] + mode[5] - mode[6] - mode[8] - mode[10] + mode[12] + mode[13] + mode[15] + mode[16] + mode[17] - mode[18]);

  n_b[15*d_LBparams.numNodes + x + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ)] =
  1.0f/36.0f * (mode[0] + mode[2] + mode[3] + mode[4] - mode[5] - mode[6] + mode[9] + mode[11] + mode[12] - mode[14] - mode[15] + mode[16] - mode[17] - mode[18]);

  n_b[16*d_LBparams.numNodes + x  + d_LBparams.dimX*((d_LBparams.dimY+y-1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*((d_LBparams.dimZ+z-1)%d_LBparams.dimZ)] =
  1.0f/36.0f * (mode[0] - mode[2] - mode[3] + mode[4] - mode[5] - mode[6] + mode[9] - mode[11] - mode[12] + mode[14] + mode[15] + mode[16] - mode[17] - mode[18]);

  n_b[17*d_LBparams.numNodes + x + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*((d_LBparams.dimZ+z-1)%d_LBparams.dimZ)] =
  1.0f/36.0f * (mode[0] + mode[2]- mode[3] + mode[4] - mode[5] - mode[6] - mode[9] + mode[11] - mode[12] - mode[14] + mode[15] + mode[16] - mode[17] - mode[18]);

  n_b[18*d_LBparams.numNodes + x + d_LBparams.dimX*((d_LBparams.dimY+y-1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ)] =
  1.0f/36.0f * (mode[0] - mode[2] + mode[3] + mode[4] - mode[5] - mode[6] - mode[9] - mode[11] + mode[12] + mode[14] - mode[15] + mode[16] - mode[17] - mode[18]);
}

// Note: suffix f
__device__ void bounce_back_boundaries (unsigned int index, unsigned int *boundary_map, float *boundary_velocities, float *n_curr, float *devBoundaryForces) {
    
  unsigned int boundaryIndex;
  float v[3]; 
  unsigned int xyz[3]; 
  float shift;
  float weight;  
  int c[3];
  float pop_to_bounce_back;
  unsigned int population;
  size_t to_index, to_index_x, to_index_y, to_index_z;
  float boundary_force[3] = {0.0f,0.0f,0.0f};
  unsigned int inverse;

  boundaryIndex = boundary_map[index];
  if (boundaryIndex != 0)
  {
    // Version 1: can assign a velocity value to each boundary
    //v[0] = boundary_velocities[(boundaryIndex-1)*3 + 0];
    //v[1] = boundary_velocities[(boundaryIndex-1)*3 + 1];
    //v[2] = boundary_velocities[(boundaryIndex-1)*3 + 2];

    // Version 2: only allow walls in the y direction to move
    if (boundaryIndex == 1) {
      v[0] = -0.5f * d_LBparams.boundaryVelocity[0];
      v[1] = 0.0f;
      v[2] = 0.0f;
    } else if (boundaryIndex == 2) {
      v[0] = 0.5f * d_LBparams.boundaryVelocity[0];
      v[1] = 0.0f;
      v[2] = 0.0f;
    } else {
      v[0] = 0.0f;
      v[1] = 0.0f;
      v[2] = 0.0f;
    }

    index_to_xyz (index, xyz);
    unsigned int x = xyz[0];
    unsigned int y = xyz[1];
    unsigned int z = xyz[2];

// TODO : PUT IN EQUILIBRIUM CONTRIBUTION TO THE BOUNCE-BACK DENSITY FOR THE BOUNDARY FORCE
// TODO : INITIALIZE BOUNDARY FORCE PROPERLY, HAS NONZERO ELEMENTS IN FIRST STEP
// TODO : SET INTERNAL BOUNDARY NODE VALUES TO ZERO

// Note:
// I have followed ESPResSo 4.1.2 and modified some of following code. 
// I am still not sure if thoses prefactors, agrid and tau, should appear or not!
// The following macro just serves as text replacement !

#define BOUNCEBACK() \
  shift = 2.0f * weight * d_LBparams.rho * (v[0]*c[0]+v[1]*c[1]+v[2]*c[2]) * 3.0f / d_LBparams.agrid * d_LBparams.tau; \
  pop_to_bounce_back = n_curr[population*d_LBparams.numNodes + index]; \
  to_index_x = (x+c[0]+d_LBparams.dimX) % d_LBparams.dimX; \
  to_index_y = (y+c[1]+d_LBparams.dimY) % d_LBparams.dimY; \
  to_index_z = (z+c[2]+d_LBparams.dimZ) % d_LBparams.dimZ; \
  to_index = to_index_x + d_LBparams.dimX*to_index_y + d_LBparams.dimX*d_LBparams.dimY*to_index_z; \
  if (boundary_map[to_index] == 0) { \
    boundary_force[0] += (2.0f * pop_to_bounce_back + shift) * c[0]; \
    boundary_force[1] += (2.0f * pop_to_bounce_back + shift) * c[1]; \
    boundary_force[2] += (2.0f * pop_to_bounce_back + shift) * c[2]; \
    n_curr[inverse*d_LBparams.numNodes + to_index] = pop_to_bounce_back + shift; \
  }
  
  // Note:
  // to_index: destination node
  // A minus sign is absorbed into the population velocity c, so the term pop_to_bounce_back + shift 
  // appears in the code above rather than pop_to_bounce_back - shift

    // the resting population does nothing, i.e., population 0.

    c[0]= 1;c[1]= 0;c[2]= 0; weight=1./18.; population= 2; inverse= 1; 
    BOUNCEBACK();
    
    c[0]=-1;c[1]= 0;c[2]= 0; weight=1./18.; population= 1; inverse= 2; 
    BOUNCEBACK();
    
    c[0]= 0;c[1]= 1;c[2]= 0; weight=1./18.; population= 4; inverse= 3; 
    BOUNCEBACK();

    c[0]= 0;c[1]=-1;c[2]= 0; weight=1./18.; population= 3; inverse= 4; 
    BOUNCEBACK();
    
    c[0]= 0;c[1]= 0;c[2]= 1; weight=1./18.; population= 6; inverse= 5; 
    BOUNCEBACK();

    c[0]= 0;c[1]= 0;c[2]=-1; weight=1./18.; population= 5; inverse= 6; 
    BOUNCEBACK(); 
    
    c[0]= 1;c[1]= 1;c[2]= 0; weight=1./36.; population= 8; inverse= 7; 
    BOUNCEBACK();
    
    c[0]=-1;c[1]=-1;c[2]= 0; weight=1./36.; population= 7; inverse= 8; 
    BOUNCEBACK();
    
    c[0]= 1;c[1]=-1;c[2]= 0; weight=1./36.; population=10; inverse= 9; 
    BOUNCEBACK();

    c[0]=-1;c[1]= 1;c[2]= 0; weight=1./36.; population= 9; inverse=10; 
    BOUNCEBACK();
    
    c[0]= 1;c[1]= 0;c[2]= 1; weight=1./36.; population=12; inverse=11; 
    BOUNCEBACK();
    
    c[0]=-1;c[1]= 0;c[2]=-1; weight=1./36.; population=11; inverse=12; 
    BOUNCEBACK();

    c[0]= 1;c[1]= 0;c[2]=-1; weight=1./36.; population=14; inverse=13; 
    BOUNCEBACK();
    
    c[0]=-1;c[1]= 0;c[2]= 1; weight=1./36.; population=13; inverse=14; 
    BOUNCEBACK();

    c[0]= 0;c[1]= 1;c[2]= 1; weight=1./36.; population=16; inverse=15; 
    BOUNCEBACK();
    
    c[0]= 0;c[1]=-1;c[2]=-1; weight=1./36.; population=15; inverse=16; 
    BOUNCEBACK();
    
    c[0]= 0;c[1]= 1;c[2]=-1; weight=1./36.; population=18; inverse=17; 
    BOUNCEBACK();
    
    c[0]= 0;c[1]=-1;c[2]= 1; weight=1./36.; population=17; inverse=18; 
    BOUNCEBACK();  
    
    atomicAdd(&devBoundaryForces[(boundaryIndex-1)*3 + 0], boundary_force[0]);
    atomicAdd(&devBoundaryForces[(boundaryIndex-1)*3 + 1], boundary_force[1]);
    atomicAdd(&devBoundaryForces[(boundaryIndex-1)*3 + 2], boundary_force[2]);
  }
}

__device__ void calc_values_in_LB_units (unsigned int index, unsigned int print_index, unsigned int *boundary_map, float *mode, float *ext_forces, LB_rho_v *d_v, LB_rho_v_pi *d_p_v) {
  
  float j[3]; 
  float modes_from_pi_eq[6]; 
  float pi[6]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};

  if (boundary_map[index] == 0) {
    /* Ensure we are working with the current values of d_v */
    update_rho_v (index, mode, ext_forces, d_v);
    d_p_v[print_index].rho  = d_v[index].rho;
    d_p_v[print_index].v[0] = d_v[index].v[0];
    d_p_v[print_index].v[1] = d_v[index].v[1];
    d_p_v[print_index].v[2] = d_v[index].v[2];

    /* stress calculation */ 
    float Rho     = d_v[index].rho;
    float inv_Rho = 1.0 / Rho;      
    /* note that d_v[index].v[] already includes the 1/2 f term, accounting for the pre- and post-collisional average */
    j[0] = Rho * d_v[index].v[0];
    j[1] = Rho * d_v[index].v[1];
    j[2] = Rho * d_v[index].v[2];

    // equilibrium part of the stress modes, which comes from 
    // the equality between modes and stress tensor components

    /* m4 = trace(pi) - rho
       m5 = pi_xx - pi_yy
       m6 = trace(pi) - 3 pi_zz
       m7 = pi_xy
       m8 = pi_xz
       m9 = pi_yz */

    // and pluggin in the Euler stress for the equilibrium:
    // pi_eq = rho_0*c_s^2*I3 + (j \otimes j)/rho
    // with I3 the 3D identity matrix and
    // rho = \trace(rho_0*c_s^2*I3), which yields

    /* m4_from_pi_eq = j.j
       m5_from_pi_eq = j_x*j_x - j_y*j_y
       m6_from_pi_eq = j.j - 3*j_z*j_z
       m7_from_pi_eq = j_x*j_y
       m8_from_pi_eq = j_x*j_z
       m9_from_pi_eq = j_y*j_z */

    // where the / Rho term has been dropped. We thus obtain: 

    modes_from_pi_eq[0] = (j[0]*j[0] + j[1]*j[1] + j[2]*j[2] ) * inv_Rho;
    modes_from_pi_eq[1] = (j[0]*j[0] - j[1]*j[1] ) * inv_Rho;
    modes_from_pi_eq[2] = (j[0]*j[0] + j[1]*j[1] + j[2]*j[2] - 3.0*j[2]*j[2]) * inv_Rho;
    modes_from_pi_eq[3] = j[0]*j[1] * inv_Rho;
    modes_from_pi_eq[4] = j[0]*j[2] * inv_Rho;
    modes_from_pi_eq[5] = j[1]*j[2] * inv_Rho;
     
    /* Now we must predict the outcome of the next collision */
    /* We immediately average pre- and post-collision.  */
    /* TODO: need a reference for this.   */

    mode[4] = modes_from_pi_eq[0] + (0.5 + 0.5*d_LBparams.gammaBulk)  * (mode[4] - modes_from_pi_eq[0]);
    mode[5] = modes_from_pi_eq[1] + (0.5 + 0.5*d_LBparams.gammaShear) * (mode[5] - modes_from_pi_eq[1]);
    mode[6] = modes_from_pi_eq[2] + (0.5 + 0.5*d_LBparams.gammaShear) * (mode[6] - modes_from_pi_eq[2]);
    mode[7] = modes_from_pi_eq[3] + (0.5 + 0.5*d_LBparams.gammaShear) * (mode[7] - modes_from_pi_eq[3]);
    mode[8] = modes_from_pi_eq[4] + (0.5 + 0.5*d_LBparams.gammaShear) * (mode[8] - modes_from_pi_eq[4]);
    mode[9] = modes_from_pi_eq[5] + (0.5 + 0.5*d_LBparams.gammaShear) * (mode[9] - modes_from_pi_eq[5]);

    // Transform the stress tensor components according to the modes that
    // correspond to those used by U. Schiller. In terms of populations this
    // expression then corresponds exactly to those in Eqs. 116 - 121 in the
    // Duenweg and Ladd paper, when these are written out in populations.
    // But to ensure this, the expression in Schiller's modes has to be different!

    pi[0] += (2.0f*(mode[0] + mode[4]) + mode[6] + 3.0f*mode[5]) / 6.0f; // xx
    pi[1] += mode[7];                                                    // xy
    pi[2] += (2.0f*(mode[0] + mode[4]) + mode[6] - 3.0f*mode[5]) / 6.0f; // yy
    pi[3] += mode[8];                                                    // xz
    pi[4] += mode[9];                                                    // yz
    pi[5] += (mode[0] + mode[4] - mode[6]) / 3.0f;                       // zz

     
    for(int i=0; i < 6; i++) {
      d_p_v[print_index].pi[i] = pi[i];
    }
  } else {
    d_p_v[print_index].rho = 0.0f;
     
    for (int i=0; i < 3; i++)
      d_p_v[print_index].v[i] = 0.0f;

    for(int i=0; i < 6; i++)
      d_p_v[print_index].pi[i] = 0.0f;
  }
}

__global__ void Print_dev_LB_Params() {

  printf("\nDim_x    Dim_y    Dim_z    Num_of_nodes\n");
  printf("%u    %u    %u    %u\n",d_LBparams.dimX, d_LBparams.dimY, d_LBparams.dimZ, d_LBparams.numNodes);
  printf("Num_of_boundaries    Boundary_vel.x    Boundary_vel.y    Boundary_vel.z\n");
  printf("%u    %f    %f    %f\n",
          d_LBparams.numBoundaries, d_LBparams.boundaryVelocity[0], d_LBparams.boundaryVelocity[1], d_LBparams.boundaryVelocity[2]);
  printf("Ext_force_flag    Ext_force_density.x    Ext_force_density.y    Ext_force_density.z\n");
  printf("%u    %f    %f    %f\n",
           d_LBparams.extForceFlag, d_LBparams.extForceDensity[0], d_LBparams.extForceDensity[1], d_LBparams.extForceDensity[2]);
  printf("Agrid    Tau    Rho\n");
  printf("%f    %f    %f\n", d_LBparams.agrid, d_LBparams.tau, d_LBparams.rho);
  printf("gamma_shear    gamma_bulk    gamma_odd    gamma_even\n");
  printf("%f    %f    %f    %f\n", d_LBparams.gammaShear, d_LBparams.gammaBulk, d_LBparams.gammaOdd, d_LBparams.gammaEven);

  printf("c_sound_sq = %f\n", c_sound_sq);
}

__global__ void PrintDevVariable () {
  printf ("The variable dimY equals to %u\n", d_LBparams.dimY);
}

__global__ void InitializeBoundaryMap (unsigned int *boundary_map) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int xyz[3];

  if (index < d_LBparams.numNodes) {
    index_to_xyz (index, xyz);
    unsigned int x = xyz[0];
    unsigned int y = xyz[1];
    unsigned int z = xyz[2];

    if (d_LBparams.numBoundaries >= 2) {
      if (y==0) {
        boundary_map[index] = 1;
      } else if (y==(d_LBparams.dimY-1)) {
        boundary_map[index] = 2;
      }
    }
    if (d_LBparams.numBoundaries == 4) {
      if (z==0) {
        boundary_map[index] = 3;
      } else if (z==(d_LBparams.dimZ-1)) {
        boundary_map[index] = 4;
      }
    }
  }   
}

__global__ void InitializeBodyForces (float *ext_forces) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  if (index < d_LBparams.numNodes) {
    //ext_forces[0*d_LBparams.numNodes + index] = d_LBparams.extForceDensity[0];
    //ext_forces[1*d_LBparams.numNodes + index] = d_LBparams.extForceDensity[1];
    //ext_forces[2*d_LBparams.numNodes + index] = d_LBparams.extForceDensity[2];   
    ext_forces[                      + index] = d_LBparams.extForceDensity[0];
    ext_forces[  d_LBparams.numNodes + index] = d_LBparams.extForceDensity[1];
    ext_forces[2*d_LBparams.numNodes + index] = d_LBparams.extForceDensity[2];   
  }
}

__global__ void calc_n_from_rho_j_pi (float *ext_forces, float *n_a, LB_rho_v *d_v) {
   /* TODO: this can handle only a uniform density, something similar, but local, 
            has to be called every time the fields are set by the user ! */ 
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  if (index < d_LBparams.numNodes)
  {
    float mode[19];

    float Rho   = d_LBparams.rho;
    float v[3]  = {0.0f, 0.0f, 0.0f};
    float pi[6] = {Rho*c_sound_sq, 0.0f, Rho*c_sound_sq, 0.0f, 0.0f, Rho*c_sound_sq};     
    float rhoc_sq = Rho * c_sound_sq;
    float avg_rho = d_LBparams.rho;
    float local_rho, local_j[3], *local_pi, trace;
     
    local_rho  = Rho;
    local_j[0] = Rho * v[0];
    local_j[1] = Rho * v[1];
    local_j[2] = Rho * v[2];
    local_pi   = pi;
     
    /** reduce the pressure tensor to the part needed here. 
        NOTE: this not true anymore for SHANCHEN 
        if the densities are not uniform. FIXME*/

    local_pi[0] -= rhoc_sq;
    local_pi[2] -= rhoc_sq;
    local_pi[5] -= rhoc_sq;
     
    trace = local_pi[0] + local_pi[2] + local_pi[5];
     
    float rho_times_coeff;
    float tmp1,tmp2;
     
    /** update the q=0 sublattice */
    n_a[0 * d_LBparams.numNodes + index] = 1.0f/3.0f * (local_rho-avg_rho) - 1.0f/2.0f*trace;
     
    /** update the q=1 sublattice */
    rho_times_coeff = 1.0f/18.0f * (local_rho-avg_rho);
     
    n_a[1 * d_LBparams.numNodes + index] = rho_times_coeff + 1.0f/6.0f*local_j[0] + 1.0f/4.0f*local_pi[0] - 1.0f/12.0f*trace;
    n_a[2 * d_LBparams.numNodes + index] = rho_times_coeff - 1.0f/6.0f*local_j[0] + 1.0f/4.0f*local_pi[0] - 1.0f/12.0f*trace;
    n_a[3 * d_LBparams.numNodes + index] = rho_times_coeff + 1.0f/6.0f*local_j[1] + 1.0f/4.0f*local_pi[2] - 1.0f/12.0f*trace;
    n_a[4 * d_LBparams.numNodes + index] = rho_times_coeff - 1.0f/6.0f*local_j[1] + 1.0f/4.0f*local_pi[2] - 1.0f/12.0f*trace;
    n_a[5 * d_LBparams.numNodes + index] = rho_times_coeff + 1.0f/6.0f*local_j[2] + 1.0f/4.0f*local_pi[5] - 1.0f/12.0f*trace;
    n_a[6 * d_LBparams.numNodes + index] = rho_times_coeff - 1.0f/6.0f*local_j[2] + 1.0f/4.0f*local_pi[5] - 1.0f/12.0f*trace;
     
    /** update the q=2 sublattice */
    rho_times_coeff = 1.0f/36.0f * (local_rho-avg_rho);
     
    tmp1 = local_pi[0] + local_pi[2];
    tmp2 = 2.0f*local_pi[1];
    n_a[ 7 * d_LBparams.numNodes+index] = rho_times_coeff + 1.0f/12.0f*(local_j[0]+local_j[1]) + 1.0f/8.0f*(tmp1+tmp2) - 1.0f/24.0f*trace;
    n_a[ 8 * d_LBparams.numNodes+index] = rho_times_coeff - 1.0f/12.0f*(local_j[0]+local_j[1]) + 1.0f/8.0f*(tmp1+tmp2) - 1.0f/24.0f*trace;
    n_a[ 9 * d_LBparams.numNodes+index] = rho_times_coeff + 1.0f/12.0f*(local_j[0]-local_j[1]) + 1.0f/8.0f*(tmp1-tmp2) - 1.0f/24.0f*trace;
    n_a[10 * d_LBparams.numNodes+index] = rho_times_coeff - 1.0f/12.0f*(local_j[0]-local_j[1]) + 1.0f/8.0f*(tmp1-tmp2) - 1.0f/24.0f*trace;
     
    tmp1 = local_pi[0] + local_pi[5];
    tmp2 = 2.0f*local_pi[3];
     
    n_a[11 * d_LBparams.numNodes + index] = rho_times_coeff + 1.0f/12.0f*(local_j[0]+local_j[2]) + 1.0f/8.0f*(tmp1+tmp2) - 1.0f/24.0f*trace;
    n_a[12 * d_LBparams.numNodes + index] = rho_times_coeff - 1.0f/12.0f*(local_j[0]+local_j[2]) + 1.0f/8.0f*(tmp1+tmp2) - 1.0f/24.0f*trace;
    n_a[13 * d_LBparams.numNodes + index] = rho_times_coeff + 1.0f/12.0f*(local_j[0]-local_j[2]) + 1.0f/8.0f*(tmp1-tmp2) - 1.0f/24.0f*trace;
    n_a[14 * d_LBparams.numNodes + index] = rho_times_coeff - 1.0f/12.0f*(local_j[0]-local_j[2]) + 1.0f/8.0f*(tmp1-tmp2) - 1.0f/24.0f*trace;
     
    tmp1 = local_pi[2] + local_pi[5];
    tmp2 = 2.0f*local_pi[4];
     
    n_a[15 * d_LBparams.numNodes + index] = rho_times_coeff + 1.0f/12.0f*(local_j[1]+local_j[2]) + 1.0f/8.0f*(tmp1+tmp2) - 1.0f/24.0f*trace;
    n_a[16 * d_LBparams.numNodes + index] = rho_times_coeff - 1.0f/12.0f*(local_j[1]+local_j[2]) + 1.0f/8.0f*(tmp1+tmp2) - 1.0f/24.0f*trace;
    n_a[17 * d_LBparams.numNodes + index] = rho_times_coeff + 1.0f/12.0f*(local_j[1]-local_j[2]) + 1.0f/8.0f*(tmp1-tmp2) - 1.0f/24.0f*trace;
    n_a[18 * d_LBparams.numNodes + index] = rho_times_coeff - 1.0f/12.0f*(local_j[1]-local_j[2]) + 1.0f/8.0f*(tmp1-tmp2) - 1.0f/24.0f*trace;
     
    /**set different seed for randomgen on every node */
//      n_a.seed[index] = para.your_seed + index;
    
    calc_m_from_n (index, n_a, mode);
    update_rho_v (index, mode, ext_forces, d_v);
  }
}

__global__ void integrate (float *n_a, float *ext_forces, LB_rho_v *d_v, float *n_b) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  float mode[19];  // the 19 moments (modes) are only temporary register values

  if (index < d_LBparams.numNodes) {
    calc_m_from_n          (index, n_a, mode);
    relax_modes            (index, ext_forces, d_v, mode);
    apply_forces           (index, ext_forces, d_v, mode); 
    normalize_modes        (mode);
    calc_n_from_modes_push (index, mode, n_b);
  }  
}

__global__ void apply_boundaries (unsigned int *boundary_map,  float *boundary_velocities, float *n_curr, float *boundary_forces) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if (index < d_LBparams.numNodes) {
    bounce_back_boundaries (index, boundary_map, boundary_velocities, n_curr, boundary_forces);
  }
}

__global__ void get_mesoscopic_values_in_LB_units (float *n_a, unsigned int *boundary_map, float *ext_forces, LB_rho_v *d_v, LB_rho_v_pi *d_p_v) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  if (index < d_LBparams.numNodes) {
    float mode[19];
    calc_m_from_n (index, n_a, mode);
    calc_values_in_LB_units (index, index, boundary_map, mode, ext_forces, d_v, d_p_v);
  }
}

extern "C"
void Initialize_LB_Parameters_dev (LBparameters *h_LBparams) {
  cuda_safe_mem(cudaMemcpyToSymbol(d_LBparams, h_LBparams, sizeof(LBparameters)));
}

extern "C"
void PrintDevParasWrapper () {
  Print_dev_LB_Params<<<1,1>>>();
  PrintDevVariable <<<1,1>>> ();
}

extern "C"
void InitializeLB () {

#define free_realloc_and_clear(var, size) \
  { \
    if ((var) != NULL) cudaFree((var)); \
    cuda_safe_mem(cudaMalloc((void**)&var, size)); \
    cudaMemset(var, 0, size); \
  } 

  cuda_safe_mem(cudaMemcpyToSymbol(d_LBparams, &h_LBparams, sizeof(LBparameters)));
  free_realloc_and_clear(devNodesA,      h_LBparams.numNodes * 19 * sizeof(float));
  free_realloc_and_clear(devNodesB,      h_LBparams.numNodes * 19 * sizeof(float));   
  free_realloc_and_clear(devBoundaryMap, h_LBparams.numNodes * sizeof(unsigned int));
  free_realloc_and_clear(devExtForces,   h_LBparams.numNodes * 3 * sizeof(float));
  free_realloc_and_clear(devBoundaryForces, h_LBparams.numBoundaries*3*sizeof(float));
  free_realloc_and_clear(devBoundaryVelocities, h_LBparams.numBoundaries*3*sizeof(float));
  size_of_rho_v     = h_LBparams.numNodes * sizeof(LB_rho_v);
  size_of_rho_v_pi  = h_LBparams.numNodes * sizeof(LB_rho_v_pi);
  free_realloc_and_clear(devRhoV, size_of_rho_v);
  /* TODO: this is a almost a copy of  device_rho_v think about eliminating it, and maybe pi can be added to device_rho_v in this case*/
  free_realloc_and_clear(print_rho_v_pi, size_of_rho_v_pi);
  // Note: discarded design
  ///**check flag if lb gpu init works*/
  //free_and_realloc(gpu_check, sizeof(int));
  //if (h_gpu_check != NULL)
  //  free (h_gpu_check);  
  //h_gpu_check = (int*)malloc(sizeof(int));
  //h_gpu_check[0] = 0;

  //hostExtForces = (float *)calloc(h_LBparams.numNodes*3, sizeof(float)); // memory needs to be released at the end !

  // values for the kernel call 
  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_LBparams.numNodes + threads_per_block * blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);

  KERNELCALL(InitializeBoundaryMap, dim_grid, threads_per_block, (devBoundaryMap));
  //cuda_safe_mem(cudaMemcpy(devBoundaryVelocities, hostBoundaryVelocities, 3*h_LBparams.numBoundaries*sizeof(float), cudaMemcpyHostToDevice));
  KERNELCALL(InitializeBodyForces, dim_grid, threads_per_block, (devExtForces));
  KERNELCALL(calc_n_from_rho_j_pi, dim_grid, threads_per_block, (devExtForces, devNodesA, devRhoV));
  intflag = 1;
  devCurrentNodes = devNodesA;
  // Note: discarded design
  //cuda_safe_mem(cudaMemcpy (h_gpu_check, gpu_check, sizeof(int), cudaMemcpyDeviceToHost));

//fprintf(stderr, "initialization of lb gpu code %i\n", h_LBparams.numNodes);

  cudaDeviceSynchronize();

  // Note: discarded design
  //#if __CUDA_ARCH__ >= 200
  //if(!h_gpu_check[0])
  //{
  //  fprintf(stderr, "initialization of lb gpu code failed! \n");
  //  errexit();
  //}
  //#endif
}

extern "C"
void UpdateLBE () {

  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_LBparams.numNodes + threads_per_block * blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  // call of fluid step
  /* NOTE: if pi is needed at every integration step, one should call an extended version 
           of the integrate kernel, or pass also devRhoV_pi and make sure that either 
           it or devRhoV are NULL depending on extended_values_flag */ 
  if (intflag == 1) {
    KERNELCALL(integrate, dim_grid, threads_per_block, (devNodesA, devExtForces, devRhoV, devNodesB));
    devCurrentNodes = devNodesB;
    intflag = 0;
  } else {
    KERNELCALL(integrate, dim_grid, threads_per_block, (devNodesB, devExtForces, devRhoV, devNodesA));
    devCurrentNodes = devNodesA;
    intflag = 1;
  }

  if (h_LBparams.numBoundaries > 0) {
    // Version 1: can assign a velocity value to each boundary
    //SetBoundaryVelocities ();
    //KERNELCALL(apply_boundaries, dim_grid, threads_per_block, (devBoundaryMap, devBoundaryVelocities, devCurrentNodes, devBoundaryForces));

    // Version 2: only allow walls in the y direction to move
    KERNELCALL(apply_boundaries, dim_grid, threads_per_block, (devBoundaryMap, devBoundaryVelocities, devCurrentNodes, devBoundaryForces));
  }
}

extern "C"
void SetBoundaryVelocities () {
  
  float *hostBoundaryVelocities;
  hostBoundaryVelocities = (float *) calloc (h_LBparams.numBoundaries*3, sizeof(float));
  hostBoundaryVelocities[0*3+0] = -0.5*h_LBparams.boundaryVelocity[0];
  hostBoundaryVelocities[0*3+1] =  0.0;
  hostBoundaryVelocities[0*3+2] =  0.0;
  hostBoundaryVelocities[1*3+0] =  0.5*h_LBparams.boundaryVelocity[0];
  hostBoundaryVelocities[1*3+1] =  0.0;
  hostBoundaryVelocities[1*3+2] =  0.0;
  cuda_safe_mem(cudaMemcpy(devBoundaryVelocities, hostBoundaryVelocities, 3*h_LBparams.numBoundaries*sizeof(float), cudaMemcpyHostToDevice));
  free (hostBoundaryVelocities);
}

extern "C"
void lb_get_values_GPU (LB_rho_v_pi *host_values) {

  // values for the kernel call
  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_LBparams.numNodes + threads_per_block * blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3(blocks_per_grid_x, blocks_per_grid_y, 1);

  KERNELCALL(get_mesoscopic_values_in_LB_units, dim_grid, threads_per_block, (devCurrentNodes, devBoundaryMap, devExtForces, devRhoV, print_rho_v_pi));
  cuda_safe_mem(cudaMemcpy(host_values, print_rho_v_pi, size_of_rho_v_pi, cudaMemcpyDeviceToHost));
}

extern "C"
void PrintFluidVelocitiesVTK (char *filePath) {

  LB_rho_v_pi *host_values;
  size_t size_of_values = h_LBparams.numNodes * sizeof(LB_rho_v_pi);
  host_values = (LB_rho_v_pi*) malloc (size_of_values);

  lb_get_values_GPU (host_values);

  FILE* fp = fopen (filePath, "w");
  fprintf (fp, "# vtk DataFile Version 2.0\nlbfluid_gpu\n"
           "ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\n"
           "ORIGIN %f %f %f\nSPACING %f %f %f\nPOINT_DATA %u\n"
           "SCALARS velocity float 3\nLOOKUP_TABLE default\n",
           h_LBparams.dimX, h_LBparams.dimY, h_LBparams.dimZ,
           h_LBparams.agrid*0.5, h_LBparams.agrid*0.5, h_LBparams.agrid*0.5,
           h_LBparams.agrid, h_LBparams.agrid, h_LBparams.agrid,
           h_LBparams.numNodes);
  for(int j=0; j < int(h_LBparams.numNodes); ++j) {
    fprintf (fp, "%f %f %f\n", host_values[j].v[0], host_values[j].v[1], host_values[j].v[2]);
  }
  fclose (fp);
  free (host_values);
}

