#include "cuda_utils.hpp"

extern "C"{
#include <stdio.h>
#include "coupling.h"
#include "lb.h"
#include "sphere_param.h"
}

/* Global variables declared in lb.h

extern LBparameters h_LBparams;
extern unsigned int *devBoundaryMap;
extern float *devExtForces;
extern float *devCurrentNodes;
extern float *devBoundaryVelocities;
extern float *hostExtForces;
*/
extern __constant__ LBparameters d_LBparams;
extern __constant__ struct sphere_param d_partParams;


__device__ void Calc_m_from_n_IBM (const unsigned int index, float *n, float *mode) {
  
  mode[0] = n[0*d_LBparams.numNodes + index]
  + n[ 1 * d_LBparams.numNodes + index] + n[ 2 * d_LBparams.numNodes + index]
  + n[ 3 * d_LBparams.numNodes + index] + n[ 4 * d_LBparams.numNodes + index]
  + n[ 5 * d_LBparams.numNodes + index] + n[ 6 * d_LBparams.numNodes + index]
  + n[ 7 * d_LBparams.numNodes + index] + n[ 8 * d_LBparams.numNodes + index]
  + n[ 9 * d_LBparams.numNodes + index] + n[10 * d_LBparams.numNodes + index]
  + n[11 * d_LBparams.numNodes + index] + n[12 * d_LBparams.numNodes + index]
  + n[13 * d_LBparams.numNodes + index] + n[14 * d_LBparams.numNodes + index]
  + n[15 * d_LBparams.numNodes + index] + n[16 * d_LBparams.numNodes + index]
  + n[17 * d_LBparams.numNodes + index] + n[18 * d_LBparams.numNodes + index];

  mode[1] = (n[1 * d_LBparams.numNodes + index] - n[2 * d_LBparams.numNodes + index])
  + (n[ 7 * d_LBparams.numNodes + index] - n[ 8 * d_LBparams.numNodes + index])
  + (n[ 9 * d_LBparams.numNodes + index] - n[10 * d_LBparams.numNodes + index])
  + (n[11 * d_LBparams.numNodes + index] - n[12 * d_LBparams.numNodes + index])
  + (n[13 * d_LBparams.numNodes + index] - n[14 * d_LBparams.numNodes + index]);

  mode[2] = (n[3 * d_LBparams.numNodes + index] - n[4 * d_LBparams.numNodes + index])
  + (n[ 7 * d_LBparams.numNodes + index] - n[ 8 * d_LBparams.numNodes + index])
  - (n[ 9 * d_LBparams.numNodes + index] - n[10 * d_LBparams.numNodes + index])
  + (n[15 * d_LBparams.numNodes + index] - n[16 * d_LBparams.numNodes + index])
  + (n[17 * d_LBparams.numNodes + index] - n[18 * d_LBparams.numNodes + index]);

  mode[3] = (n[5 * d_LBparams.numNodes + index] - n[6 * d_LBparams.numNodes + index])
  + (n[11 * d_LBparams.numNodes + index] - n[12 * d_LBparams.numNodes + index])
  - (n[13 * d_LBparams.numNodes + index] - n[14 * d_LBparams.numNodes + index])
  + (n[15 * d_LBparams.numNodes + index] - n[16 * d_LBparams.numNodes + index])
  - (n[17 * d_LBparams.numNodes + index] - n[18 * d_LBparams.numNodes + index]);
}

__global__ void ResetBodyForces (float *devExtForces) {

  const unsigned int nodeIndex = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
  
  if (nodeIndex < d_LBparams.numNodes) {
    devExtForces[                        nodeIndex] = d_LBparams.extForceDensity[0];
    devExtForces[  d_LBparams.numNodes + nodeIndex] = d_LBparams.extForceDensity[1];
    devExtForces[2*d_LBparams.numNodes + nodeIndex] = d_LBparams.extForceDensity[2];
  }
}

__global__ void ForcesIntoFluid_TwoPoint (double *devPositions, double *devForces, float *devExtForces) {

  const unsigned int markerIndex = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if (markerIndex < d_partParams.num_beads)
  {
    unsigned int offset = markerIndex*3;
    const float pos[3] = {       devPositions[offset+0], devPositions[offset+1], devPositions[offset+2]};
    const float particleForce[3] = {devForces[offset+0],    devForces[offset+1],    devForces[offset+2]};

    // First part is the same as for interpolation --> merge into a single function
    float temp_delta[6];
    float delta[8];
    int my_left[3];
    int node_index[8];
    for(int i=0; i<3; ++i)
    {
      const float scaledpos = pos[i]/d_LBparams.agrid - 0.5f;
      my_left[i] = (int)(floorf(scaledpos));
      temp_delta[3+i] = scaledpos - my_left[i];
      temp_delta[i] = 1.f - temp_delta[3+i];
    }

    delta[0] = temp_delta[0] * temp_delta[1] * temp_delta[2];
    delta[1] = temp_delta[3] * temp_delta[1] * temp_delta[2];
    delta[2] = temp_delta[0] * temp_delta[4] * temp_delta[2];
    delta[3] = temp_delta[3] * temp_delta[4] * temp_delta[2];
    delta[4] = temp_delta[0] * temp_delta[1] * temp_delta[5];
    delta[5] = temp_delta[3] * temp_delta[1] * temp_delta[5];
    delta[6] = temp_delta[0] * temp_delta[4] * temp_delta[5];
    delta[7] = temp_delta[3] * temp_delta[4] * temp_delta[5];

    // modulo for negative numbers is strange at best, shift to make sure we are positive
    const int x = my_left[0] + d_LBparams.dimX;
    const int y = my_left[1] + d_LBparams.dimY;
    const int z = my_left[2] + d_LBparams.dimZ;

    // Note: Will there be a problem when markers are just next to the boundary nodes ?
    //       The wall force prevents markers to be next to the boundary nodes
    node_index[0] = x%d_LBparams.dimX     + d_LBparams.dimX*(y%d_LBparams.dimY)     + d_LBparams.dimX*d_LBparams.dimY*(z%d_LBparams.dimZ);
    node_index[1] = (x+1)%d_LBparams.dimX + d_LBparams.dimX*(y%d_LBparams.dimY)     + d_LBparams.dimX*d_LBparams.dimY*(z%d_LBparams.dimZ);
    node_index[2] = x%d_LBparams.dimX     + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*(z%d_LBparams.dimZ);
    node_index[3] = (x+1)%d_LBparams.dimX + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*(z%d_LBparams.dimZ);
    node_index[4] = x%d_LBparams.dimX     + d_LBparams.dimX*(y%d_LBparams.dimY)     + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ);
    node_index[5] = (x+1)%d_LBparams.dimX + d_LBparams.dimX*(y%d_LBparams.dimY)     + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ);
    node_index[6] = x%d_LBparams.dimX     + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ);
    node_index[7] = (x+1)%d_LBparams.dimX + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ);

    for(int i=0; i<8; ++i)
    {
      // Atomic add is essential because this runs in parallel!
      atomicAdd(&devExtForces[0*d_LBparams.numNodes + node_index[i]], (particleForce[0] * delta[i]));
      atomicAdd(&devExtForces[1*d_LBparams.numNodes + node_index[i]], (particleForce[1] * delta[i]));
      atomicAdd(&devExtForces[2*d_LBparams.numNodes + node_index[i]], (particleForce[2] * delta[i]));
    }
  }
}

//__global__ void ForcesIntoFluid_TwoPoint_single (float *devPositions, float *devForces, float *devExtForces) {
//
//  const unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;
//
//  if (index == 0)
//  {
//    for (int markerIndex=0; markerIndex < numMarkers; markerIndex++) {
//      
//      const float pos[3] = {       devPositions[markerIndex*3+0], devPositions[markerIndex*3+1], devPositions[markerIndex*3+2]};
//      const float particleForce[3] = {devForces[markerIndex*3+0],    devForces[markerIndex*3+1],    devForces[markerIndex*3+2]};
//
////printf("marker = %d  pos=(%f %f %f)    f=(%f %f %f)\n", markerIndex, pos[0], pos[1], pos[2], particleForce[0], particleForce[1], particleForce[2]);
//
//      // First part is the same as for interpolation --> merge into a single function
//      float temp_delta[6];
//      float delta[8];
//      int my_left[3];
//      int node_index[8];
//      for(int i=0; i<3; ++i)
//      {
//        const float scaledpos = pos[i]/devLBparas.agrid - 0.5f;
//        my_left[i] = (int)(floorf(scaledpos));
//        temp_delta[3+i] = scaledpos - my_left[i];
//        temp_delta[i] = 1.f - temp_delta[3+i];
//      }
//
//      delta[0] = temp_delta[0] * temp_delta[1] * temp_delta[2];
//      delta[1] = temp_delta[3] * temp_delta[1] * temp_delta[2];
//      delta[2] = temp_delta[0] * temp_delta[4] * temp_delta[2];
//      delta[3] = temp_delta[3] * temp_delta[4] * temp_delta[2];
//      delta[4] = temp_delta[0] * temp_delta[1] * temp_delta[5];
//      delta[5] = temp_delta[3] * temp_delta[1] * temp_delta[5];
//      delta[6] = temp_delta[0] * temp_delta[4] * temp_delta[5];
//      delta[7] = temp_delta[3] * temp_delta[4] * temp_delta[5];
//
//      // modulo for negative numbers is strange at best, shift to make sure we are positive
//      const int x = my_left[0] + devLBparas.dimX;
//      const int y = my_left[1] + devLBparas.dimY;
//      const int z = my_left[2] + devLBparas.dimZ;
//
//      node_index[0] = x%devLBparas.dimX     + devLBparas.dimX*(y%devLBparas.dimY)     + devLBparas.dimX*devLBparas.dimY*(z%devLBparas.dimZ);
//      node_index[1] = (x+1)%devLBparas.dimX + devLBparas.dimX*(y%devLBparas.dimY)     + devLBparas.dimX*devLBparas.dimY*(z%devLBparas.dimZ);
//      node_index[2] = x%devLBparas.dimX     + devLBparas.dimX*((y+1)%devLBparas.dimY) + devLBparas.dimX*devLBparas.dimY*(z%devLBparas.dimZ);
//      node_index[3] = (x+1)%devLBparas.dimX + devLBparas.dimX*((y+1)%devLBparas.dimY) + devLBparas.dimX*devLBparas.dimY*(z%devLBparas.dimZ);
//      node_index[4] = x%devLBparas.dimX     + devLBparas.dimX*(y%devLBparas.dimY)     + devLBparas.dimX*devLBparas.dimY*((z+1)%devLBparas.dimZ);
//      node_index[5] = (x+1)%devLBparas.dimX + devLBparas.dimX*(y%devLBparas.dimY)     + devLBparas.dimX*devLBparas.dimY*((z+1)%devLBparas.dimZ);
//      node_index[6] = x%devLBparas.dimX     + devLBparas.dimX*((y+1)%devLBparas.dimY) + devLBparas.dimX*devLBparas.dimY*((z+1)%devLBparas.dimZ);
//      node_index[7] = (x+1)%devLBparas.dimX + devLBparas.dimX*((y+1)%devLBparas.dimY) + devLBparas.dimX*devLBparas.dimY*((z+1)%devLBparas.dimZ);
//
//      for(int i=0; i<8; ++i)
//      {
//        devExtForces[0*devLBparas.numNodes + node_index[i]] += (particleForce[0] * delta[i]);
//        devExtForces[1*devLBparas.numNodes + node_index[i]] += (particleForce[1] * delta[i]);
//        devExtForces[2*devLBparas.numNodes + node_index[i]] += (particleForce[2] * delta[i]);
//      }
//    }
//  }
//}

// Note: double !
__global__ void VelocitiesFromFluid_TwoPoint (double *devPositions, unsigned int *boundary_map, float *devBoundaryVelocities, float *n_curr, float *devExtForces, double *devVelocities) {

  const unsigned int markerIndex = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x;

  if (markerIndex < d_partParams.num_beads)
  {
    unsigned int offset = markerIndex*3;
    float pos[3] = {devPositions[offset+0], devPositions[offset+1], devPositions[offset+2]};
    float v[3] = { 0.0f };

    // ***** This part is copied from get_interpolated_velocity
    // ***** + we add the force + we consider boundaries - we remove the Shan-Chen stuff
    float temp_delta[6];
    float delta[8];
    int my_left[3];
    int node_index[8];
    float mode[4];
    #pragma unroll
    for (int i=0; i<3; ++i) {
      const float scaledpos = pos[i]/d_LBparams.agrid - 0.5f;
      my_left[i] = (int)(floorf(scaledpos));
      temp_delta[3+i] = scaledpos - my_left[i];
      temp_delta[i] = 1.f - temp_delta[3+i];
    }

    delta[0] = temp_delta[0] * temp_delta[1] * temp_delta[2];
    delta[1] = temp_delta[3] * temp_delta[1] * temp_delta[2];
    delta[2] = temp_delta[0] * temp_delta[4] * temp_delta[2];
    delta[3] = temp_delta[3] * temp_delta[4] * temp_delta[2];
    delta[4] = temp_delta[0] * temp_delta[1] * temp_delta[5];
    delta[5] = temp_delta[3] * temp_delta[1] * temp_delta[5];
    delta[6] = temp_delta[0] * temp_delta[4] * temp_delta[5];
    delta[7] = temp_delta[3] * temp_delta[4] * temp_delta[5];

    // modulo for negative numbers is strange at best, shift to make sure we are positive
    int x = my_left[0] + d_LBparams.dimX;
    int y = my_left[1] + d_LBparams.dimY;
    int z = my_left[2] + d_LBparams.dimZ;

    node_index[0] = x%d_LBparams.dimX     + d_LBparams.dimX*(y%d_LBparams.dimY)     + d_LBparams.dimX*d_LBparams.dimY*(z%d_LBparams.dimZ);
    node_index[1] = (x+1)%d_LBparams.dimX + d_LBparams.dimX*(y%d_LBparams.dimY)     + d_LBparams.dimX*d_LBparams.dimY*(z%d_LBparams.dimZ);
    node_index[2] = x%d_LBparams.dimX     + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*(z%d_LBparams.dimZ);
    node_index[3] = (x+1)%d_LBparams.dimX + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*(z%d_LBparams.dimZ);
    node_index[4] = x%d_LBparams.dimX     + d_LBparams.dimX*(y%d_LBparams.dimY)     + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ);
    node_index[5] = (x+1)%d_LBparams.dimX + d_LBparams.dimX*(y%d_LBparams.dimY)     + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ);
    node_index[6] = x%d_LBparams.dimX     + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ);
    node_index[7] = (x+1)%d_LBparams.dimX + d_LBparams.dimX*((y+1)%d_LBparams.dimY) + d_LBparams.dimX*d_LBparams.dimY*((z+1)%d_LBparams.dimZ);

    for(int i=0; i<8; ++i)
    {
       double local_rho;
       double local_j[3];

      //FIXME: The interpolated velocity is not accurate as the marker is just next to the boundary node
      //#ifdef LB_BOUNDARIES_GPU
      if (boundary_map[node_index[i]])  // boundary node
      {
        // Version 1: Bayreuth version   
        //const int boundary_index = boundary_map[node_index[i]];
        ////local_rho  = d_LBparams.rho + mode[0];
        //local_rho  = d_LBparams.rho;
        //local_j[0] = local_rho * devBoundaryVelocities[3*(boundary_index-1)+0];
        //local_j[1] = local_rho * devBoundaryVelocities[3*(boundary_index-1)+1];
        //local_j[2] = local_rho * devBoundaryVelocities[3*(boundary_index-1)+2];

        // Version 2: only allow walls in the y direction to move
        unsigned int boundary_index = boundary_map[node_index[i]];
        float boundaryVel[3];  
        if (boundary_index == 1) {
          boundaryVel[0] = -0.5f * d_LBparams.boundaryVelocity[0];
          boundaryVel[1] = 0.0f;
          boundaryVel[2] = 0.0f;
        } else if (boundary_index == 2) {
          boundaryVel[0] = 0.5f * d_LBparams.boundaryVelocity[0];
          boundaryVel[1] = 0.0f;
          boundaryVel[2] = 0.0f;
        } else {
          boundaryVel[0] = 0.0f;
          boundaryVel[1] = 0.0f;
          boundaryVel[2] = 0.0f;
        }
        local_j[0] = d_LBparams.rho * boundaryVel[0];
        local_j[1] = 0.0f;
        local_j[2] = 0.0f;
      }
      else
      //#endif
      {
        Calc_m_from_n_IBM (node_index[i], n_curr, mode);

        local_rho = d_LBparams.rho + mode[0];

        // Add the +f/2 contribution!!
        local_j[0] = mode[1] + 0.5 * devExtForces[0*d_LBparams.numNodes + node_index[i]];
        local_j[1] = mode[2] + 0.5 * devExtForces[1*d_LBparams.numNodes + node_index[i]];
        local_j[2] = mode[3] + 0.5 * devExtForces[2*d_LBparams.numNodes + node_index[i]];
      }
      // Interpolate velocity
      v[0] += delta[i]*local_j[0] / local_rho;
      v[1] += delta[i]*local_j[1] / local_rho;
      v[2] += delta[i]*local_j[2] / local_rho;
    }
    devVelocities[offset+0] = v[0];
    devVelocities[offset+1] = v[1];
    devVelocities[offset+2] = v[2];

//printf("index=%u    pos=(%f %f %f)\n",pos[0],pos[1],pos[2]);
//if(v[0] > 0.1 || v[1] > 0.1 || v[2] > 0.1)
//printf("index=%u    vel=(%f %f %f)\n",markerIndex, devMonomers[markerIndex].vel[0], devMonomers[markerIndex].vel[1], devMonomers[markerIndex].vel[2]);
  }
}

//__global__ void RearrangeOutputData() {
//
//  const unsigned int markerIndex = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
//
//  if (markerIndex < numMarkers) {
//    devMonomers[markerIndex].vel[0] = devVelocities[markerIndex*3+0];
//    devMonomers[markerIndex].vel[1] = devVelocities[markerIndex*3+1];
//    devMonomers[markerIndex].vel[2] = devVelocities[markerIndex*3+2];
//  }
//}

//__global__ void PrintForDebugging(struct monomer *devMonomers, float *devPositions, float *devForces, float *devExtForces) {
//
////  printf("Dim_x    Dim_y    Dim_z    Num_of_nodes\n");
////  printf("%u    %u    %u    %u\n",devLBparas.dimX, devLBparas.dimY, devLBparas.dimZ, devLBparas.numNodes);
////  printf("Num_of_boundaries    Boundary_vel.x    Boundary_vel.y    Boundary_vel.z\n");
////  printf("%u    %f    %f    %f\n",
////          devLBparas.numBoundaries, devLBparas.boundaryVelocity[0], devLBparas.boundaryVelocity[1], devLBparas.boundaryVelocity[2]);
////  printf("Ext_force_flag    Ext_force_density.x    Ext_force_density.y    Ext_force_density.z\n");
////  printf("%d    %f    %f    %f\n",
////          devLBparas.extForceFlag, devLBparas.extForceDensity[0], devLBparas.extForceDensity[1], devLBparas.extForceDensity[2]);
////  printf("Agrid    Tau    Rho\n");
////  printf("%f    %f    %f\n", devLBparas.agrid, devLBparas.tau, devLBparas.rho);
////  printf("gamma_shear    gamma_bulk    gamma_odd    gamma_even\n");
////  printf("%f    %f    %f    %f\n", devLBparas.gammaShear, devLBparas.gammaBulk, devLBparas.gammaOdd, devLBparas.gammaEven);
////  printf("%u\n", numMarkers);
//
////  for (int n=0; n < numMarkers; n++) {
////    printf ("index %d  pos=(%f %f %f) force=(%f %f %f) vel=(%f %f %f)\n", n, 
////           devMonomers[n].pos[0], devMonomers[n].pos[1], devMonomers[n].pos[2],
////           devMonomers[n].force[0], devMonomers[n].force[1], devMonomers[n].force[2],
////           devMonomers[n].vel[0], devMonomers[n].vel[1], devMonomers[n].vel[2]);
////  }
////  for (int n=0; n < numMarkers; n++) {
////    printf ("index %d  posDiff=(%f %f %f) forceDiff=(%f %f %f)\n", n, 
////           devMonomers[n].pos[0]-devPositions[n*3+0], devMonomers[n].pos[1]-devPositions[n*3+1], devMonomers[n].pos[2]-devPositions[n*3+2],
////           devMonomers[n].force[0]-devForces[n*3+0], devMonomers[n].force[1]-devForces[n*3+1], devMonomers[n].force[2]-devForces[n*3+2]);
////  }
//  for (int n=0; n < devLBparas.numNodes; n++) {
//    if(devExtForces[n] > 1) {
//      printf ("index %d  force=(%f %f %f)\n", n, devExtForces[0*devLBparas.numNodes+n], devExtForces[1*devLBparas.numNodes+n], devExtForces[2*devLBparas.numNodes+n]);
//    }
//  }
//}

//extern "C"
//void InitializeCoupling (const unsigned int numMarkers) {
//
//  #define free_realloc_and_clear(var, size)        \
//  {                                                \
//    if ((var) != NULL) cudaFree((var));            \
//    cuda_safe_mem(cudaMalloc((void**)&var, size)); \
//    cudaMemset(var, 0, size);                      \
//  } 
////  free_realloc_and_clear(devPositions,          numMarkers*3*sizeof(float));
////  free_realloc_and_clear(devForces,             numMarkers*3*sizeof(float));
////  free_realloc_and_clear(devVelocities,         numMarkers*3*sizeof(float));
//  cuda_safe_mem(cudaMemcpyToSymbol(devLBparas, &hostLBparas, sizeof(LBparameters)));
//  cuda_safe_mem(cudaMemcpyToSymbol(devNumMarkers, &numMarkers, sizeof(unsigned int)));
//}

extern "C"
void SpreadForceDensities (const unsigned int h_numFluidNodes, const int h_numMarkers, double *devPositions, double *devForces, float *devExtForces) {
//  if (devPositions == NULL || devForces == NULL) {
//    InitializeCoupling (numMarkers);
//  }
  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_numFluidNodes + threads_per_block * blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);
  // Note: Device variables cannot be read from the host !
  //KERNELCALL(ResetBodyForces, dim_grid, threads_per_block, (d_d_LBparams.numNodes, d_d_LBparams.extForceDensity, devExtForces));
  KERNELCALL(ResetBodyForces, dim_grid, threads_per_block, (devExtForces));

  threads_per_block = 64;
  blocks_per_grid_y = 4;
  blocks_per_grid_x = (h_numMarkers + threads_per_block * blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);
  KERNELCALL(ForcesIntoFluid_TwoPoint, dim_grid, threads_per_block, (devPositions, devForces, devExtForces));
}

extern "C"
void InterpolateMarkerVelocities (const int h_numMarkers, double *devPositions, unsigned int *devBoundaryMap, float *devBoundaryVelocities, float *devCurrentNodes, float *devExtForces, double *devVelocities) {

  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_numMarkers + threads_per_block * blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);
  KERNELCALL(VelocitiesFromFluid_TwoPoint, dim_grid, threads_per_block, (devPositions, devBoundaryMap, devBoundaryVelocities, devCurrentNodes, devExtForces, devVelocities));
}




//extern "C"
//void SpreadForceDensities_cpu (const unsigned int numMarkers, struct monomer *hostMonomers) {
//
//  if (devPositions == NULL || devForces == NULL) {
//    InitializeCoupling (numMarkers);
//  }
//  
//  cuda_safe_mem(cudaMemcpy(devMonomers, hostMonomers, numMarkers*sizeof(struct monomer), cudaMemcpyHostToDevice));
//
//  int threads_per_block = 64;
//  int blocks_per_grid_y = 4;
//  int blocks_per_grid_x = (numMarkers + threads_per_block * blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
//  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);
//
//  KERNELCALL(RearrangeInputData, dim_grid, threads_per_block, (devMonomers, devPositions, devForces));
//
//  ResetBodyForces_cpu ();
//
//  ForcesIntoFluid_TwoPoint_cpu (numMarkers, hostMonomers);
//}
//
//extern "C"
//void ResetBodyForces_cpu () {
//
//  for (int index=0; index < hostLBparas.numNodes; index++) {
//    hostExtForces[0*hostLBparas.numNodes + index] = hostLBparas.extForceDensity[0];
//    hostExtForces[1*hostLBparas.numNodes + index] = hostLBparas.extForceDensity[1];
//    hostExtForces[2*hostLBparas.numNodes + index] = hostLBparas.extForceDensity[2];
//  }
//}
//
//extern "C"
//void ForcesIntoFluid_TwoPoint_cpu (const unsigned int hostNumMarkers, struct monomer *markers) {
//
//  for (int index=0; index < hostNumMarkers; index++) {
//
//    const float pos[3] = {           markers[index].pos[0],   markers[index].pos[1],   markers[index].pos[2] };
//    const float particleForce[3] = { markers[index].force[0], markers[index].force[1], markers[index].force[2] };
//
////printf("marker = %d  pos=(%f %f %f)    f=(%f %f %f)\n", markerIndex, pos[0], pos[1], pos[2], particleForce[0], particleForce[1], particleForce[2]);
//
//    // First part is the same as for interpolation --> merge into a single function
//    float temp_delta[6];
//    float delta[8];
//    int my_left[3];
//    int node_index[8];
//    for(int i=0; i<3; ++i) {
//      const float scaledpos = pos[i]/hostLBparas.agrid - 0.5f;
//      my_left[i] = (int)(floorf(scaledpos));
//      temp_delta[3+i] = scaledpos - my_left[i];
//      temp_delta[i] = 1.f - temp_delta[3+i];
//    }
//    delta[0] = temp_delta[0] * temp_delta[1] * temp_delta[2];
//    delta[1] = temp_delta[3] * temp_delta[1] * temp_delta[2];
//    delta[2] = temp_delta[0] * temp_delta[4] * temp_delta[2];
//    delta[3] = temp_delta[3] * temp_delta[4] * temp_delta[2];
//    delta[4] = temp_delta[0] * temp_delta[1] * temp_delta[5];
//    delta[5] = temp_delta[3] * temp_delta[1] * temp_delta[5];
//    delta[6] = temp_delta[0] * temp_delta[4] * temp_delta[5];
//    delta[7] = temp_delta[3] * temp_delta[4] * temp_delta[5];
//
//    // modulo for negative numbers is strange at best, shift to make sure we are positive
//    const int x = my_left[0] + hostLBparas.dimX;
//    const int y = my_left[1] + hostLBparas.dimY;
//    const int z = my_left[2] + hostLBparas.dimZ;
//
//    node_index[0] = x%hostLBparas.dimX     + hostLBparas.dimX*(y%hostLBparas.dimY)     + hostLBparas.dimX*hostLBparas.dimY*(z%hostLBparas.dimZ);
//    node_index[1] = (x+1)%hostLBparas.dimX + hostLBparas.dimX*(y%hostLBparas.dimY)     + hostLBparas.dimX*hostLBparas.dimY*(z%hostLBparas.dimZ);
//    node_index[2] = x%hostLBparas.dimX     + hostLBparas.dimX*((y+1)%hostLBparas.dimY) + hostLBparas.dimX*hostLBparas.dimY*(z%hostLBparas.dimZ);
//    node_index[3] = (x+1)%hostLBparas.dimX + hostLBparas.dimX*((y+1)%hostLBparas.dimY) + hostLBparas.dimX*hostLBparas.dimY*(z%hostLBparas.dimZ);
//    node_index[4] = x%hostLBparas.dimX     + hostLBparas.dimX*(y%hostLBparas.dimY)     + hostLBparas.dimX*hostLBparas.dimY*((z+1)%hostLBparas.dimZ);
//    node_index[5] = (x+1)%hostLBparas.dimX + hostLBparas.dimX*(y%hostLBparas.dimY)     + hostLBparas.dimX*hostLBparas.dimY*((z+1)%hostLBparas.dimZ);
//    node_index[6] = x%hostLBparas.dimX     + hostLBparas.dimX*((y+1)%hostLBparas.dimY) + hostLBparas.dimX*hostLBparas.dimY*((z+1)%hostLBparas.dimZ);
//    node_index[7] = (x+1)%hostLBparas.dimX + hostLBparas.dimX*((y+1)%hostLBparas.dimY) + hostLBparas.dimX*hostLBparas.dimY*((z+1)%hostLBparas.dimZ);
//
//    for(int i=0; i<8; ++i)
//    {
//      hostExtForces[0*hostLBparas.numNodes + node_index[i]] += (particleForce[0] * delta[i]);
//      hostExtForces[1*hostLBparas.numNodes + node_index[i]] += (particleForce[1] * delta[i]);
//      hostExtForces[2*hostLBparas.numNodes + node_index[i]] += (particleForce[2] * delta[i]);
//    }
//  }
//  cuda_safe_mem(cudaMemcpy(devExtForces, hostExtForces, hostLBparas.numNodes*3*sizeof(float), cudaMemcpyHostToDevice));
//}

