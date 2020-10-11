extern "C"{
#include <stdio.h>
#include "initConfig.h"
#include "tools.h"
#include "neighborList.h"
#include "forces.h"
#include "integration.h"
#include "output.h"
}

static __device__ int d_step = 0;  // For initialization procedure only
static            int h_step = 0;
extern __constant__ struct sphere_param d_partParams;


__global__ void ModifyParameters () {

//  // constants: 
//  // V0_temp[], A0_temp[], V0_final[], A0_final[], totalSteps, monomers[].initLength_temp[], monomers[].initLength_final[], x0
//
//  // variables:
//  // V0[0], V0[1], A0[0], A0[1], monomers[].initLength[], monomers[].lmax[]
//  // kv[], kag[], kal[], wallConstant
//
//  int totalGrowthStep  = params->numGrowthSteps;
//  int numParticles     = params->Nsphere;
//  int numParticlesA    = params->Ntype[0];
//  int numNodesPerPartA = params->N_per_sphere[0];
//
//  params->V0[0] = params->V0_temp[0] + step * (params->V0_final[0] - params->V0_temp[0]) / totalGrowthStep;
//  params->V0[1] = params->V0_temp[1] + step * (params->V0_final[1] - params->V0_temp[1]) / totalGrowthStep;
//  params->A0[0] = params->A0_temp[0] + step * (params->A0_final[0] - params->A0_temp[0]) / totalGrowthStep;
//  params->A0[1] = params->A0_temp[1] + step * (params->A0_final[1] - params->A0_temp[1]) / totalGrowthStep;
//  params->kv[0]        = ((double)step / totalGrowthStep) * 20;
//  params->kv[1]        = ((double)step / totalGrowthStep) * 20;
//  params->kag[0]       = ((double)step / totalGrowthStep) * 200;
//  params->kag[1]       = ((double)step / totalGrowthStep) * 200;
//  params->kal[0]       = ((double)step / totalGrowthStep) * 10;   
//  params->kal[1]       = ((double)step / totalGrowthStep) * 10;   
//  params->wallConstant = ((double)step / totalGrowthStep) * 20;
//
//  //printf("step         = %d\n", step);
//  //printf("(kv kag kal) = (%f, %f, %f)\n", params->kv[0], params->kag[0], params->kal[0]);
//  //printf("wall const   = %f\n", params->wallConstant);
//  //printf("V0[0], A0[0] = %f, %f\n", params->V0[0], params->A0[0]);
//
//  step ++;
}

extern "C"
void ModifySphereParams (struct sphere_param *params) {

  // constants: 
  // V0_temp[], A0_temp[], V0_final[], A0_final[], totalSteps, monomers[].initLength_temp[], monomers[].initLength_final[], x0

  // variables:
  // V0[0], V0[1], A0[0], A0[1], monomers[].initLength[], monomers[].lmax[]
  // kv[], kag[], kal[], wallConstant

  unsigned int totalGrowthStep  = params->numGrowthSteps;

  params->V0[0] = params->V0_temp[0] + h_step * (params->V0_final[0] - params->V0_temp[0]) / totalGrowthStep;
  params->V0[1] = params->V0_temp[1] + h_step * (params->V0_final[1] - params->V0_temp[1]) / totalGrowthStep;
  params->A0[0] = params->A0_temp[0] + h_step * (params->A0_final[0] - params->A0_temp[0]) / totalGrowthStep;
  params->A0[1] = params->A0_temp[1] + h_step * (params->A0_final[1] - params->A0_temp[1]) / totalGrowthStep;
  params->kv[0]        = ((double)h_step / totalGrowthStep) * 20;
  params->kv[1]        = ((double)h_step / totalGrowthStep) * 20;
  params->kag[0]       = ((double)h_step / totalGrowthStep) * 200;
  params->kag[1]       = ((double)h_step / totalGrowthStep) * 200;
  params->kal[0]       = ((double)h_step / totalGrowthStep) * 10;   
  params->kal[1]       = ((double)h_step / totalGrowthStep) * 10;   
  params->wallConstant = ((double)h_step / totalGrowthStep) * 20;
  //printf("h_step         = %d\n", h_step);
  //printf("(kv kag kal) = (%f, %f, %f)\n", params->kv[0], params->kag[0], params->kal[0]);
  //printf("wall const   = %f\n", params->wallConstant);
  //printf("V0[0], A0[0] = %f, %f\n", params->V0[0], params->A0[0]);

  h_step++;
}


//__global__ void ModifyBondParams(struct sphere_param *d_params, int *d_numBonds, struct monomer *d_monomers) {
//
//  int numParticles  = d_partParams.Nsphere;
//  int numParticlesA = d_partParams.Ntype[0];
//  int totalSteps    = d_partParams.numGrowthSteps;
//
//  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
//  if (index < numParticles) {
//    int numNodesPerPart;
//    int offset;
//    if (index < numParticlesA) {
//      numNodesPerPart = d_partParams.N_per_sphere[0];
//      offset          = index*numNodesPerPart;
//    }
//    else {
//      numNodesPerPart = d_partParams.N_per_sphere[1];
//      offset = numParticlesA * d_partParams.N_per_sphere[0] + (index - numParticlesA)*numNodesPerPart;
//    }
//
//    for (int j=0; j < numNodesPerPart; j++) {
//      int n1 = j + offset;
//      for (int bond=1; bond <= d_numBonds[n1]; bond++)  {
//        d_monomers[n1].initLength[bond] = d_monomers[n1].initLength_temp[bond] + 
//        step * (d_monomers[n1].initLength_final[bond] - d_monomers[n1].initLength_temp[bond]) / totalSteps;
//
//        d_monomers[n1].lmax[bond] = d_monomers[n1].initLength[bond] / d_partParams.x0;
//      }    
//    }
//  }
//}

__global__ void ModifyBondParams (int *d_numBonds, struct monomer *d_monomers) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < d_partParams.num_beads) {   
    for (unsigned int bond=1; bond <= d_numBonds[index]; bond++)  {
      d_monomers[index].initLength[bond] = d_monomers[index].initLength_temp[bond] + 
      d_step * (d_monomers[index].initLength_final[bond] - d_monomers[index].initLength_temp[bond]) / d_partParams.numGrowthSteps;

      d_monomers[index].lmax[bond] = d_monomers[index].initLength[bond] / d_partParams.x0;
    }    
  }
}


extern "C"
void ModifyParameters_wrapper (int h_numBeads, int *d_numBonds, struct monomer *d_monomers) {
  
  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_numBeads + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  ModifyBondParams <<<dim_grid, threads_per_block>>> (d_numBonds, d_monomers);
//  ModifyParameters <<<1,1>>> ();
//cudaDeviceSynchronize();
}

extern "C"
void RestoreParticle_gpu (struct sphere_param h_params, double *h_foldedPos, double *h_nlistPos, int *h_numNeighbors, int *h_nlist, struct face *h_faces, struct monomer *d_monomers, struct face *d_faces, int *d_numBonds, int *d_blist, int *d_numNeighbors, int *d_nlist, double *d_pos, double *d_foldedPos, double *d_nlistPos, double *d_velocities, double *d_springForces, double *d_bendingForces, float *d_volumeForces, float *d_globalAreaForces, double *d_localAreaForces, double *d_wallForces, double *d_interparticleForces, double *d_forces    ,float *d_coms, double *d_faceCenters, double *d_normals, float *d_areas, float *d_volumes, int *d_face_pair_list) {

  int count = 0;
  fprintf (stdout,"Start restoring particles\n");
  
  for (int step=0; step <= h_params.numGrowthSteps; step++)
  {
//    if (step % 1000 == 0) {
//      char work_dir[100]={"./data"};
//      cudaMemcpy (h_foldedPos, d_foldedPos, h_params.num_beads*3*sizeof(double), cudaMemcpyDeviceToHost);  
//      WriteParticleVTK (step, work_dir, h_params, h_faces, h_foldedPos);
//    }

    ModifyParameters_wrapper (h_params.num_beads, d_numBonds, d_monomers);
    ModifySphereParams (&h_params);
    cudaMemcpyToSymbol(d_partParams, &h_params, sizeof(struct sphere_param)); 
    cudaMemcpyToSymbol(d_step, &h_step, sizeof(int));

    ComputeForces_gpu (h_params, d_monomers, d_faces, d_numBonds, d_blist, d_numNeighbors, d_nlist, d_pos, d_springForces, d_bendingForces, d_volumeForces, d_globalAreaForces, d_localAreaForces, d_wallForces, d_interparticleForces, d_forces    ,d_coms, d_faceCenters, d_normals, d_areas, d_volumes, d_face_pair_list);

    EulerUpdate_wrapper (h_params.num_beads, d_forces, d_velocities, d_pos, d_foldedPos);

    count += RenewNeighborList_gpu (h_params, h_foldedPos, h_nlistPos, h_numNeighbors, h_nlist, d_foldedPos, d_nlistPos, d_numNeighbors, d_nlist);
  }
  fprintf (stdout, "nlist is updated %d times.\n", count);
  fprintf (stdout,"End restoring particles\n\n");
}

