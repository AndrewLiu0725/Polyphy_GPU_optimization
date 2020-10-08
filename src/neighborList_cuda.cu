extern "C" {
#include <stdlib.h>
#include <stdio.h>
#include "neighborList.h"
#include "tools.h"
}
#include "cuda_runtime_api.h"

extern __constant__ struct sphere_param d_partParams;

__global__ void CheckCriterion (double *foldedPos, double *nlistPos, int *nlist, int *renewalFlag) {

  double halfSkinDis2 = 0.25 * d_partParams.nlistRenewal * d_partParams.nlistRenewal;  // (0.5*skin depth)^2

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < d_partParams.num_beads) {
    int x = index*3;
    int y = x+1;
    int z = x+2; 
    double dr[3];
    double dr2;
    if (*renewalFlag==0) 
    {
      if (d_partParams.wallFlag == 1) {
        dr[0] = n_image (foldedPos[x] - nlistPos[x], d_partParams.lx); 
        dr[1] =          foldedPos[y] - nlistPos[y];
        dr[2] = n_image (foldedPos[z] - nlistPos[z], d_partParams.lz);
      }
      else if (d_partParams.wallFlag == 2) {
        dr[0] = n_image (foldedPos[x] - nlistPos[x], d_partParams.lx); 
        dr[1] =          foldedPos[y] - nlistPos[y];
        dr[2] =          foldedPos[z] - nlistPos[z];
      }
      else {
        printf ("wall flag value is wrong in 'CheckCriterion'\n");
      }
      dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
      if (dr2 > halfSkinDis2) {
        *renewalFlag = 1;
      }
    }
    // keep foldedPos for next check and zero each node's variable of the # of neighbors
    // Should be done as the neighbor list has to be renewed.
    //nlistPos[x]     = foldedPos[x];
    //nlistPos[y]     = foldedPos[y];
    //nlistPos[z]     = foldedPos[z];
    //nlist[index*MAX_N] = 0;
  }
}

//extern "C"
//void CheckCriterion_wrapper (struct sphere_param host_partParams, int *host_renewalFlag, struct sphere_param *dev_partParams, double *dev_foldedPos, double *dev_nlistPos, int *dev_nlist, int *dev_renewalFlag) {
//
//  // hostPartParams is an extern variable 
//  int threads_per_block = 64;
//  int blocks_per_grid_y = 4;
//  int blocks_per_grid_x = (host_partParams.num_beads + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
//  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);
//
//  // Zero renewalFlag
//  *host_renewalFlag = 0;
//  cudaMemcpy (dev_renewalFlag, host_renewalFlag, sizeof(int), cudaMemcpyHostToDevice);
//
//  // devPartParams and devRenewalFlag are extern 
//  CheckCriterion <<<dim_grid, threads_per_block>>> (dev_partParams, dev_foldedPos, dev_nlistPos, dev_nlist, dev_renewalFlag); 
//
//  cudaMemcpy (host_renewalFlag, dev_renewalFlag, sizeof(int), cudaMemcpyDeviceToHost);
//
//  //printf ("CheckCriterion has been done.\n");
//}

extern "C"
int RenewNeighborList_gpu (struct sphere_param h_params, double *h_foldedPos, double *h_nlistPos, int *h_numNeighbors, int *h_nlist, double *d_foldedPos, double *d_nlistPos, int *d_numNeighbors, int *d_nlist) {

  int frequency = 0;
  int h_renewalFlag = 0;
  int *d_renewalFlag; 
  unsigned int numNodes = h_params.num_beads;

//  CheckCriterion_wrapper (host_partParams, &host_renewalFlag, dev_partParams, dev_foldedPos, dev_nlistPos, dev_nlist, dev_renewalFlag);  
  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (numNodes + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  cudaMalloc ((void**)&d_renewalFlag, sizeof(int));
  cudaMemcpy (d_renewalFlag, &h_renewalFlag, sizeof(int), cudaMemcpyHostToDevice);

  CheckCriterion <<<dim_grid, threads_per_block>>> (d_foldedPos, d_nlistPos, d_nlist, d_renewalFlag); 

  cudaMemcpy (&h_renewalFlag, d_renewalFlag, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree (d_renewalFlag);

//cudaDeviceSynchronize();

  // If the criterion is met, update the neighbor list
  if (h_renewalFlag == 1) {

    cudaMemcpy (h_foldedPos, d_foldedPos, numNodes*3*sizeof(double), cudaMemcpyDeviceToHost);

    ConstructNeighborList (h_params, h_foldedPos, h_nlistPos, h_numNeighbors, h_nlist);

    cudaMemcpy (d_numNeighbors, h_numNeighbors, numNodes*sizeof(int), cudaMemcpyHostToDevice);   
    cudaMemcpy (d_nlist,        h_nlist, numNodes*MAX_N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (d_nlistPos,     h_nlistPos, numNodes*3*sizeof(double), cudaMemcpyHostToDevice); 
    frequency = 1;
  }
  return frequency;
}

