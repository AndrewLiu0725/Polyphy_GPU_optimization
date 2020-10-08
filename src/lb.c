#include <stdio.h>
#include "lb.h"

void InitializeLBparameters (char *file_path, LBparameters *params) {
  
  FILE *file_ptr = fopen (file_path, "r");
  if (file_ptr == NULL)  
    printf("Cannot open LB parameter file\n");
  else 
    printf("LB parameter file is opened\n");

  fscanf(file_ptr, "%*s %*s %*s");
  fscanf(file_ptr, "%u %u %u", &params->dimX, &params->dimY, &params->dimZ);
  fscanf(file_ptr, "%*s %*s %*s %*s");
  fscanf(file_ptr, "%u %f %f %f", &params->numBoundaries, &params->boundaryVelocity[0], &params->boundaryVelocity[1], &params->boundaryVelocity[2]);
  fscanf(file_ptr, "%*s %*s %*s %*s");
  fscanf(file_ptr, "%u %f %f %f", &params->extForceFlag, &params->extForceDensity[0], &params->extForceDensity[1], &params->extForceDensity[2]);
  fscanf(file_ptr, "%*s %*s %*s");
  fscanf(file_ptr, "%f %f %f", &params->agrid, &params->tau, &params->rho);
  fscanf(file_ptr, "%*s %*s %*s %*s");
  fscanf(file_ptr, "%f %f %f %f", &params->gammaShear, &params->gammaBulk, &params->gammaOdd, &params->gammaEven);
  fclose(file_ptr);

  params->numNodes = (params->dimX) * (params->dimY) * (params->dimZ);

  fprintf(stdout, "\nDim_x  Dim_y  Dim_z  Num_of_nodes\n");
  fprintf(stdout, "%u    %u    %u    %u\n", params->dimX, params->dimY, params->dimZ, params->numNodes);
  fprintf(stdout, "Num_of_boundaries  Boundary_vel.x  Boundary_vel.y  Boundary_vel.z\n");
  fprintf(stdout, "%u    %f    %f    %f\n", params->numBoundaries, params->boundaryVelocity[0], params->boundaryVelocity[1], params->boundaryVelocity[2]);
  fprintf(stdout, "Ext_force_flag  Ext_force_density.x  Ext_force_density.y  Ext_force_density.z\n");
  fprintf(stdout, "%u    %f    %f    %f\n", params->extForceFlag, params->extForceDensity[0], params->extForceDensity[1], params->extForceDensity[2]);
  fprintf(stdout, "Agrid  Tau  Rho\n");
  fprintf(stdout, "%f    %f    %f\n", params->agrid, params->tau, params->rho);
  fprintf(stdout, "gamma_shear  gamma_bulk  gamma_odd  gamma_even\n");
  fprintf(stdout, "%f    %f    %f    %f\n", params->gammaShear, params->gammaBulk, params->gammaOdd, params->gammaEven);

//#ifdef LBGPU
//  cuda_safe_mem(cudaMemcpyToSymbol(d_LBparams, params, sizeof(LBparameters)));
//#endif

}

