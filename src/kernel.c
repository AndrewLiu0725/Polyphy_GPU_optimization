#include <stdio.h>
#include <time.h>
#include "kernel.h"
#include "output.h"
#include "coupling.h"
#include "integration.h"
#include "neighborList.h"
#include "forces.h"
#include "lb.h"

#include "cuda_runtime_api.h"

#ifdef PARTICLE_GPU
int LBkernel (int cycle, struct sphere_param params, double *h_foldedPos, double *d_foldedPos, char *dataFolder, struct face *h_faces, double *d_pos, unsigned int *d_boundaryMap, float *d_boundaryVelocities, float *d_currentNodes, float *d_extForces, double *d_velocities, double *h_nlistPos, int *h_nlist, double *d_nlistPos, int *d_nlist, struct monomer *d_monomers, struct face *d_faces, int *d_numBonds, int *d_blist, double *d_springForces, double *d_bendingForces, float *d_volumeForces, float *d_globalAreaForces, double *d_localAreaForces, double *d_wallForces, double *d_interparticleForces, double *d_forces, LBparameters lbParams, int *h_numNeighbors, int *d_numNeighbors    ,float *d_coms, double *d_faceCenters, double *d_normals, float *d_areas, float *d_volumes) {
#else
int LBkernel (int cycle, struct sphere_param params, double *h_foldedPos, double *d_foldedPos, char *dataFolder, struct face *h_faces, double *d_pos, unsigned int *d_boundaryMap, float *d_boundaryVelocities, float *d_currentNodes, float *d_extForces, double *d_velocities, double *h_nlistPos, int *h_nlist, double *d_nlistPos, int *d_nlist, struct monomer *d_monomers, struct face *d_faces, int *d_numBonds, int *d_blist, double *d_springForces, double *d_bendingForces, double *d_volumeForces, double *d_globalAreaForces, double *d_localAreaForces, double *d_wallForces, double *d_interparticleForces, double *d_forces, LBparameters lbParams, double *h_pos, struct monomer *h_monomers, double *h_springForces, double *h_bendingForces, double *h_volumeForces, double *h_globalAreaForces, double *h_localAreaForces, double *h_wallForces, double *h_interparticleForces, int *h_numBonds, int *h_blist, double *h_forces, int *h_numNeighbors) {
#endif

  clock_t tStart, tEnd; // for timing
  double tDiff;

  int nlistCounter = 0;

  int starting = cycle * params.numStepsPerCycle;
  int end =  (cycle+1) * params.numStepsPerCycle;
  tStart = clock();

  for (int step = starting; step < end; step++)
  {
    // @ Output data
    //if (step % params.write_time == 0) {
    //  // WriteData
    //}
    if (step % params.write_config == 0) {
      // WriteParticleVTK
      cudaMemcpy (h_foldedPos, d_foldedPos, params.num_beads*3*sizeof(double), cudaMemcpyDeviceToHost);
      WriteParticleVTK (step, dataFolder, params, h_faces, h_foldedPos);  
    }
    // WriteFluidVTK

    // @ Update marker velocities
    InterpolateMarkerVelocities (params.num_beads, d_pos, d_boundaryMap, d_boundaryVelocities, d_currentNodes, d_extForces, d_velocities);

    // @ Update marker positions
    UpdatePartice (params.num_beads, d_velocities, d_pos, d_foldedPos);

    // @ Check the neighbor list
    nlistCounter += RenewNeighborList_gpu (params, h_foldedPos, h_nlistPos, h_numNeighbors, h_nlist, d_foldedPos, d_nlistPos, d_numNeighbors, d_nlist);

    // @ Compute forces

    #ifdef PARTICLE_GPU
    ComputeForces_gpu (params, d_monomers, d_faces, d_numBonds, d_blist, d_numNeighbors, d_nlist, d_pos, d_springForces, d_bendingForces, d_volumeForces, d_globalAreaForces, d_localAreaForces, d_wallForces, d_interparticleForces, d_forces    ,d_coms, d_faceCenters, d_normals, d_areas, d_volumes);

    #else
    cudaMemcpy(h_pos, d_pos, params.num_beads*3*sizeof(double), cudaMemcpyDeviceToHost);

    ComputeForces (&params, h_monomers, h_faces, h_springForces, h_bendingForces, h_volumeForces, h_globalAreaForces, h_localAreaForces, h_wallForces, h_interparticleForces, h_pos, h_numNeighbors, h_nlist, h_numBonds, h_blist, h_forces);

    cudaMemcpy(d_forces, h_forces, params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    #endif

    // @ Spread forces to fluids
    SpreadForceDensities (lbParams.numNodes, params.num_beads, d_pos, d_forces, d_extForces);
   
    // @ Update LB populations
    UpdateLBE ();    

    // @ Time this loop
    if (step % 1000 == 0) {
      tEnd = clock ();
      tDiff = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      tStart = tEnd; 
      fprintf (stdout, "at step = %d , time elapsed since last check: %lf (sec)\n", step, tDiff);
      fflush (stdout);
    }
  }

  // @ Output checkpoint files
  cycle ++;
  // write checkpoint files

  return cycle;
}

