#ifndef KERNEL_H
#define KERNEL_H

#include "sphere_param.h"
#include "monomer.h"
#include "face.h"
#include "lb.h"
#include "config.h"

#ifdef PARTICLE_GPU
int LBkernel (int cycle, struct sphere_param params, double *h_foldedPos, double *d_foldedPos, char *dataFolder, struct face *h_faces, double *d_pos, unsigned int *d_boundaryMap, float *d_boundaryVelocities, float *d_currentNodes, float *d_extForces, double *d_velocities, double *h_nlistPos, int *h_nlist, double *d_nlistPos, int *d_nlist, struct monomer *d_monomers, struct face *d_faces, int *d_numBonds, int *d_blist, double *d_springForces, double *d_bendingForces, float *d_volumeForces, float *d_globalAreaForces, double *d_localAreaForces, double *d_wallForces, double *d_interparticleForces, double *d_forces, LBparameters lbParams, int *h_numNeighbors, int *d_numNeighbors    ,float *d_coms, double *d_faceCenters, double *d_normals, float *d_areas, float *d_volumes);
#else
int LBkernel (int cycle, struct sphere_param params, double *h_foldedPos, double *d_foldedPos, char *dataFolder, struct face *h_faces, double *d_pos, unsigned int *d_boundaryMap, float *d_boundaryVelocities, float *d_currentNodes, float *d_extForces, double *d_velocities, double *h_nlistPos, int *h_nlist, double *d_nlistPos, int *d_nlist, struct monomer *d_monomers, struct face *d_faces, int *d_numBonds, int *d_blist, double *d_springForces, double *d_bendingForces, double *d_volumeForces, double *d_globalAreaForces, double *d_localAreaForces, double *d_wallForces, double *d_interparticleForces, double *d_forces, LBparameters lbParams, double *h_pos, struct monomer *h_monomers, double *h_springForces, double *h_bendingForces, double *h_volumeForces, double *h_globalAreaForces, double *h_localAreaForces, double *h_wallForces, double *h_interparticleForces, int *h_numBonds, int *h_blist, double *h_forces, int *h_numNeighbors);
#endif


#endif

