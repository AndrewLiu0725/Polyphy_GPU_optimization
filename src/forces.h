#ifndef FORCES_H
#define FORCES_H

#include "monomer.h"
#include "face.h"
#include "sphere_param.h"

void ZeroForces (int numNodes, double *springForces, double *bendingForces, double *volumeForces, double *globalAreaForces, double *localAreaForces, double *wallForces, double *interparticleForces);

void SumForces (int numNodes, double *springForces, double *bendingForces, double *volumeForces, double *globalAreaForces, double *localAreaForces, double *wallForces, double *interparticleForces, double *forces);

void ComputeForces (struct sphere_param *sphere_pm, struct monomer *monomers, struct face *faces, double *springForces, double *bendingForces, double *volumeForces, double *globalAreaForces, double *localAreaForces, double *wallForces, double *interparticleForces, double *pos, int *numNeighbors, int *nlist, int *numBonds, int *blist, double *forces);

void SpringForce (struct sphere_param *sphere_pm, struct monomer *monomers, double *springForces, double *pos, int *numBonds, int *blist); 

void BendingForceSin (struct sphere_param *sphere_pm, struct monomer *monomers, double *bendingForces, double *pos, int *numBonds, int *blist);
 
void VolumeAreaConstraints (struct sphere_param *sphere_pm, struct monomer *monomers, struct face *faces, double *volumeForces, double *globalAreaForces, double *localAreaForces, double *pos); 

void WallForce (struct sphere_param *sphere_pm, double *pos, double *wallForces);

void InterparticleForce (struct sphere_param *sphere_pm, struct monomer *monomers, double *interparticleForces, double *pos, int *numNeighbors, int *nlist);


// cuda 
void ComputeForces_gpu (/*unsigned int h_numBeads, unsigned int h_numParticles,*/ struct sphere_param h_params, struct monomer *d_monomers, struct face *d_faces, int *d_numBonds, int *d_blist, int *d_numNeighbors, int *d_nlist, double *d_pos, double *d_springForces, double *d_bendingForces, float *d_volumeForces, float *d_globalAreaForces, double *d_localAreaForces, double *d_wallForces, double *d_interparticleForces, double *d_forces    ,float *d_coms, double *d_faceCenters, double *d_normals, float *d_areas, float *d_volumes);

#endif
