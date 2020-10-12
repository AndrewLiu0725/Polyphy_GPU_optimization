#ifndef INITCONFIG_H
#define INITCONFIG_H

#include "sphere_param.h"
#include "monomer.h"
#include "face.h"

void ModifyParameters (int step, int *numBonds, struct sphere_param *params, struct monomer *monomers);

int GenerateConfig (struct sphere_param *sphere_pm, char *work_dir, struct monomer *monomers, struct face *faces, double *pos, double *foldedPos, int *numBonds, int *blist, int ***Blist);

void ReadTemplate (int particleType, double reducedFactor, char *work_dir, double **v, int ***blist); 

void RotationMatrix (double thetaX, double thetaY, double thetaZ, double mat[3][3]);

void AssignBlist (struct sphere_param *sphere_pm, char *work_dir, int ***Blist, struct face *faces);

void ReadConfig (char *work_dir, int num_node, struct monomer *node, double *pos, double *foldedPos, int ***blist);

void ReadConfig_tmp(char *work_dir, int num_node, struct monomer *node, double *pos, double *foldedPos, int ***blist);

void SetEqPartParams (char *work_dir, struct sphere_param *sphere_pm, int *numBonds, struct monomer *vertex, struct face *faces);

void SetSpringConstants (struct sphere_param *sphere_pm, int *numBonds, struct monomer *node);

void SetReducedPartParams (double *pos, int *numBonds, int *blist, struct sphere_param *sphere_pm, struct monomer *vertex, struct face *faces);

void RestoreParticle (struct sphere_param *h_params, struct face *h_faces, double *h_foldedPos, int *h_numBonds, struct monomer *h_monomers, double *h_springForces, double *h_bendingForces, double *h_volumeForces, double *h_globalAreaForces, double *h_localAreaForces, double *h_wallForces, double *h_interparticleForces, double *h_pos, int *h_numNeighbors, int *h_nlist, int *h_blist, double *h_forces, double *h_velocities, double *h_nlistPos);


// cuda
void RestoreParticle_gpu (struct sphere_param h_params, double *h_foldedPos, double *h_nlistPos, int *h_numNeighbors, int *h_nlist, struct face *h_faces, struct monomer *d_monomers, struct face *d_faces, int *d_numBonds, int *d_blist, int *d_numNeighbors, int *d_nlist, double *d_pos, double *d_foldedPos, double *d_nlistPos, double *d_velocities, double *d_springForces, double *d_bendingForces, float *d_volumeForces, float *d_globalAreaForces, double *d_localAreaForces, double *d_wallForces, double *d_interparticleForces, double *d_forces    ,float *d_coms, double *d_faceCenters, double *d_normals, float *d_areas, float *d_volumes);

void ModifyParameters_wrapper (int h_numBeads, int *d_numBonds, struct monomer *d_monomers);
 
#endif
