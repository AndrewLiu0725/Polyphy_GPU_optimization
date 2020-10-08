#ifndef INTEGRATION_H
#define INTEGRATION_H

//#include "sphere_param.h"

void euler_update (int numBeads, double mass, int lx, int ly, int lz, double *forces, double *velocities, double *pos, double *foldedPos);

// cuda
void EulerUpdate_wrapper (int h_numBeads, double *d_forces, double *d_velocities, double *d_pos, double *d_foldedPos);
void UpdatePartice (int h_numBeads, double *d_vel, double *d_pos, double *d_foldedPos);

#endif

