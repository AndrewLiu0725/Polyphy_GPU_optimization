#include "integration.h"
#include "tools.h"

void euler_update (int numBeads, double mass, int lx, int ly, int lz, double *forces, double *velocities, double *pos, double *foldedPos) {

  # pragma omp parallel for schedule(static) 
  for(int n=0; n < numBeads; n++) {
     
    int offset = n*3; 
    velocities[offset+0] += forces[offset+0] / mass;
    velocities[offset+1] += forces[offset+1] / mass;
    velocities[offset+2] += forces[offset+2] / mass;
//mon[n].vel[0] = mon[n].force[0]/fictionalMass;
//mon[n].vel[1] = mon[n].force[1]/fictionalMass;
//mon[n].vel[2] = mon[n].force[2]/fictionalMass;

    pos[offset+0] += velocities[offset+0]; // 
    pos[offset+1] += velocities[offset+1];
    pos[offset+2] += velocities[offset+2];

    foldedPos[offset+0] = box(pos[offset+0], lx);
    // Note: #############################
    // Walls are in the y dir. by default
    // ###################################
    //mon[n].pos[1] = box(mon[n].pos_pbc[1], ly);
    foldedPos[offset+1] = pos[offset+1];
    foldedPos[offset+2] = box(pos[offset+2], lz);
  }
}

