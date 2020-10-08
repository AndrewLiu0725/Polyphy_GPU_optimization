#ifndef COUPLING_H
#define COUPLING_H

void SpreadForceDensities (const unsigned int h_numFluidNodes, const int h_numMarkers, double *devPositions, double *devForces, float *devExtForces);

void InterpolateMarkerVelocities (const int h_numMarkers, double *devPositions, unsigned int *devBoundaryMap, float *devBoundaryVelocities, float *devCurrentNodes, float *devExtForces, double *devVelocities);

#endif  // COUPLING_H
