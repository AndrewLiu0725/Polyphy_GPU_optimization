#ifndef NEIGHBORLIST_H
#define NEIGHBORLIST_H

#include "sphere_param.h"
#define MAX_N 300

void PartPosForNlist (int numBeads, double *foldedPos, double *nlistPos, int *nlist);

/*int*/ void CheckCriterion (struct sphere_param sphere_pm, double *nlistPos, double *foldedPos, int *renewalFlag);

void ConstructNeighborList (struct sphere_param sphere_pm, double *foldedPos, double *nlistPos, int *numNeighbors, int *nlist);

int RenewNeighborList (struct sphere_param sphere_pm, double *nlistPos, double *foldedPos, int *numNeighbors,  int *nlist);

void InitializeNeighborList (struct sphere_param h_params, double *h_foldedPos, double *h_nlistPos, int *h_numNeighbors, int *h_nlist);

void PrintNlist (int numNodes, int *nlist, char *filePath);


// cuda
//void CheckCriterion_wrapper (struct sphere_param host_partParams, int *host_renewalFlag, struct sphere_param *dev_partParams, double *dev_foldedPos, double *dev_nlistPos, int *dev_nlist, int *dev_renewalFlag);

int RenewNeighborList_gpu (struct sphere_param h_params, double *h_foldedPos, double *h_nlistPos, int *h_numNeighbors, int *h_nlist, double *d_foldedPos, double *d_nlistPos, int *d_numNeighbors, int *d_nlist);

#endif

