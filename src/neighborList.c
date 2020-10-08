#include "neighborList.h"
#include "tools.h"
#include <stdlib.h>
#include <stdio.h>

void PartPosForNlist (int numBeads, double *foldedPos, double *nlistPos, int *nlist) {

  for (int n=0; n < numBeads; n++) {
    int offset = n*3;
    nlistPos[offset+0] = foldedPos[offset+0];
    nlistPos[offset+1] = foldedPos[offset+1];
    nlistPos[offset+2] = foldedPos[offset+2];
    nlist[n*MAX_N] = 0;
  }
}

/*int*/ void CheckCriterion (struct sphere_param sphere_pm, double *nlistPos, double *foldedPos, int *renewalFlag) {

  double half_skin_dis2 = 0.25*sphere_pm.nlistRenewal*sphere_pm.nlistRenewal;  // (0.5*skin depth)^2

  *renewalFlag = 0;  
  #pragma omp parallel for schedule(static)   
  for (int n=0; n < sphere_pm.num_beads; n++) {
    int x = n*3;
    int y = x+1;
    int z = x+2; 
    double dr[3];
    double dr2;
    if (*renewalFlag==0) 
    {
      if (sphere_pm.wallFlag == 1) {
        dr[0] = n_image (foldedPos[x] - nlistPos[x], sphere_pm.lx); 
        dr[1] =          foldedPos[y] - nlistPos[y];
        dr[2] = n_image (foldedPos[z] - nlistPos[z], sphere_pm.lz);
      }
      else if (sphere_pm.wallFlag == 2) {
        dr[0] = n_image (foldedPos[x] - nlistPos[x], sphere_pm.lx); 
        dr[1] =          foldedPos[y] - nlistPos[y];
        dr[2] =          foldedPos[z] - nlistPos[z];
      }
      else {
        fprintf (stderr, "wall flag value is wrong in 'CheckCriterion'\n");
      }
      dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
      if (dr2 > half_skin_dis2) {
        *renewalFlag = 1;
      }
    }
    // keep foldPos for next check and zero the # of neighbors of each node
    //nlistPos[x] = foldedPos[x];
    //nlistPos[y] = foldedPos[y];
    //nlistPos[z] = foldedPos[z];
    //nlist[n*MAX_N] = 0;
  }
  //return flag;
}

void ConstructNeighborList (struct sphere_param sphere_pm, double *foldedPos, double *nlistPos, int *numNeighbors, int *nlist) {

  #define EMPTY -1

  int numBeads = sphere_pm.num_beads;
  #pragma omp parallel for schedule(static)  
	for (int n = 0; n < numBeads; n++) {
    int x = n*3;
    int y = x+1;
    int z = x+2; 
    nlistPos[x] = foldedPos[x];
    nlistPos[y] = foldedPos[y];
    nlistPos[z] = foldedPos[z];
//    nlist[n*MAX_N] = 0; 
numNeighbors[n] = 0;
	}

  double lc           = sphere_pm.cellSize;  // characteristic length determined by the inter-particle interaction
  double nlistCutoff2 = sphere_pm.nlistCutoff * sphere_pm.nlistCutoff;

  // # of cells in each direction
  int nX = sphere_pm.lx / lc; 
  int nY = sphere_pm.ly / lc;
  int nZ = sphere_pm.lz / lc;
  int numCells = nX*nY*nZ;

  // cell sizes in each direction
  double cellSize[3];
  cellSize[0] = (double) sphere_pm.lx / nX;
  cellSize[1] = (double) sphere_pm.ly / nY;
  cellSize[2] = (double) sphere_pm.lz / nZ;

  int *head;
  head = (int *)malloc(numCells*sizeof(int)); 
  if (head == NULL) {
    fprintf (stderr, "cannot allocate 'head'in 'ConstructNeighborListModified'\n");
  }
  int *list;
  list = (int *)calloc(numBeads, sizeof(int));
  if (list == NULL) {
    fprintf (stderr, "cannot allocate 'list' in 'ConstructNeighborListModified'\n");
  }

  // Rest the headers
  for (int n=0; n < numCells; n++) {
    head[n] = EMPTY;
  }

  // Scan particles into a cell to construct headers, head[], and linked lists, list[]
  for (int n=0; n < numBeads; n++) {
    int c[3];
    int offset = n*3; 
    for (int i=0; i < 3; i++) {
      c[i] = foldedPos[offset+i] / cellSize[i];
    }
    // Translate the vector cell index c[] to a scalar index
    int index = c[0]*(nY*nZ) + c[1]*nZ + c[2];
    // Link to the previous occupant
    list[n] = head[index];
    // The last one goes to the header
    head[index] = n;
  }  	

  // Scan inner cells
  for (int x1=0; x1 < nX; x1++) {
    for (int y1=0; y1 < nY; y1++) {
      for (int z1=0; z1 < nZ; z1++) {

        int index1 = x1*(nY*nZ) + y1*nZ + z1;
        // Scan the neighbor cells of cell index1 (including index1 itself)
        for (int x2 = (x1-1); x2 <= (x1+1); x2++) {
          for (int y2 = (y1-1); y2 <= (y1+1); y2++) {
            for (int z2 = (z1-1); z2 <= (z1+1); z2++) {
   
              // Calculate the scalar index of the neighbor cell
              int index2 = ((x2+nX)%nX)*nY*nZ + ((y2+nY)%nY)*nZ + (z2+nZ)%nZ;

              // Scan particle 1 in cell index1
              int n1 = head[index1];
              while (n1 != EMPTY) 
              {
                // Scan particle 2 in cell index2
                int n2 = head[index2];
                while (n2 != EMPTY) 
                {
                  if (n2 < n1)  // Avoid double counting of pair(1,2)
                  {
                    double x12, y12, z12;
                    switch (sphere_pm.wallFlag) {
                      case 1:
                      x12 = n_image (foldedPos[n1*3+0] - foldedPos[n2*3+0], sphere_pm.lx);
                      y12 =          foldedPos[n1*3+1] - foldedPos[n2*3+1];
                      z12 = n_image (foldedPos[n1*3+2] - foldedPos[n2*3+2], sphere_pm.lz);
                      break;

                      case 2:
                      x12 = n_image (foldedPos[n1*3+0] - foldedPos[n2*3+0], sphere_pm.lx);
                      y12 =          foldedPos[n1*3+1] - foldedPos[n2*3+1];
                      z12 =          foldedPos[n1*3+2] - foldedPos[n2*3+2];
                      break;

                      default:
                      fprintf (stderr, "wall_flag value is wrong in 'ConstructNeighborListModified'\n");
                      break;
                    }
                    double dis2 = x12*x12 + y12*y12 + z12*z12;
                    if (dis2 <= nlistCutoff2) {
                      //nlist[n1][0]++;    
                      //nlist[n2][0]++;
                      //nlist[n1][nlist[n1][0]] = n2; 
                      //nlist[n2][nlist[n2][0]] = n1;
//                      nlist[n1*MAX_N]++;
//                      nlist[n2*MAX_N]++;
//                      nlist[ n1*MAX_N + nlist[n1*MAX_N] ] = n2; 
//                      nlist[ n2*MAX_N + nlist[n2*MAX_N] ] = n1;
numNeighbors[n1]++;
numNeighbors[n2]++;
nlist[ n1*MAX_N + numNeighbors[n1]-1 ] = n2;
nlist[ n2*MAX_N + numNeighbors[n2]-1 ] = n1;
                      
                      //if (nlist[n1][0] > MAX_N || nlist[n2][0] > MAX_N) {
                      //  fprintf (stderr, "too many neighbors\n");
                      //  exit(1);
                      //}
//                      if (nlist[n1*MAX_N] > (MAX_N-1) || nlist[n2*MAX_N] > (MAX_N-1)) {
//                        fprintf (stderr, "# of neighbors = %d & %d. Too many neighbors\n",nlist[n1*MAX_N], nlist[n2*MAX_N]);
//                        exit(1);
//                      }
if (numNeighbors[n1] > MAX_N || numNeighbors[n2] > MAX_N) {
  fprintf (stderr, "# of neighbors = %d & %d. Too many neighbors\n", numNeighbors[n1], numNeighbors[n2]);
  exit(1);
}
                    }
                  }
                  n2 = list[n2];
                } 
                n1 = list[n1];
              }
            }
          }
        }
      }
    }
  }
  free (head);
  free (list);
}

void InitializeNeighborList (struct sphere_param h_params, double *h_foldedPos, double *h_nlistPos, int *h_numNeighbors, int *h_nlist) {

  //PartPosForNlist (h_params.num_beads, h_foldedPos, h_nlistPos, h_nlist);

  ConstructNeighborList (h_params, h_foldedPos, h_nlistPos, h_numNeighbors, h_nlist);
}

int RenewNeighborList (struct sphere_param sphere_pm, double *nlistPos, double *foldedPos, int *numNeighbors, int *nlist) {

  int frequency=0;
  int flag;  

  CheckCriterion (sphere_pm, nlistPos, foldedPos, &flag);

  // If the criterion is met, update the neighbor list
  if (flag == 1) {
    ConstructNeighborList (sphere_pm, foldedPos, nlistPos, numNeighbors, nlist);
    frequency=1; 
  }
  return frequency;
}

void PrintNlist (int numNodes, int *nlist, char *filePath) {

  FILE *stream;
  stream = fopen (filePath, "w");
  for (int n=0; n < numNodes; n++) {
    for (int i=0; i < MAX_N; i++) {
      fprintf (stream, "%d\n", nlist[n*MAX_N+i]);
    }
  }
  fclose (stream);
}

