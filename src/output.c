#include <stdio.h>
#include "output.h"
#include "math.h"

void WriteParticleVTK (int step, char *dataFolder, struct sphere_param params, struct face *faces, double *h_foldedPos) {

  int node0;
  int face0;
  int numNodes;
  int numFaces;
  char filename[200];
  FILE *pFile;
  double halfLx = 0.5*params.lx;
  double halfLy = 0.5*params.ly;
  double halfLz = 0.5*params.lz;
  
  int numNodesA = params.Ntype[0] * params.N_per_sphere[0];
  int numFacesA = params.Ntype[0] * params.face_per_sphere[0];

  node0 = 0;
  face0 = 0;  
  numNodes = numNodesA;
  numFaces = numFacesA;
  sprintf (filename, "%s/particleA_%d.vtk", dataFolder, step);  
  pFile = fopen (filename, "w");
	fprintf (pFile, "# vtk DataFile Version 2.3   \n");
	fprintf (pFile, "title particle configuration %d \n", step);
	fprintf (pFile, "ASCII                        \n\n");
	fprintf (pFile, "DATASET UNSTRUCTURED_GRID \n");
	fprintf (pFile, "POINTS %d float\n", numNodesA);

  for (int n = node0; n < numNodes; n++) {
    int offset = n*3; 
    fprintf (pFile, "%8.6lf %8.6lf %8.6lf\n", h_foldedPos[offset+0], h_foldedPos[offset+1], h_foldedPos[offset+2]);
  }
	// check if the face spans across periodic boundaries
	int nface_temp = 0;
	for (int i = face0; i < numFaces; i++) 
  {    
    if (halfLx - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 0] - h_foldedPos[ (faces[i].v[1])*3 + 0] ) < 0) {
      continue;
    } 
    if (halfLy - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 1] - h_foldedPos[ (faces[i].v[1])*3 + 1] ) < 0) {
      continue;
    } 
    if (halfLz - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 2] - h_foldedPos[ (faces[i].v[1])*3 + 2] ) < 0) {
      continue;
    }


    if (halfLx - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 0] - h_foldedPos[ (faces[i].v[2])*3 + 0] ) < 0) {
      continue;
    } 
    if (halfLy - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 1] - h_foldedPos[ (faces[i].v[2])*3 + 1] ) < 0) {
      continue;
    } 
    if (halfLz - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 2] - h_foldedPos[ (faces[i].v[2])*3 + 2] ) < 0) {
      continue;
    }


    if (halfLx - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 0] - h_foldedPos[ (faces[i].v[2])*3 + 0] ) < 0) {
      continue;
    } 
    if (halfLy - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 1] - h_foldedPos[ (faces[i].v[2])*3 + 1] ) < 0) {
      continue;
    } 
    if (halfLz - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 2] - h_foldedPos[ (faces[i].v[2])*3 + 2] ) < 0) {
      continue;
    }

    nface_temp++;
  }

  fprintf (pFile, "CELLS %d %d\n", nface_temp, nface_temp*4); // 4: (# of verteices, vertex label, vertex label, vertex label)
  for (int i=face0; i < numFaces; i++) 
  {
    int offset = i*3;  // a face has 3 verteices
    
    if (halfLx - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 0] - h_foldedPos[ (faces[i].v[1])*3 + 0] ) < 0) {
      continue;
    } 
    if (halfLy - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 1] - h_foldedPos[ (faces[i].v[1])*3 + 1] ) < 0) {
      continue;
    } 
    if (halfLz - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 2] - h_foldedPos[ (faces[i].v[1])*3 + 2] ) < 0) {
      continue;
    }


    if (halfLx - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 0] - h_foldedPos[ (faces[i].v[2])*3 + 0] ) < 0) {
      continue;
    } 
    if (halfLy - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 1] - h_foldedPos[ (faces[i].v[2])*3 + 1] ) < 0) {
      continue;
    } 
    if (halfLz - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 2] - h_foldedPos[ (faces[i].v[2])*3 + 2] ) < 0) {
      continue;
    }


    if (halfLx - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 0] - h_foldedPos[ (faces[i].v[2])*3 + 0] ) < 0) {
      continue;
    } 
    if (halfLy - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 1] - h_foldedPos[ (faces[i].v[2])*3 + 1] ) < 0) {
      continue;
    } 
    if (halfLz - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 2] - h_foldedPos[ (faces[i].v[2])*3 + 2] ) < 0) {
      continue;
    }
    // The minus node0 term is confusing. Probably it is useful when the system has 2 types of particles. 
    fprintf (pFile, "3 %d %d %d\n", faces[i].v[0] - node0, faces[i].v[1] - node0, faces[i].v[2] - node0);
  }

	fprintf (pFile, "CELL_TYPES %d\n", nface_temp);
	for (int i=0; i < nface_temp; i++) {
    fprintf (pFile, "5\n");
  }

  fclose (pFile);

  if (params.Ntype[1] > 0)
  {
    node0 = params.Ntype[0] *   params.N_per_sphere[0];
    face0 = params.Ntype[0] * params.face_per_sphere[0];
    numNodes = numNodesA + params.Ntype[1] *    params.N_per_sphere[1];
    numFaces = numFacesA + params.Ntype[1] * params.face_per_sphere[1];

    sprintf (filename, "%s/particleB_%d.vtk", dataFolder, step);  
    pFile = fopen (filename, "w");
	  fprintf (pFile, "# vtk DataFile Version 2.3   \n");
	  fprintf (pFile, "title particle configuration %d \n", step);
	  fprintf (pFile, "ASCII                        \n\n");
	  fprintf (pFile, "DATASET UNSTRUCTURED_GRID \n");
	  fprintf (pFile, "POINTS %d float\n", params.Ntype[1]*params.N_per_sphere[1]);

    for (int n = node0; n < numNodes; n++) {
      int offset = n*3; 
      fprintf (pFile, "%8.6lf %8.6lf %8.6lf\n", h_foldedPos[offset+0], h_foldedPos[offset+1], h_foldedPos[offset+2]);
    }
	  // check if the face spans across periodic boundaries
	  int nface_temp = 0;
	  for (int i = face0; i < numFaces; i++) 
    {
      int offset = i*3;  // a face has 3 verteices
    
      if (halfLx - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 0] - h_foldedPos[ (faces[i].v[1])*3 + 0] ) < 0) {
        continue;
      } 
      if (halfLy - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 1] - h_foldedPos[ (faces[i].v[1])*3 + 1] ) < 0) {
        continue;
      } 
      if (halfLz - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 2] - h_foldedPos[ (faces[i].v[1])*3 + 2] ) < 0) {
        continue;
      }
  
  
      if (halfLx - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 0] - h_foldedPos[ (faces[i].v[2])*3 + 0] ) < 0) {
        continue;
      } 
      if (halfLy - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 1] - h_foldedPos[ (faces[i].v[2])*3 + 1] ) < 0) {
        continue;
      } 
      if (halfLz - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 2] - h_foldedPos[ (faces[i].v[2])*3 + 2] ) < 0) {
        continue;
      }
  
  
      if (halfLx - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 0] - h_foldedPos[ (faces[i].v[2])*3 + 0] ) < 0) {
        continue;
      } 
      if (halfLy - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 1] - h_foldedPos[ (faces[i].v[2])*3 + 1] ) < 0) {
        continue;
      } 
      if (halfLz - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 2] - h_foldedPos[ (faces[i].v[2])*3 + 2] ) < 0) {
        continue;
      }
  
      nface_temp++;
    }

    fprintf (pFile, "CELLS %d %d\n", nface_temp, nface_temp*4); // 4: (# of verteices, vertex label, vertex label, vertex label)
    for (int i=face0; i < numFaces; i++) 
    {
      int offset = i*3;  // a face has 3 verteices
      
      if (halfLx - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 0] - h_foldedPos[ (faces[i].v[1])*3 + 0] ) < 0) {
        continue;
      } 
      if (halfLy - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 1] - h_foldedPos[ (faces[i].v[1])*3 + 1] ) < 0) {
        continue;
      } 
      if (halfLz - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 2] - h_foldedPos[ (faces[i].v[1])*3 + 2] ) < 0) {
        continue;
      }
  
  
      if (halfLx - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 0] - h_foldedPos[ (faces[i].v[2])*3 + 0] ) < 0) {
        continue;
      } 
      if (halfLy - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 1] - h_foldedPos[ (faces[i].v[2])*3 + 1] ) < 0) {
        continue;
      } 
      if (halfLz - fabs ( h_foldedPos[ (faces[i].v[1])*3 + 2] - h_foldedPos[ (faces[i].v[2])*3 + 2] ) < 0) {
        continue;
      }
  
  
      if (halfLx - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 0] - h_foldedPos[ (faces[i].v[2])*3 + 0] ) < 0) {
        continue;
      } 
      if (halfLy - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 1] - h_foldedPos[ (faces[i].v[2])*3 + 1] ) < 0) {
        continue;
      } 
      if (halfLz - fabs ( h_foldedPos[ (faces[i].v[0])*3 + 2] - h_foldedPos[ (faces[i].v[2])*3 + 2] ) < 0) {
        continue;
      }
      fprintf (pFile, "3 %d %d %d\n", faces[i].v[0] - node0, faces[i].v[1] - node0, faces[i].v[2] - node0);
    }

  	fprintf (pFile, "CELL_TYPES %d\n", nface_temp);
  	for (int i=0; i < nface_temp; i++) {
      fprintf (pFile, "5\n");
    }
  
    fclose (pFile);
  }
}

