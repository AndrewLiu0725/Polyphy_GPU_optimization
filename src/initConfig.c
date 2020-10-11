#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "initConfig.h"
#include "tools.h"
#include "neighborList.h"
#include "forces.h"
#include "integration.h"
#include "output.h"

#include "cuda_runtime_api.h"

void ModifyParameters (int step, int *numBonds, struct sphere_param *params, struct monomer *monomers) {

  int totalGrowthSteps  = params->numGrowthSteps;
  int numParticles     = params->Nsphere;
  int numParticlesA    = params->Ntype[0];
  int numNodesPerPartA = params->N_per_sphere[0];

  params->V0[0] = params->V0_temp[0] + step * (params->V0_final[0] - params->V0_temp[0]) / totalGrowthSteps;
  params->V0[1] = params->V0_temp[1] + step * (params->V0_final[1] - params->V0_temp[1]) / totalGrowthSteps;
  params->A0[0] = params->A0_temp[0] + step * (params->A0_final[0] - params->A0_temp[0]) / totalGrowthSteps;
  params->A0[1] = params->A0_temp[1] + step * (params->A0_final[1] - params->A0_temp[1]) / totalGrowthSteps;

  params->kv[0]        = ((double)step / totalGrowthSteps) * 20;
  params->kv[1]        = ((double)step / totalGrowthSteps) * 20;
  params->kag[0]       = ((double)step / totalGrowthSteps) * 200;
  params->kag[1]       = ((double)step / totalGrowthSteps) * 200;
  params->kal[0]       = ((double)step / totalGrowthSteps) * 10;   
  params->kal[1]       = ((double)step / totalGrowthSteps) * 10;   
  params->wallConstant = ((double)step / totalGrowthSteps) * 20;

  //printf("step         = %d\n", step);
  //printf("(kv kag kal) = (%f, %f, %f)\n", params->kv[0], params->kag[0], params->kal[0]);
  //printf("wall const   = %f\n", params->wallConstant);
  //printf("V0[0], A0[0] = %f, %f\n", params->V0[0], params->A0[0]);

  #pragma omp parallel for schedule(static)
  for (int index=0; index < numParticles; index++) {
    int numNodesPerPart;
    int offset;
    if (index < numParticlesA) {
      numNodesPerPart = params->N_per_sphere[0];
      offset          = index*numNodesPerPart;
    }
    else {
      numNodesPerPart = params->N_per_sphere[1];
      offset = numParticlesA * numNodesPerPartA + (index - numParticlesA)*numNodesPerPart;
    }

    for (int j=0; j < numNodesPerPart; j++) {
      int n1 = j + offset;
      for (int bond=1; bond <= numBonds[n1]; bond++)  {
        monomers[n1].initLength[bond] = monomers[n1].initLength_temp[bond] + 
        step * (monomers[n1].initLength_final[bond] - monomers[n1].initLength_temp[bond]) / totalGrowthSteps;

        monomers[n1].lmax[bond] = monomers[n1].initLength[bond] / params->x0;
      }    
    }
  }
}

int GenerateConfig (struct sphere_param *sphere_pm, char *work_dir, struct monomer *monomers, struct face *faces, double *pos, double *foldedPos, int *numBonds, int *blist, int ***Blist, int *h_node_face_id, int *h_node_face_number) {

// This function sets monomer[].pos_pbc, monomer[].pos, monomer[].blist, and monoer[].sphere_id

  double extraY = 1.5;
  double extraZ = 1.5;
  double extraR_1 = 0.5;
  double extraR_2 = 0.5;
  double ly = sphere_pm->ly;

	double radius[2];
	int Ntype[2];
	int nlevel[2];
	int Nbeads[2];
	int Nfaces[2];
	for (int i=0; i < 2 ; i++) {
		Ntype[i]  = sphere_pm->Ntype[i];
		nlevel[i] = sphere_pm->nlevel[i];
		Nbeads[i] = sphere_pm->N_per_sphere[i];
		Nfaces[i] = sphere_pm->face_per_sphere[i];
	}
	int Nsphere   = sphere_pm->Nsphere;
	int num_beads = sphere_pm->num_beads;

  double reduced_factor = 1.0;
  int    reduced_flag = 0;
	int trial_count = 0; // should be initialized for avoiding troubles.
  int trial_count_max = 100000;
	srand((unsigned)time(NULL));

	double **centers;
	double **v;            
	int ***blist_rbc;

	centers = (double**) calloc (Nsphere, sizeof(double*));
	if (centers == NULL)  fprintf (stderr, "cannot allocate centers");
	for (int n=0 ; n < Nsphere ; n++) {
		centers[n] = (double *) calloc (3, sizeof(double));
		if (centers[n] == NULL)  fprintf (stderr, "cannot allocate centers");
	}

  int maxNbeads = max(Nbeads[0], Nbeads[1]);

	v = (double **) calloc (maxNbeads, sizeof(double *));
	if (v == NULL) fprintf (stderr, "cannot allocate bond vectors");
  for (int i=0; i < maxNbeads; i++) {
	  v[i] = (double *) calloc (3, sizeof(double));
			if (v[i] == NULL) fprintf (stderr, "cannot allocate bond vectors");
	}
	
	blist_rbc = (int ***) calloc (maxNbeads, sizeof(int **));
	if (blist_rbc == NULL) fprintf (stderr, "cannot allocate blist");

	for (int i=0 ; i < maxNbeads; i++) {
	  blist_rbc[i] = (int **) calloc(MAX_BOND+1, sizeof(int *));
		if (blist_rbc[i] == NULL) fprintf (stderr, "cannot allocate blist");
		for (int j=0 ; j<=MAX_BOND ; j++) {
			blist_rbc[i][j] = (int *) calloc(3, sizeof(int));
			if (blist_rbc[i][j] == NULL) fprintf (stderr, "cannot allocate blist");
		}
	}

	// @ Generate a random (initconfig==1) or specific (initconfig==2) config
	if (sphere_pm->initconfig == 1) 
	{   
    // Allow 2 different types of particles in terms of the particle size and shape 
		for (int i=0; i < NTYPES ; i++) {  
			if (nlevel[i] == -1) {
        radius[i] = 3.91;         
      }
			else if (nlevel[i] == 2) {
        radius[i] = 3.350;   // the avg. spring length ~ 1.0 
      }
      else if(nlevel[i] == 3) {
        radius[i] = 6.7;   // the avg. spring length ~ 1.0 
      }
			else if (nlevel[i] == -3) {
        radius[i] = 7.82;
      }
      else if (nlevel[i] == 4) {
        radius[i] = 13.5;   // the avg. spring length ~ 1.0 
      }
			else if (nlevel[i] == -4) {
        radius[i] = 15.64;
      }
		}

    reduced_factor = 1.;
    reduced_flag = 0;
		for (int n=0; n < Nsphere; n++) {

      // Reduced particle sizes and generate a new config all over again after 
      // 'trial_count_max' times of attempts
   		if (trial_count > trial_count_max) {
				n = 0;   
        reduced_factor *= 0.96;
//ly += 1; // for the box resizing method; it's failed; may try again in the future 
        reduced_flag = 1;  
			}

      // variables for each particle
 			trial_count = 0;
      int particleType;  
      double n2radius;   
      int numNodesPerPart;
      int offset;
      int overlap;
       
      if (n < Ntype[0]) {
        particleType    = nlevel[0];
        n2radius        = radius[0] * reduced_factor; 
        numNodesPerPart = Nbeads[0];
        offset          = n * Nbeads[0];
      }
      else {
        particleType    = nlevel[1];
        n2radius        = radius[1] * reduced_factor; 
        numNodesPerPart = Nbeads[1];
        offset = Ntype[0]*Nbeads[0] + (n - Ntype[0])*Nbeads[1];
      }

      // Locate the particle   
			do 
			{
        overlap = FALSE;

				if (trial_count > trial_count_max) {
          printf ("Cannnot insert sphere %d within the max. # of attmpts !\n", n);
					break;  /*Leave the do-while loop and go to the if-statement containing 
                    a 'continue' statement below, finally go back to the for-loop 
                    above.*/  
				}
//        if (sphere_pm->wallFlag == 1) {
//          double tempY = n2radius + extraY; 
//          centers[n][0] = ((double)rand() / (double)RAND_MAX) * sphere_pm->lx;          // [0, lx]
//          centers[n][1] = ((double)rand() / (double)RAND_MAX) * (ly - 2*tempY) + tempY; // [n2radius+extraY ,ly-n2radius-extraY]
//          centers[n][2] = ((double)rand() / (double)RAND_MAX) *  sphere_pm->lz;         // [0, lz]
//        }
//        else if (sphere_pm->wallFlag ==2) {
//          double tempY = n2radius + extraY; 
//          double tempZ = n2radius + extraZ; 
//          centers[n][0] = ((double)rand() / (double)RAND_MAX) * sphere_pm->lx;
//          centers[n][1] = ((double)rand() / (double)RAND_MAX) * (ly-2*tempY) + tempY;
//          centers[n][2] = ((double)rand() / (double)RAND_MAX) * (sphere_pm->lz-2*tempZ) + tempZ;
//        }
//        else {
//          printf("In 'GenerateConfig' sphere_pm->wallFlag != 1 and !=2, which causes problems.\n");
//          //exit(99);
//        }
        if (sphere_pm->wallFlag == 1) {
          double tempY = n2radius + extraY; 
          centers[n][0] = ((double)rand() / (double)RAND_MAX) * sphere_pm->lx;          // [0, lx]
          centers[n][1] = ((double)rand() / (double)RAND_MAX) * (ly - 2*tempY -2) + (1 + tempY); // [1+tempY, ly-1-tempY-(1+tempY)]
          centers[n][2] = ((double)rand() / (double)RAND_MAX) *  sphere_pm->lz;         // [0, lz]
        }
        else if (sphere_pm->wallFlag ==2) {
          double tempY = n2radius + extraY; 
          double tempZ = n2radius + extraZ; 
          centers[n][0] = ((double)rand() / (double)RAND_MAX) * sphere_pm->lx;
          centers[n][1] = ((double)rand() / (double)RAND_MAX) * (ly            - 2*tempY -2) + (1 + tempY);
          centers[n][2] = ((double)rand() / (double)RAND_MAX) * (sphere_pm->lz - 2*tempZ -2) + (1 + tempZ);
        }
        else {
          printf("In 'GenerateConfig' sphere_pm->wallFlag != 1 and !=2, which causes problems.\n");
          //exit(99);
        }

        // Check particle-particle overlap 
				for (int n1=0; n1 < n; n1++) 
				{ 
          double n1radius;
          if (n1 < Ntype[0])
            n1radius = radius[0];
          else
            n1radius = radius[1];
         
          double temp = n1radius + extraR_1 + n2radius + extraR_2; 
          double criterion2 = temp * temp; 

					double dx = centers[n][0] - centers[n1][0];
					double dy = centers[n][1] - centers[n1][1];
					double dz = centers[n][2] - centers[n1][2];
          switch (sphere_pm->wallFlag)
          {
            case 1:
            dx = n_image (dx, sphere_pm->lx);
            dz = n_image (dz, sphere_pm->lz);
            break;

            case 2:
            dx = n_image (dx, sphere_pm->lx);
            break;

            default:
            printf ("wall falg value is wrong\n");
            break;
          }
					double r2 = dx*dx + dy*dy + dz*dz;

					if (r2 < criterion2) { 
				    overlap = TRUE;
            trial_count++;
            break;   
					}
				}
			} while(overlap == TRUE);
			if (trial_count > trial_count_max) {
        n=0;       // Reset the for-loop counter (should do!?).
        continue;  // Go back to the for-loop above.
      }
      
      ReadTemplate (particleType, reduced_factor, work_dir, v, blist_rbc);

			// Randomly rotate the particle
      double alpha = ((double)rand() / (double) RAND_MAX) * 360;
      double beta  = ((double)rand() / (double) RAND_MAX) * 360;
      double gamma = ((double)rand() / (double) RAND_MAX) * 360;
      double mat[3][3];
      RotationMatrix (alpha, beta, gamma, mat);

  	  for (int i=0; i < numNodesPerPart; i++) 
 			{             
        int m = i + offset;  // bead (monomer) index

        pos[m*3+0] = centers[n][0];
        pos[m*3+1] = centers[n][1];
        pos[m*3+2] = centers[n][2];
        //monomers[m].pos_pbc[0] = centers[n][0]; 
        //monomers[m].pos_pbc[1] = centers[n][1]; 
        //monomers[m].pos_pbc[2] = centers[n][2];
        for (int j=0; j < DIMS; j++) {
          for (int k=0; k < DIMS; k++) {
            //monomers[m].pos_pbc[j] += mat[j][k] * v[i][k]; 
            pos[m*3+j] += mat[j][k] * v[i][k];
          }
        }
        if (sphere_pm->wallFlag == 1) {
          //monomers[m].pos[0] = box(monomers[m].pos_pbc[0], max_x);
          //monomers[m].pos[1] =     monomers[m].pos_pbc[1];
          //monomers[m].pos[2] = box(monomers[m].pos_pbc[2], max_z); 
          foldedPos[m*3+0] = box(pos[m*3+0], sphere_pm->lx); 
          foldedPos[m*3+1] =     pos[m*3+1];
          foldedPos[m*3+2] = box(pos[m*3+2], sphere_pm->lz);
        }
        else if (sphere_pm->wallFlag == 2) {
          //monomers[m].pos[0] = box(monomers[m].pos_pbc[0], max_x);
          //monomers[m].pos[1] =     monomers[m].pos_pbc[1];
          //monomers[m].pos[2] =     monomers[m].pos_pbc[2]; 
          foldedPos[m*3+0] = box(pos[m*3+0], sphere_pm->lx); 
          foldedPos[m*3+1] =     pos[m*3+1];
          foldedPos[m*3+2] =     pos[m*3+2];
        }
        else {
          printf ("In 'GenerateConfig', sphere_pm->wallFlag != 1 or !=2 is not allowed at the moment.\n");
        }

//  			monomers[m].blist[0][0] = blist_rbc[i][0][0];
//        blist[m][0][0] = blist_rbc[i][0][0];
        numBonds[m] = blist_rbc[i][0][0]; 

  			for (int j=1 ; j<=blist_rbc[i][0][0] ; j++) 
  			{
  			  //monomers[m].blist[j][0] = offset + blist_rbc[i][j][0];
  				//monomers[m].blist[j][1] = offset + blist_rbc[i][j][1];
  				//monomers[m].blist[j][2] = offset + blist_rbc[i][j][2];
//  			  blist[m][j][0] = offset + blist_rbc[i][j][0];
//  				blist[m][j][1] = offset + blist_rbc[i][j][1];
//  				blist[m][j][2] = offset + blist_rbc[i][j][2];
          blist[m*18+(j-1)*3+0] = offset + blist_rbc[i][j][0];
          blist[m*18+(j-1)*3+1] = offset + blist_rbc[i][j][1];
          blist[m*18+(j-1)*3+2] = offset + blist_rbc[i][j][2];
  			}
  			monomers[m].sphere_id = n;
  		}				

//      printf("%d particles inserted\n", n+1);    
		}
	}
	else if (sphere_pm->initconfig == 2) 
	{   
    // Allow 2 different types of particles in terms of the particle size and shape 
		for (int i=0; i < NTYPES ; i++) {  
			if (nlevel[i] == -1) {
        radius[i] = 3.91;         
      }
			else if (nlevel[i] == 2) {
        radius[i] = 3.350;   // the avg. spring length ~ 1.0 
      }
      else if(nlevel[i] == 3) {
        radius[i] = 6.7;   // the avg. spring length ~ 1.0 
      }
			else if (nlevel[i] == -3) {
        radius[i] = 7.82;
      }
      else if (nlevel[i] == 4) {
        radius[i] = 13.5;   // the avg. spring length ~ 1.0 
      }
			else if (nlevel[i] == -4) {
        radius[i] = 15.64;
      }
		}

    // read the position and orientation of each particle
    FILE *pFile;
    char filename[150];
    sprintf (filename, "%s/init/posOrient.dat", work_dir);
    pFile = fopen (filename, "r");

		for (int n=0; n < Nsphere; n++) {

      double alpha, beta, gamma;
      fscanf (pFile, "%lf %lf %lf %lf %lf %lf", &centers[n][0], &centers[n][1], &centers[n][2], &alpha, &beta, &gamma);

      // variables for each particle
      int particleType;  
      double n2radius;   
      int numNodesPerPart;
      int offset;
      int overlap;
       
      if (n < Ntype[0]) {
        particleType    = nlevel[0];
        n2radius        = radius[0] * reduced_factor; 
        numNodesPerPart = Nbeads[0];
        offset          = n * Nbeads[0];
      }
      else {
        particleType    = nlevel[1];
        n2radius        = radius[1] * reduced_factor; 
        numNodesPerPart = Nbeads[1];
        offset = Ntype[0]*Nbeads[0] + (n - Ntype[0])*Nbeads[1];
      }
      // Check particle-particle overlap 
      overlap = FALSE;
			for (int n1=0; n1 < n; n1++) 
			{ 
        double n1radius;
        if (n1 < Ntype[0])
          n1radius = radius[0];
        else
          n1radius = radius[1];

        double temp = n1radius + extraR_1 + n2radius + extraR_2; 
        double criterion2 = temp * temp; 
				double dx = centers[n][0] - centers[n1][0];
				double dy = centers[n][1] - centers[n1][1];
				double dz = centers[n][2] - centers[n1][2];
        // n_image function is to address pbc
        switch (sphere_pm->wallFlag) {
          case 1:
            dx = n_image (dx, sphere_pm->lx);
            dz = n_image (dz, sphere_pm->lz);
            break;
          case 2:
            dx = n_image (dx, sphere_pm->lx);
            break;
          default:
            printf ("wall falg value is wrong\n");
            break;
        }
				double r2 = dx*dx + dy*dy + dz*dz;

				if (r2 < criterion2) { 
				  overlap = TRUE;
          break;   
			  }
			}
      if (overlap == TRUE) {
        fprintf (stdout, "Particles may overlap!\n");
      }
      // Read the shape of pre-described particle
      ReadTemplate (particleType, reduced_factor, work_dir, v, blist_rbc);

			// Rotate the particle by creating rotation matrix, mat
      double mat[3][3];
      RotationMatrix (alpha, beta, gamma, mat);

  	  for (int i=0; i < numNodesPerPart; i++) 
 			{       
        int m = i + offset;  // bead (monomer) index
        //monomers[m].pos_pbc[0] = centers[n][0]; 
        //monomers[m].pos_pbc[1] = centers[n][1]; 
        //monomers[m].pos_pbc[2] = centers[n][2];
        pos[m*3+0] = centers[n][0];
        pos[m*3+1] = centers[n][1];
        pos[m*3+2] = centers[n][2];    

        // project and rotate
        for (int j=0; j < DIMS; j++) {
          for (int k=0; k < DIMS; k++) {
            //monomers[m].pos_pbc[j] += mat[j][k] * v[i][k]; 
            pos[m*3+j] += mat[j][k] * v[i][k]; 
          }
        }
        // foldedPos is the pos in pbc
        if (sphere_pm->wallFlag == 1) {
          //monomers[m].pos[0] = box(monomers[m].pos_pbc[0], max_x);
          //monomers[m].pos[1] =     monomers[m].pos_pbc[1];
          //monomers[m].pos[2] = box(monomers[m].pos_pbc[2], max_z); 
          foldedPos[m*3+0] = box(pos[m*3+0], sphere_pm->lx);
          foldedPos[m*3+1] =     pos[m*3+1];
          foldedPos[m*3+2] = box(pos[m*3+2], sphere_pm->lz);
        }
        else if (sphere_pm->wallFlag == 2) {
          //monomers[m].pos[0] = box(monomers[m].pos_pbc[0], max_x);
          //monomers[m].pos[1] =     monomers[m].pos_pbc[1];
          //monomers[m].pos[2] =     monomers[m].pos_pbc[2]; 
          foldedPos[m*3+0] = box(pos[m*3+0], sphere_pm->lx);
          foldedPos[m*3+1] =     pos[m*3+1];
          foldedPos[m*3+2] =     pos[m*3+2];
        }
        else {
          printf ("In 'GenerateConfig', sphere_pm->wallFlag != 1 or !=2 is not allowed at the moment.\n");
        }

//  			monomers[m].blist[0][0] = blist_rbc[i][0][0];
//  			blist[m][0][0] = blist_rbc[i][0][0];
        numBonds[m] = blist_rbc[i][0][0];

  			for (int j=1 ; j<=blist_rbc[i][0][0] ; j++) 
  			{
//  			  monomers[m].blist[j][0] = offset + blist_rbc[i][j][0];
//  				monomers[m].blist[j][1] = offset + blist_rbc[i][j][1];
//  				monomers[m].blist[j][2] = offset + blist_rbc[i][j][2];
//  			  blist[m][j][0] = offset + blist_rbc[i][j][0];
//  				blist[m][j][1] = offset + blist_rbc[i][j][1];
//  				blist[m][j][2] = offset + blist_rbc[i][j][2];
          blist[m*18+(j-1)*3+0] = offset + blist_rbc[i][j][0];
          blist[m*18+(j-1)*3+1] = offset + blist_rbc[i][j][1];
          blist[m*18+(j-1)*3+2] = offset + blist_rbc[i][j][2];
  			}
  			monomers[m].sphere_id = n;
  		}				

      printf("%d particles inserted\n", n+1);    
		}
    fclose (pFile);
	}
  // @ Generate a config by reading the config. file
//	else if(sphere_pm->initconfig == 3) 
//	{
//    reduced_flag = 0;
//
//#ifdef DISAGGREGATION
//ReadConfig_tmp (work_dir, sphere_pm->num_beads, monomers, pos, foldedPos, blist);
//#else
//ReadConfig (work_dir, sphere_pm->num_beads, monomers, pos, foldedPos, blist);
//
//#endif
//  }
fprintf (stdout, "initconfig option 3 is turnned off\n"); 

  // @ Generate a bond list
  //
  // Note: ############################################
  // There is another bond list stored in "blist[][]"
  // It's not necessary to have two bond lists.
  // ##################################################
  AssignBlist (sphere_pm, work_dir, Blist, faces, h_node_face_id, h_node_face_number); 
  // Note:
  // merged into AssignBlist
  // @ Store labels of beads making up a face to "face[]"
  //SetFace (sphere_pm, monomers, faces);                

  for (int n=0; n < Nsphere; n++)
    free (centers[n]);
  free (centers);

  for (int i=0; i < maxNbeads; i++)
    free (v[i]);
  free (v);

  for (int i=0; i < maxNbeads; i++)
    for (int j=0; j <= MAX_BOND; j++)
      free (blist_rbc[i][j]);
  for (int i=0; i < maxNbeads; i++)
    free (blist_rbc[i]);
  free (blist_rbc);

  printf("reduced factor = %f\n", reduced_factor);
//printf("increased height = %f\n", ly-max_y);
	return reduced_flag;
//return ly;
}

void ReadTemplate (int particleType, double reducedFactor, char *work_dir, double **v, int ***blist) {

	char filename[200];
	FILE *stream;
  int numNodes;

  switch(particleType)
  {
    case -1:
    numNodes = 162; 
	  sprintf (filename, "%s/init/n2_biconcave.dat", work_dir);
    break;  
    case -3:
    numNodes = 642;
	  sprintf (filename, "%s/init/n3_biconcave.dat", work_dir);
    break;
    case 2:
    numNodes = 162;
	  sprintf (filename, "%s/init/n2_sphere.dat", work_dir);
    break;
    case 3:
    numNodes = 642;
	  sprintf (filename, "%s/init/n3_sphere.dat", work_dir);
    break;
    case 4:
    numNodes = 2562; 
	  sprintf (filename, "%s/init/n4_sphere.dat", work_dir);
    break;
    case -4:
    numNodes = 2562;
	  sprintf (filename, "%s/init/n4_biconcave.dat", work_dir);
    break;
    default:
    printf ("invalid particle type in 'ReadTemplate\n'");
    break;
  }
  stream = fopen (filename, "r");
	for (int i=0; i < numNodes; i++) {
	  fscanf (stream, "%le %le %le", &v[i][0], &v[i][1], &v[i][2]);
    v[i][0] *= reducedFactor;
    v[i][1] *= reducedFactor;
    v[i][2] *= reducedFactor;
  }
	for (int i=0; i < numNodes; i++) {
		fscanf(stream, "%*s %d", &blist[i][0][0]);
		for (int j=1; j<=blist[i][0][0]; j++) {
			fscanf(stream, "%d %d %d", &blist[i][j][0], &blist[i][j][1], &blist[i][j][2]);
    }
	}
	fclose(stream);
}

void RotationMatrix (double thetaX, double thetaY, double thetaZ, double mat[3][3]) {

  double alpha = thetaX / 180.0 * M_PI;
  double beta  = thetaY / 180.0 * M_PI;
  double gamma = thetaZ / 180.0 * M_PI;

  mat[0][0] = cos(gamma)*cos(beta);  mat[0][1] = cos(gamma)*sin(beta)*sin(alpha) - sin(gamma)*cos(alpha);  mat[0][2] = cos(gamma)*sin(beta)*cos(alpha) + sin(gamma)*sin(alpha);
  mat[1][0] = sin(gamma)*cos(beta);  mat[1][1] = sin(gamma)*sin(beta)*sin(alpha) + cos(gamma)*cos(alpha);  mat[1][2] = sin(gamma)*sin(beta)*cos(alpha) - cos(gamma)*sin(alpha);
  mat[2][0] = -sin(beta);            mat[2][1] = cos(beta)*sin(alpha);                                     mat[2][2] = cos(beta)*cos(alpha);
}

void AssignBlist (struct sphere_param *sphere_pm, char *work_dir, int ***Blist, struct face *faces, int *h_node_face_id, int *h_node_face_number) {

	char filename[200];
	FILE *stream;

  for (int n=0; n < sphere_pm->Nsphere; n++) {

    int particleType;
  	int numBeadsPerPart;
    int numFacesPerPart;
    int beadOffset;
    int faceOffset;
    if (n < sphere_pm->Ntype[0]) {
      particleType    = sphere_pm->nlevel[0];
      numBeadsPerPart = sphere_pm->N_per_sphere[0];
      numFacesPerPart = sphere_pm->face_per_sphere[0];  
      beadOffset = n*sphere_pm->N_per_sphere[0];
      faceOffset = n*sphere_pm->face_per_sphere[0];
    }
    else {
      particleType    = sphere_pm->nlevel[1];
      numBeadsPerPart = sphere_pm->N_per_sphere[1];
      numFacesPerPart = sphere_pm->face_per_sphere[1];
      beadOffset = sphere_pm->Ntype[0]*   sphere_pm->N_per_sphere[0] + (n-sphere_pm->Ntype[0])*   sphere_pm->N_per_sphere[1];
      faceOffset = sphere_pm->Ntype[0]*sphere_pm->face_per_sphere[0] + (n-sphere_pm->Ntype[0])*sphere_pm->face_per_sphere[1];
    }

    switch (particleType) {
      case -1:
        sprintf (filename, "%s/init/n2Blist.dat", work_dir);
        break;
      case 2:
        sprintf (filename, "%s/init/n2Blist.dat", work_dir);
        break;
      case 3:
        sprintf (filename, "%s/init/n3Blist.dat", work_dir);
        break;
      case -3:
        sprintf (filename, "%s/init/n3Blist.dat", work_dir);
        break;
      case 4:
        sprintf (filename, "%s/init/n4Blist.dat", work_dir);
        break;
      case -4:
        sprintf (filename, "%s/init/n4Blist.dat", work_dir);
        break;
      default:
        printf ("non-available particle type in 'AssignBlist'\n");
        break;
    }
    
    stream = fopen (filename, "r");
	  for (int i=0; i < numBeadsPerPart; i++) {
      int index = i + beadOffset; 
		  fscanf (stream, "%*s %d", &Blist[index][0][0]);
		  for (int j=1; j <= Blist[index][0][0]; j++) {
			  fscanf (stream, "%d %d %d", &Blist[index][j][0], &Blist[index][j][1], &Blist[index][j][2]);

        Blist[index][j][0] += beadOffset;
        Blist[index][j][1] += beadOffset;
        Blist[index][j][2] += beadOffset;
      }
	  }
	  fclose(stream);

    switch (particleType) {
      case -1:
        sprintf (filename, "%s/init/n2faceList.dat", work_dir);
        break;
      case 2:
        sprintf (filename, "%s/init/n2faceList.dat", work_dir);
        break;
      case 3:
        sprintf (filename, "%s/init/n3faceList.dat", work_dir);
        break;
      case -3:
        sprintf (filename, "%s/init/n3faceList.dat", work_dir);
        break;
      case 4:
        sprintf (filename, "%s/init/n4faceList.dat", work_dir);
        break;
      case -4:
        sprintf (filename, "%s/init/n4faceList.dat", work_dir);
        break;
      default:
        printf ("non-available particle type in 'AssignBlist'\n");
        break;
    }    

    stream = fopen (filename, "r");
    for (int i=0; i < numFacesPerPart; i++) {
      int index = i + faceOffset; // the index of face
      faces[index].sphere_id = n;
      fscanf (stream, "%d %d %d", &faces[index].v[0], &faces[index].v[1], &faces[index].v[2]);

      faces[index].v[0] += beadOffset;
      faces[index].v[1] += beadOffset;
      faces[index].v[2] += beadOffset;

      h_node_face_id[faces[index].v[0]*MAX_BOND + h_node_face_number[faces[index].v[0]]] = index;
      h_node_face_number[faces[index].v[0]] += 1;
      h_node_face_id[faces[index].v[1]*MAX_BOND + h_node_face_number[faces[index].v[1]]] = index;
      h_node_face_number[faces[index].v[1]] += 1;
      h_node_face_id[faces[index].v[2]*MAX_BOND + h_node_face_number[faces[index].v[2]]] = index;
      h_node_face_number[faces[index].v[2]] += 1;
      
    }
    fclose (stream);
  }
}

void ReadConfig (char *work_dir, int num_node, struct monomer *node, double *pos, double *foldedPos, int ***blist) {
  // sphere_id, pos_pbc[], pos[], blist[][] are set

  char file_name[100];
  sprintf(file_name, "%s/init/init_config.dat", work_dir);
  FILE *stream; 
  stream = fopen(file_name, "r");
  for(int n=0; n < num_node; n++) {
    int offset = n*3; 
    fscanf(stream, "%u", &node[n].sphere_id);
    fscanf(stream, "%le %le %le", &pos[offset+0],       &pos[offset+1],       &pos[offset+2]);
    fscanf(stream, "%le %le %le", &foldedPos[offset+0], &foldedPos[offset+1], &foldedPos[offset+2]);
    fscanf(stream, "%d", &blist[n][0][0]);
    for(int i=1; i <= blist[n][0][0]; i++) {
      fscanf(stream, "%d %d %d", &blist[n][i][0], &blist[n][i][1], &blist[n][i][2]);
    }
  }
  fclose(stream);
}

void ReadConfig_tmp(char *work_dir, int num_node, struct monomer *node, double *pos, double *foldedPos, int ***blist) {
  // sphere_id, pos_pbc[], pos[], blist[][] are set

  char file_name[100];
  sprintf(file_name, "%s/init/finalConfig.dat", work_dir);
  FILE *stream; 
  stream = fopen(file_name, "r");
  for(int n=0; n < num_node; n++) {
    int offset = n*3;
    fscanf(stream, "%u", &node[n].sphere_id);
    fscanf(stream, "%le %le %le", &pos[offset+0],       &pos[offset+1],       &pos[offset+2]);
    fscanf(stream, "%le %le %le", &foldedPos[offset+0], &foldedPos[offset+1], &foldedPos[offset+2]);
    fscanf(stream, "%d", &blist[n][0][0]);
    for(int i=1; i <= blist[n][0][0]; i++) {
      fscanf(stream, "%d %d %d", &blist[n][i][0], &blist[n][i][1], &blist[n][i][2]);
    }
  }
  fclose(stream);
}

void SetEqPartParams (char *work_dir, struct sphere_param *sphere_pm, int *numBonds, struct monomer *vertex, struct face *faces) {

  // TODO: #########################################################################
  // Currently this function should be called after monomer[].blist is set. I should
  // reorganize the code in a way that monomer's data members can more logically be 
  // initialized or set
  // ###############################################################################

  // initLength, lmax, initAngle, area_0, initLength_final, V0_final, A0_final

  FILE *stream; 
  int numParticles  = sphere_pm->Nsphere;
  int numParticlesA = sphere_pm->Ntype[0];

  for (int n=0; n < numParticles; n++)
  {
    int particleShape;
    int type; // 0 or 1
    int numNodes;
    int numFaces;
    int nodeOffset;
    int faceOffset;
    char filename[150];
    if (n < numParticlesA) {
      type = 0;
      particleShape = sphere_pm->nlevel[0];
      numNodes      = sphere_pm->N_per_sphere[0];
      numFaces      = sphere_pm->face_per_sphere[0];
      nodeOffset    = n*numNodes;
      faceOffset    = n*numFaces; 
    }
    else {
      type = 1;
      particleShape = sphere_pm->nlevel[1];
      numNodes      = sphere_pm->N_per_sphere[1];
      numFaces      = sphere_pm->face_per_sphere[1];
      nodeOffset = numParticlesA*sphere_pm->N_per_sphere[0]    + (n-numParticlesA)*sphere_pm->N_per_sphere[1];
      faceOffset = numParticlesA*sphere_pm->face_per_sphere[0] + (n-numParticlesA)*sphere_pm->face_per_sphere[1];
    }
    switch (particleShape) 
    {
      case -1:
      sprintf (filename, "%s/init/shapePara_n2rbc.dat", work_dir);
      break;
      case -3:
      sprintf (filename, "%s/init/shapePara_n3rbc.dat", work_dir);
      break;
      case 2:
      sprintf (filename, "%s/init/shapePara_n2sphere.dat", work_dir);
      break;   
      case 3:
      sprintf (filename, "%s/init/shapePara_n3sphere.dat", work_dir);
      break;   
      case 4:
      sprintf (filename, "%s/init/shapePara_n4sphere.dat", work_dir);
      break;   
      case -4:
      sprintf (filename, "%s/init/shapePara_n4rbc.dat", work_dir);
      break;   
    }
    stream = fopen (filename, "r");
    fscanf (stream, "%le %le", &sphere_pm->V0[type], &sphere_pm->A0[type]);
    sphere_pm->V0_final[type] = sphere_pm->V0[type];
    sphere_pm->A0_final[type] = sphere_pm->A0[type];
//printf ("(V0, A0, V0final, A0final)=(%lf, %lf, %lf, %lf)\n", sphere_pm->V0[0], sphere_pm->A0[0], sphere_pm->V0_final[0], sphere_pm->A0_final[0]);
//printf ("(V0, A0, V0final, A0final)=(%lf, %lf, %lf, %lf)\n", sphere_pm->V0[1], sphere_pm->A0[1], sphere_pm->V0_final[1], sphere_pm->A0_final[1]);
    for (int i=0; i < numNodes; i++) 
    {
      int index = i + nodeOffset; 

      for (int j=1; j <= numBonds[index]; j++) 
      {
        fscanf (stream, "%le %le %le", &vertex[index].initLength[j], &vertex[index].lmax[j], &vertex[index].initAngle[j]);
        vertex[index].initLength_final[j] = vertex[index].initLength[j]; 
      }
    }
    for (int i=0; i < numFaces; i++) 
    {
      int index = i + faceOffset;
      fscanf (stream,"%le", &faces[index].area_0);
    }
    fclose(stream);
  }
}

void SetSpringConstants (struct sphere_param *sphere_pm, int *numBonds, struct monomer *node) {

  // TODO: #########################################################################
  // Currently this function should be called after monomer[].blist is set. I should
  // reorganize the code in a way that monomer's data members can more logically be 
  // initialized or set
  // ###############################################################################

  double x0 = sphere_pm->x0;
  double x0_2 = x0*x0;
  double x0_3 = x0*x0*x0; 
  double one_m_x0 = 1 - sphere_pm->x0;
  double one_m_x0_2 = one_m_x0 * one_m_x0;
  double one_m_x0_3 = one_m_x0 * one_m_x0 * one_m_x0;
  int numNodesA = sphere_pm->Ntype[0]*sphere_pm->N_per_sphere[0];
 
  for(int n=0; n < sphere_pm->num_beads; n++)
  {
    double shearModulus; 
    if (n < numNodesA) {
      shearModulus = sphere_pm->shearModulus[0]*sphere_pm->kT;
    }
    else {
      shearModulus = sphere_pm->shearModulus[1]*sphere_pm->kT;
    }
    for(int m=1; m <= numBonds[n]; m++)
    {
      double l0 = node[n].initLength[m];
      double l0_2 = l0*l0;
      double factor = ((0.25*sqrt(3)/l0) * ((0.5*x0/one_m_x0_3) - (0.25/one_m_x0_2) + 0.25)) +
                      (3*sqrt(3)*(4*x0_3-9*x0_2+6*x0) / (16*l0*one_m_x0_2));
      node[n].kT_inversed_persis[m] = shearModulus / factor; 
      node[n].kp[m] = 0.25*l0_2*node[n].kT_inversed_persis[m]*(4*x0_3-9*x0_2+6*x0) / one_m_x0_2;
      //node[n].lmax[m] = l0 / x0;
    }
  }
}

void SetReducedPartParams (double *pos, int *numBonds, int *blist, struct sphere_param *sphere_pm, struct monomer *vertex, struct face *faces) {

  // initLength_temp, lmax, initAngle_temp, V0_temp, A0_temp

  for(int n1=0; n1 < sphere_pm->num_beads; n1++)
  {
    
    for(int label=1; label <= numBonds[n1]; label++)
    {
      // calculate initial bond length
//      int n2 = blist[n1][label][0];
//      int n3 = blist[n1][label][1];
//      int n4 = blist[n1][label][2];
      int n2 = blist[n1*18+(label-1)*3+0];
      int n3 = blist[n1*18+(label-1)*3+1];
      int n4 = blist[n1*18+(label-1)*3+2];

      double bond[3];
      double bondLength2 = 0.;
      for (int d=0; d < 3; d++) {
        //bond[d] = vertex[n1].pos_pbc[d] - vertex[n2].pos_pbc[d];
        bond[d] = pos[n1*3+d] - pos[n2*3+d];
        bondLength2 += bond[d]*bond[d];
      }
      vertex[n1].initLength_temp[label] = sqrt(bondLength2);
      vertex[n1].lmax[label] = vertex[n1].initLength_temp[label] / sphere_pm->x0;

      // calculate initial bending angle 
//      double x31[3], x21[3], x41[3];
//      double normal1[3], normal2[3];
//      double cross[3], crossNorm;
//      for(int d=0; d < 3; d++) {
//        //x31[d] = vertex[n3].pos_pbc[d] - vertex[n1].pos_pbc[d];
//        //x21[d] = vertex[n2].pos_pbc[d] - vertex[n1].pos_pbc[d];
//        //x41[d] = vertex[n4].pos_pbc[d] - vertex[n1].pos_pbc[d];
//        x31[d] = pos[n3*3+d] - pos[n1*3+d];
//        x21[d] = pos[n2*3+d] - pos[n1*3+d];
//        x41[d] = pos[n4*3+d] - pos[n1*3+d];
//      }
//      product(x31,x21, normal1);
//      product(x21,x41, normal2);
//      product(normal1, normal2, cross);
//
//      crossNorm = cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2];
//      crossNorm = sqrt(crossNorm);
//
//      vertex[n1].initAngle_temp[label] = atan2(crossNorm, iproduct(normal1,normal2));
//      // correction! Theta could be negtive!
//      double orient = cross[0]*x21[0] + cross[1]*x21[1] + cross[2]*x21[2];
//      if (orient > 0) {
//        vertex[n1].initAngle_temp[label] *= -1;
//      }
    }
  }
  // Calculate V0_temp and A0_temp
  int nodeOffset;
  int faceOffset;
  int numNodes;
  int numFaces;
  double com[3]={0};
  double area=0.;
  double volume=0.;

  nodeOffset = 0;
  faceOffset = 0;
  numNodes = sphere_pm->N_per_sphere[0];
  numFaces = sphere_pm->face_per_sphere[0]; 
  
  for (int j=0; j < numNodes; j++) {
    int n1 = j + nodeOffset;
    //com[0] += vertex[n1].pos_pbc[0];  
    //com[1] += vertex[n1].pos_pbc[1];  
    //com[2] += vertex[n1].pos_pbc[2];  
    com[0] += pos[n1*3+0];  
    com[1] += pos[n1*3+1];  
    com[2] += pos[n1*3+2];  
  }
  com[0] /=  numNodes;
  com[1] /=  numNodes;
  com[2] /=  numNodes;    

  for(int nface=0; nface < numFaces; nface++) {
    int m = nface + faceOffset; 
    int n1 = faces[m].v[0];
    int n2 = faces[m].v[1];
    int n3 = faces[m].v[2];
    double dr[3], q1[3], q2[3], normal[3];
    for(int d=0; d < 3; d++) {
      //dr[d] = vertex[n1].pos_pbc[d] - com[d];
      //q1[d] = vertex[n3].pos_pbc[d] - vertex[n1].pos_pbc[d];
      //q2[d] = vertex[n2].pos_pbc[d] - vertex[n1].pos_pbc[d];
      dr[d] = pos[n1*3+d] - com[d];
      q1[d] = pos[n3*3+d] - pos[n1*3+d];
      q2[d] = pos[n2*3+d] - pos[n1*3+d];
    }
    product(q1, q2, normal);
    // correction
    //volume +=fabs(dr[0]*normal[0]+dr[1]*normal[1]+dr[2]*normal[2])/6.;
    volume += (dr[0]*normal[0]+dr[1]*normal[1]+dr[2]*normal[2])/6.;
    area += sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2])*0.5;
  }
  sphere_pm->V0_temp[0] = volume;  
  sphere_pm->A0_temp[0] = area;   
   

  if(sphere_pm->Ntype[1] > 0) {
    nodeOffset = sphere_pm->Ntype[0]*sphere_pm->N_per_sphere[0];
    faceOffset = sphere_pm->Ntype[0]*sphere_pm->face_per_sphere[0];
    numNodes = sphere_pm->N_per_sphere[1];
    numFaces = sphere_pm->face_per_sphere[1];   
    com[0] = 0.;
    com[1] = 0.;
    com[2] = 0.;
    area   = 0.;
    volume = 0.;

    for (int j=0; j < numNodes; j++) {
      int n1 = j + nodeOffset;
      //com[0] += vertex[n1].pos_pbc[0];  
      //com[1] += vertex[n1].pos_pbc[1];  
      //com[2] += vertex[n1].pos_pbc[2];  
      com[0] += pos[n1*3+0];  
      com[1] += pos[n1*3+1];  
      com[2] += pos[n1*3+2];
    }
    com[0] /=  numNodes;
    com[1] /=  numNodes;
    com[2] /=  numNodes;    

    for(int nface=0; nface < numFaces; nface++) {
      int m = nface + faceOffset; 
      int n1 = faces[m].v[0];
      int n2 = faces[m].v[1];
      int n3 = faces[m].v[2];
      double dr[3], q1[3], q2[3], normal[3];
      for(int d=0; d < 3; d++) {
        //dr[d] = vertex[n1].pos_pbc[d] - com[d];
        //q1[d] = vertex[n3].pos_pbc[d] - vertex[n1].pos_pbc[d];
        //q2[d] = vertex[n2].pos_pbc[d] - vertex[n1].pos_pbc[d];
        dr[d] = pos[n1*3+d] - com[d];
        q1[d] = pos[n3*3+d] - pos[n1*3+d];
        q2[d] = pos[n2*3+d] - pos[n1*3+d];
      }
      product(q1, q2, normal);
      // correction
      //volume +=fabs(dr[0]*normal[0]+dr[1]*normal[1]+dr[2]*normal[2])/6.;
      volume += (dr[0]*normal[0]+dr[1]*normal[1]+dr[2]*normal[2])/6.;
      area += sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2])*0.5;
    }
    sphere_pm->V0_temp[1] = volume;  
    sphere_pm->A0_temp[1] = area;   
  } 
}

void RestoreParticle (struct sphere_param *h_params, struct face *h_faces, double *h_foldedPos, int *h_numBonds, struct monomer *h_monomers, double *h_springForces, double *h_bendingForces, double *h_volumeForces, double *h_globalAreaForces, double *h_localAreaForces, double *h_wallForces, double *h_interparticleForces, double *h_pos, int *h_numNeighbors, int *h_nlist, int *h_blist, double *h_forces, double *h_velocities, double *h_nlistPos) {

  int count = 0;
  fprintf (stdout,"Start restoring particles\n");
  
  for (int step=0; step <= h_params->numGrowthSteps; step++)
  {
    if (step % 1000 == 0) {
      char work_dir[100]=".";  
      WriteParticleVTK (step, work_dir, *h_params, h_faces, h_foldedPos);
    }

    ModifyParameters (step, h_numBonds, h_params, h_monomers);

    ComputeForces (h_params, h_monomers, h_faces, h_springForces, h_bendingForces, h_volumeForces, h_globalAreaForces, h_localAreaForces, h_wallForces, h_interparticleForces, h_pos, h_numNeighbors, h_nlist, h_numBonds, h_blist, h_forces);

    euler_update (h_params->num_beads, h_params->fictionalMass, h_params->lx, h_params->ly, h_params->lz, h_forces, h_velocities, h_pos, h_foldedPos);

    count += RenewNeighborList (*h_params, h_nlistPos, h_foldedPos, h_numNeighbors, h_nlist);
  }
  fprintf (stdout, "nlist is updated %d times.\n", count);
  fprintf (stdout,"End restoring particles\n\n");
}

void ConstructFacePairList (struct sphere_param *h_params, struct face *h_faces, int *h_node_face_id, int *h_node_face_number, int *h_face_pair_list, int *blist, int *numBonds){
  int current_num_face_pair = 0;
  int n1, n2, n3;
  int i, j, k, l;
  int F;
  for (i = 0; i < h_params->nfaces; i ++){
    n1 = h_faces[i].v[0];
    n2 = h_faces[i].v[1];
    n3 = h_faces[i].v[2];

    // edge (n1, n2)
    for (j = 0; j < h_node_face_number[n1]; j ++){
      F = h_node_face_id[n1*MAX_BOND+j];
      for (k = 0; k < h_node_face_number[n2]; k ++){
        if ((F == h_node_face_id[n2*MAX_BOND+k]) && (F > i)){
          h_face_pair_list[FPSIZE*current_num_face_pair] = i;
          h_face_pair_list[FPSIZE*current_num_face_pair+1] = F;
          for (l = 0; l < numBonds[n1]; l ++){
            if (blist[n1*MAX_BOND*3 + l*3] == n2){
              h_face_pair_list[FPSIZE*current_num_face_pair+2] = (l+1);
              h_face_pair_list[FPSIZE*current_num_face_pair+3] = n1;
              h_face_pair_list[FPSIZE*current_num_face_pair+4] = n2;
              h_face_pair_list[FPSIZE*current_num_face_pair+5] = blist[n1*MAX_BOND*3 + l*3 + 1];
              h_face_pair_list[FPSIZE*current_num_face_pair+6] = blist[n1*MAX_BOND*3 + l*3 + 2];
            }
          }
          current_num_face_pair += 1;
        }
      }
    }
    // edge (n2, n3)
    for (j = 0; j < h_node_face_number[n2]; j ++){
      F = h_node_face_id[n2*MAX_BOND+j];
      for (k = 0; k < h_node_face_number[n3]; k ++){
        if ((F == h_node_face_id[n3*MAX_BOND+k]) && (F > i)){
          h_face_pair_list[FPSIZE*current_num_face_pair] = i;
          h_face_pair_list[FPSIZE*current_num_face_pair+1] = F;
          for (l = 0; l < numBonds[n2]; l ++){
            if (blist[n2*MAX_BOND*3 + l*3] == n3){
              h_face_pair_list[FPSIZE*current_num_face_pair+2] = (l+1);
              h_face_pair_list[FPSIZE*current_num_face_pair+3] = n2;
              h_face_pair_list[FPSIZE*current_num_face_pair+4] = n3;
              h_face_pair_list[FPSIZE*current_num_face_pair+5] = blist[n2*MAX_BOND*3 + l*3 + 1];
              h_face_pair_list[FPSIZE*current_num_face_pair+6] = blist[n2*MAX_BOND*3 + l*3 + 2];
            }
          }
          current_num_face_pair += 1;
        }
      }
    }
    // edge (n1, n3)
    for (j = 0; j < h_node_face_number[n1]; j ++){
      F = h_node_face_id[n1*MAX_BOND+j];
      for (k = 0; k < h_node_face_number[n3]; k ++){
        if ((F == h_node_face_id[n3*MAX_BOND+k]) && (F > i)){
          h_face_pair_list[FPSIZE*current_num_face_pair] = i;
          h_face_pair_list[FPSIZE*current_num_face_pair+1] = F;
          for (l = 0; l < numBonds[n1]; l ++){
            if (blist[n1*MAX_BOND*3 + l*3] == n3){
              h_face_pair_list[FPSIZE*current_num_face_pair+2] = (l+1);
              h_face_pair_list[FPSIZE*current_num_face_pair+3] = n1;
              h_face_pair_list[FPSIZE*current_num_face_pair+4] = n3;
              h_face_pair_list[FPSIZE*current_num_face_pair+5] = blist[n1*MAX_BOND*3 + l*3 + 1];
              h_face_pair_list[FPSIZE*current_num_face_pair+6] = blist[n1*MAX_BOND*3 + l*3 + 2];
            }
          }
          current_num_face_pair += 1;
        }
      }
    }
  }
  h_params->num_face_pair = current_num_face_pair;
}