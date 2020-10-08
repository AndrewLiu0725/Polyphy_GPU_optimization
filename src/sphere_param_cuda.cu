extern "C"{
#include <stdio.h>
#include "sphere_param.h"
}

__constant__ struct sphere_param d_partParams;

extern "C"
void InitializeDevPartParams (struct sphere_param *h_ptrPartParams) {

  cudaMemcpyToSymbol(d_partParams, h_ptrPartParams, sizeof(struct sphere_param));
}

//extern "C"
//void InitializeParticleParameters (char *filePath, struct sphere_param *params) {
//
//  FILE *ptr = fopen (filePath, "r");
//  if (ptr == NULL)
//    fprintf(stderr, "Cannot open particle parameter file\n");
//  else
//    fprintf(stdout, "Particle parameter file is opened\n");
//
//  fscanf (ptr, "%*s %*s %*s ");
//	fscanf (ptr, "%d %d %d", &params->lx, &params->ly, &params->lz);           
//	fscanf (ptr, "%*s %*s %*s %*s ");
//  fscanf (ptr, "%d %d %d %d", &params->cycle, &params->numStepsPerCycle, &params->write_time, &params->write_config);
//	fscanf (ptr, "%*s %*s %*s %*s %*s %*s ");
//	fscanf (ptr, "%d %d %d %d %d %d", &params->initconfig, &params->wallFlag, &params->Ntype[0], &params->nlevel[0], &params->Ntype[1], &params->nlevel[1]);
//	fscanf (ptr, "%*s %*s %*s %*s ");
//	fscanf (ptr, "%lf %d %d %d", &params->kT, &params->springType[0], &params->springType[1], &params->interparticle);
//  fscanf (ptr, "%*s %*s %*s %*s %*s ");
//  fscanf (ptr, "%lf %lf %lf %lf %lf", &params->x0, &params->shearModulus[0], &params->springConst[0], &params->shearModulus[1], &params->springConst[1]);
//	fscanf (ptr, "%*s %*s ");
//	fscanf (ptr, "%lf %lf", &params->kc[0], &params->kc[1]);
//	fscanf (ptr, "%*s %*s ");
//	fscanf (ptr, "%lf %lf", &params->kv[0], &params->kv[1]);
//	fscanf (ptr, "%*s %*s ");
//	fscanf (ptr, "%lf %lf", &params->kag[0], &params->kag[1]);
//	fscanf (ptr, "%*s %*s ");
//	fscanf (ptr, "%lf %lf", &params->kal[0], &params->kal[1]);
//	fscanf (ptr, "%*s %*s %*s ");
//	fscanf (ptr, "%lf %lf %lf", &params->eps, &params->eqLJ, &params->cutoffLJ);
//	fscanf (ptr, "%*s %*s %*s %*s ");
//	fscanf (ptr, "%lf %lf %lf %lf", &params->depthMorse, &params->widthMorse, &params->eqMorse, &params->cutoffMorse);
//	fscanf (ptr, "%*s %*s ");
//	fscanf (ptr, "%lf %lf", &params->wallConstant, &params->wallForceDis);
//  fscanf (ptr, "%*s %*s %*s ");
//  fscanf (ptr, "%lf %lf %lf", &params->nlistCutoff, &params->cellSize, &params->nlistRenewal);
//  fscanf (ptr, "%*s %*s ");
//  fscanf (ptr, "%d %lf", &params->numGrowthSteps, &params->fictionalMass);
//  fclose (ptr);
//
//  // The input is inversed x0. So, inverse it again.
//  params->x0 = 1. / params->x0;
//
//  // TODO ###################################
//  // Should prevent wrong parameter values !
//  // @ Set N_per_sphere and face_per_sphere
//  // ########################################
//  for (int n=0; n < 2; n++) { 
//	  int temp;
//	  if (params->nlevel[n] >= 0) // for a sphere
//	  {
//	    params->N_per_sphere[n] = 12;
//		  params->face_per_sphere[n] = 20;
//		  temp = 30;
//
//		  for (int i=0 ; i < params->nlevel[n] ; i++) 
//		  {
//		    params->N_per_sphere[n] += temp;
//			  params->face_per_sphere[n] *= 4;
//			  temp *= 4;
//		  }
//	  }
//  }
//	if(params->nlevel[0] == -1)   // for a RBC 
//	{
//		params->N_per_sphere[0] = 162;
//		params->face_per_sphere[0] = 320;
//	}
//	else if(params->nlevel[0] == -3)   // for a RBC with a higher resolution
//	{
//		params->N_per_sphere[0] = 642;
//		params->face_per_sphere[0] = 1280;
//	}
//	else if(params->nlevel[0] == -4)   // for a RBC with a higher resolution
//	{
//		params->N_per_sphere[0] = 2562;
//		params->face_per_sphere[0] = 5120;
//	}
//
//	if(params->nlevel[1] == -1)   // for a RBC 
//	{
//		params->N_per_sphere[1] = 162;
//		params->face_per_sphere[1] = 320;
//	}
//	else if(params->nlevel[1] == -3)   // for a RBC with a higher resolution
//	{
//		params->N_per_sphere[1] = 642;
//		params->face_per_sphere[1] = 1280;
//	}
//	else if(params->nlevel[1] == -4)   // for a RBC with a higher resolution
//	{
//		params->N_per_sphere[1] = 2562;
//		params->face_per_sphere[1] = 5120;
//	}
//  // @ Set Nsphere, num_beads, and nfaces
//	params->Nsphere   = params->Ntype[0] + params->Ntype[1];
//	params->num_beads = params->Ntype[0]*params->N_per_sphere[0]    + params->Ntype[1]*params->N_per_sphere[1];
//	params->nfaces    = params->Ntype[0]*params->face_per_sphere[0] + params->Ntype[1]*params->face_per_sphere[1];
//
//  fprintf (stdout, "Box size = (%d, %d, %d)\n", params->lx, params->ly, params->lz);
//  fprintf (stdout, "Total step = (# of cycle) * (# of step/per cycle) = %d * %d = %d\n", params->cycle, params->numStepsPerCycle, params->cycle*params->numStepsPerCycle);
//  fprintf (stdout, "Output data every                                             %d steps\n", params->write_time);
//  fprintf (stdout, "Output particle VTK files evey                                %d steps\n", params->write_config); 
//  fprintf (stdout, "(Initialization flag, Wall flag) = (%d, %d)\n", params->initconfig, params->wallFlag);
//  fprintf (stdout, "Particle numbers        (A,B) = (%d, %d)\n",  params->Ntype[0],  params->Ntype[1]);
//  fprintf (stdout, "Particle types          (A,B) = (%d, %d)\n", params->nlevel[0], params->nlevel[1]);
//  fprintf (stdout, "# of nodes per particle (A,B) = (%d, %d)\n",    params->N_per_sphere[0],    params->N_per_sphere[1]);
//  fprintf (stdout, "# of faces per particle (A,B) = (%d, %d)\n", params->face_per_sphere[0], params->face_per_sphere[1]);
//  fprintf (stdout, "# of particles      = %d\n", params->Nsphere);
//  fprintf (stdout, "# of particle nodes = %d\n", params->num_beads);
//  fprintf (stdout, "# of faces          = %d\n", params->nfaces);
//  fprintf (stdout, "Energy scale = %lf\n", params->kT);
//  fprintf (stdout, "Spring type (A, B)         = (%d, %d)\n", params->springType[0], params->springType[1]);
//  fprintf (stdout, "Inter-particle interaction = %d\n", params->interparticle);
//  fprintf (stdout, "x0 = l0/lmax = %lf\n", params->x0);
//  fprintf (stdout, "Shear modulus          (A, B) = (%lf, %lf)\n", params->shearModulus[0], params->shearModulus[1]);
//  fprintf (stdout, "Spring constant        (A, B) = (%lf, %lf)\n",  params->springConst[0],  params->springConst[1]);
//  fprintf (stdout, "Bending constant       (A, B) = (%lf, %lf)\n",  params->kc[0],  params->kc[1]);
//  fprintf (stdout, "Volume constranit      (A, B) = (%lf, %lf)\n",  params->kv[0],  params->kv[1]);
//  fprintf (stdout, "Global area constraint (A, B) = (%lf, %lf)\n", params->kag[0], params->kag[1]);
//  fprintf (stdout, "Local area constant    (A, B) = (%lf, %lf)\n", params->kal[0], params->kal[1]);
//  fprintf (stdout, "LJ potential    (eps, zeroForceDis, cutoff)      = (%lf, %lf, %lf)\n", params->eps, params->eqLJ, params->cutoffLJ);
//  fprintf (stdout, "Morse potential (D, width, zeroForceDis, cutoff) = (%lf, %lf, %lf, %lf)\n",params->depthMorse, params->widthMorse, params->eqMorse, params->cutoffMorse);
//  fprintf (stdout, "Wall force (forceConstant, dis) = (%lf, %lf)\n", params->wallConstant, params->wallForceDis);
//  fprintf (stdout, "Neighbor list parameters (verlet cutoff, cell size, Renewal threshold) = (%lf, %lf, %lf)\n", params->nlistCutoff, params->cellSize, params->nlistRenewal);
//  fprintf (stdout, "(Growth steps, Frictional mass) = (%d, %lf)\n\n", params->numGrowthSteps, params->fictionalMass);
//
//}

//extern "C"
//void InitializeDevPartParams () {
//
//  cudaMemcpy (devPartParams, &hostPartParams, sizeof(struct sphere_param), cudaMemcpyHostToDevice);
//  //cudaMemcpyToSymbol (devPartParams, &hostPartParams, sizeof(struct sphere_param));
//}

__global__ void PrintDevPartParams (struct sphere_param *devPartParams) {

  printf ("Box size = (%d, %d, %d)\n", devPartParams->lx, devPartParams->ly, devPartParams->lz);
  printf ("Total step = (# of cycle) * (# of step/per cycle) = %d * %d = %d\n", devPartParams->cycle, devPartParams->numStepsPerCycle, devPartParams->cycle*devPartParams->numStepsPerCycle);
  printf ("Output data every                                             %d steps\n", devPartParams->write_time);
  printf ("Output particle VTK files evey                                %d steps\n", devPartParams->write_config); 
  printf ("(Initialization flag, Wall flag) = (%d, %d)\n", devPartParams->initconfig, devPartParams->wallFlag);
  printf ("Particle numbers        (A,B) = (%d, %d)\n",  devPartParams->Ntype[0],  devPartParams->Ntype[1]);
  printf ("Particle types          (A,B) = (%d, %d)\n", devPartParams->nlevel[0], devPartParams->nlevel[1]);
  printf ("# of nodes per particle (A,B) = (%d, %d)\n",    devPartParams->N_per_sphere[0],    devPartParams->N_per_sphere[1]);
  printf ("# of faces per particle (A,B) = (%d, %d)\n", devPartParams->face_per_sphere[0], devPartParams->face_per_sphere[1]);
  printf ("# of particles      = %d\n", devPartParams->Nsphere);
  printf ("# of particle nodes = %d\n", devPartParams->num_beads);
  printf ("# of faces          = %d\n", devPartParams->nfaces);
  printf ("Energy scale = %lf\n", devPartParams->kT);
  printf ("Spring type (A, B)         = (%d, %d)\n", devPartParams->springType[0], devPartParams->springType[1]);
  printf ("Inter-particle interaction = %d\n", devPartParams->interparticle);
  printf ("x0 = l0/lmax = %lf\n", devPartParams->x0);
  printf ("Shear modulus          (A, B) = (%lf, %lf)\n", devPartParams->shearModulus[0], devPartParams->shearModulus[1]);
  printf ("Spring constant        (A, B) = (%lf, %lf)\n",  devPartParams->springConst[0],  devPartParams->springConst[1]);
  printf ("Bending constant       (A, B) = (%lf, %lf)\n",  devPartParams->kc[0],  devPartParams->kc[1]);
  printf ("Volume constranit      (A, B) = (%lf, %lf)\n",  devPartParams->kv[0],  devPartParams->kv[1]);
  printf ("Global area constraint (A, B) = (%lf, %lf)\n", devPartParams->kag[0], devPartParams->kag[1]);
  printf ("Local area constant    (A, B) = (%lf, %lf)\n", devPartParams->kal[0], devPartParams->kal[1]);
  printf ("LJ potential    (eps, zeroForceDis, cutoff)      = (%lf, %lf, %lf)\n", devPartParams->eps, devPartParams->eqLJ, devPartParams->cutoffLJ);
  printf ("Morse potential (D, width, zeroForceDis, cutoff) = (%lf, %lf, %lf, %lf)\n",devPartParams->depthMorse, devPartParams->widthMorse, devPartParams->eqMorse, devPartParams->cutoffMorse);
  printf ("Wall force (forceConstant, dis) = (%lf, %lf)\n", devPartParams->wallConstant, devPartParams->wallForceDis);
  printf ("Neighbor list parameters (verlet cutoff, cell size, Renewal threshold) = (%lf, %lf, %lf)\n", devPartParams->nlistCutoff, devPartParams->cellSize, devPartParams->nlistRenewal);
  printf ("(Growth steps, Frictional mass) = (%d, %lf)\n", devPartParams->numGrowthSteps, devPartParams->fictionalMass);
}

extern "C"
void PrintDevVariables (struct sphere_param *devPartParams) {
  PrintDevPartParams<<<1,1>>>(devPartParams);
  cudaDeviceSynchronize();  // If this line is commented out, the information will not be printted on the screen !
}

//extern "C"
//void PrintParams () {
//
//  struct sphere_param params;
//  cudaMemcpy (&params, devPartParams, sizeof(struct sphere_param), cudaMemcpyDeviceToHost);  
//
//  printf ("Box size = (%d, %d, %d)\n", params.lx, params.ly, params.lz);
//  printf ("Total step = (# of cycle) * (# of step/per cycle) = %d * %d = %d\n", params.cycle, params.numStepsPerCycle, params.cycle*params.numStepsPerCycle);
//  printf ("Output data every                                             %d steps\n", params.write_time);
//  printf ("Output particle VTK files evey                                %d steps\n", params.write_config); 
//  printf ("(Initialization flag, Wall flag) = (%d, %d)\n", params.initconfig, params.wallFlag);
//  printf ("Particle numbers        (A,B) = (%d, %d)\n",  params.Ntype[0],  params.Ntype[1]);
//  printf ("Particle types          (A,B) = (%d, %d)\n", params.nlevel[0], params.nlevel[1]);
//  printf ("# of nodes per particle (A,B) = (%d, %d)\n",    params.N_per_sphere[0],    params.N_per_sphere[1]);
//  printf ("# of faces per particle (A,B) = (%d, %d)\n", params.face_per_sphere[0], params.face_per_sphere[1]);
//  printf ("# of particles      = %d\n", params.Nsphere);
//  printf ("# of particle nodes = %d\n", params.num_beads);
//  printf ("# of faces          = %d\n", params.nfaces);
//  printf ("Energy scale = %lf\n", params.kT);
//  printf ("Spring type (A, B)         = (%d, %d)\n", params.springType[0], params.springType[1]);
//  printf ("Inter-particle interaction = %d\n", params.interparticle);
//  printf ("x0 = l0/lmax = %lf\n", params.x0);
//  printf ("Shear modulus          (A, B) = (%lf, %lf)\n", params.shearModulus[0], params.shearModulus[1]);
//  printf ("Spring constant        (A, B) = (%lf, %lf)\n",  params.springConst[0],  params.springConst[1]);
//  printf ("Bending constant       (A, B) = (%lf, %lf)\n",  params.kc[0],  params.kc[1]);
//  printf ("Volume constranit      (A, B) = (%lf, %lf)\n",  params.kv[0],  params.kv[1]);
//  printf ("Global area constraint (A, B) = (%lf, %lf)\n", params.kag[0], params.kag[1]);
//  printf ("Local area constant    (A, B) = (%lf, %lf)\n", params.kal[0], params.kal[1]);
//  printf ("LJ potential    (eps, zeroForceDis, cutoff)      = (%lf, %lf, %lf)\n", params.eps, params.eqLJ, params.cutoffLJ);
//  printf ("Morse potential (D, width, zeroForceDis, cutoff) = (%lf, %lf, %lf, %lf)\n",params.depthMorse, params.widthMorse, params.eqMorse, params.cutoffMorse);
//  printf ("Wall force (forceConstant, dis) = (%lf, %lf)\n", params.wallConstant, params.wallForceDis);
//  printf ("Neighbor list parameters (verlet cutoff, cell size, Renewal threshold) = (%lf, %lf, %lf)\n", params.nlistCutoff, params.cellSize, params.nlistRenewal);
//  printf ("(Growth steps, Frictional mass) = (%d, %lf)\n\n", params.numGrowthSteps, params.fictionalMass);
//
//}

