#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime_api.h"

#include "sphere_param.h"
#include "initConfig.h"
#include "tools.h"
#include "neighborList.h"

#include "common.h"

#include "lb.h"
#include "kernel.h"
#include "config.h"

int main () {

cudaSetDevice(1);
  
  char *work_dir = ".";  // Note: This setting may go wrong ! 
  int reduced_flag = 0;
  int cycle = 0;

  struct sphere_param  h_params;
 	struct monomer      *h_monomers;
	struct face         *h_faces;
  double              *h_pos;
  double              *h_foldedPos;
  double              *h_nlistPos;
  double              *h_springForces;
  double              *h_bendingForces;
  double              *h_volumeForces;
  double              *h_globalAreaForces;
  double              *h_localAreaForces;
  double              *h_wallForces;
  double              *h_interparticleForces;
  double              *h_forces;
  double              *h_velocities;
  int                 *h_numNeighbors;
  int                 *h_nlist;
  int                 *h_numBonds;
  int                 *h_blist;
  int               ***h_Blist;

  //struct sphere_param *d_params;
 	struct monomer      *d_monomers;
	struct face         *d_faces;
  double              *d_pos;
  double              *d_foldedPos;
  double              *d_nlistPos;
  double              *d_springForces;
  double              *d_bendingForces;
  float              *d_volumeForces;
  float              *d_globalAreaForces;
  double              *d_localAreaForces;
  double              *d_wallForces;
  double              *d_interparticleForces;
  double              *d_forces;
  double              *d_velocities;
  int                 *d_numNeighbors;
  int                 *d_nlist;
  int                 *d_numBonds;
  int                 *d_blist;
  //  int    ***Blist;
  float *d_coms;
  double *d_faceCenters;
  double *d_normals;
  float *d_areas;
  float *d_volumes;


  char partParamPath[150] = {"./init/parameter.dat"};
  InitializeParticleParameters (partParamPath, &h_params);
  char lbParamPath[100] = {"./init/lb_parameters.dat"};
  InitializeLBparameters (lbParamPath, &h_LBparams);
  char dataFolder[100] = {"./data"};
/*
Initialize_LB_Parameters_dev (&h_LBparams);
PrintDevParasWrapper ();
//PrintConstVariableFromOtherFile_wrapper ();
cudaDeviceReset();
exit(0);
*/

  // TODO #####################################################################
  // Parameters for initialization are hard coded
  // In the future I may use the box resize method to generate a init. config.
  // ##########################################################################
  int    initSpringType    = 2;
  int    initInterparticle = 1;
  double initSpringConst   = 11.547;
  double init_kc           = 1.62;
  double init_kv           = 20.0;
  double init_kag          = 200.0;
  double init_kal          = 10.0;
  double initWallConst     = 20.0;
  double init_eps          = 0.005652;
  double init_eqLJ         = 1.0;
  double init_cutoffLJ     = 1.0;
  double initCellSize      = init_cutoffLJ;
  double initNlistCutoff   = init_cutoffLJ+1;

  int tempSpringType[2];
  int tempInterparticle = h_params.interparticle; 
  double tempSpringConst[2];
  double temp_kc[2];
  double temp_kv[2];
  double temp_kag[2];
  double temp_kal[2];
  double tempWallConst       = h_params.wallConstant; 
  double temp_eps            = h_params.eps;
  double temp_eqLJ           = h_params.eqLJ;
  double temp_cutoffLJ       = h_params.cutoffLJ;
  double tempCellSize        = h_params.cellSize;
  double tempNlistCutoff     = h_params.nlistCutoff;
  tempSpringType[0]          = h_params.springType[0];
  tempSpringType[1]          = h_params.springType[1];
  tempSpringConst[0]         = h_params.springConst[0];
  tempSpringConst[1]         = h_params.springConst[1];
  temp_kc[0]                 = h_params.kc[0];
  temp_kc[1]                 = h_params.kc[1];
  temp_kv[0]                 = h_params.kv[0];
  temp_kv[1]                 = h_params.kv[1];
  temp_kag[0]                = h_params.kag[0];
  temp_kag[1]                = h_params.kag[1];
  temp_kal[0]                = h_params.kal[0];
  temp_kal[1]                = h_params.kal[1];

  h_params.springType[0]    = initSpringType;       
  h_params.springType[1]    = initSpringType;       
  h_params.interparticle    = initInterparticle;  
  h_params.springConst[0]   = initSpringConst;
  h_params.springConst[0]   = initSpringConst;
  h_params.kc[0]            = init_kc;
  h_params.kc[1]            = init_kc;
  h_params.kv[0]            = init_kv;
  h_params.kv[1]            = init_kv;
  h_params.kag[0]           = init_kag;
  h_params.kag[1]           = init_kag;
  h_params.kal[0]           = init_kal;
  h_params.kal[1]           = init_kal;
  h_params.wallConstant     = initWallConst;
  h_params.eps              = init_eps;
  h_params.eqLJ             = init_eqLJ;
  h_params.cutoffLJ         = init_cutoffLJ;
  h_params.cellSize         = initCellSize;
  h_params.nlistCutoff      = initNlistCutoff;


	if (h_params.Nsphere > 0) 
  {
    h_monomers = (struct monomer*)malloc(h_params.num_beads*sizeof(struct monomer));
		if(h_monomers == NULL) {
			printf("Memory allocation for monomers failed\n");
			exit(2);
		}

		h_faces = (struct face*)malloc(h_params.nfaces*sizeof(struct face));
		if(h_faces == NULL) {
			printf("Memory allocation for monomers failed\n");
			exit(2);
		}

    h_pos = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_pos == NULL) {
      printf ("Memory allocation for foldedPos failed\n");
      exit(2);
    }   

    h_foldedPos = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_foldedPos == NULL) {
      printf ("Memory allocation for foldedPos failed\n");
      exit(2);
    }

    h_nlistPos = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_nlistPos == NULL) {
      printf ("Memory allocation for nlistPos failed\n");
      exit(2);
    }

h_numNeighbors = (int *) malloc (h_params.num_beads*sizeof(int));
if (h_numNeighbors == NULL) {
  printf ("Memory allocation for h_numNeighbors failed\n");
  exit(2);
}

    h_nlist = (int *) malloc (h_params.num_beads*MAX_N*sizeof(int));
    if (h_nlist == NULL) {
      fprintf (stderr, "Memory allocation for nlist failed\n");
      exit(2);
    }
 
    h_numBonds = (int *) malloc (h_params.num_beads*sizeof(int));
    if (h_numBonds == NULL) {
      fprintf (stderr, "Memory allocation for numBonds failed\n");
    }
 
    h_blist = (int *) malloc (h_params.num_beads*6*3*sizeof(int)); 
    if (h_blist == NULL) {
      fprintf (stderr, "Memory allocation for blist failed\n");
    }
   
    h_Blist = (int ***) malloc (h_params.num_beads*sizeof(int **));
    if (h_Blist == NULL) fprintf (stderr, "Memory allocation for Blist failed\n");
    for (int n=0; n < h_params.num_beads; n++) {
      h_Blist[n] = (int **) malloc((MAX_BOND+1)*sizeof(int *));
      if (h_Blist[n] == NULL) fprintf (stderr, "Memory allocation for Blist failed\n"); 
      for (int m=0; m <= MAX_BOND; m++) {
        h_Blist[n][m] = (int *) malloc (3*sizeof(int));
        if (h_Blist[n][m] == NULL) fprintf (stderr, "Memory allocation for blist failed\n");  
      }
    }  

    for (int n=0; n < h_params.num_beads; n++) {
      h_monomers[n].updatedFlag = FALSE;
    } 

    h_forces = (double *)calloc(h_params.num_beads*3, sizeof(double));
    if (h_forces == NULL) {
      fprintf(stderr,"Memory allocation for forces failed\n");
      exit(2);   
    }

    h_springForces = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_springForces == NULL) {
      fprintf(stderr,"Memory allocation for springForces failed\n");
      exit(2);   
    }

    h_bendingForces = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_bendingForces == NULL) {
      fprintf(stderr,"Memory allocation for bendingForces failed\n");
      exit(2);   
    }

    h_volumeForces = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_volumeForces == NULL) {
      fprintf(stderr,"Memory allocation for volumeForces failed\n");
      exit(2);   
    }

    h_globalAreaForces = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_globalAreaForces == NULL) {
      fprintf(stderr,"Memory allocation for globalAreaForces failed\n");
      exit(2);   
    }

    h_localAreaForces = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_localAreaForces == NULL) {
      fprintf(stderr,"Memory allocation for localAreaForces failed\n");
      exit(2);   
    }

    h_wallForces = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_wallForces == NULL) {
      fprintf(stderr,"Memory allocation for wallForces failed\n");
      exit(2);   
    }

    h_interparticleForces = (double *)malloc(h_params.num_beads*3*sizeof(double));
    if (h_interparticleForces == NULL) {
      fprintf(stderr,"Memory allocation for interparticleForces failed\n");
      exit(2);   
    }
 
    h_velocities = (double *)calloc(h_params.num_beads*3, sizeof(double));
    if (h_velocities == NULL) {
      fprintf(stderr,"Memory allocation for velocities failed\n");
      exit(2);
    }   

    //#ifdef PARTICLE_GPU
    //cudaMalloc((void**)&d_params, sizeof(struct sphere_param));
    cudaMalloc((void**)&d_monomers, h_params.num_beads*sizeof(struct monomer));
    cudaMalloc((void**)&d_faces,       h_params.nfaces*sizeof(struct face));
    CHECK(cudaMalloc((void**)&d_pos,       h_params.num_beads*3*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_foldedPos, h_params.num_beads*3*sizeof(double)));
    cudaMalloc((void**)&d_nlistPos,        h_params.num_beads*3*sizeof(double));
    cudaMalloc((void**)&d_springForces,        h_params.num_beads*3*sizeof(double));
    cudaMalloc((void**)&d_bendingForces,       h_params.num_beads*3*sizeof(double));
    cudaMalloc((void**)&d_volumeForces,        h_params.num_beads*3*sizeof(float));
    cudaMalloc((void**)&d_globalAreaForces,    h_params.num_beads*3*sizeof(float));
    cudaMalloc((void**)&d_localAreaForces,     h_params.num_beads*3*sizeof(double));
    cudaMalloc((void**)&d_wallForces,          h_params.num_beads*3*sizeof(double));
    cudaMalloc((void**)&d_interparticleForces, h_params.num_beads*3*sizeof(double));
    CHECK(cudaMalloc((void**)&d_velocities,    h_params.num_beads*3*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_forces,        h_params.num_beads*3*sizeof(double)));
    cudaMalloc((void**)&d_numNeighbors, h_params.num_beads*sizeof(int));       
    cudaMalloc((void**)&d_nlist, h_params.num_beads*MAX_N*sizeof(int));
    cudaMalloc((void**)&d_numBonds, h_params.num_beads*sizeof(int));
    cudaMalloc((void**)&d_blist, h_params.num_beads*6*3*sizeof(int));




    cudaMalloc((void**)&d_coms, h_params.Nsphere*3*sizeof(float));
    cudaMalloc((void**)&d_areas, h_params.Nsphere*sizeof(float));
    cudaMalloc((void**)&d_volumes, h_params.Nsphere*sizeof(float));
    cudaMalloc((void**)&d_faceCenters, (h_params.Ntype[0]*h_params.face_per_sphere[0]+h_params.Ntype[1]*h_params.face_per_sphere[1])*3*sizeof(double));
    cudaMalloc((void**)&d_normals, (h_params.Ntype[0]*h_params.face_per_sphere[0]+h_params.Ntype[1]*h_params.face_per_sphere[1])*3*sizeof(double));
    cudaMemset (d_coms, 0, h_params.Nsphere*3*sizeof(float));
    cudaMemset (d_areas, 0, h_params.Nsphere*sizeof(float));
    cudaMemset (d_volumes, 0, h_params.Nsphere*sizeof(float));
    cudaMemset (d_faceCenters, 0, (h_params.Ntype[0]*h_params.face_per_sphere[0]+h_params.Ntype[1]*h_params.face_per_sphere[1])*3*sizeof(double));
    cudaMemset (d_normals, 0, (h_params.Ntype[0]*h_params.face_per_sphere[0]+h_params.Ntype[1]*h_params.face_per_sphere[1])*3*sizeof(double));




    //#endif

    reduced_flag = GenerateConfig (&h_params, work_dir, h_monomers, h_faces, h_pos, h_foldedPos, h_numBonds, h_blist, h_Blist);

    SetEqPartParams (work_dir, &h_params, h_numBonds, h_monomers, h_faces);

    SetSpringConstants (&h_params, h_numBonds, h_monomers);   

    // @ Restore particle
    //if (reduced_flag == 1)  // Restore the particle size and parameter values
    //{
    SetReducedPartParams (h_pos, h_numBonds, h_blist, &h_params, h_monomers, h_faces);

    InitializeNeighborList (h_params, h_foldedPos, h_nlistPos, h_numNeighbors, h_nlist);

    //#ifdef PARTICLE_GPU
    InitializeDevPartParams (&h_params);
    //Initialize_LB_Parameters_dev (&h_LBparams);  // It is done in 'InitializeLB'.
    //cudaMemcpy (d_params,   &h_params, sizeof(struct sphere_param), cudaMemcpyHostToDevice);  
    CHECK(cudaMemcpy (d_monomers,     h_monomers,  h_params.num_beads*sizeof(struct monomer), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy (d_faces,        h_faces,     h_params.nfaces*sizeof(struct face),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy (d_pos,          h_pos,       h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy (d_foldedPos,    h_foldedPos, h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy (d_nlistPos,     h_nlistPos,  h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy (d_numNeighbors, h_numNeighbors, h_params.num_beads*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy (d_nlist,        h_nlist,     h_params.num_beads*MAX_N*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy (d_numBonds,     h_numBonds,  h_params.num_beads*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy (d_blist,        h_blist,     h_params.num_beads*6*3*sizeof(int), cudaMemcpyHostToDevice));

    //cudaMemcpy (d_springForces,        h_springForces,        h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy (d_bendingForces,       h_bendingForces,       h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy (d_volumeForces,        h_volumeForces,        h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy (d_globalAreaForces,    h_globalAreaForces,    h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy (d_localAreaForces,     h_localAreaForces,     h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy (d_wallForces,          h_wallForces,          h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy (d_interparticleForces, h_interparticleForces, h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy (d_forces,              h_forces,              h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy (d_velocities,          h_velocities,          h_params.num_beads*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset (d_springForces,        0, h_params.num_beads*3*sizeof(double));
    cudaMemset (d_bendingForces,       0, h_params.num_beads*3*sizeof(double));
cudaMemset (d_volumeForces,        0, h_params.num_beads*3*sizeof(float));
cudaMemset (d_globalAreaForces,    0, h_params.num_beads*3*sizeof(float));
    cudaMemset (d_localAreaForces,     0, h_params.num_beads*3*sizeof(double));
    cudaMemset (d_wallForces,          0, h_params.num_beads*3*sizeof(double));
    cudaMemset (d_interparticleForces, 0, h_params.num_beads*3*sizeof(double));
    cudaMemset (d_velocities,          0, h_params.num_beads*3*sizeof(double));
    cudaMemset (d_forces,              0, h_params.num_beads*3*sizeof(double));
    //#endif

    if (reduced_flag == 1) 
    {
//      #ifdef PARTICLE_GPU
      RestoreParticle_gpu (h_params, h_foldedPos, h_nlistPos, h_numNeighbors, h_nlist, h_faces, d_monomers, d_faces, d_numBonds, d_blist, d_numNeighbors, d_nlist, d_pos, d_foldedPos, d_nlistPos, d_velocities, d_springForces, d_bendingForces, d_volumeForces, d_globalAreaForces, d_localAreaForces, d_wallForces, d_interparticleForces, d_forces    ,d_coms, d_faceCenters, d_normals, d_areas, d_volumes);      
//      #else
//      RestoreParticle (&h_params, h_faces, h_foldedPos, h_numBonds, h_monomers, h_springForces, h_bendingForces, h_volumeForces, h_globalAreaForces, h_localAreaForces, h_wallForces, h_interparticleForces, h_pos, h_nlist, h_blist, h_forces, h_velocities, h_nlistPos);
//      #endif
    }
    // Retrieve parameter values inputted in parameter.dat
    // TODO: In the future I may use the box resize method to generate the init. config.
    h_params.springType[0]  = tempSpringType[0];       
    h_params.springType[1]  = tempSpringType[0];       
    h_params.interparticle  = tempInterparticle;  
    h_params.springConst[0] = tempSpringConst[0];
    h_params.springConst[1] = tempSpringConst[1];
    h_params.kc[0]          = temp_kc[0];
    h_params.kc[1]          = temp_kc[1];
    h_params.kv[0]          = temp_kv[0];
    h_params.kv[1]          = temp_kv[1];
    h_params.kag[0]         = temp_kag[0];
    h_params.kag[1]         = temp_kag[1];
    h_params.kal[0]         = temp_kal[0];
    h_params.kal[1]         = temp_kal[1];
    h_params.wallConstant   = tempWallConst;
    h_params.eps            = temp_eps;
    h_params.eqLJ           = temp_eqLJ;
    h_params.cutoffLJ       = temp_cutoffLJ;
    h_params.cellSize       = tempCellSize;
    h_params.nlistCutoff    = tempNlistCutoff;
    //#ifdef PARTICLE_GPU
    InitializeDevPartParams (&h_params);
    //#endif

CHECK(cudaMemcpy (h_monomers,  d_monomers,  h_params.num_beads*sizeof(struct monomer), cudaMemcpyDeviceToHost));
  }

  // @ Initialize LB
  #ifdef LB_GPU
  InitializeLB ();
  #endif

  // @ Main loop
  // cycle should be declared before 'ReadCheckpointFiles'
  if (cycle == h_params.cycle)  h_params.cycle *= 2;
  else if (cycle > h_params.cycle) h_params.cycle += cycle;
 
	while (cycle < h_params.cycle) {
    fprintf (stdout, "cycle = %d; %d cycle starts\n",cycle, cycle+1);
    fflush (stdout);

    #ifdef PARTICLE_GPU
    cycle = LBkernel (cycle, h_params, h_foldedPos, d_foldedPos, dataFolder, h_faces, d_pos, devBoundaryMap, devBoundaryVelocities, devCurrentNodes, devExtForces, d_velocities, h_nlistPos, h_nlist, d_nlistPos, d_nlist, d_monomers, d_faces, d_numBonds, d_blist, d_springForces, d_bendingForces, d_volumeForces, d_globalAreaForces, d_localAreaForces, d_wallForces, d_interparticleForces, d_forces, h_LBparams, h_numNeighbors, d_numNeighbors    ,d_coms, d_faceCenters, d_normals, d_areas, d_volumes);
    #else  
    cycle = LBkernel (cycle, h_params, h_foldedPos, d_foldedPos, dataFolder, h_faces, d_pos, devBoundaryMap, devBoundaryVelocities, devCurrentNodes, devExtForces, d_velocities, h_nlistPos, h_nlist, d_nlistPos, d_nlist, d_monomers, d_faces, d_numBonds, d_blist, d_springForces, d_bendingForces, d_volumeForces, d_globalAreaForces, d_localAreaForces, d_wallForces, d_interparticleForces, d_forces, h_LBparams, h_pos, h_monomers, h_springForces, h_bendingForces, h_volumeForces, h_globalAreaForces, h_localAreaForces, h_wallForces, h_interparticleForces, h_numBonds, h_blist, h_forces, h_numNeighbors);
    #endif
  }
  
  #if defined(PARTICLE_GPU) || defined(LB_GPU)
  cudaDeviceReset();
  #endif

  return 0;
}

