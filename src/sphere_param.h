#ifndef SPHERE_PARAM_H
#define SPHERE_PARAM_H

#define DOUBLE double

struct sphere_param {

  int lx;
  int ly;
  int lz;
  int cycle;
  int numStepsPerCycle;
  int write_time;      
  int write_config;    

  int initconfig;      
  int wallFlag;
//  int flow_type;
//  double wall_vel_diff;
//  double centerline_vel;
//  double max_strain_amp;
//  double sweep_freq;

  int springType[2];   
  int interparticle;
  int Ntype[2]; // # of particles per type
  int nlevel[2]; // can be negtive                           
  int N_per_sphere[2];    
  int face_per_sphere[2]; 
  int Nsphere;   
  int num_beads; 
  int nfaces;    

  DOUBLE V0[2];        
  DOUBLE A0[2];   
  DOUBLE V0_temp[2]; 
  DOUBLE A0_temp[2];  
  DOUBLE V0_final[2];
  DOUBLE A0_final[2];

  DOUBLE kT;        
  DOUBLE x0;
  DOUBLE shearModulus[2];
  DOUBLE springConst[2];
  DOUBLE kc[2];
  DOUBLE kv[2];
  DOUBLE kag[2];
  DOUBLE kal[2];
  DOUBLE wallForceDis;  
  DOUBLE wallConstant;

  DOUBLE eps;
  DOUBLE eqLJ;
  DOUBLE cutoffLJ;

  DOUBLE depthMorse;
  DOUBLE widthMorse;
  DOUBLE eqMorse;
  DOUBLE cutoffMorse;  

  DOUBLE cellSize;
  DOUBLE nlistCutoff;
  DOUBLE nlistRenewal;

  int numGrowthSteps;
  DOUBLE fictionalMass;

//  int numsteps;  
//int MD_steps;  
//double dt;    
//int write_fluid;     




//  int verlet;                                    /* verlet update type */
//  int relax_time;                                /* sphere relaxation time */
//  double sigma_k;                                /* Kuhn segment length in units of lattice spacing */
//  double H_fene[2], Q_fene[2];         /* FENE parameters */
//  double k_bend[2];                         /* bending force strength */
//  double Ss;                                     /* Ideal chain bead radius */
//  double monmass;                                /* monomer mass */
//  double fric;                                   /* friction coef. on a point force */
//  double evcutoff;                               /* cutoff for exc. vol. interactions */
//  double nks;                                    /* # of kuhn seg. / spring */
//  double tempscale;                              /* ratio between input and output monomer temperature */
//  double f_ext[3];                            /* extern force acting on the monomers */
};

void InitializeParticleParameters (char *filePath, struct sphere_param *partParams);
void PrintDevVariables (struct sphere_param *devPartParams);
 
// cuda
void InitializeDevPartParams (struct sphere_param *h_ptrPartParams);

#endif
