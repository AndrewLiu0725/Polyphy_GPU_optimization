#ifndef LB_H
#define LB_H

typedef struct {

  float agrid;
  float tau;
  float rho;
  float gammaShear;
  float gammaBulk;
  float gammaOdd;
  float gammaEven;
  unsigned int dimX;
  unsigned int dimY;
  unsigned int dimZ;
  unsigned int numNodes;
  unsigned int numBoundaries;
  float boundaryVelocity[3];
  unsigned int extForceFlag;
  float extForceDensity[3];

} LBparameters;

typedef struct {

  float rho;
  float v[3];

} LB_rho_v;

typedef struct { 

  float rho;
  float v[3];
  float pi[6];

} LB_rho_v_pi;

extern LBparameters h_LBparams;
// Note: error: identifier "__constant__" is undefined 
//extern __constant__ LBparameters d_LBparams;
extern unsigned int *devBoundaryMap;
extern float *devExtForces;
extern float *devCurrentNodes;
extern float *devBoundaryVelocities;
extern float *hostExtForces;

void InitializeLBparameters (char *file_path, LBparameters *params);

void Print_Parameter_Values ();

void PrintDevParasWrapper ();

void InitializeLB ();

void UpdateLBE ();

void SetBoundaryVelocities ();

void lb_get_values_GPU (LB_rho_v_pi *host_values);

void PrintFluidVelocitiesVTK (char *filePath);

void PrintBoundaryMapVTK (LBparameters lbParams);

void PrintFluidDensitiesVTK (LBparameters h_lbParams);

void Initialize_LB_Parameters_dev (LBparameters *h_lbParams);

//__global__ void Print_dev_LB_Params();

//__device__ void index_to_xyz (unsigned int index, unsigned int *xyz);
//
//__device__ void calc_m_from_n (unsigned int index, float *n_a, float *mode);
//
//__device__ void update_rho_v (unsigned int index, float *mode, float *ext_forces, LB_rho_v *d_v);
//
//__device__ void relax_modes (unsigned int index, float *ext_forces, LB_rho_v *d_v, float *mode);
//
//__device__ void apply_forces (unsigned int index, float *ext_forces, LB_rho_v *d_v, float *mode);
//
//__device__ void reset_LB_forces (unsigned int index, float *ext_forces);
//
//__device__ void normalize_modes (float* mode);
//
//__device__ void calc_n_from_modes_push (unsigned int index, float *mode, float *n_b);
//
//__device__ void bounce_back_boundaries (unsigned int index, unsigned int *boundary_map, float *boundary_velocities, float *n_curr, float *devBoundary_forces);
//
//__device__ void calc_values_in_LB_units (unsigned int index, unsigned int print_index, unsigned int *boundary_map, float *mode, float *ext_forces, LB_rho_v *d_v, LB_rho_v_pi *d_p_v);
//
//__global__ void InitializeBoundaryMap (unsigned int* boundary_map);
//__global__ void InitializeBodyForces (float *ext_forces);
//__global__ void calc_n_from_rho_j_pi (float *ext_forces, float *n_a, LB_rho_v *d_v);

//__global__ void integrate (float *n_a, float *ext_forces, LB_rho_v *d_v, float *n_b);
//__global__ void apply_boundaries (unsigned int *boundary_map,  float *boundary_velocities, float *n_curr, float *boundary_forces);
//__global__ void get_mesoscopic_values_in_LB_units (float *n_a, unsigned int *boundary_map, float *ext_forces, LB_rho_v *d_v, LB_rho_v_pi *d_p_v);


#endif // LB_H
