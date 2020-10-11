extern "C"{
#include "forces.h"
#include "tools.h"
#include "neighborList.h"
#include "stdio.h"
}

extern __constant__ struct sphere_param d_partParams;

__device__ void VectorProduct (double a[3], double b[3], double c[3]) {

  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ void VectorProduct_float (float a[3], float b[3], float c[3]) {

  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}

__global__ void ZeroForces (double *springForces, double *bendingForces, float *volumeForces, float *globalAreaForces, double *localAreaForces, double *wallForces, double *interparticleForces) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  if (index <  d_partParams.num_beads) {
    unsigned int x = index*3;
    unsigned int y = x+1;
    unsigned int z = x+2;
    springForces       [x] = 0.0;
    springForces       [y] = 0.0;
    springForces       [z] = 0.0;

    bendingForces      [x] = 0.0;
    bendingForces      [y] = 0.0;
    bendingForces      [z] = 0.0;

    volumeForces       [x] = 0.0f;
    volumeForces       [y] = 0.0f;
    volumeForces       [z] = 0.0f;

    globalAreaForces   [x] = 0.0f;
    globalAreaForces   [y] = 0.0f;
    globalAreaForces   [z] = 0.0f;

    localAreaForces    [x] = 0.0;
    localAreaForces    [y] = 0.0;
    localAreaForces    [z] = 0.0;

    wallForces         [x] = 0.0;
    wallForces         [y] = 0.0;
    wallForces         [z] = 0.0;

    interparticleForces[x] = 0.0;
    interparticleForces[y] = 0.0;
    interparticleForces[z] = 0.0;
  }  
}

__global__ void SumForces (double *springForces, double *bendingForces, float *volumeForces, float *globalAreaForces, double *localAreaForces, double *wallForces, double *interparticleForces, double *forces) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < d_partParams.num_beads) {
    unsigned int x = index*3; 
    unsigned int y = x+1;
    unsigned int z = x+2;   
    forces[x] = springForces[x] + bendingForces[x] + volumeForces[x] + globalAreaForces[x] + localAreaForces[x] + wallForces[x] + interparticleForces[x];
    forces[y] = springForces[y] + bendingForces[y] + volumeForces[y] + globalAreaForces[y] + localAreaForces[y] + wallForces[y] + interparticleForces[y];
    forces[z] = springForces[z] + bendingForces[z] + volumeForces[z] + globalAreaForces[z] + localAreaForces[z] + wallForces[z] + interparticleForces[z];
  }
}

__global__ void SpringForce (struct monomer *monomers, int *numBonds, int *blist, double *pos, double *springForces) {

  // The thread grid is large enough to cover the entire data array, so no need to use grid-stride loop
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < d_partParams.num_beads)  
	{
    unsigned short springType;
    double springConst;

    // Ntype attr is # of particles per type
    // Decide the spring type of that monomer
    if (index < (d_partParams.Ntype[0]*d_partParams.N_per_sphere[0])) {  
      springType = d_partParams.springType[0];
      springConst = -2.0 * d_partParams.springConst[0] * d_partParams.kT;
      // varibles for the WLC-POW spring is set in function 'SetSpringConst'
    }
    else {
      springType = d_partParams.springType[1]; 
      springConst = -2.0 * d_partParams.springConst[1] * d_partParams.kT;
      // variables for the WLC-POW spring is set in function 'SetSpringConst'.
    }
		unsigned int n1 = index;
    unsigned int n1offset = n1*3;
                           
    for (unsigned int k=1; k <= numBonds[n1]; k++) 
    {
      //int n2 = blist[n1][k][0];
      unsigned int n2 = blist[n1*6*3+(k-1)*3+0];
      unsigned int n2offset = n2*3;
      double length_eq = monomers[n1].initLength[k];
      double q12[3];
      q12[0] = pos[n1offset+0] - pos[n2offset+0];
      q12[1] = pos[n1offset+1] - pos[n2offset+1];
      q12[2] = pos[n1offset+2] - pos[n2offset+2];
      double q12mag;  
      q12mag = q12[0]*q12[0] + q12[1]*q12[1] + q12[2]*q12[2];
      q12mag = sqrt(q12mag); 
      q12[0] /= q12mag;
      q12[1] /= q12mag;
      q12[2] /= q12mag;

      double mag;  
      double force[3];
      if (springType == 1) {
        double x = q12mag / monomers[n1].lmax[k];
        double x2 = x*x;
        double x3 = x*x*x;       
        mag = -0.25*monomers[n1].kT_inversed_persis[k] * (4*x3-9*x2+6*x) / (1+x2-2*x) + monomers[n1].kp[k] / (q12mag*q12mag);
      }
      else if (springType == 2) {
        // Note: mag = (-partial U / partial r) 
        //mag = -2.0 * spring_const * (q12mag-length_eq);
        mag = springConst * (q12mag-length_eq);
      }
      force[0] = mag * q12[0];
      force[1] = mag * q12[1];
      force[2] = mag * q12[2];
      springForces[n1offset+0] += force[0];
      springForces[n1offset+1] += force[1];
      springForces[n1offset+2] += force[2];
      springForces[n2offset+0] -= force[0];
      springForces[n2offset+1] -= force[1];
      springForces[n2offset+2] -= force[2];
    }
    
  }
}

__global__ void BendingForce (struct monomer *monomers, int *numBonds, int *blist, double *pos, double *bendingForces, double *faceNormals, int *face_pair_list) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < d_partParams.num_face_pair)
	{
    double kb;
    int F1 = face_pair_list[FPSIZE*index]; // face 1 id
    int F2 = face_pair_list[FPSIZE*index+1]; // face 2 id

    if (F1 < (d_partParams.Ntype[0]*d_partParams.face_per_sphere[0])) {  
      kb = 2.0/sqrt(3.0)*d_partParams.kc[0]*d_partParams.kT;
    }
    else {
      kb = 2.0/sqrt(3.0)*d_partParams.kc[1]*d_partParams.kT; 
    }

    unsigned int n1 = face_pair_list[FPSIZE*index+3];
    unsigned int n1offset = n1*3; 

    double theta0 = monomers[n1].initAngle[face_pair_list[FPSIZE*index+2]];
    unsigned int n2 = face_pair_list[FPSIZE*index+4];
    unsigned int n3 = face_pair_list[FPSIZE*index+5];
    unsigned int n4 = face_pair_list[FPSIZE*index+6];
    unsigned int n2offset = n2*3;
    unsigned int n3offset = n3*3;
    unsigned int n4offset = n4*3;

    double v1[3], v2[3], v3[3], v4[3];
    v1[0]=pos[n1offset];
    v1[1]=pos[n1offset+1];
    v1[2]=pos[n1offset+2];

    v2[0]=pos[n2offset];
    v2[1]=pos[n2offset+1];
    v2[2]=pos[n2offset+2];

    v3[0]=pos[n3offset];
    v3[1]=pos[n3offset+1];
    v3[2]=pos[n3offset+2];

    v4[0]=pos[n4offset];
    v4[1]=pos[n4offset+1];
    v4[2]=pos[n4offset+2];

    double a31[3], a21[3], a41[3];
    a31[0] = v3[0] - v1[0];
    a31[1] = v3[1] - v1[1];
    a31[2] = v3[2] - v1[2];
    a21[0] = v2[0] - v1[0];
    a21[1] = v2[1] - v1[1];
    a21[2] = v2[2] - v1[2];
    a41[0] = v4[0] - v1[0];
    a41[1] = v4[1] - v1[1];
    a41[2] = v4[2] - v1[2];

    double normal1[3], normal2[3];
    normal1[0] = faceNormals[F1*3];
    normal1[1] = faceNormals[F1*3+1];
    normal1[2] = faceNormals[F1*3+2];

    normal2[0] = faceNormals[F2*3];
    normal2[1] = faceNormals[F2*3+1];
    normal2[2] = faceNormals[F2*3+2];

    double normal1_mag, normal2_mag;
    normal1_mag = sqrt(normal1[0]*normal1[0] + normal1[1]*normal1[1] + normal1[2]*normal1[2]);
    normal2_mag = sqrt(normal2[0]*normal2[0] + normal2[1]*normal2[1] + normal2[2]*normal2[2]);

    normal1[0] = normal1[0] / normal1_mag;
    normal1[1] = normal1[1] / normal1_mag;
    normal1[2] = normal1[2] / normal1_mag;
    normal2[0] = normal2[0] / normal2_mag; 
    normal2[1] = normal2[1] / normal2_mag; 
    normal2[2] = normal2[2] / normal2_mag; 

    double orient[3];
    VectorProduct(normal1, normal2, orient);
    double value = sqrt((orient[0]*orient[0] + orient[1]*orient[1] + orient[2]*orient[2]));
    double scaler = normal1[0]*normal2[0] + normal1[1]*normal2[1] + normal1[2]*normal2[2];
    double theta = atan2(value, scaler);
    // Note: ###################################################################
    // When RBC particles are applied this program crashes if theta is computed 
    // by acos !
    // #########################################################################
    /*if(scaler > 1.) scaler=1;
    double theta = acos(scaler);*/

    double cosTheta = cos(theta);
    double sign = orient[0]*a21[0] + orient[1]*a21[1] + orient[2]*a21[2];
    //double temp = sin(theta_nr);
    if (sign > 0) theta = -1.0*theta;
    double delta_theta = theta - theta0;
    double sinTheta = sin(theta);
    double factor;

    factor =  kb * sin(delta_theta) / sinTheta;  

    double n12[3], n21[3];
    n12[0] = normal1[0] - cosTheta*normal2[0];
    n12[1] = normal1[1] - cosTheta*normal2[1];
    n12[2] = normal1[2] - cosTheta*normal2[2];
    n21[0] = normal2[0] - cosTheta*normal1[0];
    n21[1] = normal2[1] - cosTheta*normal1[1];
    n21[2] = normal2[2] - cosTheta*normal1[2];

    double a12[3], a14[3], a23[3], a42[3];
    a12[0] = -a21[0];
    a12[1] = -a21[1];
    a12[2] = -a21[2];
    a14[0] = -a41[0];
    a14[1] = -a41[1];
    a14[2] = -a41[2];
    a23[0] = v2[0] - v3[0];
    a23[1] = v2[1] - v3[1];
    a23[2] = v2[2] - v3[2];
    a42[0] = v4[0] - v2[0];
    a42[1] = v4[1] - v2[1];
    a42[2] = v4[2] - v2[2];

    double term3[3], term21[3], term22[3], term11[3], term12[3], term4[3];
    VectorProduct(n21, a12, term3);
    VectorProduct(n21, a31, term21);
    VectorProduct(n12, a14, term22);
    VectorProduct(n21, a23, term11);
    VectorProduct(n12, a42, term12);
    VectorProduct(n12, a21, term4);

    double pre1, pre2;
    pre1 = factor / normal1_mag;
    pre2 = factor / normal2_mag;

    double f1[3], f2[3], f3[3], f4[3];
    f3[0] = pre1 * term3[0];
    f3[1] = pre1 * term3[1];
    f3[2] = pre1 * term3[2];

    f2[0] = pre1 * term21[0] + pre2 * term22[0];
    f2[1] = pre1 * term21[1] + pre2 * term22[1];
    f2[2] = pre1 * term21[2] + pre2 * term22[2];

    f1[0] = pre1 * term11[0] + pre2 * term12[0];
    f1[1] = pre1 * term11[1] + pre2 * term12[1];
    f1[2] = pre1 * term11[2] + pre2 * term12[2];

    f4[0] = pre2 * term4[0];    
    f4[1] = pre2 * term4[1];    
    f4[2] = pre2 * term4[2];    

    bendingForces[n1offset+0] += f1[0];          
    bendingForces[n1offset+1] += f1[1];          
    bendingForces[n1offset+2] += f1[2];
    
    bendingForces[n2offset+0] += f2[0];          
    bendingForces[n2offset+1] += f2[1];          
    bendingForces[n2offset+2] += f2[2];          

    bendingForces[n3offset+0] += f3[0];          
    bendingForces[n3offset+1] += f3[1];          
    bendingForces[n3offset+2] += f3[2];          

    bendingForces[n4offset+0] += f4[0];          
    bendingForces[n4offset+1] += f4[1];          
    bendingForces[n4offset+2] += f4[2];        
		
  }
}

__global__ void ZeroCOM (float *coms, float *volumes, float *areas) {

  unsigned int index = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if (index < d_partParams.Nsphere) {
    coms[index*3]=0.0f;
    coms[index*3+1]=0.0f;
    coms[index*3+2]=0.0f;
    volumes[index]=0.0f;
    areas[index]=0.0f; 
  }
}

__global__ void COM (double *pos, float *com) {

  unsigned int index = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if (index < d_partParams.num_beads)
  {
    unsigned int partIdx;
    //int partIdx;
    unsigned int numNodesPerPart;
    if (index < d_partParams.Ntype[0] * d_partParams.N_per_sphere[0]) {
      partIdx = index / d_partParams.N_per_sphere[0];
      //partIdx = (int)index / d_partParams.N_per_sphere[0];
      numNodesPerPart = d_partParams.N_per_sphere[0];    
    } else {
      partIdx =  d_partParams.Ntype[0] + (index - d_partParams.Ntype[0]*d_partParams.N_per_sphere[0]) / d_partParams.N_per_sphere[1];
      //partIdx =  d_partParams.Ntype[0] + (int)(index - d_partParams.Ntype[0]*d_partParams.N_per_sphere[0]) / d_partParams.N_per_sphere[1];
      numNodesPerPart = d_partParams.N_per_sphere[1];   
    }


    //printf("partIdx=%d  nodePos=(%f %f %f)\n", partIdx, (float)(pos[index*3]/numNodesPerPart), (float)(pos[index*3+1]/numNodesPerPart), (float)(pos[index*3+2]/numNodesPerPart));
   
    // atomic operation is necessary !!!
    atomicAdd(&com[partIdx*3],   ((float)pos[index*3]   / (float)numNodesPerPart));
    atomicAdd(&com[partIdx*3+1], ((float)pos[index*3+1] / (float)numNodesPerPart));
    atomicAdd(&com[partIdx*3+2], ((float)pos[index*3+2] / (float)numNodesPerPart));
  }
}

__global__ void PrintCOM (float *coms, float *volumes, float *areas) {

  unsigned int index = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if (index < d_partParams.Nsphere) {
    printf("particle %d  vol=%f  area=%f  com = (%f %f %f)\n", index, volumes[index], areas[index], coms[index*3], coms[index*3+1], coms[index*3+2]);
  }
}

__global__ void FaceNormal (struct face *faces, double *pos, double *faceNormal){

  unsigned int index = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

  if (index < (d_partParams.Ntype[0]*d_partParams.face_per_sphere[0] + d_partParams.Ntype[1]*d_partParams.face_per_sphere[1])){
    unsigned int n1 = faces[index].v[0];    
    unsigned int n2 = faces[index].v[2];
    unsigned int n3 = faces[index].v[1];

    double v1[3], v2[3], v3[3];
    double normal[3];
    double v31[3], v21[3];

    v1[0]=pos[n1*3];
    v1[1]=pos[n1*3+1];
    v1[2]=pos[n1*3+2];

    v2[0]=pos[n2*3];
    v2[1]=pos[n2*3+1];
    v2[2]=pos[n2*3+2];

    v3[0]=pos[n3*3];
    v3[1]=pos[n3*3+1];
    v3[2]=pos[n3*3+2];

    v21[0] = v2[0] - v1[0];
    v21[1] = v2[1] - v1[1];
    v21[2] = v2[2] - v1[2];
  
    v31[0] = v3[0] - v1[0];
    v31[1] = v3[1] - v1[1];
    v31[2] = v3[2] - v1[2];

    VectorProduct(v21, v31, normal);

    // face noremal  
    faceNormal[index*3]   = normal[0];
    faceNormal[index*3+1] = normal[1];
    faceNormal[index*3+2] = normal[2];
  }
}

__global__ void VolumeAreas (struct face *faces, double *pos, float *com, double *faceCenter, double *faceNormal, float *area, float *volume) {

  unsigned int index = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  //unsigned int numFaces = d_partParams.Ntype[0]*d_partParams.face_per_sphere[0] + d_partParams.Ntype[1]*d_partParams.face_per_sphere[1];
  if (index < d_partParams.Ntype[0]*d_partParams.face_per_sphere[0] + d_partParams.Ntype[1]*d_partParams.face_per_sphere[1])
  {
    unsigned int partIdx;
    if (index < d_partParams.Ntype[0]*d_partParams.face_per_sphere[0]) {
      partIdx = index / d_partParams.face_per_sphere[0];
    } else {
      partIdx = d_partParams.Ntype[0] + (index - d_partParams.Ntype[0]*d_partParams.face_per_sphere[0]) / d_partParams.face_per_sphere[1];
    }
    unsigned int n1 = faces[index].v[0];
    unsigned int n2 = faces[index].v[2];
    unsigned int n3 = faces[index].v[1];

    float v1[3], v2[3], v3[3];
    float dr[3];   
    float normal[3];
    float center[3];

    v1[0]=pos[n1*3];
    v1[1]=pos[n1*3+1];
    v1[2]=pos[n1*3+2];

    v2[0]=pos[n2*3];
    v2[1]=pos[n2*3+1];
    v2[2]=pos[n2*3+2];

    v3[0]=pos[n3*3];
    v3[1]=pos[n3*3+1];
    v3[2]=pos[n3*3+2];

    center[0] = com[partIdx*3]; 
    center[1] = com[partIdx*3+1];
    center[2] = com[partIdx*3+2];

    dr[0] = v1[0] - center[0];
    dr[1] = v1[1] - center[1];
    dr[2] = v1[2] - center[2];

    // face noremal  
    normal[0] = faceNormal[index*3];
    normal[1] = faceNormal[index*3+1];
    normal[2] = faceNormal[index*3+2];

    // face center
    faceCenter[index*3]   = (v1[0] + v2[0] + v3[0])/3.0 - center[0];
    faceCenter[index*3+1] = (v1[1] + v2[1] + v3[1])/3.0 - center[1];
    faceCenter[index*3+2] = (v1[2] + v2[2] + v3[2])/3.0 - center[2];

    // local area
    faces[index].area = 0.5 * sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    // global area
    atomicAdd(&area[partIdx], (float)(faces[index].area));
    // particle volume
    atomicAdd(&volume[partIdx], (float)((dr[0]*normal[0] + dr[1]*normal[1] + dr[2]*normal[2])/6.));
  }
}

__global__ void VolumeAreaForces (float *volumes, struct face *faces, float *areas, double *pos, double *faceCenters, double *normals, float *volumeForces, float *globalAreaForces, double *localAreaForces) {

  unsigned int index = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  //__shared__ double s_normals[64];
  //__shared__ double s_faceCenters[64];
  //  __shared__ double s_volumes[64];
  //  s_volumes[threadIdx.x] = volumes[index];
  //  __syncthreads();

  if (index < d_partParams.Ntype[0]*d_partParams.face_per_sphere[0]+d_partParams.Ntype[1]*d_partParams.face_per_sphere[1])
  {
    float kv;
    float kag;
    float kal;
    float V0;
    float A0;
    unsigned int partIdx;       
      
    if (index < d_partParams.Ntype[0]*d_partParams.face_per_sphere[0]) {
      kv  = d_partParams.kv[0];
      kag = d_partParams.kag[0];
      kal = d_partParams.kal[0];        
      V0  = d_partParams.V0[0]; 
      A0  = d_partParams.A0[0];
      partIdx = index / d_partParams.face_per_sphere[0];  
    } else {
      kv  = d_partParams.kv[1];
      kag = d_partParams.kag[1];
      kal = d_partParams.kal[1];
      V0  = d_partParams.V0[1];
      A0  = d_partParams.A0[1];
      partIdx = d_partParams.Ntype[0] + (index - d_partParams.Ntype[0]*d_partParams.face_per_sphere[0]) / d_partParams.Ntype[1];
    }
    kv = kv * -1.0 * ((volumes[partIdx]/V0)-1.0) / 6.0;
    kag = kag*(-0.25)/faces[index].area*((areas[partIdx] / A0)             - 1.0);
    kal = kal*(-0.25)/faces[index].area*((faces[index].area / faces[index].area_0) - 1.0);

    unsigned int n1 = faces[index].v[0]; 
    unsigned int n2 = faces[index].v[1]; 
    unsigned int n3 = faces[index].v[2]; 

    float v1[3];
    float v2[3];
    float v3[3];

    float v23[3];
    float v31[3];
    float v12[3];

    float faceCenter[3];
    float normal[3];
    float normal_3[3];

    float temp1[3];
    float temp2[3];
    float temp3[3];

    v1[0] = pos[n1*3];
    v1[1] = pos[n1*3+1];
    v1[2] = pos[n1*3+2];
    v2[0] = pos[n2*3];
    v2[1] = pos[n2*3+1];
    v2[2] = pos[n2*3+2];
    v3[0] = pos[n3*3];
    v3[1] = pos[n3*3+1];
    v3[2] = pos[n3*3+2];

    v23[0] = v2[0] - v3[0];
    v23[1] = v2[1] - v3[1];
    v23[2] = v2[2] - v3[2];

    v31[0] = v3[0] - v1[0];
    v31[1] = v3[1] - v1[1];
    v31[2] = v3[2] - v1[2];

    v12[0] = v1[0] - v2[0];
    v12[1] = v1[1] - v2[1];
    v12[2] = v1[2] - v2[2];

    faceCenter[0] = faceCenters[index*3]; 
    faceCenter[1] = faceCenters[index*3+1];
    faceCenter[2] = faceCenters[index*3+2];

    //VectorProduct(faceCenter[faceOffset+j], v23, temp1);
    //VecotrProduct(faceCenter[faceOffset+j], v31, temp2);
    //VectorProduct(faceCenter[faceOffset+j], v12, temp3);
    temp1[0] = faceCenter[1]*v23[2] - faceCenter[2]*v23[1];
    temp1[1] = faceCenter[2]*v23[0] - faceCenter[0]*v23[2];
    temp1[2] = faceCenter[0]*v23[1] - faceCenter[1]*v23[0];

    temp2[0] = faceCenter[1]*v31[2] - faceCenter[2]*v31[1];
    temp2[1] = faceCenter[2]*v31[0] - faceCenter[0]*v31[2];
    temp2[2] = faceCenter[0]*v31[1] - faceCenter[1]*v31[0];

    temp3[0] = faceCenter[1]*v12[2] - faceCenter[2]*v12[1];
    temp3[1] = faceCenter[2]*v12[0] - faceCenter[0]*v12[2];
    temp3[2] = faceCenter[0]*v12[1] - faceCenter[1]*v12[0];

    normal[0] = normals[index*3];
    normal[1] = normals[index*3+1];
    normal[2] = normals[index*3+2];

    normal_3[0] = normal[0] / 3.0;
    normal_3[1] = normal[1] / 3.0;
    normal_3[2] = normal[2] / 3.0;

    // volume constraint  
    atomicAdd(&volumeForces[n1*3],   (float)(kv * (normal_3[0] + temp1[0])));
    atomicAdd(&volumeForces[n1*3+1], (float)(kv * (normal_3[1] + temp1[1])));
    atomicAdd(&volumeForces[n1*3+2], (float)(kv * (normal_3[2] + temp1[2])));

    atomicAdd(&volumeForces[n2*3],   (float)(kv * (normal_3[0] + temp2[0])));
    atomicAdd(&volumeForces[n2*3+1], (float)(kv * (normal_3[1] + temp2[1])));
    atomicAdd(&volumeForces[n2*3+2], (float)(kv * (normal_3[2] + temp2[2])));

    atomicAdd(&volumeForces[n3*3],   (float)(kv * (normal_3[0] + temp3[0])));
    atomicAdd(&volumeForces[n3*3+1], (float)(kv * (normal_3[1] + temp3[1])));
    atomicAdd(&volumeForces[n3*3+2], (float)(kv * (normal_3[2] + temp3[2])));

    // global and local area constraints
    temp1[0] = normal[1]*v23[2] - normal[2]*v23[1];
    temp1[1] = normal[2]*v23[0] - normal[0]*v23[2];
    temp1[2] = normal[0]*v23[1] - normal[1]*v23[0];

    temp2[0] = normal[1]*v31[2] - normal[2]*v31[1];
    temp2[1] = normal[2]*v31[0] - normal[0]*v31[2];
    temp2[2] = normal[0]*v31[1] - normal[1]*v31[0];

    temp3[0] = normal[1]*v12[2] - normal[2]*v12[1];
    temp3[1] = normal[2]*v12[0] - normal[0]*v12[2];
    temp3[2] = normal[0]*v12[1] - normal[1]*v12[0];

    atomicAdd(&globalAreaForces[n1*3],   (float)((kag+kal)*temp1[0]));
    atomicAdd(&globalAreaForces[n1*3+1], (float)((kag+kal)*temp1[1]));
    atomicAdd(&globalAreaForces[n1*3+2], (float)((kag+kal)*temp1[2]));
    atomicAdd(&globalAreaForces[n2*3],   (float)((kag+kal)*temp2[0]));
    atomicAdd(&globalAreaForces[n2*3+1], (float)((kag+kal)*temp2[1]));
    atomicAdd(&globalAreaForces[n2*3+2], (float)((kag+kal)*temp2[2]));
    atomicAdd(&globalAreaForces[n3*3],   (float)((kag+kal)*temp3[0]));
    atomicAdd(&globalAreaForces[n3*3+1], (float)((kag+kal)*temp3[1]));
    atomicAdd(&globalAreaForces[n3*3+2], (float)((kag+kal)*temp3[2]));
    //    localAreaForces[n1]   += kal*temp1[0];
    //    localAreaForces[n1+1] += kal*temp1[1];
    //    localAreaForces[n1+2] += kal*temp1[2];
    //    localAreaForces[n2]   += kal*temp2[0];
    //    localAreaForces[n2+1] += kal*temp2[1];
    //    localAreaForces[n2+2] += kal*temp2[2];
    //    localAreaForces[n3]   += kal*temp3[0];
    //    localAreaForces[n3+1] += kal*temp3[1];
    //    localAreaForces[n3+2] += kal*temp3[2];
  }
}

// Is this function not used anymore?
__global__ void VolumeAreaConstraints (struct face *faces, double *pos, double *volumeForces, double *globalAreaForces, double *localAreaForces) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < d_partParams.Nsphere)
	{
    double kv; 
    double kag;
    double kal;
    double V0;
    double A0;  
    unsigned short numNodesPerParticle; 
    unsigned short numFacesPerParticle; 
    unsigned int nodeOffset;
    unsigned int faceOffset;

    if (index < d_partParams.Ntype[0]) {  
      kv  = d_partParams.kv[0];
      kag = d_partParams.kag[0];
      kal = d_partParams.kal[0]; 
      V0  = d_partParams.V0[0];
      A0  = d_partParams.A0[0];   
      numNodesPerParticle = d_partParams.N_per_sphere[0];
      numFacesPerParticle = d_partParams.face_per_sphere[0];   
      nodeOffset = index*d_partParams.N_per_sphere[0];
      faceOffset = index*d_partParams.face_per_sphere[0];
    }
    else {
      kv  = d_partParams.kv[1];
      kag = d_partParams.kag[1];
      kal = d_partParams.kal[1]; 
      V0  = d_partParams.V0[1];
      A0  = d_partParams.A0[1];   
      numNodesPerParticle = d_partParams.N_per_sphere[1];
      numFacesPerParticle = d_partParams.face_per_sphere[1];   
      nodeOffset = d_partParams.Ntype[0] * d_partParams.N_per_sphere[0] +    (index-d_partParams.Ntype[0]) * d_partParams.N_per_sphere[1];
      faceOffset = d_partParams.Ntype[0] * d_partParams.face_per_sphere[0] + (index-d_partParams.Ntype[0]) * d_partParams.face_per_sphere[1];
    }

    // Use COM instead?
    double com[3] = {0.0};
    double area   = 0.0;
    double volume = 0.0;
		for (unsigned short j=0; j < numNodesPerParticle; j++) 
		{
			unsigned int n = j + nodeOffset; 
      //unsigned int offset = n*3;  
			com[0] += pos[n*3+0];
			com[1] += pos[n*3+1];
			com[2] += pos[n*3+2];
		}
		com[0] = com[0] / numNodesPerParticle;
		com[1] = com[1] / numNodesPerParticle;
		com[2] = com[2] / numNodesPerParticle;

		for (unsigned short j=0; j < numFacesPerParticle; j++)
		{
			unsigned int nface = j + faceOffset;
			unsigned int n1 = faces[nface].v[0];
			unsigned int n2 = faces[nface].v[1];
			unsigned int n3 = faces[nface].v[2];
      double v1[3];
      double v2[3];
      double v3[3];  
			double dr[3], q12[3], q13[3], normal[3];  
      v1[0] = pos[n1*3];
      v1[1] = pos[n1*3+1];
      v1[2] = pos[n1*3+2];
      v2[0] = pos[n2*3];
      v2[1] = pos[n2*3+1];
      v2[2] = pos[n2*3+2];
      v3[0] = pos[n3*3];
      v3[1] = pos[n3*3+1];
      v3[2] = pos[n3*3+2];

			dr[0] = v1[0] - com[0];  // origin is at the position of COM
			dr[1] = v1[1] - com[1];
			dr[2] = v1[2] - com[2];

			q12[0] = v2[0] - v1[0];
			q12[1] = v2[1] - v1[1];
			q12[2] = v2[2] - v1[2];
			q13[0] = v3[0] - v1[0];
			q13[1] = v3[1] - v1[1];
			q13[2] = v3[2] - v1[2];
			VectorProduct(q13, q12, normal);  // "normal" points outward viewing from the centroid of the particle
			faces[nface].area = 0.5 * sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
			area += faces[nface].area;
			// Note: ###############################################################################################
			// I think it's reasonable to not take the absolute value! Thank about how to calculate a torus' volume. 
			// #####################################################################################################
			//particle[i].volume += fabs(dr[0]*normal[0]+dr[1]*normal[1]+dr[2]*normal[2]);
			volume += (dr[0]*normal[0]+dr[1]*normal[1]+dr[2]*normal[2]);
		}
		volume /= 6.;

		double vol   = -kv  * ((volume / V0) - 1.) / 6.;

		for(unsigned short j=0; j < numFacesPerParticle; j++) 
		{
			unsigned int nface = j + faceOffset;
			double areaG = -0.25 * kag / faces[nface].area * ((area / A0) - 1.);
			//double areaL = -0.25 * kal * (faces[nface].area - faces[nface].area_0) / (faces[nface].area_0 * faces[nface].area);
      double areaL = -0.25 * kal / faces[nface].area * ((faces[nface].area / faces[nface].area_0) - 1.); 
			unsigned int n1 = faces[nface].v[0];
			unsigned int n2 = faces[nface].v[1];
			unsigned int n3 = faces[nface].v[2];
      double v1[3];
      double v2[3];
      double v3[3];
			double v23[3], v31[3], v12[3];
      
      v1[0] = pos[n1*3];
      v1[1] = pos[n1*3+1];
      v1[2] = pos[n1*3+2];

      v2[0] = pos[n2*3];
      v2[1] = pos[n2*3+1];
      v2[2] = pos[n2*3+2];

      v3[0] = pos[n3*3];
      v3[1] = pos[n3*3+1];
      v3[2] = pos[n3*3+2];
 
			v23[0] = v2[0] - v3[0];
			v23[1] = v2[1] - v3[1];
			v23[2] = v2[2] - v3[2];

			v31[0] = v3[0] - v1[0];
			v31[1] = v3[1] - v1[1];
			v31[2] = v3[2] - v1[2];

			v12[0] = v1[0] - v2[0];
			v12[1] = v1[1] - v2[1];
			v12[2] = v1[2] - v2[2];

			double normalVector[3]; 
			VectorProduct(v31, v12, normalVector);
      normalVector[0] *= -1.0;
      normalVector[1] *= -1.0;
      normalVector[2] *= -1.0;

			// Note: ############################################################################
			// The origin is at the center of mass of the particle. Should make sure it's right!
			// ##################################################################################
			double faceCenter[3];                          
			//faceCenter[0] = (monomers[n1].pos_pbc[0]+monomers[n2].pos_pbc[0]+monomers[n3].pos_pbc[0]) / 3.;
			//faceCenter[1] = (monomers[n1].pos_pbc[1]+monomers[n2].pos_pbc[1]+monomers[n3].pos_pbc[1]) / 3.;
			//faceCenter[2] = (monomers[n1].pos_pbc[2]+monomers[n2].pos_pbc[2]+monomers[n3].pos_pbc[2]) / 3.;
			faceCenter[0] = (v1[0] + v2[0] + v3[0]) / 3. - com[0];
			faceCenter[1] = (v1[1] + v2[1] + v3[1]) / 3. - com[1];
			faceCenter[2] = (v1[2] + v2[2] + v3[2]) / 3. - com[2];

			double temp1[3], temp2[3], temp3[3]; 
			//double force1[3], force2[3], force3[3];

			// Volume constraint
			double normal_3[3];
			normal_3[0] = normalVector[0] / 3.;
			normal_3[1] = normalVector[1] / 3.;
			normal_3[2] = normalVector[2] / 3.;
			VectorProduct(faceCenter, v23, temp1);
			VectorProduct(faceCenter, v31, temp2);
			VectorProduct(faceCenter, v12, temp3); 

			volumeForces[n1*3]   += vol * (normal_3[0] + temp1[0]);
			volumeForces[n1*3+1] += vol * (normal_3[1] + temp1[1]);
			volumeForces[n1*3+2] += vol * (normal_3[2] + temp1[2]);

			volumeForces[n2*3]   += vol * (normal_3[0] + temp2[0]);
			volumeForces[n2*3+1] += vol * (normal_3[1] + temp2[1]);
			volumeForces[n2*3+2] += vol * (normal_3[2] + temp2[2]);

			volumeForces[n3*3]   += vol * (normal_3[0] + temp3[0]);
			volumeForces[n3*3+1] += vol * (normal_3[1] + temp3[1]);       
			volumeForces[n3*3+2] += vol * (normal_3[2] + temp3[2]);     

			// Global area constraint
			VectorProduct(normalVector, v23, temp1);
			VectorProduct(normalVector, v31, temp2);
			VectorProduct(normalVector, v12, temp3);

			globalAreaForces[n1*3]   += (areaG+areaL) * temp1[0];
			globalAreaForces[n1*3+1] += (areaG+areaL) * temp1[1];
			globalAreaForces[n1*3+2] += (areaG+areaL) * temp1[2];

			globalAreaForces[n2*3]   += (areaG+areaL) * temp2[0];
			globalAreaForces[n2*3+1] += (areaG+areaL) * temp2[1];
			globalAreaForces[n2*3+2] += (areaG+areaL) * temp2[2];

			globalAreaForces[n3*3]   += (areaG+areaL) * temp3[0];
			globalAreaForces[n3*3+1] += (areaG+areaL) * temp3[1];
			globalAreaForces[n3*3+2] += (areaG+areaL) * temp3[2];

			// Local area constraint
      //			localAreaForces[n1*3]   += areaL * temp1[0];
      //			localAreaForces[n1*3+1] += areaL * temp1[1];
      //			localAreaForces[n1*3+2] += areaL * temp1[2];

      //			localAreaForces[n2*3]   += areaL * temp2[0];
      //			localAreaForces[n2*3+1] += areaL * temp2[1];
      //			localAreaForces[n2*3+2] += areaL * temp2[2];

      //			localAreaForces[n3*3]   += areaL * temp3[0];
      //			localAreaForces[n3*3+1] += areaL * temp3[1];
      //			localAreaForces[n3*3+2] += areaL * temp3[2];
		}
	}
}

__global__ void WallForce_gpu (double *pos, double *wallForces) {

  //	double boxCenter[3] = {0.5*d_partParams.lx, 0.5*d_partParams.ly, 0.5*d_partParams.lz};
  //	double end_fluid_node_y = 0.5*(d_partParams.ly-1.); // measured from the box center: (max -0.5) - 0.5*max
  //	double end_fluid_node_z = 0.5*(d_partParams.lz-1.); // measured from the box center
  //  double topWally = end_fluid_node_y - d_partParams.wallForceDis;
  //  double topWallz = end_fluid_node_z - d_partParams.wallForceDis;
  //  double topWally = 0.5*(d_partParams.ly-1.) - d_partParams.wallForceDis;
  //  double topWallz = 0.5*(d_partParams.lz-1.) - d_partParams.wallForceDis;
  //  double bottomWally = -topWally;
  //  double bottomWallz = -topWallz;
  double topWally = (d_partParams.ly-1) - d_partParams.wallForceDis - (0.5*d_partParams.ly);
  double topWallz = (d_partParams.lz-1) - d_partParams.wallForceDis - (0.5*d_partParams.lz);
  double bottomWally = - topWally;
  double bottomWallz = - topWallz;

  double wallConst = d_partParams.wallConstant * d_partParams.kT;

  // cubic potential
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < d_partParams.num_beads)
  {
    double posTemp[3], force[3], delta[3];
    unsigned int offset = index*3;    
    posTemp[1] = pos[offset+1] - (0.5*d_partParams.ly);
    posTemp[2] = pos[offset+2] - (0.5*d_partParams.lz);

    if (d_partParams.wallFlag==1) {
      if (posTemp[1] < bottomWally) {
        delta[1] = bottomWally - posTemp[1];
        force[1] = wallConst*delta[1]*delta[1];
      }
      else if (posTemp[1] > topWally) {
        delta[1] = posTemp[1] - topWally;
        force[1] = - wallConst*delta[1]*delta[1];
      }
      else {
        force[1] = 0.;
      }
      wallForces[offset+1] = force[1]; 
    }
    else if (d_partParams.wallFlag == 2) {
      if (posTemp[1] < bottomWally) {
        delta[1] = bottomWally - posTemp[1];
        force[1] = wallConst*delta[1]*delta[1];
      }
      else if (posTemp[1] > topWally) {
        delta[1] = posTemp[1] - topWally;
        force[1] = - wallConst*delta[1]*delta[1];
      }
      else {
        force[1] = 0.;
      }
      if (posTemp[2] < bottomWallz) {
        delta[2] = bottomWallz - posTemp[2];
        force[2] = wallConst*delta[2]*delta[2];
      }
      else if (posTemp[2] > topWallz) {
        delta[2] = posTemp[2] - topWallz;
        force[2] = - wallConst*delta[2]*delta[2];
      }
      else {
        force[2] = 0.;
      }
      wallForces[offset+1] = force[1];
      wallForces[offset+2] = force[2]; 
    }
  }
}

__global__ void InterparticleForce (struct monomer *monomers, int *numNeighbors, int *nlist, double *pos, double *interparticleForces) {

  // TODO
  // 1) Use another array to store each node's sphere_id or obtain the sphere_id by simple computation if it is possible.
  // 2) Access the neighbor list more efficiently 
  // 3) Redesign the algorithm for the neighbor list construction to avoid using a fixed array size for all nodes' lists

  // will be deleted !
  //Float sigma = 2.0*radius;
  //double cutoff = 1.122 * sigma;
  //Float eps = d_partParams.kT/sigma; // there should be a sigma in denominator?
  //double sigma = 1 / 1.122;
  //double cutoff = 1.;

	// L-J potential
	//double eps    = d_partParams.eps * d_partParams.kT; 
	//double sigma  = d_partParams.eqLJ / pow(2., 1./6.);
	//double cutoff = d_partParams.cutoffLJ;

	// Morse potential
	//double r0           = d_partParams.eqMorse;
	//double beta         = d_partParams.widthMorse;
	//double epsMorse     = d_partParams.depthMorse * d_partParams.kT;
	//double cutoff_morse = d_partParams.cutoffMorse;

  //__shared__ short int numNeighb_shared[64];

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  //numNeighb_shared[threadIdx.x] = numNeighbors[index];
  //__syncthreads();

  if (index < d_partParams.num_beads)
  {
    // Zero stress variables
    // TODO: ####################################################################
    // To restructure 'monomers' think about keeping 'stress_int' in another way
    // ##########################################################################
    monomers[index].stress_int[0][0]=0.;
    monomers[index].stress_int[0][1]=0.;
    monomers[index].stress_int[0][2]=0.;
    monomers[index].stress_int[1][0]=0.;
    monomers[index].stress_int[1][1]=0.;
    monomers[index].stress_int[1][2]=0.;
    monomers[index].stress_int[2][0]=0.;
    monomers[index].stress_int[2][1]=0.;
    monomers[index].stress_int[2][2]=0.;

    unsigned short partIdx1, partIdx2;
    if (index < d_partParams.Ntype[0]*d_partParams.N_per_sphere[0]) {
      partIdx1 = index / d_partParams.N_per_sphere[0];
    } else {
      partIdx1 = d_partParams.Ntype[0] + (index-d_partParams.Ntype[0]*d_partParams.N_per_sphere[0])/d_partParams.N_per_sphere[1];
    }

    //unsigned int n1offset = index*3;

    //		for (unsigned int i=1; i <= nlist[index*MAX_N]; i++) 
    for (unsigned int i=0; i < numNeighbors[index]; i++)
    //    for (unsigned int i=0; i < numNeighb_shared[threadIdx.x]; i++)
    {
			unsigned int n2 = nlist[index*MAX_N+i];
      //unsigned int n2offset = n2*3;   

      if (n2 < d_partParams.Ntype[0]*d_partParams.N_per_sphere[0]) {
        partIdx2 = n2 / d_partParams.N_per_sphere[0];
      } else {
        partIdx2 = d_partParams.Ntype[0] + (n2-d_partParams.Ntype[0]*d_partParams.N_per_sphere[0])/d_partParams.N_per_sphere[1];
      }

      //			if (monomers[index].sphere_id != monomers[n2].sphere_id) 
      //      if (fabs( float(index-n2)) < 162) //doesn't work !? should take absolute value
      if (partIdx1 != partIdx2)
      {
				double q12[3], q12mag=0., force[3]={0.};    
				// calculate the monomer-monomer distance
				q12[0] = pos[index*3+0] - pos[n2*3+0];
				q12[1] = pos[index*3+1] - pos[n2*3+1];
				q12[2] = pos[index*3+2] - pos[n2*3+2];
				if (d_partParams.wallFlag == 1) {
				  q12[0] = n_image(q12[0], d_partParams.lx);
				  q12[2] = n_image(q12[2], d_partParams.lz);
				}
				else if (d_partParams.wallFlag==2) {
				  q12[0] = n_image(q12[0], d_partParams.lx);
				}
				else {
				  printf ("Wall flag value is not allowed.\n");
				}
				q12mag = q12[0]*q12[0] + q12[1]*q12[1] + q12[2]*q12[2];
				q12mag = sqrt(q12mag);

				if (d_partParams.interparticle == 1) // LJ potential
				{
					if (q12mag < d_partParams.cutoffLJ) {
						//double temp   = sigma / q12mag;
            double temp   = (d_partParams.eqLJ / pow(2., 1./6.)) / q12mag;  
            double temp6  = temp*temp*temp*temp*temp*temp;
            double temp12 = temp*temp*temp*temp*temp*temp*temp*temp*temp*temp*temp*temp;	
						//force[0] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[0]/q12mag;
					  //force[1] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[1]/q12mag;
            //force[2] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[2]/q12mag;
						force[0] = 24.0*(d_partParams.eps*d_partParams.kT)*(2.0*temp12 - temp6)/q12mag * q12[0]/q12mag;
					  force[1] = 24.0*(d_partParams.eps*d_partParams.kT)*(2.0*temp12 - temp6)/q12mag * q12[1]/q12mag;
            force[2] = 24.0*(d_partParams.eps*d_partParams.kT)*(2.0*temp12 - temp6)/q12mag * q12[2]/q12mag;
					}
				}
				else if (d_partParams.interparticle == 2) // Morse potential 
				{
					//double dr = r0 - q12mag;
					double betaDr = d_partParams.widthMorse*(d_partParams.eqMorse - q12mag);
					if (q12mag < d_partParams.cutoffMorse) {
				    //double mag = epsMorse*2*d_partParams.widthMorse*(exp(betaDr) - exp(2.*betaDr)); // derivative of potential, du/dr
				    double mag = (d_partParams.depthMorse*d_partParams.kT)*2*d_partParams.widthMorse*(exp(betaDr) - exp(2.*betaDr)); // derivative of potential, du/dr
						force[0] =  -mag * q12[0] / q12mag;  // force = -du/dr
						force[1] =  -mag * q12[1] / q12mag; 
						force[2] =  -mag * q12[2] / q12mag;
					}
				}
				else if (d_partParams.interparticle == 3)  // blend of WCA and Morse potentials
				{
					//double dr = r0-q12mag;
					//double beta_dr = beta*dr;
          double beta_dr = d_partParams.widthMorse*(d_partParams.eqMorse - q12mag);
					if (q12mag < d_partParams.cutoffMorse) {
						//double mag = epsMorse*2*d_partParams.widthMorse*(exp(beta_dr) - exp(2*beta_dr)); // derivative of potential, du/dr
						double mag = (d_partParams.depthMorse*d_partParams.kT)*2*d_partParams.widthMorse*(exp(beta_dr) - exp(2*beta_dr)); // derivative of potential, du/dr
						//double forceTemp[3];
						//forceTemp[0] = -mag * q12[0] / q12mag; // force = - du/dr
						//forceTemp[1] = -mag * q12[1] / q12mag;
						//forceTemp[2] = -mag * q12[2] / q12mag;
						//force[0] +=  forceTemp[0];  
						//force[1] +=  forceTemp[1];
						//force[2] +=  forceTemp[2];
            force[0] -= (mag * q12[0] / q12mag);
            force[1] -= (mag * q12[1] / q12mag);
            force[2] -= (mag * q12[2] / q12mag); 
					}
					if (q12mag < d_partParams.cutoffLJ) {
						//double temp = sigma / q12mag;
            double temp = (d_partParams.eqLJ / pow(2., 1./6.)) / q12mag;
            double temp6  = temp*temp*temp*temp*temp*temp;
            double temp12 = temp*temp*temp*temp*temp*temp*temp*temp*temp*temp*temp*temp;       
						//double forceTemp[3];
						//forceTemp[0] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[0]/q12mag;
						//forceTemp[1] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[1]/q12mag;
						//forceTemp[2] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[2]/q12mag;
						//force[0] += forceTemp[0];
						//force[1] += forceTemp[1];
						//force[2] += forceTemp[2];  
            //force[0] += (24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[0]/q12mag);
            //force[1] += (24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[1]/q12mag);
            //force[2] += (24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[2]/q12mag);
            force[0] += (24.0*(d_partParams.eps*d_partParams.kT)*(2.0*temp12 - temp6)/q12mag * q12[0]/q12mag);
            force[1] += (24.0*(d_partParams.eps*d_partParams.kT)*(2.0*temp12 - temp6)/q12mag * q12[1]/q12mag);
            force[2] += (24.0*(d_partParams.eps*d_partParams.kT)*(2.0*temp12 - temp6)/q12mag * q12[2]/q12mag);
					}
				}
        else {
          printf ("Inter-particle interaction type is not available!\n");
        }

				interparticleForces[index*3]   += force[0];  
				interparticleForces[index*3+1] += force[1];  
				interparticleForces[index*3+2] += force[2];

				for (int m=0; m < 3; m++) { // Calculate particle stress
					for (int n=0; n < 3; n++) { 
				    monomers[index].stress_int[m][n] += 0.5*q12[m]*force[n];
						//monomers[n1].stress_int_v2[m][n] -= 0.5*q12[m]*force[n];
					}
				}
			}
		}
	}
}

extern "C"
void ComputeForces_gpu (/*unsigned int h_numBeads, unsigned int h_numParticles,*/ struct sphere_param h_params, struct monomer *d_monomers, struct face *d_faces, int *d_numBonds, int *d_blist, int *d_numNeighbors, int *d_nlist, double *d_pos, double *d_springForces, double *d_bendingForces, float *d_volumeForces, float *d_globalAreaForces, double *d_localAreaForces, double *d_wallForces, double *d_interparticleForces, double *d_forces    ,float *d_coms, double *d_faceCenters, double *d_normals, float *d_areas, float *d_volumes, int *d_face_pair_list) {

  int h_numParticles = h_params.Nsphere;
  int h_numBeads     = h_params.num_beads;
  int h_numFaces     = h_params.Ntype[0]*h_params.face_per_sphere[0]+h_params.Ntype[1]*h_params.face_per_sphere[1];
  int h_numFacePairs = h_params.num_face_pair;

  int numStreams = 5;
  cudaStream_t *streams = (cudaStream_t *) malloc(numStreams*sizeof(cudaStream_t));
  for (int i=0; i < numStreams; i++) {
    cudaStreamCreate (&(streams[i]));
  }

  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_numBeads + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  ZeroForces <<<dim_grid, threads_per_block>>> (d_springForces, d_bendingForces, d_volumeForces, d_globalAreaForces, d_localAreaForces, d_wallForces, d_interparticleForces);

  threads_per_block = 64;
  blocks_per_grid_y = 4;
  blocks_per_grid_x = (h_numFaces + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  FaceNormal <<<dim_grid, threads_per_block>>> (d_faces, d_pos, d_normals);

  threads_per_block = 64;
  blocks_per_grid_y = 4;
  blocks_per_grid_x = (h_numBeads + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  SpringForce <<<dim_grid, threads_per_block , 0, streams[0]>>> (d_monomers, d_numBonds, d_blist, d_pos, d_springForces);

  threads_per_block = 64;
  blocks_per_grid_y = 4;
  blocks_per_grid_x = (h_numFacePairs + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  BendingForce <<<dim_grid, threads_per_block , 0, streams[1]>>> (d_monomers, d_numBonds, d_blist, d_pos, d_bendingForces, d_normals, d_face_pair_list);

  threads_per_block = 64;
  blocks_per_grid_y = 4;
  blocks_per_grid_x = (h_numParticles + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  ZeroCOM <<<dim_grid, threads_per_block, 0, streams[2]>>> (d_coms, d_volumes, d_areas);

  threads_per_block = 64;
  blocks_per_grid_y = 4;
  blocks_per_grid_x = (h_numBeads + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);
  
  COM <<<dim_grid, threads_per_block, 0, streams[2]>>> (d_pos, d_coms);

  threads_per_block = 64;
  blocks_per_grid_y = 4;
  blocks_per_grid_x = (h_numFaces + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);
  
  VolumeAreas <<<dim_grid, threads_per_block,0,streams[2]>>> (d_faces, d_pos, d_coms, d_faceCenters, d_normals, d_areas, d_volumes);

  //PrintCOM <<<1,64,0,streams[2]>>> (d_coms, d_volumes, d_areas);
  
  VolumeAreaForces <<<dim_grid, threads_per_block,0,streams[2]>>> (d_volumes, d_faces, d_areas, d_pos, d_faceCenters, d_normals, d_volumeForces, d_globalAreaForces, d_localAreaForces);


  threads_per_block = 64;
  blocks_per_grid_y = 4;
  blocks_per_grid_x = (h_numBeads + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);
 
  WallForce_gpu <<<dim_grid, threads_per_block , 0, streams[3]>>> (d_pos, d_wallForces);

  InterparticleForce <<<dim_grid, threads_per_block, 0, streams[4]>>> (d_monomers, d_numNeighbors, d_nlist, d_pos, d_interparticleForces);

  SumForces <<<dim_grid, threads_per_block>>> (d_springForces, d_bendingForces, d_volumeForces, d_globalAreaForces, d_localAreaForces, d_wallForces, d_interparticleForces, d_forces);
  for (int i=0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }
  free (streams);

//cudaDeviceSynchronize();
}