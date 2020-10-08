extern "C"{
#include "integration.h"
#include "tools.h"
#include <stdio.h>
#include "sphere_param.h"
#include "lb.h"
}

extern __constant__ struct sphere_param d_partParams;
extern __constant__ LBparameters d_LBparams;

/* void fluc_vel(Float *force); */

//void verlet_update(struct monomer *mon, struct face *faces, struct sphere_param *sphere_pm, 
//                   Float ***velcs_df, int n_step, VSLStreamStatePtr rngstream)
//{
//	extern int max_x, max_y, max_z;
//	extern int wall_flag;
//	double maxsize[3];
//	long i,j;
//	int d;
//	int num_beads = sphere_pm->num_beads;
//	int overlap;
//	double dt = sphere_pm->dt;
//	double h_dt = dt/2.0;
//	double h_dt2= h_dt*h_dt;
//	double trialpos[DIMS], trialvel[DIMS];
//	double *fforce;
//	double dr[DIMS];
//	double wallx, wally, wallz;
//	double mon_mass = Rho_Fl*sphere_pm->monmass;
//	double x = sphere_pm->fric*dt/sphere_pm->monmass;
//	double midpt[DIMS];
//	FILE *stream;
//  extern char *work_dir;  // Modification 20170321
//  char fileName[100];
//	maxsize[0]=(double)max_x;
//	maxsize[1]=(double)max_y;
//	maxsize[2]=(double)max_z;
//
//	fforce=(Float *)calloc(DIMS, sizeof(Float));
//	if(fforce == 0) fatal_err("cannot allocate force", -1);
//
//	for(j=0; j<DIMS; j++)
//		midpt[j]=(maxsize[j]-1.0)/2.0;
//
//	wallx = (maxsize[0]+1.0)/2.0-1.5*mon[0].radius;
//	wally = (maxsize[1])/2.0-1.5*mon[0].radius;
//	wallz = (maxsize[2])/2.0-1.5*mon[0].radius;
//
//	if(sphere_pm->verlet == 0) /* no position update */
//	{
//		get_forces(sphere_pm, mon, faces);
//
//		for(i=0;i<num_beads;i++) {
//			for(d=0; d<DIMS; d++) {
//				mon[i].vel[d] += mon[i].force[d]/mon_mass*dt;
//				dr[d] = dt*trialvel[d];
//				mon[i].pos[d]=box(mon[i].pos_pbc[d], maxsize[d]);
//			}
//		}
//	} 
//	else if(sphere_pm->verlet==1)  /* velocity verlet update */
//	{
//		for(i=0;i < num_beads;i++)	
//		{
//			for(d=0; d < DIMS; d++) {
//				dr[d] = dt * mon[i].vel[d] + 2.0 * h_dt2 * mon[i].force0[d]/mon_mass;
//				mon[i].pos_pbc[d] = mon[i].pos_pbc[d] + dr[d];
//				mon[i].pos[d] = box(mon[i].pos_pbc[d], maxsize[d]);
//
//        //if(isnan(mon[i].pos_pbc[d]))
//        //{
//        //  sprintf(fileName,"%s/data/nan_warning.dat",work_dir);
//        //  stream=fopen(fileName,"w");
//        //  fprintf(stream,"step = %d  momomer %d\n",n_step,i);
//        //  fprintf(stream,"spring force = (%f, %f, %f)\n",mon[i].force_spring[0], 
//        //          mon[i].force_spring[1],mon[i].force_spring[2]);
//        //  fprintf(stream,"bending force = (%f, %f, %f)\n",mon[i].force_bend[0],
//        //          mon[i].force_bend[1],mon[i].force_bend[2]);
//        //  fprintf(stream,"wall force = (%f, %f, %f)\n",mon[i].force_wall[0],
//        //          mon[i].force_wall[1],mon[i].force_wall[2]);
//        //  fprintf(stream,"drag foce = (%f, %f, %f)\n",mon[i].drag[0],
//        //          mon[i].drag[1],mon[i].drag[2]);
//        //  fprintf(stream,"inter-particle foce = (%f, %f, %f)\n",mon[i].force_pp[0],
//        //          mon[i].force_pp[1],mon[i].force_pp[2]);
//        //  fprintf(stream,"intra-particle nonbonded foce = (%f, %f, %f)\n",
//        //          mon[i].force_nonbonded_intra[0], mon[i].force_nonbonded_intra[1],
//        //          mon[i].force_nonbonded_intra[2]);
//        //  fclose(stream);
//        //  exit(18);
//        //}
//			}
//		}
//   
//    //if(n_step > 11999) 
//    //  ArtificialShift(sphere_pm, mon);  // Modification 20170323
//    // Modification 20170423
//		get_forces_hi(sphere_pm, mon, faces, velcs_df, n_step, rngstream);
//
//		for(i=0;i < num_beads;i++) 
//		{
//			for(d=0; d < DIMS; d++) {
//				mon[i].vel[d] += h_dt * (mon[i].force0[d]+mon[i].force[d])/mon_mass;
//				mon[i].force0[d]=mon[i].force[d];
//			}
//		}
//	}
//	else if(sphere_pm->verlet == 2) /* explicit 1st order */
//	{
//		get_forces(sphere_pm, mon, faces);
//
//		for(i=0;i<num_beads;i++) {
//			for(d=0; d<DIMS; d++) {
//				trialvel[d] = (mon[i].vel[d]+dt*mon[i].force[d]/mon_mass);
//				dr[d] = dt*(trialvel[d]+mon[i].vel[d])/2.0;
//				trialpos[d]=mon[i].pos_pbc[d]+dr[d];
//			}
//
//			for(d=0; d<DIMS; d++) {
//				mon[i].vel[d]=trialvel[d];
//				mon[i].pos_pbc[d]=trialpos[d];
//				mon[i].pos[d]=box(mon[i].pos_pbc[d], maxsize[d]);
//			}
//		}
//	}
//	else if(sphere_pm->verlet == 3) /* implicit 1st order */
//	{
//		for(i=0;i<num_beads;i++) {
//			for(d=0; d<DIMS; d++) {
//				mon[i].pos_tmp[d]=mon[i].pos_pbc[d];
//				mon[i].pos_pbc[d] +=mon[i].vel[d]*dt;
//				mon[i].pos[d]=box(mon[i].pos_pbc[d], maxsize[d]);
//			}
//		}
//
//		get_forces(sphere_pm, mon, faces);
//
//		for(i=0;i<num_beads;i++) {
//			for(d=0; d<DIMS; d++) {
//				trialvel[d] = (mon[i].vel[d]+dt*(mon[i].force[d])/mon_mass);
//				dr[d] = dt*(trialvel[d]+mon[i].vel[d])/2.0;
//				mon[i].vel[d]=trialvel[d];
//				mon[i].pos_pbc[d]=mon[i].pos_tmp[d]+dr[d];
//				mon[i].pos[d]=box(mon[i].pos_pbc[d], maxsize[d]);
//			}
//
//			/*
//				 for(d=0; d<DIMS; d++) {
//				 trialvel[d] = (mon[i].vel[d]+dt*(mon[i].f_int[d]+mon[i].f_fluc[d])/mon_mass)/(1+x);
//				 dr[d] = dt*(trialvel[d]+mon[i].vel[d])/2.0;
//				 mon[i].vel[d]=trialvel[d];
//				 mon[i].pos_pbc[d]=mon[i].pos_tmp[d]+dr[d];
//				 mon[i].pos[d]=box(mon[i].pos_pbc[d], maxsize[d]);
//				 }
//
//				 for(d=0; d<DIMS; d++) {
//				 mon[i].vel[d]=(1+x)/(1+2*x)*(mon[i].vel[d]+mon[i].fluid_vel[d]*x/(1+x)+
//				 dt*(mon[i].f_int[d]/mon_mass+mon[i].f_fluc[d]/mon_mass/(1+x)));
//				 dr[d] = dt*mon[i].vel[d];
//				 mon[i].pos_pbc[d]=mon[i].pos_tmp[d]+dr[d];
//				 mon[i].pos[d]=box(mon[i].pos_pbc[d], maxsize[d]);
//				 }
//			 */
//		}
//	}
//
//	free(fforce);
//}

//double adams_bashforth (double y, double derivOld, double derivNew, double dt) 
//{
//  double yNew;
//  yNew = y + (1.5*derivNew - 0.5*derivOld) * dt;
//  return yNew;
//}

//double euler_method (double y, double deriv, double dt) 
//{
//  double yNew;
//  yNew = y + deriv * dt;
//  return yNew;
//}

//void update_position_attached (int numBead, double dt, struct monomer *mon) 
//{
//  extern int max_x, max_y, max_z;
//  double maxsize[DIMS] = {max_x, max_y, max_z};
//  extern char *work_dir;
//  //int label[128];
//  //int num_vertex;
//  char filename[200];
//  FILE *stream;
//  int num_attached_bead = 128;
//
//  sprintf (filename, "%s/init/attached_bead.dat", work_dir);
//  stream = fopen (filename, "r");
//  //fscanf (stream, "%d\n", &num_vertex);
//
//  for(int n=0; n < num_attached_bead; n++)
//  {
//    int temp;
//    fscanf (stream, "%d\n", &temp);
//    mon[temp].updatedFlag=2;
//  }
//  fclose (stream);
//
//  for(int n=0; n < 642; n++)
//  {
//    mon[n].updatedFlag=2;
//  }
//
//
//
////  #pragma omp parallel for schedule(static) 
//  for(int n=0; n < numBead; n++) 
//  {
//    if (mon[n].updatedFlag==2)
//    {
//      mon[n].pos_pbc[0] = euler_method(mon[n].pos_pbc[0], 0., dt);
//      mon[n].pos_pbc[1] = euler_method(mon[n].pos_pbc[1], 0., dt);
//      mon[n].pos_pbc[2] = euler_method(mon[n].pos_pbc[2], 0., dt);
//
//      mon[n].pos[0] = box(mon[n].pos_pbc[0], maxsize[0]);
//      mon[n].pos[1] = mon[n].pos_pbc[1];
//      mon[n].pos[2] = box(mon[n].pos_pbc[2], maxsize[2]);
//    }
//
//    else if(mon[n].updatedFlag==FALSE)
//    {
//      mon[n].pos_pbc[0] = euler_method(mon[n].pos_pbc[0], mon[n].vel[0], dt);
//      mon[n].pos_pbc[1] = euler_method(mon[n].pos_pbc[1], mon[n].vel[1], dt);
//      mon[n].pos_pbc[2] = euler_method(mon[n].pos_pbc[2], mon[n].vel[2], dt);
//
//      mon[n].pos[0] = box(mon[n].pos_pbc[0], maxsize[0]);
//      mon[n].pos[1] = mon[n].pos_pbc[1];
//      mon[n].pos[2] = box(mon[n].pos_pbc[2], maxsize[2]);
//
//      //mon[n].vel_old[0] = mon[n].vel[0];
//      //mon[n].vel_old[1] = mon[n].vel[1];
//      //mon[n].vel_old[2] = mon[n].vel[2];
//    }
//    //mon[n].updatedFlag = FALSE;  // Modification 20170723 move to get_force.c
//    // Modification 20170818
//    else if (mon[n].updatedFlag==TRUE)
//    { 
//      mon[n].vel[0] = mon[n].vel_temp[0];
//      mon[n].vel[1] = mon[n].vel_temp[1];
//      mon[n].vel[2] = mon[n].vel_temp[2];
//
//      mon[n].pos_pbc[0] = mon[n].pos_temp[0];
//      mon[n].pos_pbc[1] = mon[n].pos_temp[1];
//      mon[n].pos_pbc[2] = mon[n].pos_temp[2];
//
//      mon[n].pos[0] = box(mon[n].pos_pbc[0], maxsize[0]);
//      mon[n].pos[1] = mon[n].pos_pbc[1];
//      mon[n].pos[2] = box(mon[n].pos_pbc[2], maxsize[2]);
//        
//      //mon[n].vel_old[0] = mon[n].vel[0];
//      //mon[n].vel_old[1] = mon[n].vel[1];
//      //mon[n].vel_old[2] = mon[n].vel[2];
//    }
//  }
//}


__global__ void EulerUpdate (double *forces, double *velocities, double *pos, double *foldedPos) {

  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < d_partParams.num_beads) { 
    int offset = index*3; 
    velocities[offset+0] += forces[offset+0] / d_partParams.fictionalMass;
    velocities[offset+1] += forces[offset+1] / d_partParams.fictionalMass;
    velocities[offset+2] += forces[offset+2] / d_partParams.fictionalMass;
//mon[n].vel[0] = mon[n].force[0]/fictionalMass;
//mon[n].vel[1] = mon[n].force[1]/fictionalMass;
//mon[n].vel[2] = mon[n].force[2]/fictionalMass;

    pos[offset+0] += velocities[offset+0]; // * dt (dt=1)
    pos[offset+1] += velocities[offset+1];
    pos[offset+2] += velocities[offset+2];

    foldedPos[offset+0] = box(pos[offset+0], d_partParams.lx);
    // Note: #############################
    // Walls are in the y dir. by default
    // ###################################
    //mon[n].pos[1] = box(mon[n].pos_pbc[1], ly);
    foldedPos[offset+1] = pos[offset+1];
    foldedPos[offset+2] = box(pos[offset+2], d_partParams.lz);

//    printf("index=%d  v=(%f, %f, %f)  pos=(%f, %f, %f)  foldedPos=(%f, %f, %f)\n", index, velocities[offset+0], velocities[offset+1], velocities[offset+2], pos[offset+0], pos[offset+1], pos[offset+2], foldedPos[offset+0], foldedPos[offset+1], foldedPos[offset+2]);

//    printf("index=%d  force=(%f, %f, %f)\n", index, forces[offset+0], forces[offset+1], forces[offset+2]);
  }
}

extern "C"
void EulerUpdate_wrapper (int h_numBeads, double *d_forces, double *d_velocities, double *d_pos, double *d_foldedPos) {

  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_numBeads + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  EulerUpdate<<<dim_grid, threads_per_block>>> (d_forces, d_velocities, d_pos, d_foldedPos);

//cudaDeviceSynchronize();

}

__global__ void EulerMethod (double *vel, double *pos, double *foldedPos) {

  // Note:
  // dt is set to 1.0. When d_LBparams.tau != 1, one may need to correct this function
 
  unsigned int index = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x; 
  if (index < d_partParams.num_beads) {
    int offset = index*3;
    pos[offset]   = pos[offset]   + vel[offset]   * d_LBparams.tau;
    pos[offset+1] = pos[offset+1] + vel[offset+1] * d_LBparams.tau;
    pos[offset+2] = pos[offset+2] + vel[offset+2] * d_LBparams.tau;

    foldedPos[offset]   = box(pos[offset],   d_partParams.lx);
    foldedPos[offset+1] =     pos[offset+1];
    foldedPos[offset+2] = box(pos[offset+2], d_partParams.lz);
    
    //if (pos[offset+1] > top_fluid_node || pos[offset+1] < 0.5) {
    //  flag=1; 
    //}
  }
  //if (flag==1) {
  //  printf ("Membrane nodes are too close to the walls! The program terminates.\n");
  //  exit(18);
  //}
}

extern "C"
void UpdatePartice (int h_numBeads, double *d_vel, double *d_pos, double *d_foldedPos) {

  int threads_per_block = 64;
  int blocks_per_grid_y = 4;
  int blocks_per_grid_x = (h_numBeads + threads_per_block*blocks_per_grid_y - 1) / (threads_per_block * blocks_per_grid_y);
  dim3 dim_grid = make_uint3 (blocks_per_grid_x, blocks_per_grid_y, 1);

  EulerMethod<<<dim_grid, threads_per_block>>> (d_vel, d_pos, d_foldedPos);

}

