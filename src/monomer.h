#ifndef STRUCT_H
#define STRUCT_H

#define DIMS     3
#define MAX_BOND 6

struct monomer                                  
{
  double initLength         [MAX_BOND+1];  
  double initLength_temp    [MAX_BOND+1];
  double initLength_final   [MAX_BOND+1];
  double initAngle          [MAX_BOND+1];
  double initAngle_temp     [MAX_BOND+1];
  double lmax               [MAX_BOND+1];
  double kT_inversed_persis [MAX_BOND+1];
  double kp                 [MAX_BOND+1];
  double stress_int[3][3];
  //int fluid_neighbor[8][3];
  //int weighting[8][3];

  int  sphere_id;                     
  int updatedFlag; 

  //double pos_pbc[3]; // real position
  //double pos[3]; // folded position
  //double vel[3];    
  //double force[3];         
  //double pos_lst[DIMS]; // will be discarded soon
  //double pos_temp[3];          
  //double vel_temp[3];         
  //double vel_old[3];  // for 2nd order integration method


  //int list[MAX_N+1]; // Neighbor list 
  //int blist[MAX_BOND+1][3]; // bond list (first element is the bonded monomer; second and third are faces) 
  //int Blist[MAX_BOND+1][3]; // all the bonds of a vertex
  //int faceList[MAX_BOND+1];

//  int type;                                      /* sphere type */
//  double radius;                                 /* monomer radius */
//  double rho;                                    /* monomer density */
//  double pos0[DIMS];                             /* monomer position */

//  int num_proj;
//  double Dis[MAX_F+1];
//  double f_aggre[DIMS];
//  double curve;
//  double normal[DIMS];

//  double force_spring[DIMS];
//  double force_bend[DIMS];
//  double force_vol[DIMS];
//  double force_areaL[3];
//  double force_areaG[3];
//  double force_wall[DIMS];
//  double force_inter[3];
//  double force_interA[3];
//  double force_interR[3];
//  double force_ev[DIMS];
//  double force_fluc[DIMS];
//  double force_face[DIMS];
//  double force_nonbonded_intra[3];
//  double stress_spring[3][3];
//  double stress_bend[3][3];
//  double stress_vol[3][3];
//  double stress_areaL[3][3];
//  double stress_areaG[3][3];
//  double stress_wall[3][3];
//  double f_ext[DIMS];                            /* extern force acting on the monomers */
//  double f_int[DIMS];                            /* internal forces */
//  double f_fluc[DIMS];                           /* fluctuation forces */
//  double stress[DIMS][DIMS];
//  double force_pp[3];
//  double drag[DIMS];
   //double stress_int_v2[3][3];
//  double dr; // 20161129
//  double v_diff[3]; // 20170125
//  double pos_com[DIMS]; // 20170220
//  double force0[DIMS];
//  double fricforce[DIMS];                        /* friction force on the monomer */
//  double fluid_vel[DIMS];                        /* fluid velocity at the monomer position */
//  double pos_old[DIMS];                          /* pbc monomer pos */
//  double pos_tmp[DIMS];                          /* pbc monomer pos */
//  double dr2;                                    /* pbc monomer displacement */
};

#endif

