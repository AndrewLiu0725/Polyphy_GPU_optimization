#include <math.h>
#include <stdio.h>
#include "forces.h"
#include "tools.h"
#include "neighborList.h"

void ZeroForces (int numNodes, double *springForces, double *bendingForces, double *volumeForces, double *globalAreaForces, double *localAreaForces, double *wallForces, double *interparticleForces) {

  #pragma omp parallel for schedule(static)  
  for (int n=0; n < numNodes; n++)
  {
    int x = n*3;
    int y = x+1;
    int z = x+2;

    springForces       [x] = 0.;
    springForces       [y] = 0.;
    springForces       [z] = 0.;

    bendingForces      [x] = 0.;
    bendingForces      [y] = 0.;
    bendingForces      [z] = 0.;

    volumeForces       [x] = 0.;
    volumeForces       [y] = 0.;
    volumeForces       [z] = 0.;

    globalAreaForces   [x] = 0.;
    globalAreaForces   [y] = 0.;
    globalAreaForces   [z] = 0.;

    localAreaForces    [x] = 0.;
    localAreaForces    [y] = 0.;
    localAreaForces    [z] = 0.;

    wallForces         [x] = 0.;
    wallForces         [y] = 0.;
    wallForces         [z] = 0.;

    interparticleForces[x] = 0.;
    interparticleForces[y] = 0.;
    interparticleForces[z] = 0.;
  }
}

void SumForces (int numNodes, double *springForces, double *bendingForces, double *volumeForces, double *globalAreaForces, double *localAreaForces, double *wallForces, double *interparticleForces, double *forces) {

  #pragma omp parallel for schedule(static)
  for (int n=0; n < numNodes; n++)
  {
    int x = n*3; 
    int y = x+1;
    int z = x+2;   
    forces[x] =        springForces[x] +   bendingForces[x] + volumeForces[x] + 
                   globalAreaForces[x] + localAreaForces[x] +   wallForces[x] +
                interparticleForces[x];
    forces[y] =        springForces[y] +   bendingForces[y] + volumeForces[y] + 
                   globalAreaForces[y] + localAreaForces[y] +   wallForces[y] +
                interparticleForces[y];
    forces[z] =        springForces[z] +   bendingForces[z] + volumeForces[z] + 
                   globalAreaForces[z] + localAreaForces[z] +   wallForces[z] +
                interparticleForces[z];
  }
}

void ComputeForces (struct sphere_param *sphere_pm, struct monomer *monomers, struct face *faces, double *springForces, double *bendingForces, double *volumeForces, double *globalAreaForces, double *localAreaForces, double *wallForces, double *interparticleForces, double *pos, int *numNeighbors, int *nlist, int *numBonds, int *blist, double *forces) {

  ZeroForces (sphere_pm->num_beads, springForces, bendingForces, volumeForces, globalAreaForces, localAreaForces, wallForces, interparticleForces);

  SpringForce (sphere_pm, monomers, springForces, pos, numBonds, blist);

  BendingForceSin (sphere_pm, monomers, bendingForces, pos, numBonds, blist);

  VolumeAreaConstraints (sphere_pm, monomers, faces, volumeForces, globalAreaForces, localAreaForces, pos);

  WallForce (sphere_pm, pos, wallForces);

  InterparticleForce (sphere_pm, monomers, interparticleForces, pos, numNeighbors, nlist);

  SumForces (sphere_pm->num_beads, springForces, bendingForces, volumeForces, globalAreaForces, localAreaForces, wallForces, interparticleForces, forces);  
}

void SpringForce (struct sphere_param *sphere_pm, struct monomer *monomers, double *springForces, double *pos, int *numBonds, int *blist) {

  int numParticles  = sphere_pm->Nsphere;
  int numParticlesA = sphere_pm->Ntype[0];
 
  #pragma omp parallel for schedule(static) 
  for (int i=0; i < numParticles; i++) 
	{
    int springType;
    double springConst;
    int numNodesPerParticle; 
    int nodeOffset;

    if (i < numParticlesA) {  
      springType = sphere_pm->springType[0];
      springConst = -2.0 * sphere_pm->springConst[0] * sphere_pm->kT;
      // varibles for the WLC-POW spring is set in function 'SetSpringConst'
      numNodesPerParticle = sphere_pm->N_per_sphere[0];
      nodeOffset = i*sphere_pm->N_per_sphere[0];
    }
    else {
      springType = sphere_pm->springType[1]; 
      springConst = -2.0 * sphere_pm->springConst[1] * sphere_pm->kT;
      // variables for the WLC-POW spring is set in function 'SetSpringConst'.
      numNodesPerParticle = sphere_pm->N_per_sphere[1];
      nodeOffset = numParticlesA *     sphere_pm->N_per_sphere[0] + 
                   (i-numParticlesA) * sphere_pm->N_per_sphere[1];
    }

		for (int j=0; j < numNodesPerParticle; j++) 
		{
		  int n1 = j + nodeOffset;
      int n1offset = n1*3;
      // TODO: reorganize blist                     
			for (int k=1; k <= numBonds[n1]; k++) 
			{
//			  int n2 = blist[n1][k][0]; blist[n1*18+(k-1)*3+0];
			  int n2 = blist[n1*18+(k-1)*3+0];
        int n2offset = n2*3;
				double length_eq = monomers[n1].initLength[k];
				double q12[3];
        // TODO: reorganize pos_pbc[]
				//q12[0] = monomers[n1].pos_pbc[0] - monomers[n2].pos_pbc[0];
				//q12[1] = monomers[n1].pos_pbc[1] - monomers[n2].pos_pbc[1];
				//q12[2] = monomers[n1].pos_pbc[2] - monomers[n2].pos_pbc[2];
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
}

void BendingForceSin (struct sphere_param *sphere_pm, struct monomer *monomers, double *bendingForces, double *pos, int *numBonds, int *blist) {
  
  // TODO: reorganize blist and pos_pbc[]
  int numNodes = sphere_pm->num_beads;
  int numParticlesA = sphere_pm->Ntype[0];
  int numParticles  = sphere_pm->Nsphere;

  #pragma omp parallel for schedule(static) 
	for (int i=0; i < numParticles; i++) 
	{
    double kb; 
    int numNodesPerParticle; 
    int nodeOffset;

    if (i < numParticlesA) {  
      kb = 2.0/sqrt(3)*sphere_pm->kc[0]*sphere_pm->kT;
//kb = 2.0*sqrt(3)*sphere_pm->kc[0]*sphere_pm->kT;
      numNodesPerParticle = sphere_pm->N_per_sphere[0];
      nodeOffset = i*sphere_pm->N_per_sphere[0];
    }
    else {
      kb = 2.0/sqrt(3)*sphere_pm->kc[1]*sphere_pm->kT;
      numNodesPerParticle = sphere_pm->N_per_sphere[1];
      nodeOffset = numParticlesA *     sphere_pm->N_per_sphere[0] + 
                   (i-numParticlesA) * sphere_pm->N_per_sphere[1];
    }

		for (int j=0; j < numNodesPerParticle; j++) 
		{
			int n1 = j + nodeOffset;
      int n1offset = n1*3; 

			for (int k=1; k <= numBonds[n1]; k++) 
			{
        double theta0 = monomers[n1].initAngle[k];
//        int n2 = blist[n1][k][0];
//	      int n3 = blist[n1][k][1];
//		    int n4 = blist[n1][k][2];
        int n2 = blist[n1*18+(k-1)*3+0];
	      int n3 = blist[n1*18+(k-1)*3+1];
		    int n4 = blist[n1*18+(k-1)*3+2];
        int n2offset = n2*3;
        int n3offset = n3*3;
        int n4offset = n4*3;

				double a31[3], a21[3], a41[3];
				a31[0] = pos[n3offset+0] - pos[n1offset+0];
				a31[1] = pos[n3offset+1] - pos[n1offset+1];
				a31[2] = pos[n3offset+2] - pos[n1offset+2];
				a21[0] = pos[n2offset+0] - pos[n1offset+0];
				a21[1] = pos[n2offset+1] - pos[n1offset+1];
				a21[2] = pos[n2offset+2] - pos[n1offset+2];
				a41[0] = pos[n4offset+0] - pos[n1offset+0];
				a41[1] = pos[n4offset+1] - pos[n1offset+1];
				a41[2] = pos[n4offset+2] - pos[n1offset+2];

				double normal1[3], normal2[3];
				product(a31, a21, normal1);
				product(a21, a41, normal2);

				double normal1_mag, normal2_mag;
				normal1_mag = sqrt(normal1[0]*normal1[0] + normal1[1]*normal1[1] + 
                          normal1[2]*normal1[2]);
				normal2_mag = sqrt(normal2[0]*normal2[0] + normal2[1]*normal2[1] + 
                          normal2[2]*normal2[2]);

				normal1[0] = normal1[0] / normal1_mag;
				normal1[1] = normal1[1] / normal1_mag;
				normal1[2] = normal1[2] / normal1_mag;
				normal2[0] = normal2[0] / normal2_mag; 
				normal2[1] = normal2[1] / normal2_mag; 
				normal2[2] = normal2[2] / normal2_mag; 

        double orient[3];
        product(normal1, normal2, orient);
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
//        delta_theta = sin(delta_theta);

        double factor;
//        if(sign > 0)
//          factor = -kb * delta_theta / temp;
//        else
//          factor =  kb * delta_theta / temp;  

        factor =  kb * sin(delta_theta) / sinTheta;  
//
//        double cosTheta = normal1[0]*normal2[0] + normal1[1]*normal2[1] + 
//                          normal1[2]*normal2[2];
//				double theta = acos(cosTheta);
//
//				double orient[3]; 
//				product(normal1, normal2, orient);
//
//        // Determine the sign of the angle, which determines the force direction
//        // if sign > 0 theta_nr = -theta_nr, otherwise theta_nr = theta_nr
//      	double sign = orient[0]*a21[0] + orient[1]*a21[1] + orient[2]*a21[2];
//				if (sign > 0) { 
//          theta = -1.0*theta; 
//        }
//        // After the sign is determined, compute delta theta and sin(theta)
//        double sinTheta = sin(theta);
////				double deltaTheta = theta - theta0;
//double deltaTheta = theta;
//
//        // factor = -kb * sin(deltaTheta) / -sqrt(1-cos(theta)*cos(theta))
////        double factor = kb * sin(deltaTheta) / sinTheta;
//        // if the small angle approximation is preffered:
//        // sin(theta- theta0) ~ theta - theta0 
//double factor = kb * deltaTheta / sinTheta;
//
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
				a23[0] = pos[n2offset+0] - pos[n3offset+0];
				a23[1] = pos[n2offset+1] - pos[n3offset+1];
				a23[2] = pos[n2offset+2] - pos[n3offset+2];
				a42[0] = pos[n4offset+0] - pos[n2offset+0];
				a42[1] = pos[n4offset+1] - pos[n2offset+1];
				a42[2] = pos[n4offset+2] - pos[n2offset+2];

				double term3[3], term21[3], term22[3], term11[3], term12[3], term4[3];
				product(n21, a12, term3);
				product(n21, a31, term21);
				product(n12, a14, term22);
				product(n21, a23, term11);
				product(n12, a42, term12);
				product(n12, a21, term4);

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
  }
}

void VolumeAreaConstraints (struct sphere_param *sphere_pm, struct monomer *monomers, struct face *faces, double *volumeForces, double *globalAreaForces, double *localAreaForces, double *pos) {

  // TODO: reorganize 'face'

  int numNodes      = sphere_pm->num_beads;
  int numParticles  = sphere_pm->Nsphere;
  int numParticlesA = sphere_pm->Ntype[0];

  #pragma omp parallel for schedule(static) 
	for (int i=0; i < numParticles; i++) 
	{
    double kv; 
    double kag;
    double kal;
    double V0;
    double A0;  
    int numNodesPerParticle; 
    int numFacesPerParticle; 
    int nodeOffset;
    int faceOffset;

    if (i < numParticlesA) {  
      kv  = sphere_pm->kv[0];
      kag = sphere_pm->kag[0];
      kal = sphere_pm->kal[0]; 
      V0  = sphere_pm->V0[0];
      A0  = sphere_pm->A0[0];   
      numNodesPerParticle = sphere_pm->N_per_sphere[0];
      numFacesPerParticle = sphere_pm->face_per_sphere[0];   
      nodeOffset = i*sphere_pm->N_per_sphere[0];
      faceOffset = i*sphere_pm->face_per_sphere[0];
    }
    else {
      kv  = sphere_pm->kv[1];
      kag = sphere_pm->kag[1];
      kal = sphere_pm->kal[1]; 
      V0  = sphere_pm->V0[1];
      A0  = sphere_pm->A0[1];   
      numNodesPerParticle = sphere_pm->N_per_sphere[1];
      numFacesPerParticle = sphere_pm->face_per_sphere[1];   
      nodeOffset = numParticlesA     * sphere_pm->N_per_sphere[0] + 
                   (i-numParticlesA) * sphere_pm->N_per_sphere[1];
      faceOffset = numParticlesA     * sphere_pm->face_per_sphere[0] + 
                   (i-numParticlesA) * sphere_pm->face_per_sphere[1];
    }

    double com[3] = {0.0};
    double area   = 0.0;
    double volume = 0.0;
		for (int j=0; j < numNodesPerParticle; j++) 
		{
			int n = j + nodeOffset; 
      int offset = n*3;  
			com[0] += pos[offset+0];
			com[1] += pos[offset+1];
			com[2] += pos[offset+2];
		}
		com[0] = com[0] / numNodesPerParticle;
		com[1] = com[1] / numNodesPerParticle;
		com[2] = com[2] / numNodesPerParticle;

		for (int j=0; j < numFacesPerParticle; j++)
		{
			int nface = j + faceOffset;
			int n1 = faces[nface].v[0];
			int n2 = faces[nface].v[1];
			int n3 = faces[nface].v[2];
      int n1offset = n1*3;
      int n2offset = n2*3;
      int n3offset = n3*3; 
			double dr[3], q12[3], q13[3], normal[3];  
			dr[0] = pos[n1offset+0] - com[0];  // origin is at the position of COM
			dr[1] = pos[n1offset+1] - com[1];
			dr[2] = pos[n1offset+2] - com[2];
			q12[0] = pos[n2offset+0] - pos[n1offset+0];
			q12[1] = pos[n2offset+1] - pos[n1offset+1];
			q12[2] = pos[n2offset+2] - pos[n1offset+2];
			q13[0] = pos[n3offset+0] - pos[n1offset+0];
			q13[1] = pos[n3offset+1] - pos[n1offset+1];
			q13[2] = pos[n3offset+2] - pos[n1offset+2];
			product(q13, q12, normal);  // "normal" points outward viewing from the centroid of the particle
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

		for(int j=0; j < numFacesPerParticle; j++) 
		{
			int nface = j + faceOffset;
			double areaG = -0.25 * kag / faces[nface].area * ((area / A0) - 1.);
			//double areaL = -0.25 * kal * (faces[nface].area - faces[nface].area_0) / (faces[nface].area_0 * faces[nface].area);
      double areaL = -0.25 * kal / faces[nface].area * ((faces[nface].area / faces[nface].area_0) - 1.); 
			int n1 = faces[nface].v[0];
			int n2 = faces[nface].v[1];
			int n3 = faces[nface].v[2];
      int n1offset = n1*3;
      int n2offset = n2*3;
      int n3offset = n3*3;
 
			double v23[DIMS], v31[DIMS], v12[DIMS], v21[DIMS];
			v23[0] = pos[n2offset+0] - pos[n3offset+0];
			v23[1] = pos[n2offset+1] - pos[n3offset+1];
			v23[2] = pos[n2offset+2] - pos[n3offset+2];

			v31[0] = pos[n3offset+0] - pos[n1offset+0];
			v31[1] = pos[n3offset+1] - pos[n1offset+1];
			v31[2] = pos[n3offset+2] - pos[n1offset+2];

			v12[0] = pos[n1offset+0] - pos[n2offset+0];
			v12[1] = pos[n1offset+1] - pos[n2offset+1];
			v12[2] = pos[n1offset+2] - pos[n2offset+2];

			v21[0] = -v12[0];
			v21[1] = -v12[1];
			v21[2] = -v12[2];

			double normalVector[3]; 
			product(v31, v21, normalVector);

			// Note: ############################################################################
			// The origin is at the center of mass of the particle. Should make sure it's right!
			// ##################################################################################
			double faceCenter[DIMS];                          
			//faceCenter[0] = (monomers[n1].pos_pbc[0]+monomers[n2].pos_pbc[0]+monomers[n3].pos_pbc[0]) / 3.;
			//faceCenter[1] = (monomers[n1].pos_pbc[1]+monomers[n2].pos_pbc[1]+monomers[n3].pos_pbc[1]) / 3.;
			//faceCenter[2] = (monomers[n1].pos_pbc[2]+monomers[n2].pos_pbc[2]+monomers[n3].pos_pbc[2]) / 3.;
			faceCenter[0] = (pos[n1offset+0] + pos[n2offset+0] + pos[n3offset+0]) / 3. - com[0];
			faceCenter[1] = (pos[n1offset+1] + pos[n2offset+1] + pos[n3offset+1]) / 3. - com[1];
			faceCenter[2] = (pos[n1offset+2] + pos[n2offset+2] + pos[n3offset+2]) / 3. - com[2];

			double temp1[DIMS], temp2[DIMS], temp3[DIMS]; 
			double force1[DIMS], force2[DIMS], force3[DIMS];

			// Volume constraint
			double normal_3[3];
			normal_3[0] = normalVector[0] / 3.;
			normal_3[1] = normalVector[1] / 3.;
			normal_3[2] = normalVector[2] / 3.;
			product(faceCenter, v23, temp1);
			product(faceCenter, v31, temp2);
			product(faceCenter, v12, temp3); 

			force1[0] = vol * (normal_3[0] + temp1[0]);
			force1[1] = vol * (normal_3[1] + temp1[1]);
			force1[2] = vol * (normal_3[2] + temp1[2]);

			force2[0] = vol * (normal_3[0] + temp2[0]);
			force2[1] = vol * (normal_3[1] + temp2[1]);
			force2[2] = vol * (normal_3[2] + temp2[2]);

			force3[0] = vol * (normal_3[0] + temp3[0]);
			force3[1] = vol * (normal_3[1] + temp3[1]);
			force3[2] = vol * (normal_3[2] + temp3[2]);

			volumeForces[n1offset+0] += force1[0];
			volumeForces[n1offset+1] += force1[1];
			volumeForces[n1offset+2] += force1[2];

			volumeForces[n2offset+0] += force2[0];
			volumeForces[n2offset+1] += force2[1];
			volumeForces[n2offset+2] += force2[2];

			volumeForces[n3offset+0] += force3[0];
			volumeForces[n3offset+1] += force3[1];       
			volumeForces[n3offset+2] += force3[2];       

			// Global area constraint
			product(normalVector, v23, temp1);
			product(normalVector, v31, temp2);
			product(normalVector, v12, temp3);

			force1[0] = areaG * temp1[0];
			force1[1] = areaG * temp1[1];
			force1[2] = areaG * temp1[2];

			force2[0] = areaG * temp2[0];
			force2[1] = areaG * temp2[1];
			force2[2] = areaG * temp2[2];

			force3[0] = areaG * temp3[0];
			force3[1] = areaG * temp3[1];
			force3[2] = areaG * temp3[2];

			globalAreaForces[n1offset+0] += force1[0];
			globalAreaForces[n1offset+1] += force1[1];
			globalAreaForces[n1offset+2] += force1[2];

			globalAreaForces[n2offset+0] += force2[0];
			globalAreaForces[n2offset+1] += force2[1];
			globalAreaForces[n2offset+2] += force2[2];

			globalAreaForces[n3offset+0] += force3[0];
			globalAreaForces[n3offset+1] += force3[1];
			globalAreaForces[n3offset+2] += force3[2];

			// Local area constraint
			force1[0] = areaL * temp1[0];
			force1[1] = areaL * temp1[1];
			force1[2] = areaL * temp1[2];

			force2[0] = areaL * temp2[0];
			force2[1] = areaL * temp2[1];
			force2[2] = areaL * temp2[2];

			force3[0] = areaL * temp3[0];
			force3[1] = areaL * temp3[1];
			force3[2] = areaL * temp3[2];

			localAreaForces[n1offset+0] += force1[0];
			localAreaForces[n1offset+1] += force1[1];
			localAreaForces[n1offset+2] += force1[2];

			localAreaForces[n2offset+0] += force2[0];
			localAreaForces[n2offset+1] += force2[1];
			localAreaForces[n2offset+2] += force2[2];

			localAreaForces[n3offset+0] += force3[0];
			localAreaForces[n3offset+1] += force3[1];
			localAreaForces[n3offset+2] += force3[2];
		}
	}
}

void WallForce (struct sphere_param *sphere_pm, double *pos, double *wallForces) {

	int numNodes = sphere_pm->num_beads;
	//double end_fluid_node_y = 0.5*(channelHeight-1.); // measured from the box center: (max -0.5) - 0.5*max
	//double end_fluid_node_z = 0.5*(maxsize[2]-1.); // measured from the box center
//  double topWally = 0.5*(sphere_pm->ly-1.) - sphere_pm->wallForceDis;
//  double topWallz = 0.5*(sphere_pm->lz-1.) - sphere_pm->wallForceDis;
//  double bottomWally = -topWally;
//  double bottomWallz = -topWallz;
  double topWally = (sphere_pm->ly-1) - sphere_pm->wallForceDis - (0.5*sphere_pm->ly);
  double topWallz = (sphere_pm->lz-1) - sphere_pm->wallForceDis - (0.5*sphere_pm->lz);
  double bottomWally = -topWally;
  double bottomWallz = -topWallz;

  double wallConst = sphere_pm->wallConstant * sphere_pm->kT;

  // cubic potential
  #pragma omp parallel for schedule(static) 
  for (int n1=0; n1 < numNodes; n1++)
  {
    double posTemp[3], force[3], delta[3];
    int offset = n1*3;    
    posTemp[1] = pos[offset+1] - (0.5*sphere_pm->ly);
    posTemp[2] = pos[offset+2] - (0.5*sphere_pm->lz);

    if (sphere_pm->wallFlag==1) {
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
    else if (sphere_pm->wallFlag == 2) {
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

void InterparticleForce (struct sphere_param *sphere_pm, struct monomer *monomers, double *interparticleForces, double *pos, int *numNeighbors, int *nlist) {

	//Float sigma = 2.0*radius;
	//double cutoff = 1.122 * sigma;
	//Float eps = sphere_pm->kT/sigma; // there should be a sigma in denominator?
	//double sigma = 1 / 1.122;
	//double cutoff = 1.;

	// L-J potential
	double eps    = sphere_pm->eps * sphere_pm->kT; 
	double sigma  = sphere_pm->eqLJ / pow(2., 1./6.);
	double cutoff = sphere_pm->cutoffLJ;

	// Morse potential
	double r0           = sphere_pm->eqMorse;
	double beta         = sphere_pm->widthMorse;
	double epsMorse     = sphere_pm->depthMorse * sphere_pm->kT;
	double cutoff_morse = sphere_pm->cutoffMorse;

	// WCA potential for initilization procedure
//	double eps_init=0.005652;
//	double sigma_init=1/pow(2.,1./6.);
//	double cutoff_init=1.;

	int numBeads = sphere_pm->num_beads;

  #pragma omp parallel for schedule(static) 
	for (int n1=0; n1 < numBeads; n1++) 
  {
    // Zero stress variables
    // Note: ####################################################################
    // To restructure 'monomers' think about keeping 'stress_int' in another way
    // ##########################################################################
    monomers[n1].stress_int[0][0]=0.;
    monomers[n1].stress_int[0][1]=0.;
    monomers[n1].stress_int[0][2]=0.;
    monomers[n1].stress_int[1][0]=0.;
    monomers[n1].stress_int[1][1]=0.;
    monomers[n1].stress_int[1][2]=0.;
    monomers[n1].stress_int[2][0]=0.;
    monomers[n1].stress_int[2][1]=0.;
    monomers[n1].stress_int[2][2]=0.;

    int n1offset = n1*3;

    // TODO: ###############################
    // monomer.list[] and monomer.sphere_id
    // ##################################### 
//		for (int index=1; index <= nlist[n1*MAX_N]; index++) 
    for (int index=0; index < numNeighbors[n1]; index++)
    {
			int n2 = nlist[n1*MAX_N + index];
      int n2offset = n2*3;   
			if (monomers[n1].sphere_id != monomers[n2].sphere_id) 
      {
				double q12[3], q12mag=0., force[3]={0.};    
				// calculate the monomer-monomer distance
				q12[0] = pos[n1offset+0] - pos[n2offset+0];
				q12[1] = pos[n1offset+1] - pos[n2offset+1];
				q12[2] = pos[n1offset+2] - pos[n2offset+2];
				if (sphere_pm->wallFlag == 1) {
				  q12[0] = n_image(q12[0], sphere_pm->lx);
				  q12[2] = n_image(q12[2], sphere_pm->lz);
				}
				else if (sphere_pm->wallFlag==2) {
				  q12[0] = n_image(q12[0], sphere_pm->lx);
				  //q12[1] = n_image(q12[1],maxsize[1]);
          //q12[2] = n_image(q12[2],maxsize[2]);   
				}
				else {
				  fprintf (stderr,"Wall flag value is not allowed.\n");
				}
				q12mag = q12[0]*q12[0] + q12[1]*q12[1] + q12[2]*q12[2];
				q12mag = sqrt(q12mag);

				if (sphere_pm->interparticle == 1) // LJ potential
				{
					if (q12mag < cutoff) {
						double temp   = sigma / q12mag;
            double temp6  = temp*temp*temp*temp*temp*temp;
            double temp12 = temp*temp*temp*temp*temp*temp*temp*temp*temp*temp*temp*temp;	
						force[0] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[0]/q12mag;
					  force[1] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[1]/q12mag;
            force[2] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[2]/q12mag;
					}
				}
				else if (sphere_pm->interparticle == 2) // Morse potential 
				{
					double dr = r0 - q12mag;
					double betaDr = beta*dr;
					if (q12mag < cutoff_morse) {
				    double mag = epsMorse*2*beta*(exp(betaDr) - exp(2.*betaDr)); // derivative of potential, du/dr
						force[0] =  -mag * q12[0] / q12mag;  // force = -du/dr
						force[1] =  -mag * q12[1] / q12mag; 
						force[2] =  -mag * q12[2] / q12mag;
					}
				}
				else if (sphere_pm->interparticle == 3)  // blend of WCA and Morse potentials
				{
					double dr = r0-q12mag;
					double beta_dr = beta*dr;
					if (q12mag < cutoff_morse) {
						double mag = epsMorse*2*beta*(exp(beta_dr) - exp(2*beta_dr)); // derivative of potential, du/dr
						double forceTemp[3];
						forceTemp[0] = -mag * q12[0] / q12mag; // force = - du/dr
						forceTemp[1] = -mag * q12[1] / q12mag;
						forceTemp[2] = -mag * q12[2] / q12mag;
						force[0] +=  forceTemp[0];   
						force[1] +=  forceTemp[1]; 
						force[2] +=  forceTemp[2];
					}
					if (q12mag < cutoff) {
						double temp = sigma / q12mag;
            double temp6  = temp*temp*temp*temp*temp*temp;
            double temp12 = temp*temp*temp*temp*temp*temp*temp*temp*temp*temp*temp*temp;       
						double forceTemp[3];
						forceTemp[0] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[0]/q12mag;
						forceTemp[1] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[1]/q12mag;
						forceTemp[2] = 24.0*eps*(2.0*temp12 - temp6)/q12mag * q12[2]/q12mag;
						force[0] += forceTemp[0];
						force[1] += forceTemp[1];
						force[2] += forceTemp[2];  
					}
				}
        else {
          fprintf (stderr,"Inter-particle interaction type is not available!\n");
        }

				interparticleForces[n1offset+0] += force[0];  
				interparticleForces[n1offset+1] += force[1];  
				interparticleForces[n1offset+2] += force[2];

				for (int m=0; m < 3; m++) { // Calculate particle stress
					for (int n=0; n < 3; n++) { 
				    monomers[n1].stress_int[m][n] += 0.5*q12[m]*force[n];
						//monomers[n1].stress_int_v2[m][n] -= 0.5*q12[m]*force[n];
					}
				}
			}
		}
	}
}

//void PullingForce (char *work_dir, struct monomer *mon, double force) {
//
//  int num_pulled_bead = 256;
//  int num_pulled_top_right = 64;
//  char filename[200];
//  FILE *stream;
//
//  sprintf (filename, "%s/init/pulled_bead.dat", work_dir);
//  stream = fopen (filename, "r");  
//  for (int n=0; n < num_pulled_bead; n++) {
//    int temp;  
//    fscanf (stream, "%d\n", &temp);
//    mon[temp].force[1] += (force/num_pulled_bead);
//  }
//  fclose (stream);
//
//  //sprintf (filename, "%s/init/pulled_top_right.dat", work_dir);
//  //stream = fopen (filename, "r");  
//  //for(int n=0; n < num_pulled_top_right; n++)
//  //{
//  //  int temp;  
//  //  fscanf (stream, "%d\n", &temp);
//  //  mon[temp].force[1] += (force/num_pulled_top_right);
//  //}
//  //fclose (stream);
//
//  for(int n=0; n < 642; n++)
//  {
//    mon[n].force[0]=0.;
//    mon[n].force[1]=0.;
//    mon[n].force[2]=0.;
//  }
//
//
////  sprintf (filename, "%s/init/upper.dat", work_dir);
////  stream = fopen (filename, "r");
////  fscanf (stream, "%d\n", &num_vertex);
////  for(int n=0; n < num_vertex; n++)
////  {
////    fscanf (stream, "%d\n", &label[n]);
////    label[n] += 642;
////  }
////  fclose (stream);
////
////  for(int n=0; n < num_vertex; n++)
////  {
////    mon[ label[n] ].force[1] += force/128.0;
////  }
////
////  sprintf (filename, "%s/init/attached.dat", work_dir);
////  stream = fopen (filename, "r");
////  fscanf (stream, "%d\n", &num_vertex);
////  for(int n=0; n < num_vertex; n++)
////  {
////    fscanf (stream, "%d\n", &label[n]);    
////  }
////  fclose (stream);
////
////  for(int n=0; n < num_vertex; n++)
////  {
////    mon[ label[n] ].force[1] -= force/128.0;
////  }
//}

