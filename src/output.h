#ifndef OUTPUT_H
#define OUTPUT_H

#include "sphere_param.h"
#include "face.h"

void WriteParticleVTK (int step, char *work_dir, struct sphere_param params, struct face *faces, double *h_foldedPos);

#endif

