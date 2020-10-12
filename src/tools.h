#ifndef TOOLS_H
#define TOOLS_H

#define  n_int(x)       ( (x) >= (0) ? (int)((x)+0.5) : (int)((x)-0.5))
#define  s_int(x)       ( (x) >= (0) ? (int)(x) : (int)((x)-1.0))
#define  sgn(x)         ( (x) <  (0) ? (-1) : (1))
#define  max(a, b)      ( (a) >  (b) ? (a)  : (b))
#define  min(a, b)      ( (a) <  (b) ? (a)  : (b))
#define  box(a, b)      ( (a) - s_int((a)/(b))*(b))
#define  box2(a, b)     ( (a) >= (0.0) ? (a) - (int)((a)/(b))*(b): (a)-(int)((a)/(b)-1.0)*(b))
#define  n_image(a, b)  ( (a) - n_int((a)/(b))*(b))
#define  range(x, a, b) (((x) >= (a) && (x) < (b)) ? (1) : (0))
#define  cycle(x, a)  ( (x) >= 0 ? (x) % (a) : (x) % (a) + (a) )    // syfan 151101

#define Float  double
#define TRUE   1
#define FALSE  0
//#define  M_PI  3.1415926535897932384626433832795029L 
#define  NTYPES           2          /* # of types of particles */
#define  MAX_B         8192          /* Buffer size for msg passing */
#define  MAX_L         5000          /* Max # link cells */
#define  Pi            (acos(-1.0))
#define WRITE_BUFFER     10          /* buffer for output */
#define  MAX_DP         500          /* Max # of deformable particles */
#define  MAX_W          100          /* Max # warnings */
#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

void file_name (char *name, char *work_dir, int task_number);

//void warning (char *warning_msg);

void fatal_err (char *error_msg, int flag);

//void error_chk ();

void product(double a[3], double b[3], double c[3]);

double iproduct(double a[3], double b[3]);

#endif
