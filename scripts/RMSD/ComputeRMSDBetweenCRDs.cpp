#include <cstdio>
#include <cstdlib>
#include <cstdarg>

#include <cstring>
#include <cctype>

#include <cmath>

#include <cassert>

#include <dirent.h>
#include <cerrno>

#include <climits>

#include <ctime>

#include <sys/stat.h>
    
#include "rmsd.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
  // svd from lapack
  void dgesvd_(char*,char*,int*,int*,double*,int*,double*,double*,int*,double*,
           int*,double*,int*,int*);
}

//usage: list_file dir_name
#define USAGE  "\nUsage: %s <refCRDFile> <compareCRDFile> <nrConfs> <RMSDFile> <AlignedCRDFile> " 
int main(int argc, char** argv)
{
    if(argc < 6)
    {
  fprintf(stdout, "%s\n", USAGE);
  exit(1);
    }
    
    char* refCRDFile     = argv[1];
    char* CRDFile        = argv[2];
    int   nrConfs        = atoi(argv[3]);
    char* RMSDFile       = argv[4];
    char* AlignedCRDFile = argv[5];    
        
    //read ref crd
    int ref_size           = 10;
    double* ref_coords = (double*) calloc(ref_size, sizeof(double));    
    FILE* refCRDFp = fopen(refCRDFile, "r");
    
    int curr_size = 0;    
    while(fscanf(refCRDFp, "%lf", ref_coords+curr_size) != EOF)
    {
  curr_size++;
  if(curr_size == ref_size)
  {
      ref_size += 10;
      ref_coords = (double*) realloc(ref_coords, ref_size * sizeof(double));    
  }
    }
    if(curr_size < ref_size)
    {
  ref_size = curr_size; 
  ref_coords = (double*) realloc(ref_coords, ref_size * sizeof(double));   
    } 
        
    fclose(refCRDFp);    

    fprintf(stdout, "%d coordinates read from reference crd file\n", ref_size);
    
    //allocate, read, compute, and write
    double** coords = (double**) calloc(nrConfs, sizeof(double*));    
    double* rmsds   = (double*) calloc(nrConfs, sizeof(double));
    
    FILE* CRDFp = fopen(CRDFile, "r");
    FILE* RMSDFp = fopen(RMSDFile, "w");
    
    for(int i =0; i < nrConfs; i++)
    {
        coords[i] = (double*) calloc(ref_size, sizeof(double));
        for(int j = 0; j < ref_size; j++)
            assert(fscanf(CRDFp, "%lf", (coords[i])+j) != EOF);
  
        rmsds[i] = rmsd(ref_size, ref_coords, coords[i]);
        fprintf(RMSDFp, "%4.3f\n", rmsds[i]);
    }

    fclose(CRDFp);
    fclose(RMSDFp); 
    
    fprintf(stdout, "%d frames/confs now aligned\n", nrConfs);
    
    int n_dofs = ref_size/3;
    assert(ref_size == n_dofs * 3);
    //otherwise, these are not atom x,y,z coordinates
 

    free(ref_coords);
    for(int i = 0; i < nrConfs; i++)
  free(coords[i]);
    free(coords);
    free(rmsds);

    return 0;
}

// n: number of degrees of freedom of each point
// x, y: pointers to arrays with the points
// compute rmsd between two conformations after alignment. If the rmsd
// exceeds threshold, the special value INF is returned.
double rmsd(int n, double* x, double* y)
{
  int i;
  double comx[3], comy[3], C[9], *v, *w, *e,*w_prime;
  static int three=3;
  
  // compute centers of mass
  comx[0]=comx[1]=comx[2]=comy[0]=comy[1]=comy[2]=0.;
  for (i=0; i<n; i+=3) {
    comx[0]+=x[i]; comx[1]+=x[i+1]; comx[2]+=x[i+2];
    comy[0]+=y[i]; comy[1]+=y[i+1]; comy[2]+=y[i+2];
  }
  comx[0]*=3./n; comx[1]*=3./n; comx[2]*=3./n;
  comy[0]*=3./n; comy[1]*=3./n; comy[2]*=3./n;

  // compute covariance matrix
  memset(C,0,9*sizeof(double));
  v = new double[n];
  w = new double[n];
  for (i=0; i<n; i+=3) {
    v[i] = x[i]-comx[0]; v[i+1] = x[i+1]-comx[1]; v[i+2] = x[i+2]-comx[2];
    w[i] = y[i]-comy[0]; w[i+1] = y[i+1]-comy[1]; w[i+2] = y[i+2]-comy[2];
    C[0] += v[i]*w[i];   C[1] += v[i]*w[i+1];   C[2] += v[i]*w[i+2];
    C[3] += v[i+1]*w[i]; C[4] += v[i+1]*w[i+1]; C[5] += v[i+1]*w[i+2];
    C[6] += v[i+2]*w[i]; C[7] += v[i+2]*w[i+1]; C[8] += v[i+2]*w[i+2];
  }

  // compute SVD of C
  int lwork=30, info=0;
  double S[3], U[9], VT[9], work[30];
  dgesvd_("A","A",&three,&three,C,&three,S,U,&three,VT,&three,work,&lwork,
      &info);

  // compute rotation: rot=U*VT
  double rot[9];
  rot[0] = U[0]*VT[0] + U[3]*VT[1] + U[6]*VT[2];
  rot[1] = U[1]*VT[0] + U[4]*VT[1] + U[7]*VT[2];
  rot[2] = U[2]*VT[0] + U[5]*VT[1] + U[8]*VT[2];
  rot[3] = U[0]*VT[3] + U[3]*VT[4] + U[6]*VT[5];
  rot[4] = U[1]*VT[3] + U[4]*VT[4] + U[7]*VT[5];
  rot[5] = U[2]*VT[3] + U[5]*VT[4] + U[8]*VT[5];
  rot[6] = U[0]*VT[6] + U[3]*VT[7] + U[6]*VT[8];
  rot[7] = U[1]*VT[6] + U[4]*VT[7] + U[7]*VT[8];
  rot[8] = U[2]*VT[6] + U[5]*VT[7] + U[8]*VT[8];
  // make sure rot is a proper rotation, check determinant
  if ((rot[1]*rot[5]-rot[2]*rot[4])*rot[6]
      + (rot[2]*rot[3]-rot[0]*rot[5])*rot[7]
      + (rot[0]*rot[4]-rot[1]*rot[3])*rot[8] < 0) {
    rot[0] -= 2*U[6]*VT[2]; rot[1] -= 2*U[7]*VT[2]; rot[2] -= 2*U[8]*VT[2];
    rot[3] -= 2*U[6]*VT[5]; rot[4] -= 2*U[7]*VT[5]; rot[5] -= 2*U[8]*VT[5];
    rot[6] -= 2*U[6]*VT[8]; rot[7] -= 2*U[7]*VT[8]; rot[8] -= 2*U[8]*VT[8];
  }
  
  //transform w by rotation matrix
  w_prime = new double[n];
  for (i=0; i<n; i+=3) {
    w_prime[i]   = rot[0]*w[i] + rot[1]*w[i+1] + rot[2]*w[i+2];
    w_prime[i+1] = rot[3]*w[i] + rot[4]*w[i+1] + rot[5]*w[i+2];
    w_prime[i+2] = rot[6]*w[i] + rot[7]*w[i+1] + rot[8]*w[i+2];
  }
  
  //compute residuals and summ them up
  e = new double[n];
  double dist = 0.0;
  for (i=0; i<n; i+=3) {
      e[i]     = v[i] - w_prime[i];
      e[i+1]   = v[i+1] - w_prime[i+1];
      e[i+2]   = v[i+2] - w_prime[i+2];
      
      dist += e[i]*e[i] + e[i+1]*e[i+1] + e[i+2]*e[i+2];
  }
  
  //compute rmsd
  if (dist!=HUGE_VAL) 
      dist = sqrt(dist*3./n);

  //overwrite y coordinates with transformed ones
  for (i=0; i<n; i+=3) {
      y[i]   = w_prime[i];
      y[i+1] = w_prime[i+1];
      y[i+2] = w_prime[i+2];  
  }
  
  delete[] v;
  delete[] w;
  delete[] w_prime;
  delete[] e;
  
  //return rmsd
  return dist;
}

