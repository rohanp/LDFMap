/*
  This file has my implementation of the LAPACK routine dgesdd for
  C++.  This program solves for the singular value decomposition of a
  rectangular matrix A.  The function call is of the form

    void dgesdd(double **A, int m, int n, double *S, double *U, double *VT)

    A: the m by n matrix that we are decomposing
    m: the number of rows in A
    n: the number of columns in A (generally, n<m)
    S: a min(m,n) element array to hold the singular values of A
    U: a [m, min(m,n)] element rectangular array to hold the right
       singular vectors of A.  These vectors will be the columns of U,
       so that U[i][j] is the ith element of vector j.
    VT: a [min(m,n), n] element rectangular array to hold the left
        singular vectors of A.  These vectors will be the rows of VT
	(it is a transpose of the vector matrix), so that VT[i][j] is
	the jth element of vector i.

  Note that S, U, and VT must be initialized before calling this
  routine, or there will be an error.  Here is a quick sample piece of
  code to perform this initialization; in many cases, it can be lifted
  right from here into your program.
  
    S = new double[minmn];
    U = new double*[m]; for (int i=0; i<m; i++) U[i] = new double[minmn];
    VT = new double*[minmn]; for (int i=0; i<minmn; i++) VT[i] = new double[n];

  Scot Shaw
  24 January 2000 */

void dgesvd(double **A, int m, int n, double *S, double **U, double **VT);

double *dgesvd_ctof(double **in, int rows, int cols);
void dgesvd_ftoc(double *in, double **out, int rows, int cols);

extern "C" void dgesvd_(char *jobu, char *jobvt, int *m, int *n,
			double *a, int *lda, double *s, double *u,
			int *ldu, double *vt, int *ldvt, double *work,
			int *lwork, int *info);

void dgesvd(double **A, int m, int n, double *S, double **U, double **VT)
{
  char jobu, jobvt;
  int lda, ldu, ldvt, lwork, info;
  double *a, *u, *vt, *work;

  int minmn, maxmn;

  jobu = 'S'; /* Specifies options for computing U.
		 A: all M columns of U are returned in array U;
		 S: the first min(m,n) columns of U (the left
		    singular vectors) are returned in the array U;
		 O: the first min(m,n) columns of U (the left
		    singular vectors) are overwritten on the array A;
		 N: no columns of U (no left singular vectors) are
		    computed. */

  jobvt = 'S'; /* Specifies options for computing VT.
		  A: all N rows of V**T are returned in the array
		     VT;
		  S: the first min(m,n) rows of V**T (the right
		     singular vectors) are returned in the array VT;
		  O: the first min(m,n) rows of V**T (the right
		     singular vectors) are overwritten on the array A;
		  N: no rows of V**T (no right singular vectors) are
		     computed. */

  lda = m; // The leading dimension of the matrix a.
  a = dgesvd_ctof(A, lda, n); /* Convert the matrix A from double pointer
			  C form to single pointer Fortran form. */

  ldu = m;

  /* Since A is not a square matrix, we have to make some decisions
     based on which dimension is shorter. */

  if (m>=n) { minmn = n; maxmn = m; } else { minmn = m; maxmn = n; }

  ldu = m; // Left singular vector matrix
  u = new double[ldu*minmn];

  ldvt = minmn; // Right singular vector matrix
  vt = new double[ldvt*n];

  lwork = 5*maxmn; // Set up the work array, larger than needed.
  work = new double[lwork];

  dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, S, u,
	  &ldu, vt, &ldvt, work, &lwork, &info);

  dgesvd_ftoc(u, U, ldu, minmn);
  dgesvd_ftoc(vt, VT, ldvt, n);
  
  delete a;
  delete u;
  delete vt;
  delete work;
}

double* dgesvd_ctof(double **in, int rows, int cols)
{
  double *out;
  int i, j;

  out = new double[rows*cols];
  for (i=0; i<rows; i++) for (j=0; j<cols; j++) out[i+j*rows] = in[i][j];
  return(out);
}

void dgesvd_ftoc(double *in, double **out, int rows, int cols)
{
  int i, j;

  for (i=0; i<rows; i++) for (j=0; j<cols; j++) out[i][j] = in[i+j*rows];
}
