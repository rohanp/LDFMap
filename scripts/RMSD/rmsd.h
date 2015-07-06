
#ifndef RMSD_H
#define RMSD_H

// n: number of degrees of freedom of each point
// x, y: pointers to arrays with the points
// compute rmsd between two conformations/sets of points after
//alignment. If the rmsd exceeds threshold, the special value INF is
//returned.
double rmsd(int n, double* x, double* y);

#endif
