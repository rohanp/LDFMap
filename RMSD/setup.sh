# setup.sh
# Make the "myext" Python Module ("myext.so")
CC="gcc"   \
CXX="g++"   \
CFLAGS="-I. -I../../../DEPENDENCIES/python2.7/inc -I../../../DEPENDENCIES/gsl-1.15"   \
LDFLAGS="L~/newLDFMap/RMSD/headers"   \
python setup.py build_ext --inplace