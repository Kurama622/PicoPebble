node_num=$1
cd build;
make;
mpirun -np $node_num ./example | tee mpi_run.txt;
cd -
