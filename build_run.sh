node_num=$1
pushd build;
make;
mpirun -np $node_num ./example | tee mpi_run.txt;
popd
