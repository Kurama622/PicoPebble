#include "mpi/typeTraits.hpp"
#include <mpi.h>

using namespace DeepLearningFramework;

MPI_Datatype MPIDataTypeTrait<int>::mpi_type = MPI_INT;

MPI_Datatype MPIDataTypeTrait<int64_t>::mpi_type = MPI_INT64_T;

MPI_Datatype MPIDataTypeTrait<float>::mpi_type = MPI_FLOAT;
