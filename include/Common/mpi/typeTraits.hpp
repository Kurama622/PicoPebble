#pragma once

#include <mpi.h>
namespace DeepLearningFramework {
template <typename T> struct MPIDataTypeTrait {
  static MPI_Datatype mpi_type;
};

template <> struct MPIDataTypeTrait<int> {
  static MPI_Datatype mpi_type;
};

template <> struct MPIDataTypeTrait<int64_t> {
  static MPI_Datatype mpi_type;
};

template <> struct MPIDataTypeTrait<float> {
  static MPI_Datatype mpi_type;
};

template <typename T> MPI_Datatype getMPIDataType() {
  return MPIDataTypeTrait<T>::mpi_type;
}

} // namespace DeepLearningFramework
