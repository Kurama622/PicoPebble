#pragma once

#include "typeTraits.hpp"
#include <mpi.h>
#include <string>
#include <vector>

namespace DeepLearningFramework {

class MPIController {
public:
  MPIController() : mpi_rank(-1), mpi_size(-1), mpi_comm(MPI_COMM_WORLD) {
    mpiInit();
  };

  ~MPIController() { mpiFinalize(); };

  int &mpiRank() { return mpi_rank; }

  int &mpiSize() { return mpi_size; };

  void mpiInit() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
  };

  template <typename T>
  int mpiScatterv(const std::vector<T> &sendbuf, const int counts[],
                  std::vector<T> &recvbuf, int recvcount, int root) {

    std::vector<int> displs(mpi_size);
    if (mpi_rank == 0) {
      displs[0] = 0;
      for (int i = 1; i < mpi_size; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
      }
    }

    return MPI_Scatterv(sendbuf.data(), counts, displs.data(),
                        getMPIDataType<T>(), recvbuf.data(), recvcount,
                        getMPIDataType<T>(), root, mpi_comm);
  }

  template <typename T>
  void mpiScatter(const T send_data[], int send_count, T &recv_data,
                  int recv_count, int root) {
    MPI_Scatter(send_data, send_count, getMPIDataType<T>(), &recv_data,
                recv_count, getMPIDataType<T>(), root, mpi_comm);
  }

  void mpiFinalize() { MPI_Finalize(); };

private:
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm;
};
} // namespace DeepLearningFramework
