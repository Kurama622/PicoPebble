#pragma once

#include "globalState.hpp"
#include "mpi/mpiController.hpp"
#include <iomanip>
#include <mpi.h>
#include <sstream>
#include <string>

namespace DeepLearningFramework {
inline void initialize() {
  MPIController &global_controller = globalController();
}

inline void finalize() {}

inline std::string formatString(int number) {
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0') << number;
  return ss.str();
}

} // namespace DeepLearningFramework
