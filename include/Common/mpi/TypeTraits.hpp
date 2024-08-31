///////////////////////////////////////////////////////////////////////////
//
// PicoPebble - A lightweight distributed machine learning training framework for beginners
//
///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024. All rights reserved.
//
// Licensed under the MIT License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://opensource.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////////
#pragma once

#include <mpi.h>
#include <stdint.h>

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

template <> struct MPIDataTypeTrait<uint32_t> {
  static MPI_Datatype mpi_type;
};

template <> struct MPIDataTypeTrait<float> {
  static MPI_Datatype mpi_type;
};

template <typename T> MPI_Datatype getMPIDataType() {
  return MPIDataTypeTrait<T>::mpi_type;
}

} // namespace DeepLearningFramework
