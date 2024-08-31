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
#include "mpi/TypeTraits.hpp"
#include <mpi.h>

using namespace DeepLearningFramework;

MPI_Datatype MPIDataTypeTrait<int>::mpi_type = MPI_INT;

MPI_Datatype MPIDataTypeTrait<int64_t>::mpi_type = MPI_INT64_T;

MPI_Datatype MPIDataTypeTrait<uint32_t>::mpi_type = MPI_UINT32_T;

MPI_Datatype MPIDataTypeTrait<float>::mpi_type = MPI_FLOAT;
