#ifndef COMMON_H
#define COMMON_H


#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <memory>
#include <string>
#include <sstream>


#include <cuda.h>
#include <cublas.h>
#include <cusparse.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <mpi.h>


#define CUDA_CHECK(call) {                                                 \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " ("     \
                  << __FILE__ << ":" << __LINE__ << ")" << std::endl;      \
        exit(err);                                                         \
    }                                                                      \
}

namespace inferno {


namespace globals {

    int mpi_rank;
    int mpi_world_size;

    int n_devs;

    nvshmemx_init_attr_t attr;

}


class DeviceManager
{

public:



	/* Map this PE to a GPU */
    static void SetDevice()
    {
        cudaGetDeviceCount(&(globals::n_devs));
        cudaSetDevice(globals::mpi_rank % globals::n_devs);

        //TODO: Check to make sure only one PE per device
    }

};



void inferno_init()
{
    /* MPI */
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &(globals::mpi_rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(globals::mpi_world_size));

	/* GPU setup */
	DeviceManager::SetDevice();

    /* NVSHMEM */
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &globals::attr);


	
}


void inferno_finalize()
{
    nvshmem_finalize();
    MPI_Finalize();

}

}

#endif
