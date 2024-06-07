
#include "inferno.hpp"

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>


using namespace inferno;

int main(int argc, char ** argv)
{
    int rank; int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    inferno_init();

	if (rank==0)
		std::cout<<"Running inferno"<<std::endl;

	DistSpMatCsr<int64_t, double> A(2,2,1, true);


    inferno_finalize();

    return 0;
}
