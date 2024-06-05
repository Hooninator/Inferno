
#include "inferno.hpp"

using namespace inferno;

int main(int argc, char ** argv)
{
    inferno_init();

	if (upcxx::rank_me()==0)
		std::cout<<"Running inferno"<<std::endl;

	DistSpMatCsr<int64_t, double> A(2,2,1, true);


    inferno_finalize();

    return 0;
}
