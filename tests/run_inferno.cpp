
#include "inferno.hpp"

using namespace inferno;

int main(int argc, char ** argv)
{
    inferno_init();

    std::cout<<"Running inferno"<<std::endl;

    inferno_finalize();

    return 0;
}
