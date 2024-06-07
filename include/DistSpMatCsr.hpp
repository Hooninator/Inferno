
#ifndef SPMATCSR_HPP
#define SPMATCSR_HPP

#include "common.h"
#include "utils.hpp"
#include "ProcessCube.hpp"

namespace inferno{

template <typename I, typename N>
class DistSpMatCsr 
{
public:



	// row, column, val
	using coo_triple = std::tuple<I, I, N>;

    DistSpMatCsr(int proc_rows, int proc_cols, int proc_layers, bool row_split)
    {

        /* Process cube setup */
        assert(proc_rows==proc_cols);
        assert(is_pow_2(proc_rows*proc_cols));

        proc_cube.reset(new ProcessCube(proc_rows, proc_cols, proc_layers));

        MPI_Comm_rank( MPI_COMM_WORLD, &(this->rank));
        MPI_Comm_size( MPI_COMM_WORLD, &(this->world_size));

        assert(world_size == proc_cube->get_total_procs());

        row_split = row_split;

    }


    void read_mm(std::string& path)
    {
        std::vector<coo_triple> tuples;

        std::ifstream mm_file(path.c_str());

        std::string line;

        bool header_done = false;

        while (std::getline(mm_file, line)) {

            // Skip header
            if (line.find("%")) continue;
            
            std::istringstream iss(line);

            // First line after header is rows, cols, nnz
            if (!header_done) {
                iss>>rows>>cols>>nnz;
                header_done = true;
                continue;
            }
                

            I row; 
            I col;
            N val;

            iss >> row >> col >> val;
            tuples.emplace_back(row, col, val);

        }

        mm_file.close();

        /* Distribute tuples according to 3D distribution */

        if (row_split) {

            std::vector<std::vector<coo_triple>> send_tuples(this->world_size);

        } else {

        }


    }


    int map_triple(coo_triple& triple)
    {
        if (row_split) {

            I row = std::get<0>(triple);
            I col = std::get<1>(triple);

        } else {

        }
    }


    void local_spgemm() {/* TODO */}

    
    void get_remote_tile()
    {
        /* TODO */
    }


private:

    // rank info
    int rank;
    int world_size;

    // Process cube 
    std::shared_ptr<ProcessCube> proc_cube;

    // Global matrix info
    I rows;
    I cols;
    I nnz;
    
    // distributed objects containing pointers to csr arrays

    // Am I split along rows or columns?
    bool row_split;


};


}//inferno

#endif
