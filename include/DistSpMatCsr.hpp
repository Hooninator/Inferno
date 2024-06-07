
#ifndef SPMATCSR_HPP
#define SPMATCSR_HPP

#include "common.h"
#include "utils.hpp"
#include "ProcessCube.hpp"
#include "MPITypes.h"
#include "Log.hpp"

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


    void read_mm(const char * path)
    {
        std::vector<coo_triple> tuples;

        /* First, get nnz, dim info, and header offset */
        I * send_buf = new I[4];
        send_buf[3] = I(0);

        if (globals::mpi_rank==0) {

		    std::string line;	

            std::ifstream mm_file(path);

            while (std::getline(mm_file, line)) {

                std::cout<<line<<std::endl;

                send_buf[3] += (strlen(line.c_str()) + 1);

                // Skip header
                if (line.find('%')!=std::string::npos) continue;
                
                std::istringstream iss(line);

                // First line after header is rows, cols, nnz
                iss>>send_buf[0]>>send_buf[1]>>send_buf[2];
                break;
                    
            }

            mm_file.close();
        }

        MPI_Bcast(send_buf, 4, MPIType<I>(), 0, MPI_COMM_WORLD);

        if (globals::mpi_rank==0)
            std::cout<<send_buf[0]<<std::endl;

        rows = send_buf[0];
        cols = send_buf[1];
        nnz = send_buf[2];

        MPI_Offset header_offset = send_buf[3];

        delete[] send_buf;

        /* Begin MPI IO */
        MPI_File file_handle;
        MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);

        if (globals::mpi_rank==0)
            std::cout<<"Opened file"<<std::endl;

        /* Compute offset info */
        MPI_Offset total_bytes;
        MPI_File_get_size(file_handle, &total_bytes);

        if (globals::mpi_rank==0)
            std::cout<<"total bytes: "<<total_bytes<<std::endl;

        MPI_Offset my_offset = (header_offset) + (( ( total_bytes - header_offset ) / globals::mpi_world_size) * globals::mpi_rank);

        std::cout<<"Rank "<<globals::mpi_rank<<" offset: "<<my_offset<<std::endl;

        int num_bytes = ((total_bytes - header_offset) / globals::mpi_world_size);  
        char *buf = new char[(size_t)(num_bytes*1.5 + 1)];//*1.5 ensures we have enough space to read in edge lines
                                                          //
        if (globals::mpi_rank==0)
            std::cout<<"num_bytes: "<<num_bytes<<std::endl;

        MPI_File_read_at(file_handle, my_offset, buf, num_bytes, MPI_CHAR, MPI_STATUS_IGNORE);

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout<<"Made it past file read"<<std::endl;

#ifdef DEBUG
        logptr->OFS()<<"Partition of file"<<std::endl;
        logptr->OFS()<<std::string(buf, num_bytes)<<std::endl;
#endif

        /* Check to see if I miraculously hit a newline at the end */
        if (buf[num_bytes - 1] != '\n') {
            char buf2[num_bytes];
            MPI_File_read_at(file_handle, my_offset+num_bytes, buf2, num_bytes, MPI_CHAR, MPI_STATUS_IGNORE);

            std::string buf2_str(buf2, num_bytes);
            size_t newline_idx = buf2_str.find('\n'); //index of next newline

            strncpy(buf, buf2, newline_idx+1); //copy up to next newline into buf

            num_bytes += (newline_idx + 1);
        }
        buf[num_bytes - 1] = '\0'; 

        /* Parse my lines */
        auto local_tuples = this->parse_mm_lines(num_bytes, buf);
        delete[] buf;
        


        /* Distribute tuples according to 3D distribution */
        std::vector<std::vector<coo_triple>> send_tuples(this->world_size);

        /* Map global tuple indices to local indices */


    }


    std::vector<coo_triple> parse_mm_lines(const int num_bytes, char * buf)
    {
        std::vector<coo_triple> tuples;

        char * curr = buf;

        while (*curr!='\n') {
            if (*curr=='\0') {
                ERROR("Found null terminator when parsing mm file for some reason");
            }
            curr++; //advance until we find a newline
        }

        curr++; //advance one more byte, should be at start of line
        
        std::string buf_str(curr);
        
        size_t pos = 0;

        // Find end of this line
        size_t next_eol = buf_str.find('\n', pos);
        while ( next_eol != std::string::npos) {
            
            // Copy line into std::string 
            std::string line(curr+pos, next_eol);
            std::istringstream iss(line);

            // Make the tuple
            I row; I col; N val;
            iss>>row>>col>>val;
            tuples.emplace_back(row,col,val);

            // Advance to start of next line and find the end of the next line
            pos += (next_eol + 1);
            next_eol = buf_str.find('\n', pos);
        }

        return tuples;

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
