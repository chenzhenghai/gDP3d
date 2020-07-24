#include "CudaCompactMesh.h"
#include "CudaMesh.h"

void compactMesh(
	int& out_numofpoint,
	double*& out_pointlist,
	double*& out_weightlist,
	RealD& t_pointlist,
	RealD& t_weightlist,
	int& out_numoftriface,
	int*& out_trifacelist,
	double*& out_trifacecent,
	IntD& t_trifacelist,
	RealD& t_trifacecent,
	TriStatusD& t_tristatus,
	TetHandleD& t_tri2tetlist,
	int& out_numoftet,
	int& out_numoftet_indomain,
	int*& out_tetlist,
	tetstatus*& out_tetstatus,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus
)
{
	IntD t_sizes, t_indices, t_list;
	RealD t_list1;
	TetStatusD t_list2;
	int numberofthreads, numberofblocks;

	out_numofpoint = t_pointlist.size() / 3;
	out_pointlist = new double[3 * out_numofpoint];
	out_weightlist = new double[out_numofpoint];
	cudaMemcpy(out_pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * out_numofpoint * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_weightlist, thrust::raw_pointer_cast(&t_weightlist[0]), out_numofpoint * sizeof(double), cudaMemcpyDeviceToHost);

	int last_triface = t_tristatus.size();
	t_sizes.resize(last_triface);
	t_indices.resize(last_triface);
	thrust::fill(t_sizes.begin(), t_sizes.end(), 1);
	thrust::replace_if(t_sizes.begin(), t_sizes.end(), t_tristatus.begin(), isEmptyTri(), 0);
	thrust::exclusive_scan(t_sizes.begin(), t_sizes.end(), t_indices.begin());
	out_numoftriface = thrust::reduce(t_sizes.begin(), t_sizes.end());
	out_trifacelist = new int[3 * out_numoftriface];
	out_trifacecent = new double[3 * out_numoftriface];
	t_list.resize(3 * out_numoftriface);
	t_list1.resize(3 * out_numoftriface);

	numberofthreads = last_triface;
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelCompactTriface << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_trifacecent[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_sizes[0]),
		thrust::raw_pointer_cast(&t_indices[0]),
		thrust::raw_pointer_cast(&t_list[0]),
		thrust::raw_pointer_cast(&t_list1[0]),
		numberofthreads
		);
	cudaMemcpy(out_trifacelist, thrust::raw_pointer_cast(&t_list[0]), 3 * out_numoftriface * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_trifacecent, thrust::raw_pointer_cast(&t_list1[0]), 3 * out_numoftriface * sizeof(double), cudaMemcpyDeviceToHost);

	int last_tet = t_tetstatus.size();
	t_sizes.resize(last_tet);
	t_indices.resize(last_tet);
	thrust::fill(t_sizes.begin(), t_sizes.end(), 1);
	numberofthreads = last_tet;
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	kernelCompactTet_Phase1 << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_sizes[0]),
		numberofthreads
		);

	thrust::exclusive_scan(t_sizes.begin(), t_sizes.end(), t_indices.begin());
	out_numoftet = thrust::reduce(t_sizes.begin(), t_sizes.end());
	out_tetlist = new int[4 * out_numoftet];
	out_tetstatus = new tetstatus[out_numoftet];
	t_list.resize(4 * out_numoftet);
	t_list2.resize(out_numoftet);

	kernelCompactTet_Phase2 << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_sizes[0]),
		thrust::raw_pointer_cast(&t_indices[0]),
		thrust::raw_pointer_cast(&t_list[0]),
		thrust::raw_pointer_cast(&t_list2[0]),
		numberofthreads
		);
	cudaMemcpy(out_tetlist, thrust::raw_pointer_cast(&t_list[0]), 4 * out_numoftet * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_tetstatus, thrust::raw_pointer_cast(&t_list2[0]), out_numoftet * sizeof(tetstatus), cudaMemcpyDeviceToHost);

	int numoftets_indomain = 0;
	for (int i = 0; i < out_numoftet; i++)
	{
		if (out_tetstatus[i].isInDomain())
			numoftets_indomain++;
	}
	out_numoftet_indomain = numoftets_indomain;
}