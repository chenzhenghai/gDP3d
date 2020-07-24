#include "CudaInitStatus.h"
#include "CudaMesh.h"
#include "CudaAnimation.h"
#include <time.h>

void initTriTetQuality(
	RealD& t_pointlist,
	PointTypeD& t_pointtypelist,
	RealD& t_weightlist,
	IntD& t_trifacelist,
	RealD& t_trifacecent,
	TriStatusD& t_tristatus,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	MESHCR* criteria,
	int numoftrifaces,
	int numoftets
)
{
	// Init triface quality
	int numberofblocks = (ceil)((float)numoftrifaces / BLOCK_SIZE);
	kernelInitTriQuality << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_weightlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_trifacecent[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		criteria->facet_angle,
		criteria->facet_size,
		criteria->facet_distance,
		numoftrifaces
		);

	// Init tet quality
	numberofblocks = (ceil)((float)numoftets / BLOCK_SIZE);
	kernelInitTetQuality << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_weightlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		criteria->cell_radius_edge_ratio,
		criteria->cell_size,
		numoftets
		);
}

// Init triface and its neighbors
void initTristatus(
	IntD& t_aabbnodeleft,
	IntD& t_aabbnoderight,
	RealD& t_aabbnodebbs,
	RealD& t_aabbpmcoord,
	RealD& t_aabbpmbbs,
	RealD& t_pointlist,
	PointTypeD& t_pointtypelist,
	RealD& t_weightlist,
	IntD& t_trifacelist,
	RealD& t_trifacecent,
	TriStatusD& t_tristatus,
	TetHandleD& t_tri2tetlist,
	IntD& t_tetlist,
	TriHandleD& t_tet2trilist,
	TetHandleD& t_neighborlist,
	MESHCR* criteria,
	int& numoftrifaces,
	int numoftets
)
{

#ifdef GQM3D_INIT_PROFILING
	clock_t tv[2];
	tv[0] = clock();
#endif

	IntD trifacecount(4 * numoftets, 0);
	IntD trifaceindices(4 * numoftets, 0);
	RealD trifaceipt(4 * 3 * numoftets);

#ifdef GQM3D_INIT_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        initTristatus vector initialization time = %f\n", (REAL)(tv[1] - tv[0]));
	tv[0] = tv[1];
#endif

	// Mark initial trifaces
	int numberofblocks = (ceil)((float)numoftets / BLOCK_SIZE);
	kernelMarkInitialTrifaces << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_aabbnodeleft[0]),
		thrust::raw_pointer_cast(&t_aabbnoderight[0]),
		thrust::raw_pointer_cast(&t_aabbnodebbs[0]),
		thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
		thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_weightlist[0]),
		thrust::raw_pointer_cast(&trifacecount[0]),
		thrust::raw_pointer_cast(&trifaceipt[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		numoftets
		);

#ifdef GQM3D_INIT_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelMarkInitialTrifaces time = %f\n", (REAL)(tv[1] - tv[0]));
	tv[0] = tv[1];
#endif

	// Set up memory
	numoftrifaces = thrust::count(trifacecount.begin(), trifacecount.end(), 1);
	printf("        Number of trifaces = %d\n", numoftrifaces);
	t_trifacelist.resize(3 * numoftrifaces);
	t_trifacecent.resize(3 * numoftrifaces);
	t_tristatus.resize(numoftrifaces, tristatus(1));
	t_tri2tetlist.resize(2 * numoftrifaces);
	thrust::exclusive_scan(trifacecount.begin(), trifacecount.end(), trifaceindices.begin());

#ifdef GQM3D_INIT_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        initTristatus prepare vector time = %f\n", (REAL)(tv[1] - tv[0]));
	tv[0] = tv[1];
#endif

	// Append triface lists
	kernelAppendInitialTrifaces << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&trifacecount[0]),
		thrust::raw_pointer_cast(&trifaceindices[0]),
		thrust::raw_pointer_cast(&trifaceipt[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_weightlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_trifacecent[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		criteria->facet_angle,
		criteria->facet_size,
		criteria->facet_distance,
		numoftets
		);

#ifdef GQM3D_INIT_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelAppendInitialTrifaces time = %f\n", (REAL)(tv[1] - tv[0]));
#endif

	int numofbadfacet = thrust::count_if(t_tristatus.begin(), t_tristatus.end(), isBadTri());
	printf("        Number of bad trifaces = %d\n", numofbadfacet);
}

void initTetstatus(
	IntD& t_aabbnodeleft,
	IntD& t_aabbnoderight,
	RealD& t_aabbnodebbs,
	RealD& t_aabbpmcoord,
	RealD& t_aabbpmbbs,
	RealD& t_pointlist,
	RealD& t_weightlist,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	MESHCR* criteria,
	REAL aabb_diglen,
	int numoftets
)
{

#ifdef GQM3D_INIT_PROFILING
	clock_t tv[2];
	tv[0] = clock();
#endif

	// Init tet status
	int numberofblocks = (ceil)((float)numoftets / BLOCK_SIZE);
	kernelInitTetStatus << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_aabbnodeleft[0]),
		thrust::raw_pointer_cast(&t_aabbnoderight[0]),
		thrust::raw_pointer_cast(&t_aabbnodebbs[0]),
		thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
		thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_weightlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		criteria->cell_radius_edge_ratio,
		criteria->cell_size,
		aabb_diglen,
		numoftets
		);

#ifdef GQM3D_INIT_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelInitTetStatus time = %f\n", (REAL)(tv[1] - tv[0]));
	int numofcells = thrust::count_if(t_tetstatus.begin(), t_tetstatus.end(), isTetInDomain());
	printf("        Number of tets in domain = %d\n", numofcells);
	int numofbadcells = thrust::count_if(t_tetstatus.begin(), t_tetstatus.end(), isBadTet());
	printf("        Number of bad tets in domain = %d\n", numofbadcells);
#endif
}

void initTetstatus2(
	IntD& t_aabbnodeleft,
	IntD& t_aabbnoderight,
	RealD& t_aabbnodebbs,
	RealD& t_aabbpmcoord,
	RealD& t_aabbpmbbs,
	RealD& t_pointlist,
	RealD& t_weightlist,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	MESHCR* criteria,
	MESHIO* inputmesh,
	int numoftets
)
{
	//gpuMemoryCheck();

#ifdef GQM3D_INIT_PROFILING
	clock_t tv[2];
	tv[0] = clock();
#endif

	int curhandlesize;
	TetHandleD::iterator last_iterator;
	IntD t_domaincount(numoftets, 0);// counter for in/out domain test
	TetHandleD t_domainhandle;
	RealD t_domainsegment;
	IntD t_domainthreadlist;
	IntD t_domainnode;
	int numberofthreads;
	int numberofblocks;
	int basenum = 256;
	int winsize = 3000 * basenum;

	double free_mb;
	getFreeMemory(free_mb);
#ifdef GQM3D_INIT_DEBUG
	printf("free_mb = %lf\n", free_mb);
#endif
	int maxreserve = 3 * 100 * 1000 * 1000;
	int reservesize = free_mb * 1024 * 1024 / 3 / sizeof(tethandle);
	if (reservesize > maxreserve)
		reservesize = maxreserve;
#ifdef GQM3D_INIT_DEBUG
	printf("reservesize = %d\n", reservesize);
#endif
	try
	{
		t_domainhandle.reserve(reservesize);
		t_domainnode.reserve(reservesize);
	}
	catch (thrust::system_error &e)
	{
		// output an error message and exit
		std::cerr << "Error: " << e.what() << std::endl;
		exit(-1);
	}

	//gpuMemoryCheck();

	int offset = 0;
	while (true)
	{
		numberofthreads = numoftets - offset;
		if (numberofthreads > winsize)
			numberofthreads = winsize;
		else if (numberofthreads <= 0)
			break;

#ifdef GQM3D_INIT_DEBUG
		printf("numberofthreads = %d, offset = %d, winsize = %d\n", numberofthreads, offset, winsize);
#endif

		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		t_domainhandle.resize(numberofthreads);
		thrust::fill(t_domainhandle.begin(), t_domainhandle.begin() + numberofthreads, tethandle(-1, 11));
		kernelInitDomainHandle_Tet << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_domainhandle[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			offset,
			numberofthreads
			);

#ifdef GQM3D_INIT_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		last_iterator =
			thrust::remove_if(t_domainhandle.begin(), t_domainhandle.begin() + numberofthreads, isInvalidTetHandle());
		curhandlesize = thrust::distance(t_domainhandle.begin(), last_iterator);
#ifdef GQM3D_INIT_DEBUG
		printf("curhandlesize = %d\n", curhandlesize);
#endif
		try
		{
			t_domainsegment.resize(6 * curhandlesize);
			t_domainthreadlist.resize(curhandlesize); // thread indice list to store new tet thread indice
		}
		catch (thrust::system_error &e)
		{
			// output an error message and exit
			std::cerr << "Error: " << e.what() << std::endl;
			exit(-1);
		}

		numberofthreads = curhandlesize;
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelInitDomainSegment_Tet << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_aabbnodebbs[0]),
			thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
			thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
			thrust::raw_pointer_cast(&t_domainhandle[0]),
			thrust::raw_pointer_cast(&t_domainsegment[0]),
			thrust::raw_pointer_cast(&t_domainthreadlist[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_weightlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			inputmesh->aabb_diglen,
			numberofthreads
			);

#ifdef GQM3D_INIT_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		last_iterator = // remove degenerate cases and fast check cases
			thrust::remove_if(t_domainhandle.begin(), t_domainhandle.begin() + numberofthreads, isInvalidTetHandle());
		curhandlesize = thrust::distance(t_domainhandle.begin(), last_iterator);
#ifdef GQM3D_INIT_DEBUG
		printf("curhandlesize = %d\n", curhandlesize);
#endif
		if (curhandlesize == 0)
			return;
		t_domainnode.resize(t_domainhandle.size());
		thrust::fill(t_domainnode.begin(), t_domainnode.begin() + curhandlesize, 1);
		numberofthreads = curhandlesize;
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

		int numofemptyhandleslot;
		int domainexpanditer = 0;
		bool halfwinsize = false;
		while (true)
		{
#ifdef GQM3D_INIT_DEBUG
			printf("Domain search iteration = %d, curhandlesize = %d\n", domainexpanditer, curhandlesize);
#endif
			kernelDomainSegmentAndBoxCheck << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_aabbnodebbs[0]),
				thrust::raw_pointer_cast(&t_domainhandle[0]),
				thrust::raw_pointer_cast(&t_domainnode[0]),
				thrust::raw_pointer_cast(&t_domainsegment[0]),
				numberofthreads
				);

			// remove the handles that do not intersect with node bounding boxes 
			typedef thrust::zip_iterator<thrust::tuple<TetHandleD::iterator, IntD::iterator>> ZipIterator;
			ZipIterator first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_domainhandle.begin(), t_domainnode.begin()));
			auto last_iterator =
				thrust::remove_if(first_iterator,
					thrust::make_zip_iterator(thrust::make_tuple(t_domainhandle.begin() + numberofthreads, t_domainnode.begin() + numberofthreads)),
					isInvalidDomainTuple());
			curhandlesize = thrust::distance(first_iterator, last_iterator);
			if (curhandlesize == 0)
				break;

			if (domainexpanditer == inputmesh->aabb_level)
				break;

			// prepare enough space for new handles and nodes
			numofemptyhandleslot = t_domainhandle.size() - curhandlesize;
			if (numofemptyhandleslot < curhandlesize)
			{
				try
				{
					if (2 * curhandlesize > reservesize) // possible to run out of memory
					{
						// half the window size
#ifdef GQM3D_INIT_DEBUG
						printf("half the window size\n");
#endif
						winsize /= 2;
						halfwinsize = true;
						break;
					}

					t_domainhandle.resize(2 * curhandlesize);
					t_domainnode.resize(2 * curhandlesize);
				}
				catch (thrust::system_error &e)
				{
					// output an error message and exit
					std::cerr << "Error: " << e.what() << std::endl;
					exit(-1);
				}
			}

			thrust::fill(t_domainhandle.begin() + curhandlesize, t_domainhandle.begin() + 2 * curhandlesize, tethandle(-1, 11));
			thrust::fill(t_domainnode.begin() + curhandlesize, t_domainnode.begin() + 2 * curhandlesize, 0);

			numberofthreads = curhandlesize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelDomainHandleAppend << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_aabbnodeleft[0]),
				thrust::raw_pointer_cast(&t_aabbnoderight[0]),
				thrust::raw_pointer_cast(&t_domainhandle[0]),
				thrust::raw_pointer_cast(&t_domainnode[0]),
				numberofthreads
				);

			curhandlesize = 2 * curhandlesize;
			numberofthreads = curhandlesize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

			domainexpanditer++;
		}

		if (halfwinsize)
			continue;

		numberofthreads = curhandlesize;
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelDomainSegmentAndPrimitiveCheck_Tet << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
			thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
			thrust::raw_pointer_cast(&t_domainhandle[0]),
			thrust::raw_pointer_cast(&t_domainnode[0]),
			thrust::raw_pointer_cast(&t_domainsegment[0]),
			thrust::raw_pointer_cast(&t_domaincount[0]),
			thrust::raw_pointer_cast(&t_domainthreadlist[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_weightlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			numberofthreads
			);

#ifdef GQM3D_INIT_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		numberofthreads = numoftets - offset;
		if (numberofthreads > winsize)
			numberofthreads = winsize;
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelSetTetStatus << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_domaincount[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_weightlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			criteria->cell_radius_edge_ratio,
			criteria->cell_size,
			offset,
			numberofthreads
			);

#ifdef GQM3D_INIT_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		offset += winsize;
	}

#ifdef GQM3D_INIT_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        Tet in/out domain test time = %f\n", (REAL)(tv[1] - tv[0]));
#endif
}