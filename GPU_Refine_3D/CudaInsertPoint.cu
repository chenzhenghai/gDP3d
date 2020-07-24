#include "CudaInsertPoint.h"
#include "CudaMesh.h"
#include "CudaAnimation.h"
#include <math_constants.h>
#include <time.h>

#ifdef GQM3D_PROFILING
	std::map<std::string, double> inserttimer;
#endif

#ifdef GQM3D_LOOP_PROFILING
	std::map<std::string, double> looptimer;
#endif

int insertPoint(
	IntD& t_aabbnodeleft,
	IntD& t_aabbnoderight,
	RealD& t_aabbnodebbs,
	RealD& t_aabbpmcoord,
	RealD& t_aabbpmbbs,
	RealD& t_pointlist,
	RealD& t_weightlist,
	PointTypeD& t_pointtypelist,
	IntD& t_pointpmt,
	IntD& t_trifacelist,
	RealD& t_trifacecent,
	TetHandleD& t_tri2tetlist,
	TriStatusD& t_tristatus,
	IntD& t_trifacepmt,
	IntD& t_tetlist,
	TetHandleD& t_neighborlist,
	TriHandleD& t_tet2trilist,
	TetStatusD& t_tetstatus,
	IntD& t_insertidxlist,
	IntD& t_threadmarker,
	int numofbadelements,
	int numofbadtriface,
	int numofbadtet,
	int& numofpoints,
	int& numoftriface,
	int& numoftet,
	MESHCR* criteria,
	MESHIO* inputmesh,
	MESHBH* behavior,
	int insertmode, // 0: subface only, 1: mix subface and tet
	int iter
)
{
#ifdef GQM3D_OUTPUTMESH
	//if (iter == 0)
	//{
	//	outputAllFacets(
	//		t_pointlist,
	//		t_trifacelist,
	//		t_tristatus,
	//		t_tri2tetlist,
	//		t_tetstatus,
	//		"animation/Jacket_facet_after_edge_protection_gpu",
	//		-1
	//	);

	//	outputInternalMesh(
	//		t_pointlist,
	//		t_trifacelist,
	//		t_tristatus,
	//		t_tetlist,
	//		t_tetstatus,
	//		"animation/Jacket_facet_after_edge_protection_gpu",
	//		-1
	//	);
	//}
#endif

#ifdef GQM3D_CHECKMEMORY
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

#ifdef GQM3D_PROFILING
	clock_t tv[2];
	cudaDeviceSynchronize();
	tv[0] = clock();
#endif

#ifdef GQM3D_LOOP_PROFILING
	clock_t ltv[2];
#endif

	internalmesh* drawmesh = behavior->drawmesh;
	int numofinsertpt = numofbadelements;
	REAL aabb_diglen = inputmesh->aabb_diglen;
	int aabb_level = inputmesh->aabb_level;
	int aabb_pmnum = inputmesh->numofaabbpms;

	// Initialization
	int numberofwonfacets;
	int numberofnewtets_facet;
	int numberofthreads;
	int numberofblocks;
	IntD t_threadlist; // active thread list
	UInt64D t_tetmarker(numoftet, 0); // marker for tets. Used for cavity.

	RealD t_insertptlist(3*numofinsertpt);
	IntD t_priority(numofinsertpt, 0);
	RealD t_priorityreal(numofinsertpt, 0.0); // store real temporarily

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        vector initialization time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["vector initialization"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Compute Steiner points and priorities
	numberofblocks = (ceil)((float)numofinsertpt / BLOCK_SIZE);
	kernelComputePriorities << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_priorityreal[0]),
		numofinsertpt
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelComputePriorities time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelComputePriorities"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Sort element indices by priorites and
	// pick the first N elements where N = behavior->maxbadelements
	if (behavior->filtermode == 2 && behavior->maxbadelements > 0 && numofbadelements > behavior->maxbadelements)
	{
		if (behavior->filterstatus == 1)
			behavior->filterstatus = 2;

		int numberofloser;
		if (numofbadtriface > numofbadtet)
		{
			numberofloser = numofbadtriface - behavior->maxbadelements;
			if (numberofloser > 0)
			{
				thrust::sort_by_key(t_insertidxlist.begin(), t_insertidxlist.begin() + numofbadtriface,
					t_priorityreal.begin());
				thrust::fill(t_threadmarker.begin(), t_threadmarker.begin() + numberofloser, -1);
			}
		}
		else
		{
			numberofloser = numofbadtet - behavior->maxbadelements;
			if (numberofloser > 0)
			{
				thrust::sort_by_key(t_insertidxlist.begin() + numofbadtriface, t_insertidxlist.end(),
					t_priorityreal.begin() + numofbadtriface);
				thrust::fill(t_threadmarker.begin() + numofbadtriface, t_threadmarker.begin() + numofbadtriface + numberofloser, -1);
			}
		}

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        fast filtering - sorting time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["fast filtering - sorting"] += (REAL)(tv[1] - tv[0]);
		tv[0] = tv[1];
#endif
	}
	else
	{
		if (behavior->filterstatus == 2)
			behavior->filterstatus = 3;
	}

	kernelComputeSteinerPoints << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_weightlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_trifacecent[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		numofinsertpt
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelComputeSteinerPoints time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelComputeSteinerPoints"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Modify priorities and convert them into integers
	// Make sure triface > tet
	double priority_min[2], priority_max[2], priority_offset[2] = {0, 0};
	thrust::pair<RealD::iterator, RealD::iterator> priority_pair;
	if (numofbadtet > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_priorityreal.begin() + numofbadtriface,
				t_priorityreal.end());
		priority_min[1] = *priority_pair.first;
		priority_max[1] = *priority_pair.second;
		priority_offset[1] = 0;
#ifdef GQM3D_DEBUG
		printf("MinMax Real priorities for tet: %lf, %lf\n", priority_min[1], priority_max[1]);
		printf("Offset: %lf\n", priority_offset[1]);
#endif
	}

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        min-max tet priority time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["min-max tet priority"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	if (numofbadtriface > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_priorityreal.begin(),
				t_priorityreal.begin() + numofbadtriface);
		priority_min[0] = *priority_pair.first;
		priority_max[0] = *priority_pair.second;
		if (numofbadtet > 0)
			priority_offset[0] = priority_max[1] + priority_offset[1] + 10 - priority_min[0];
		else
			priority_offset[0] = 0;
#ifdef GQM3D_DEBUG
			printf("MinMax Real priorities for subface: %lf, %lf\n", priority_min[0], priority_max[0]);
			printf("Offset: %lf\n", priority_offset[0]);
#endif
	}

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        min-max subface priority time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["min-max subface priority"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	kernelModifyPriority << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_priorityreal[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		priority_offset[0],
		priority_offset[1],
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numofbadtriface,
		numofinsertpt
		);

#ifdef GQM3D_DEBUG
	if (numofbadtriface > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_priorityreal.begin(),
				t_priorityreal.begin() + numofbadtriface);
		priority_min[0] = *priority_pair.first;
		priority_max[0] = *priority_pair.second;
		printf("MinMax Real priorities for subface: %lf, %lf\n", priority_min[0], priority_max[0]);
	}

	if (numofbadtet > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_priorityreal.begin() + numofbadtriface,
				t_priorityreal.end());
		priority_min[1] = *priority_pair.first;
		priority_max[1] = *priority_pair.second;
		printf("MinMax Real priorities for tet: %lf, %lf\n", priority_min[1], priority_max[1]);
	}

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelModifyPriority time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelModifyPriority"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	freeVec(t_priorityreal);

	if (behavior->filtermode == 2 && behavior->maxbadelements > 0 && numofbadelements > behavior->maxbadelements)
	{
		int gridlength = 150; // ^3
		int totalgridsize = gridlength * gridlength * gridlength;
		if (t_tetmarker.size() < totalgridsize)
			t_tetmarker.resize(totalgridsize, 0);

		int range_left, range_right;
		if (numofbadtriface > numofbadtet)
		{
			range_left = 0;
			range_right = numofbadtriface;
		}
		else
		{
			range_left = numofbadtriface;
			range_right = numofbadelements;
		}

		double step_x = (inputmesh->aabb_xmax - inputmesh->aabb_xmin) / gridlength;
		double step_y = (inputmesh->aabb_ymax - inputmesh->aabb_ymin) / gridlength;
		double step_z = (inputmesh->aabb_zmax - inputmesh->aabb_zmin) / gridlength;

		kernelGridFiltering << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_priority[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			range_left,
			range_right,
			step_x,
			step_y,
			step_z,
			inputmesh->aabb_xmin,
			inputmesh->aabb_ymin,
			inputmesh->aabb_zmin,
			gridlength
			);

		thrust::fill(t_tetmarker.begin(), t_tetmarker.begin() + totalgridsize, 0);
#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        fast filtering - grid time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["fast filtering - grid"] += (REAL)(tv[1] - tv[0]);
		tv[0] = tv[1];
#endif
	}

	// Update working thread list
	numberofthreads = updateActiveListByMarker(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (numberofthreads == 0)
		return 0;

	if (behavior->verbose >= 1)
		printf("        After Steiner point and priority, numberofthreads = %d(#%d, #%d, #%d)\n", 
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));

#ifdef GQM3D_CHECKMEMORY
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        update working thread list time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["update working thread list"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Locate Steiner points
	thrust::device_vector<locateresult> t_pointlocation(numofinsertpt, UNKNOWN);
	TetHandleD t_searchtet(numofinsertpt, tethandle(-1, 11));

	kernelLocatePoint << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_weightlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		thrust::raw_pointer_cast(&t_pointlocation[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelLocatePoint time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelLocatePoint"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// update working thread list
	numberofthreads = updateActiveListByMarker(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (behavior->verbose >= 1)
		printf("        After point location, numberofthreads = %d(#%d, #%d, #%d)\n", 
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
		return 0;

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        update working thread list time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["update working thread list"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	if (drawmesh != NULL && drawmesh->animation)
	{
		if(iter == drawmesh->iter_tet)
			outputStartingFrame(
				drawmesh,
				t_pointlist,
				t_tetlist,
				t_tetstatus,
				t_threadlist,
				t_insertidxlist,
				t_insertptlist,
				t_searchtet,
				-1,
				-1,
				iter
			);
	}

#ifdef GQM3D_CHECKMEMORY
	//cudaDeviceSetLimit(cudaLimitStackSize, 0); // free memory used by kernel
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

	//IntD t_linklistcur(numofinsertpt, -1);
	IntD t_cavethreadidx;
	//IntD t_threadfinishmarker(numofinsertpt);

	TetHandleD t_caveoldtetlist; // linklist to record interior tets
	IntD t_caveoldtetprev, t_caveoldtetnext;
	IntD t_caveoldtethead(numofinsertpt, -1), t_caveoldtettail(numofinsertpt, -1);
	TetHandleD t_cavetetlist; // linklist to record tets in expanding cavities
	IntD t_cavetetprev, t_cavetetnext;
	IntD t_cavetethead(numofinsertpt, -1), t_cavetettail(numofinsertpt, -1);

	// Adatively reserve memory space
	if (!behavior->R5)
	{
		// size and fac would fluctuate
		if (behavior->caveoldtetsizefac > 3.0)
			behavior->caveoldtetsizefac = 1.5;
		if (behavior->cavetetsizefac > 3.0)
			behavior->cavetetsizefac = 1.5;
		if (behavior->cavebdrysizefac > 3.0)
			behavior->cavebdrysizefac = 1.5;
	}
	int resoldtetsize = behavior->caveoldtetsize * behavior->caveoldtetsizefac;
	//printf("caveoldtetsize = %d, caveoldtetsizefac = %lf, resoldtetsize = %d\n",
	//	behavior->caveoldtetsize, behavior->caveoldtetsizefac, resoldtetsize);
	t_caveoldtetlist.reserve(resoldtetsize);
	t_caveoldtetprev.reserve(resoldtetsize);
	t_caveoldtetnext.reserve(resoldtetsize);
	int restetsize = behavior->cavetetsize * behavior->cavetetsizefac;
	//printf("cavetetsize = %d, cavetetsizefac = %lf, restetsize = %d\n",
	//	behavior->cavetetsize, behavior->cavetetsizefac, restetsize);
	t_cavetetlist.reserve(restetsize);
	t_cavetetprev.reserve(restetsize);
	t_cavetetnext.reserve(restetsize);

	// Compute initial cavity starting points
	int oldsize, newsize;
	int initialcavitysize/*, initialsubcavitysize*/;
	IntD t_initialcavitysize(numofinsertpt, MAXINT);
	IntD t_initialcavityindices(numofinsertpt, -1);

	// set losers' cavity and scavity sizes to zero
	thrust::replace_if(t_initialcavitysize.begin(), t_initialcavitysize.end(), t_threadmarker.begin(), isNegativeInt(), 0);

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        cavity vector initialization time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["cavity vector initialization"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Form initial cavities
	// mark and count the initial cavities
	// mark tets using original thread indices
	kernelMarkAndCountInitialCavity << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_pointlocation[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_initialcavitysize[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelMarkAndCountInitialCavity time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelMarkAndCountInitialCavity"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// update working thread list
	numberofthreads = updateActiveListByMarker(t_threadmarker, t_threadlist, t_threadmarker.size());
	if (behavior->verbose >= 1)
		printf("        After initial cavity marking, numberofthreads = %d(#%d, #%d, #%d)\n", 
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
	{
		// This should not error
		printf("Error: 0 threads after marking initial cavities!\n");
		exit(0);
	}

#ifdef GQM3D_CHECKMEMORY
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        update working thread list time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["update working thread list"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// compute total size and indices for intital cavities
	thrust::exclusive_scan(t_initialcavitysize.begin(), t_initialcavitysize.end(), t_initialcavityindices.begin());
	initialcavitysize = t_initialcavityindices[numofinsertpt - 1] + t_initialcavitysize[numofinsertpt - 1];

#ifdef GQM3D_DEBUG
	printf("Initial cavity size = %d\n", initialcavitysize);
#endif

	// init cavity linklists
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	t_caveoldtetlist.resize(initialcavitysize);
	t_caveoldtetprev.resize(initialcavitysize);
	t_caveoldtetnext.resize(initialcavitysize);
	int expandfactor = 4;
	t_cavetetlist.resize(expandfactor * initialcavitysize);
	t_cavetetprev.resize(expandfactor * initialcavitysize);
	t_cavetetnext.resize(expandfactor * initialcavitysize);
	t_cavethreadidx.resize(expandfactor * initialcavitysize);
	thrust::fill(t_cavethreadidx.begin(), t_cavethreadidx.end(), -1);

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        prepare vector for intial cavity time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["prepare vector for intial cavity"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	kernelInitCavityLinklist << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_pointlocation[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_initialcavityindices[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetprev[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_caveoldtettail[0]),
		thrust::raw_pointer_cast(&t_cavetetlist[0]),
		thrust::raw_pointer_cast(&t_cavetetprev[0]),
		thrust::raw_pointer_cast(&t_cavetetnext[0]),
		thrust::raw_pointer_cast(&t_cavetethead[0]),
		thrust::raw_pointer_cast(&t_cavetettail[0]),
		thrust::raw_pointer_cast(&t_cavethreadidx[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	freeVec(t_pointlocation);
	freeVec(t_initialcavitysize);
	freeVec(t_initialcavityindices);

	if (drawmesh != NULL && drawmesh->animation)
	{
		// Output animation frame mesh
		if (iter == drawmesh->iter_tet)
		{
			outputCavityFrame(
				drawmesh,
				t_pointlist,
				t_tetlist,
				t_tetmarker,
				t_threadmarker,
				t_caveoldtetlist,
				t_caveoldtetnext,
				t_caveoldtethead,
				0,
				0,
				iter,
				-1,
				initialcavitysize
			);
		}
	}

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelInitCavityLinklist = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelInitCavityLinklist"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Expand Initial Cavity
	// Every iteration, test if current tet in cavetetlist is included in cavity
	// If it is, expand cavetetlist and caveoldtetlist, otherwise expand cavebdrylist
	TetHandleD t_cavebdrylist;
	IntD t_cavebdryprev, t_cavebdrynext;
	IntD t_cavebdryhead(numofinsertpt, -1), t_cavebdrytail(numofinsertpt, -1);

	// Adatively reserve memory space
	int resbdrysize = behavior->cavebdrysize * behavior->cavebdrysizefac;
	//printf("cavebdrysize = %d, cavebdrysizefac = %lf, resbdrysize = %d\n",
	//	behavior->cavebdrysize, behavior->cavebdrysizefac, resbdrysize);
	t_cavebdrylist.reserve(resbdrysize);
	t_cavebdryprev.reserve(resbdrysize);
	t_cavebdrynext.reserve(resbdrysize);

	int cavetetcurstartindex = 0;
	int cavetetstartindex = t_cavetetlist.size();
	int caveoldtetstartindex = t_caveoldtetlist.size();
	int cavebdrystartindex = 0;
	int cavetetexpandsize = cavetetstartindex, caveoldtetexpandsize, cavebdryexpandsize;

	numberofthreads = cavetetexpandsize; // each thread works on one tet in cavetetlist
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	IntD t_cavetetexpandsize(numberofthreads, 0);
	IntD t_caveoldtetexpandsize(numberofthreads, 0);
	IntD t_cavebdryexpandsize(numberofthreads, 0);
	IntD t_cavetetexpandindices(numberofthreads, -1);
	IntD t_caveoldtetexpandindices(numberofthreads, -1);
	IntD t_cavebdryexpandindices(numberofthreads, -1);
	IntD t_cavetetthreadidx;
	IntD t_caveoldtetthreadidx;
	IntD t_cavebdrythreadidx;

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        cavity expanding vector initialization = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["cavity expanding vector initialization"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

#ifdef GQM3D_LOOP_PROFILING
	double cavity_total_time = 0;
#endif
	int iteration = 0;
	while (true)
	{
		if (iteration > behavior->maxcavity) // Too large cavities. Stop and mark as unsplittable elements
		{
			kernelLargeCavityCheck << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_insertptlist[0]),
				thrust::raw_pointer_cast(&t_cavethreadidx[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_tetstatus[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			break;
		}

#ifdef GQM3D_LOOP_PROFILING
		double iter_total_time = 0;
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("          iteration = %d, survivors = %d, number of tuples to check = %d\n", 
			iteration, 
			thrust::count_if(t_threadmarker.begin(), t_threadmarker.end(), isNotNegativeInt()),
			numberofthreads);
#endif
		cudaDeviceSynchronize();
		ltv[0] = clock();
#endif

		// Check if current tet is included in cavity
		kernelCavityExpandingCheck << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_weightlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_cavetetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetprev[0]),
			thrust::raw_pointer_cast(&t_cavetetnext[0]),
			thrust::raw_pointer_cast(&t_cavetethead[0]),
			thrust::raw_pointer_cast(&t_cavetettail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
			thrust::raw_pointer_cast(&t_caveoldtetprev[0]),
			thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
			thrust::raw_pointer_cast(&t_caveoldtethead[0]),
			thrust::raw_pointer_cast(&t_caveoldtettail[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryprev[0]),
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdryhead[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			thrust::raw_pointer_cast(&t_priority[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			cavetetcurstartindex,
			numberofthreads
			);

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["kernelCavityExpandingCheck"] += (REAL)(ltv[1] - ltv[0]);
		iter_total_time += (REAL)(ltv[1] - ltv[0]);
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            kernelCavityExpandingCheck time = %lf\n", (REAL)(ltv[1] - ltv[0]));
#endif
		ltv[0] = ltv[1];
#endif

		kernelCorrectExpandingSize << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["kernelCorrectExpandingSize"] += (REAL)(ltv[1] - ltv[0]);
		iter_total_time += (REAL)(ltv[1] - ltv[0]);
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            kernelCorrectExpandingSize time = %lf\n", (REAL)(ltv[1] - ltv[0]));
#endif
		ltv[0] = ltv[1];
#endif

		thrust::exclusive_scan(
			thrust::make_zip_iterator(thrust::make_tuple(t_cavetetexpandsize.begin(), t_caveoldtetexpandsize.begin(), t_cavebdryexpandsize.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(t_cavetetexpandsize.end(), t_caveoldtetexpandsize.end(), t_cavebdryexpandsize.end())),
			thrust::make_zip_iterator(thrust::make_tuple(t_cavetetexpandindices.begin(), t_caveoldtetexpandindices.begin(), t_cavebdryexpandindices.begin())),
			thrust::make_tuple(0, 0, 0),
			PrefixSumTupleOP());

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["exclusive_scan for expanding"] += (REAL)(ltv[1] - ltv[0]);
		iter_total_time += (REAL)(ltv[1] - ltv[0]);
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            exclusive_scan for expanding time = %lf\n", (REAL)(ltv[1] - ltv[0]));
#endif
		ltv[0] = ltv[1];
#endif

		// Count expanding sizes
		cavetetexpandsize = t_cavetetexpandindices[numberofthreads - 1] + t_cavetetexpandsize[numberofthreads - 1];
		caveoldtetexpandsize = t_caveoldtetexpandindices[numberofthreads - 1] + t_caveoldtetexpandsize[numberofthreads - 1];
		cavebdryexpandsize = t_cavebdryexpandindices[numberofthreads - 1] + t_cavebdryexpandsize[numberofthreads - 1];

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["count sizes for expanding"] += (REAL)(ltv[1] - ltv[0]);
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            count sizes for expanding time = %lf\n", (REAL)(ltv[1] - ltv[0]));
#endif
		ltv[0] = ltv[1];
#endif

#ifdef GQM3D_DEBUG
		if (behavior->verbose >= 3)
			printf("          Iteration = %d, expand tet = %d, oldtet = %d, bdrytet = %d, survivor = %d, ",
				iteration, cavetetexpandsize, caveoldtetexpandsize, cavebdryexpandsize,
				thrust::count_if(t_threadmarker.begin(), t_threadmarker.end(), isNotNegativeInt()));
#endif

		// Prepare memeory
		oldsize = t_cavetetlist.size();
		newsize = oldsize + cavetetexpandsize;
		t_cavetetlist.resize(newsize);
		t_cavetetprev.resize(newsize);
		t_cavetetnext.resize(newsize);
		t_cavetetthreadidx.resize(cavetetexpandsize);
		oldsize = t_caveoldtetlist.size();
		newsize = oldsize + caveoldtetexpandsize;
		t_caveoldtetlist.resize(newsize);
		t_caveoldtetprev.resize(newsize);
		t_caveoldtetnext.resize(newsize);
		t_caveoldtetthreadidx.resize(caveoldtetexpandsize);
		oldsize = t_cavebdrylist.size();
		newsize = oldsize + cavebdryexpandsize;
		t_cavebdrylist.resize(newsize);
		t_cavebdryprev.resize(newsize);
		t_cavebdrynext.resize(newsize);
		t_cavebdrythreadidx.resize(cavebdryexpandsize);

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["resize vector for expanding"] += (REAL)(ltv[1] - ltv[0]);
		iter_total_time += (REAL)(ltv[1] - ltv[0]);
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            resize vector for expanding time = %lf\n", (REAL)(ltv[1] - ltv[0]));
#endif
		ltv[0] = ltv[1];
#endif

		// Set threadidx list
		kernelCavityExpandingSetThreadidx << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavetetthreadidx[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandindices[0]),
			thrust::raw_pointer_cast(&t_caveoldtetthreadidx[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavebdrythreadidx[0]),
			numberofthreads
			);

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["kernelCavityExpandingSetThreadidx"] += (REAL)(ltv[1] - ltv[0]);
		iter_total_time += (REAL)(ltv[1] - ltv[0]);
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            kernelCavityExpandingSetThreadidx time = %lf\n", (REAL)(ltv[1] - ltv[0]));
#endif
		ltv[0] = ltv[1];
#endif

		// Mark and append expanding tets
		kernelCavityExpandingMarkAndAppend << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_cavetetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetprev[0]),
			thrust::raw_pointer_cast(&t_cavetetnext[0]),
			thrust::raw_pointer_cast(&t_cavetethead[0]),
			thrust::raw_pointer_cast(&t_cavetettail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavetetthreadidx[0]),
			cavetetstartindex,
			cavetetexpandsize,
			thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
			thrust::raw_pointer_cast(&t_caveoldtetprev[0]),
			thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
			thrust::raw_pointer_cast(&t_caveoldtethead[0]),
			thrust::raw_pointer_cast(&t_caveoldtettail[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandindices[0]),
			thrust::raw_pointer_cast(&t_caveoldtetthreadidx[0]),
			caveoldtetstartindex,
			caveoldtetexpandsize,
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryprev[0]),
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdryhead[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavebdrythreadidx[0]),
			cavebdrystartindex,
			cavebdryexpandsize,
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			thrust::raw_pointer_cast(&t_priority[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			cavetetcurstartindex,
			numberofthreads
			);

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["kernelCavityExpandingMarkAndAppend"] += (REAL)(ltv[1] - ltv[0]);
		iter_total_time += (REAL)(ltv[1] - ltv[0]);
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            kernelCavityExpandingMarkAndAppend time = %lf\n", (REAL)(ltv[1] - ltv[0]));
#endif
		ltv[0] = ltv[1];
#endif

		kernelCavityExpandingUpdateListTails << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_cavetetnext[0]),
			thrust::raw_pointer_cast(&t_cavetettail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			cavetetstartindex,
			thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
			thrust::raw_pointer_cast(&t_caveoldtettail[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandindices[0]),
			caveoldtetstartindex,
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			cavebdrystartindex,
			numberofthreads
			);

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["kernelCavityExpandingUpdateListTails"] += (REAL)(ltv[1] - ltv[0]);
		iter_total_time += (REAL)(ltv[1] - ltv[0]);
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            kernelCavityExpandingUpdateListTails time = %lf\n", (REAL)(ltv[1] - ltv[0]));
#endif
		ltv[0] = ltv[1];
#endif

		if (drawmesh != NULL && drawmesh->animation)
		{
			// Output animation frame mesh
			if (iter == drawmesh->iter_tet)
			{
				outputCavityFrame(
					drawmesh,
					t_pointlist,
					t_tetlist,
					t_tetmarker,
					t_threadmarker,
					t_caveoldtetlist,
					t_caveoldtetnext,
					t_caveoldtethead,
					-1,
					-1,
					iter,
					iteration,
					caveoldtetexpandsize
				);

				if(iteration >= behavior->maxcavity - 10)
					outputLargeCavityFrame(
						drawmesh,
						t_pointlist,
						t_tetlist,
						t_tetmarker,
						t_threadmarker,
						t_cavethreadidx,
						t_caveoldtetlist,
						t_caveoldtetnext,
						t_caveoldtethead,
						-1,
						-1,
						iter,
						iteration,
						caveoldtetexpandsize
					);
			}
		}

		// Update working thread list
		numberofthreads = cavetetexpandsize;
		iteration++;
		if (numberofthreads == 0)
			break;

		// Update variables
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		cavetetcurstartindex = cavetetstartindex;
		cavetetstartindex = t_cavetetlist.size();
		caveoldtetstartindex = t_caveoldtetlist.size();
		cavebdrystartindex = t_cavebdrylist.size();

		// Set threadidx list
		t_cavethreadidx.resize(numberofthreads);
		thrust::copy(t_cavetetthreadidx.begin(), t_cavetetthreadidx.end(), t_cavethreadidx.begin());

		// Reset expanding lists
		t_cavetetexpandsize.resize(numberofthreads);
		thrust::fill(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end(), 0);
		t_cavetetexpandindices.resize(numberofthreads);

		t_caveoldtetexpandsize.resize(numberofthreads);
		thrust::fill(t_caveoldtetexpandsize.begin(), t_caveoldtetexpandsize.end(), 0);
		t_caveoldtetexpandindices.resize(numberofthreads);
		
		t_cavebdryexpandsize.resize(numberofthreads);
		thrust::fill(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), 0);
		t_cavebdryexpandindices.resize(numberofthreads);

#ifdef GQM3D_LOOP_PROFILING
		cudaDeviceSynchronize();
		ltv[1] = clock();
		looptimer["prepare memory for next expanding"] += (REAL)(ltv[1] - ltv[0]);
		iter_total_time += (REAL)(ltv[1] - ltv[0]);
		cavity_total_time += iter_total_time;
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
		printf("            prepare memory for next expanding time = %lf\n", (REAL)(ltv[1] - ltv[0]));
		printf("          total time = %lf\n", iter_total_time);
#endif
		ltv[0] = ltv[1];
#endif

	}
#ifdef GQM3D_LOOP_PROFILING
#ifdef GQM3D_LOOP_PROFILING_VERBOSE
	printf("          cavity growing total time = %lf\n", cavity_total_time);
#endif
#endif

#ifdef GQM3D_CHECKMEMORY
	printf("Before release cavity memory\n");
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

	// Update cavetet, caveoldtet, cavebdry sizes and factors
	behavior->cavetetsizefac = t_cavetetlist.size() * 1.0 / behavior->cavetetsize + 0.02;
	behavior->cavetetsize = t_cavetetlist.size();
	//printf("cavetetsize = %d, cavetetsizefac = %lf\n", behavior->cavetetsize, behavior->cavetetsizefac);
	behavior->caveoldtetsizefac = t_caveoldtetlist.size() * 1.0 / behavior->caveoldtetsize + 0.02;
	behavior->caveoldtetsize = t_caveoldtetlist.size();
	//printf("caveoldtetsize = %d, caveoldtetsizefac = %lf\n", behavior->caveoldtetsize, behavior->caveoldtetsizefac);
	behavior->cavebdrysizefac = t_cavebdrylist.size() * 1.0 / behavior->cavebdrysize + 0.02;
	behavior->cavebdrysize = t_cavebdrylist.size();
	//printf("cavebdrysize = %d, cavebdrysizefac = %lf\n", behavior->cavebdrysize, behavior->cavebdrysizefac);
	if (behavior->filterstatus == 3)
	{
		behavior->cavetetsizefac = 1.1;
		behavior->caveoldtetsizefac = 1.1;
		behavior->cavebdrysizefac = 1.1;
		behavior->filterstatus = 1;
	}

	freeVec(t_priority);
	freeVec(t_cavethreadidx);
	freeVec(t_cavetetlist);
	freeVec(t_cavetetprev);
	freeVec(t_cavetetnext);
	freeVec(t_cavetethead);
	freeVec(t_cavetettail);
	freeVec(t_cavetetexpandsize);
	freeVec(t_caveoldtetexpandsize);
	freeVec(t_cavebdryexpandsize);
	freeVec(t_cavetetexpandindices);
	freeVec(t_caveoldtetexpandindices);
	freeVec(t_cavebdryexpandindices);
	freeVec(t_cavetetthreadidx);
	freeVec(t_caveoldtetthreadidx);
	freeVec(t_cavebdrythreadidx);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        cavity expanding time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["cavity expanding"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Update working threadlist to winners
	numberofthreads = updateActiveListByMarker(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if(behavior->verbose >= 1)
		printf("        After expanding cavity, numberofthreads = %d(#%d, #%d, #%d), total expanding iteration = %d\n", 
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2),
			iteration);
	if (numberofthreads == 0)
		return 1;

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        update working threadlist time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["update working thread list"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

//	if (drawmesh != NULL && drawmesh->animation)
//	{
//		// Output animation frame mesh
//		if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
//			iter_2 == drawmesh->iter_tet)
//		{
//			outputCavityFrame(
//				drawmesh,
//				t_pointlist,
//				t_trifacelist,
//				t_tristatus,
//				t_tri2tetlist,
//				t_tetlist,
//				t_neighborlist,
//				t_tetmarker,
//				t_threadmarker,
//				t_caveoldtetlist,
//				t_caveoldtetnext,
//				t_caveoldtethead,
//				iter_0,
//				iter_1,
//				iter_2,
//				-1,
//				0
//			);
//		}
//	}

#ifdef GQM3D_CHECKMEMORY
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

	// Collect subfaces in cavities
	TriHandleD t_cavetetshlist;
	IntD t_cavetetshprev, t_cavetetshnext;
	IntD t_cavetetshhead(numofinsertpt, -1), t_cavetetshtail(numofinsertpt, -1);
	IntD t_cavetetshsize(numberofthreads, 0), t_cavetetshindices(numberofthreads, -1);
	int cavetetshsize;

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        subface vector initialization = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["subface vector initialization"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	kernelCountCavitySubfaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_cavetetshsize[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelCountCavitySubfaces = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelCountCavitySubfaces"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	thrust::exclusive_scan(t_cavetetshsize.begin(), t_cavetetshsize.end(), t_cavetetshindices.begin());
	//cavetetshsize = thrust::reduce(t_cavetetshsize.begin(), t_cavetetshsize.end());
	cavetetshsize = t_cavetetshindices[numberofthreads - 1] + t_cavetetshsize[numberofthreads - 1];
	t_cavetetshlist.resize(cavetetshsize);
	t_cavetetshprev.resize(cavetetshsize);
	t_cavetetshnext.resize(cavetetshsize);

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        prepare subface vector time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["prepare subface vector"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	kernelAppendCavitySubfaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshprev[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_cavetetshtail[0]),
		thrust::raw_pointer_cast(&t_cavetetshsize[0]),
		thrust::raw_pointer_cast(&t_cavetetshindices[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	thrust::pair<IntD::iterator, IntD::iterator> shnum_pair;
	shnum_pair =
		thrust::minmax_element(
			t_cavetetshsize.begin(),
			t_cavetetshsize.end());
	int maxshnum = *shnum_pair.second;
	printf("Maximum number of facets in cavity = %d\n", maxshnum);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	freeVec(t_cavetetshsize);
	freeVec(t_cavetetshindices);

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelAppendCavitySubfaces time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelAppendCavitySubfaces"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// remove the cavities that share subfaces
	numberofthreads = updateActiveListByMarker(t_threadmarker, t_threadlist, t_threadmarker.size());
	if (behavior->verbose >= 1)
		printf("        After boundary checking, numberofthreads = %d(#%d, #%d, #%d)\n",
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
	{
		// This should not happen
		printf("Error: 0 threads after boundary checking!\n");
		exit(0);
	}

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        update working threadlist time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["update working thread list"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Subface encroachment test
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelCheckSubfaceEncroachment << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_weightlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_trifacecent[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelCheckSubfaceEncroachment time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelCheckSubfaceEncroachment"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// remove encroaching points
	numberofthreads = updateActiveListByMarker(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (behavior->verbose >= 1)
		printf("        After subface encroachment check, numberofthreads = %d(#%d, #%d, #%d)\n", 
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
		return 1;

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        update working thread list time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["update working thread list"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

//	if (drawmesh != NULL && drawmesh->animation)
//	{
//		// Output animation frame mesh
//		if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
//			iter_2 == drawmesh->iter_tet)
//		{
//			outputCavityFrame(
//				drawmesh,
//				t_pointlist,
//				t_trifacelist,
//				t_tristatus,
//				t_tri2tetlist,
//				t_tetlist,
//				t_neighborlist,
//				t_tetmarker,
//				t_threadmarker,
//				t_caveoldtetlist,
//				t_caveoldtetnext,
//				t_caveoldtethead,
//				iter_0,
//				iter_1,
//				iter_2,
//				-1,
//				0
//			);
//
//			outputCavityFrame(
//				drawmesh,
//				t_pointlist,
//				t_tetlist,
//				t_tetmarker,
//				t_threadmarker,
//				t_caveoldtetlist,
//				t_caveoldtetnext,
//				t_caveoldtethead,
//				iter_0,
//				iter_1,
//				iter_2,
//				-1,
//				0
//			);
//		}
//	}

	// Refinement elements check
	// The new point is inserted by Delaunay refinement, i.e., it is the 
	//   circumcenter of a tetrahedron, or a subface, or a segment.
	//   Do not insert this point if the tetrahedron, or subface, or segment
	//   is not inside the final cavity.
	kernelValidateRefinementElements << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelValidateRefinementElements time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelValidateRefinementElements"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	if (insertmode == 1)
	{
		kernelRecomputeTrifaceCenter << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_aabbnodeleft[0]),
			thrust::raw_pointer_cast(&t_aabbnoderight[0]),
			thrust::raw_pointer_cast(&t_aabbnodebbs[0]),
			thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
			thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
			thrust::raw_pointer_cast(&t_trifacecent[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_trifacepmt[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_weightlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			aabb_diglen,
			numberofthreads
			);
	}

	numberofwonfacets = thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1);
	numberofthreads = updateActiveListByMarker(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if(behavior->verbose >= 1)
		printf("        After validating cavity, numberofthreads = %d(#%d, #%d, #%d)\n", 
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			numberofwonfacets,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
		return 1;

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        update working thread list time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["update working thread list"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

#ifdef GQM3D_CHECKMEMORY
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

//	if (drawmesh != NULL && drawmesh->animation)
//	{
//		// Output animation frame mesh
//		if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
//			iter_2 == drawmesh->iter_tet)
//		{
//			outputCavityFrame(
//				drawmesh,
//				t_pointlist,
//				t_tetlist,
//				t_tetmarker,
//				t_threadmarker,
//				t_caveoldtetlist,
//				t_caveoldtetnext,
//				t_caveoldtethead,
//				iter_0,
//				iter_1,
//				iter_2,
//				-1,
//				0
//			);
//		}
//	}
//
//	if (drawmesh != NULL && !drawmesh->animation)
//	{
//		if (drawmesh->iter_seg != -1 || drawmesh->iter_subface != -1 || drawmesh->iter_tet != -1)
//		{
//			if (iter_0 == drawmesh->iter_seg  && iter_1 == drawmesh->iter_subface
//				&& iter_2 == drawmesh->iter_tet)
//			{
//				printf("Copy mesh to drawmesh!\n");
//
//				outputTmpMesh(
//					drawmesh,
//					t_pointlist, t_pointtypelist,
//					t_seglist, t_segstatus,
//					t_trifacelist, t_tristatus,
//					t_tetlist, t_tetstatus,
//					t_insertidxlist, t_insertptlist,
//					t_threadlist, t_threadmarker,
//					t_cavebdrylist, t_cavebdrynext, t_cavebdryhead,
//					0);
//
//				return 0;
//			}
//		}
//	}

	// Insert points into list
	oldsize = t_weightlist.size();
	int oldpointsize = oldsize;
	newsize = oldsize + numberofthreads;
	t_pointlist.resize(3 * newsize);
	t_weightlist.resize(newsize, 0.0); // 0 weight for new insert points
	t_pointtypelist.resize(newsize);
	t_pointpmt.resize(newsize, -1);

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        prepare new point vector time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["prepare new point vector"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	kernelInsertNewPoints << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_pointpmt[0]),
		thrust::raw_pointer_cast(&t_trifacepmt[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		oldpointsize,
		numberofthreads
		);

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelInsertNewPoints time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelInsertNewPoints"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// Create new tetrahedra to fill the cavity
	IntD t_tetexpandsize(numberofthreads, 0);
	IntD t_tetexpandindice(numberofthreads, -1);
	int tetexpandsize = 0;

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        countnewtet vector initialization time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["countnewtet vector initialization"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	kernelCountNewTets << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_tetexpandsize[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelCountNewTets time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelCountNewTets"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	//tetexpandsize = thrust::reduce(t_tetexpandsize.begin(), t_tetexpandsize.end());
	thrust::exclusive_scan(t_tetexpandsize.begin(), t_tetexpandsize.end(), t_tetexpandindice.begin());
	tetexpandsize = t_tetexpandindice[numberofthreads - 1] + t_tetexpandsize[numberofthreads - 1];
	if (insertmode == 1 && numberofwonfacets > 0)
		numberofnewtets_facet = t_tetexpandindice[numberofwonfacets - 1] + t_tetexpandsize[numberofwonfacets - 1];

#ifdef GQM3D_DEBUG
	printf("Tet expanding size = %d\n", tetexpandsize);
#endif

	IntD t_emptytetslots;
	int numberofemptyslot = updateEmptyTetList(t_tetstatus, t_emptytetslots);
	if (numberofemptyslot < tetexpandsize) // dont have enough empty slots, extend lists
	{
		oldsize = t_tetstatus.size();
		newsize = oldsize + tetexpandsize - numberofemptyslot;
		try
		{
			t_tetlist.resize(4 * newsize, -1);
			t_neighborlist.resize(4 * newsize, tethandle(-1, 11));
			t_tet2trilist.resize(4 * newsize, trihandle(-1, 0));
			t_tetstatus.resize(newsize, tetstatus(0));
		}
		catch (thrust::system_error &e)
		{
			// output an error message and exit
			std::cerr << "Error: " << e.what() << std::endl;
			exit(-1);
		}
		numberofemptyslot = updateEmptyTetList(t_tetstatus, t_emptytetslots);
	}

#ifdef GQM3D_DEBUG
	printf("numberofemptyslot = %d\n", numberofemptyslot);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        prepare new tet vector time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["prepare new tet vector"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	IntD t_newtetthreadindices(tetexpandsize, -1); // used to update tetstatus later on
	kernelInsertNewTets << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_tetexpandindice[0]),
		thrust::raw_pointer_cast(&t_emptytetslots[0]),
		thrust::raw_pointer_cast(&t_newtetthreadindices[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		oldpointsize,
		numberofthreads
		);

	freeVec(t_tetexpandsize);
	freeVec(t_tetexpandindice);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelInsertNewTets time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelInsertNewTets"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Connect adjacent new tetrahedra together
	kernelConnectNewTetNeighbors << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_CHECKMEMORY
	printf("After inserting new elements\n");
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelConnectNewTetNeighbors time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelConnectNewTetNeighbors"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	// Update tristatus and tetstatus, and
	// check and add new trifaces
	int triexpandsize = 0;
	IntD t_triexpandsize(4 * tetexpandsize, 0);
	IntD t_triexpandindice(4 * tetexpandsize, 0);
	IntD t_tripmtidx(4 * tetexpandsize, -1);
	RealD t_trifaceipt(4 * 3 * tetexpandsize);
	IntD t_emptytrislots;

#ifdef GQM3D_CHECKMEMORY
	printf("After triface expanding vector initialization\n");
	cudaDeviceSynchronize();
	gpuMemoryCheck();
	//checkVectorSize(t_triexpandsize, "t_triexpandsize", int);
	//checkVectorSize(t_triexpandindice, "t_triexpandindice", int);
	//checkVectorSize(t_trifaceipt, "t_trifaceipt", REAL);
	//checkVectorSize(t_emptytetslots, "t_emptytrislots", int);
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        tri and tet status vector initialization time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["tri and tet status vector initializatio"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	if (behavior->aabbmode == 1)
	{
		//printf("tetexpandsize = %d\n", tetexpandsize);
		numberofthreads = tetexpandsize;
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

		kernelUpdateTriAndTetStatus_Phase1 << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_aabbnodeleft[0]),
			thrust::raw_pointer_cast(&t_aabbnoderight[0]),
			thrust::raw_pointer_cast(&t_aabbnodebbs[0]),
			thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
			thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
			thrust::raw_pointer_cast(&t_emptytetslots[0]),
			thrust::raw_pointer_cast(&t_newtetthreadindices[0]),
			thrust::raw_pointer_cast(&t_triexpandsize[0]),
			thrust::raw_pointer_cast(&t_trifaceipt[0]),
			thrust::raw_pointer_cast(&t_tripmtidx[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_weightlist[0]),
			thrust::raw_pointer_cast(&t_pointpmt[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			criteria->cell_radius_edge_ratio,
			criteria->cell_size,
			aabb_diglen,
			aabb_pmnum,
			insertmode,
			behavior->aabbshortcut,
			numberofthreads
			);

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        kernelUpdateTriAndTetStatus_Phase1 time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["kernelUpdateTriAndTetStatus_Phase1"] += (REAL)(tv[1] - tv[0]);
		tv[0] = tv[1];
#endif

		//triexpandsize = thrust::reduce(t_triexpandsize.begin(), t_triexpandsize.end());
		thrust::exclusive_scan(t_triexpandsize.begin(), t_triexpandsize.end(), t_triexpandindice.begin());
		triexpandsize = t_triexpandindice[4 * tetexpandsize - 1] + t_triexpandsize[4 * tetexpandsize - 1];
#ifdef GQM3D_DEBUG
		printf("Tri expanding size = %d\n", triexpandsize);
#endif

		//IntD t_emptytrislots;
		numberofemptyslot = updateEmptyTriList(t_tristatus, t_emptytrislots);
		if (numberofemptyslot < triexpandsize) // dont have enough empty slots, extend lists
		{
			oldsize = t_tristatus.size();
			newsize = oldsize + triexpandsize - numberofemptyslot;
			try
			{
				t_trifacelist.resize(3 * newsize, -1);
				t_trifacecent.resize(3 * newsize);
				t_tri2tetlist.resize(2 * newsize, tethandle(-1, 11));
				t_tristatus.resize(newsize, tristatus(0));
				t_trifacepmt.resize(newsize, -1);
			}
			catch (thrust::system_error &e)
			{
				// output an error message and exit
				std::cerr << "Error: " << e.what() << std::endl;
				exit(-1);
			}
			numberofemptyslot = updateEmptyTriList(t_tristatus, t_emptytrislots);
		}

#ifdef GQM3D_DEBUG
		printf("numberofemptyslot = %d\n", numberofemptyslot);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        prepare new subface vector time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["prepare new subface vector"] += (REAL)(tv[1] - tv[0]);
		tv[0] = tv[1];
#endif

		kernelUpdateTriAndTetStatus_Phase2 << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_emptytetslots[0]),
			thrust::raw_pointer_cast(&t_triexpandsize[0]),
			thrust::raw_pointer_cast(&t_triexpandindice[0]),
			thrust::raw_pointer_cast(&t_emptytrislots[0]),
			thrust::raw_pointer_cast(&t_trifaceipt[0]),
			thrust::raw_pointer_cast(&t_tripmtidx[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_pointtypelist[0]),
			thrust::raw_pointer_cast(&t_weightlist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_trifacecent[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_trifacepmt[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tet2trilist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			criteria->facet_angle,
			criteria->facet_size,
			criteria->facet_distance,
			numberofthreads
			);

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        kernelUpdateTriAndTetStatus_Phase2 time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["kernelUpdateTriAndTetStatus_Phase2"] += (REAL)(tv[1] - tv[0]);
		tv[0] = tv[1];
#endif
	}
	else if (behavior->aabbmode == 2)
	{
		int curhandlesize;
		TetHandleD::iterator last_iterator;
		IntD t_domaincount(tetexpandsize, 0);// counter for in/out domain test
		TetHandleD t_domainhandle;
		RealD t_domainsegment;
		IntD t_domainthreadlist;
		IntD t_domainnode;
		int winsize = tetexpandsize;

		//printf("aabbhandlesize = %d, aabbhandlesizefac = %lf\n", behavior->aabbhandlesize, behavior->aabbhandlesizefac);
		int rechandlesize = behavior->aabbhandlesizefac * behavior->aabbhandlesize;
		double free_mb;
		getFreeMemory(free_mb);
		int maxreservesize = free_mb * 1024 * 1024 / 6 / sizeof(tethandle); // 1/6 of the available memory
		//printf("rechandlesize = %d, maxreservesize = %d\n", rechandlesize, maxreservesize);
		if (rechandlesize < maxreservesize)
		{
			t_domainhandle.reserve(rechandlesize);
			t_domainnode.reserve(rechandlesize);
		}
		else
		{
			//printf("Enter memory save mode\n");
			t_domainhandle.reserve(maxreservesize);
			t_domainnode.reserve(maxreservesize);
			if (behavior->aabbwinsize == -1)
				behavior->aabbwinsize = tetexpandsize;
			else
				winsize = behavior->aabbwinsize; // set proper winsize
		}

		try
		{
			if (t_tetmarker.size() < 4 * tetexpandsize) // use temporarily for unique distance marking
				t_tetmarker.resize(4 * tetexpandsize);
			thrust::fill(t_tetmarker.begin(), t_tetmarker.begin() + 4 * tetexpandsize, MAXULL);
		}
		catch (thrust::system_error &e)
		{
			// output an error message and exit
			std::cerr << "Error: " << e.what() << std::endl;
			exit(-1);
		}

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        domain vector initialization time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["domain vector initialization"] += (REAL)(tv[1] - tv[0]);
		tv[0] = clock();
#endif
		int offset = 0;
		int oldaabbhandlesize = behavior->aabbhandlesize;
		behavior->aabbhandlesize = 0;
		while (true)
		{
			numberofthreads = tetexpandsize - offset;
			if (numberofthreads > winsize)
				numberofthreads = winsize;
			else if (numberofthreads <= 0)
				break;

			//printf("tetexpandsize = %d, numberofthreads = %d, offset = %d, winsize = %d\n", tetexpandsize, numberofthreads, offset, winsize);

			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			if (insertmode == 0)
			{
				t_domainhandle.resize(5 * numberofthreads);
			}
			else
			{
				if (numberofwonfacets == 0) // nothing to do
					break;
				t_domainhandle.resize(5 * numberofnewtets_facet);
			}
			thrust::fill(t_domainhandle.begin(), t_domainhandle.end(), tethandle(-1, 11));
			kernelInitDomainHandle << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_emptytetslots[0]),
				thrust::raw_pointer_cast(&t_newtetthreadindices[0]),
				thrust::raw_pointer_cast(&t_domainhandle[0]),
				thrust::raw_pointer_cast(&t_domaincount[0]),
				thrust::raw_pointer_cast(&t_tetlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tetstatus[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				insertmode,
				offset,
				numberofthreads
				);

#ifdef GQM3D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
			cudaDeviceSynchronize();
			tv[1] = clock();
			printf("        kernelInitDomainHandle time = %f\n", (REAL)(tv[1] - tv[0]));
			inserttimer["kernelInitDomainHandle"] += (REAL)(tv[1] - tv[0]);
			tv[0] = clock();
#endif

			last_iterator =
				thrust::remove_if(t_domainhandle.begin(), t_domainhandle.end(), isInvalidTetHandle());
			curhandlesize = thrust::distance(t_domainhandle.begin(), last_iterator);
			//printf("curhandlesize = %d\n", curhandlesize);
			if (curhandlesize == 0)
			{
				offset += winsize;
				continue;
			}

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

#ifdef GQM3D_PROFILING
			cudaDeviceSynchronize();
			tv[1] = clock();
			printf("        prepare for domain segment vector time = %f\n", (REAL)(tv[1] - tv[0]));
			inserttimer["prepare for domain segment vector"] += (REAL)(tv[1] - tv[0]);
			tv[0] = clock();
#endif

			numberofthreads = curhandlesize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelInitDomainSegment << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_aabbnodebbs[0]),
				thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
				thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
				thrust::raw_pointer_cast(&t_domainhandle[0]),
				thrust::raw_pointer_cast(&t_domainsegment[0]),
				thrust::raw_pointer_cast(&t_domainthreadlist[0]),
				thrust::raw_pointer_cast(&t_triexpandsize[0]),
				thrust::raw_pointer_cast(&t_trifaceipt[0]),
				thrust::raw_pointer_cast(&t_tripmtidx[0]),
				thrust::raw_pointer_cast(&t_emptytetslots[0]),
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_weightlist[0]),
				thrust::raw_pointer_cast(&t_pointpmt[0]),
				thrust::raw_pointer_cast(&t_tetlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				aabb_diglen,
				aabb_pmnum,
				insertmode,
				behavior->aabbshortcut,
				numberofthreads
				);

#ifdef GQM3D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
			cudaDeviceSynchronize();
			tv[1] = clock();
			printf("        kernelInitDomainSegment time = %f\n", (REAL)(tv[1] - tv[0]));
			inserttimer["kernelInitDomainSegment"] += (REAL)(tv[1] - tv[0]);
			tv[0] = clock();
#endif

			last_iterator = // remove degenerate cases and fast check cases
				thrust::remove_if(t_domainhandle.begin(), t_domainhandle.begin() + numberofthreads, isInvalidTetHandle());
			curhandlesize = thrust::distance(t_domainhandle.begin(), last_iterator);
			//printf("curhandlesize = %d\n", curhandlesize);
			if (curhandlesize == 0)
			{
				offset += winsize;
				continue;
			}
			t_domainnode.resize(t_domainhandle.size());
			thrust::fill(t_domainnode.begin(), t_domainnode.begin() + curhandlesize, 1);
			numberofthreads = curhandlesize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

#ifdef GQM3D_PROFILING
			cudaDeviceSynchronize();
			tv[1] = clock();
			printf("        prepare vector for domain search time = %f\n", (REAL)(tv[1] - tv[0]));
			inserttimer["prepare vector for domain search"] += (REAL)(tv[1] - tv[0]);
			tv[0] = clock();
#endif

#ifdef GQM3D_CHECKMEMORY
			printf("Domain search initialization\n");
			cudaDeviceSynchronize();
			gpuMemoryCheck();
			//checkVectorSize(t_domaincount, "t_domaincount", int);
			//checkVectorSize(t_domainhandle, "t_domainhandle", tethandle);
			//checkVectorSize(t_domainnode, "t_domainnode", int);
			//checkVectorSize(t_domainsegment, "t_domainsegment", REAL);
			//checkVectorSize(t_domainthreadlist, "t_domainthreadlist", int);
#endif

			int numofemptyhandleslot;
			int domainexpanditer = 0;
			int maxhandlesize = curhandlesize;
			bool halfwinsize = false;
			while (true)
			{
				if (curhandlesize > maxhandlesize)
					maxhandlesize = curhandlesize;

#ifdef GQM3D_LOOP_PROFILING
				cudaDeviceSynchronize();
				ltv[0] = clock();
#endif
				//printf("Domain search iteration = %d\n", domainexpanditer);
				kernelDomainSegmentAndBoxCheck << <numberofblocks, BLOCK_SIZE >> > (
					thrust::raw_pointer_cast(&t_aabbnodebbs[0]),
					thrust::raw_pointer_cast(&t_domainhandle[0]),
					thrust::raw_pointer_cast(&t_domainnode[0]),
					thrust::raw_pointer_cast(&t_domainsegment[0]),
					numberofthreads
					);

#ifdef GQM3D_DEBUG
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_LOOP_PROFILING
				cudaDeviceSynchronize();
				ltv[1] = clock();
				looptimer["kernelDomainSegmentAndBoxCheck"] += (REAL)(ltv[1] - ltv[0]);
				ltv[0] = ltv[1];
#endif
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

#ifdef GQM3D_LOOP_PROFILING
				cudaDeviceSynchronize();
				ltv[1] = clock();
				looptimer["remove_if for domain search"] += (REAL)(ltv[1] - ltv[0]);
				ltv[0] = ltv[1];
#endif

				// It seems that in some very rare case, t_domainnode would exit with positive elements
				if (domainexpanditer == aabb_level)
					break;

				// prepare enough space for new handles and nodes
				numofemptyhandleslot = t_domainhandle.size() - curhandlesize;
				if (numofemptyhandleslot < curhandlesize)
				{
					try
					{
						if (2 * curhandlesize > maxreservesize) // possible to run out of memory
						{
							// half the window size
							//printf("half the window size\n");
							winsize /= 2;
							halfwinsize = true;
							behavior->aabbwinsize = winsize;
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

#ifdef GQM3D_LOOP_PROFILING
				cudaDeviceSynchronize();
				ltv[1] = clock();
				looptimer["resize handle and node vector"] += (REAL)(ltv[1] - ltv[0]);
				ltv[0] = ltv[1];
#endif

				thrust::fill(t_domainhandle.begin() + curhandlesize, t_domainhandle.begin() + 2 * curhandlesize, tethandle(-1, 11));
				thrust::fill(t_domainnode.begin() + curhandlesize, t_domainnode.begin() + 2 * curhandlesize, 0);

#ifdef GQM3D_LOOP_PROFILING
				cudaDeviceSynchronize();
				ltv[1] = clock();
				looptimer["fill handle and node vector"] += (REAL)(ltv[1] - ltv[0]);
				ltv[0] = ltv[1];
#endif

				numberofthreads = curhandlesize;
				numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
				kernelDomainHandleAppend << <numberofblocks, BLOCK_SIZE >> > (
					thrust::raw_pointer_cast(&t_aabbnodeleft[0]),
					thrust::raw_pointer_cast(&t_aabbnoderight[0]),
					thrust::raw_pointer_cast(&t_domainhandle[0]),
					thrust::raw_pointer_cast(&t_domainnode[0]),
					numberofthreads
					);

#ifdef GQM3D_DEBUG
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_LOOP_PROFILING
				cudaDeviceSynchronize();
				ltv[1] = clock();
				looptimer["kernelDomainHandleAppend"] += (REAL)(ltv[1] - ltv[0]);
#endif

				curhandlesize = 2 * curhandlesize;
				numberofthreads = curhandlesize;
				numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

				domainexpanditer++;
			}

			if (halfwinsize)
				continue;

#ifdef GQM3D_CHECKMEMORY
			printf("After Domain search\n");
			cudaDeviceSynchronize();
			gpuMemoryCheck();
#endif

#ifdef GQM3D_PROFILING
			cudaDeviceSynchronize();
			tv[1] = clock();
			printf("        domain search time = %f\n", (REAL)(tv[1] - tv[0]));
			inserttimer["domain search"] += (REAL)(tv[1] - tv[0]);
			tv[0] = clock();
#endif
			behavior->aabbhandlesize += maxhandlesize; // record the largest handle size to use in next iteration

			if (curhandlesize == 0)
			{
				offset += winsize;
				continue;
			}

			//printf("curhandlesize = %d, tetexpandsize = %d, t_tetmarker.size() = %d, t_insertptlist.size() = %d, oldhandlesize = %d, handlesizefac = %lf\n", 
			//	curhandlesize, tetexpandsize, t_tetmarker.size(), t_insertptlist.size(), behavior->aabbhandlesize, behavior->aabbhandlesizefac);

			numberofthreads = curhandlesize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

#ifdef GQM3D_PROFILING
			cudaDeviceSynchronize();
			tv[1] = clock();
			printf("        prepare vector for intersection marking time = %f\n", (REAL)(tv[1] - tv[0]));
			inserttimer["prepare vector for intersection marking"] += (REAL)(tv[1] - tv[0]);
			tv[0] = clock();
#endif

			kernelDomainSegmentAndPrimitiveCheck << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
				thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
				thrust::raw_pointer_cast(&t_domainhandle[0]),
				thrust::raw_pointer_cast(&t_domainnode[0]),
				thrust::raw_pointer_cast(&t_domainsegment[0]),
				thrust::raw_pointer_cast(&t_domaincount[0]),
				thrust::raw_pointer_cast(&t_domainthreadlist[0]),
				thrust::raw_pointer_cast(&t_triexpandsize[0]),
				thrust::raw_pointer_cast(&t_emptytetslots[0]),
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_weightlist[0]),
				thrust::raw_pointer_cast(&t_tetlist[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				numberofthreads
				);

#ifdef GQM3D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
			cudaDeviceSynchronize();
			tv[1] = clock();
			printf("        kernelDomainSegmentAndPrimitiveCheck time = %f\n", (REAL)(tv[1] - tv[0]));
			inserttimer["kernelDomainSegmentAndPrimitiveCheck"] += (REAL)(tv[1] - tv[0]);
			tv[0] = clock();
#endif

			kernelDomainSetTriCenter << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_aabbpmcoord[0]),
				thrust::raw_pointer_cast(&t_aabbpmbbs[0]),
				thrust::raw_pointer_cast(&t_domainhandle[0]),
				thrust::raw_pointer_cast(&t_domainnode[0]),
				thrust::raw_pointer_cast(&t_domainsegment[0]),
				thrust::raw_pointer_cast(&t_domainthreadlist[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				thrust::raw_pointer_cast(&t_trifaceipt[0]),
				thrust::raw_pointer_cast(&t_tripmtidx[0]),
				numberofthreads
				);

#ifdef GQM3D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_CHECKMEMORY
			printf("After intersection points calculation\n");
			cudaDeviceSynchronize();
			gpuMemoryCheck();
#endif

#ifdef GQM3D_PROFILING
			cudaDeviceSynchronize();
			tv[1] = clock();
			printf("        kernelDomainSetTriCenter time = %f\n", (REAL)(tv[1] - tv[0]));
			inserttimer["kernelDomainSetTriCenter"] += (REAL)(tv[1] - tv[0]);
			tv[0] = clock();
#endif

			offset += winsize;
		}

		if (oldaabbhandlesize != 0)
			behavior->aabbhandlesizefac = behavior->aabbhandlesize*1.0 / oldaabbhandlesize + 0.02;
		else
			behavior->aabbhandlesizefac = 1.002;

		thrust::exclusive_scan(t_triexpandsize.begin(), t_triexpandsize.end(), t_triexpandindice.begin());
		triexpandsize = t_triexpandindice[4 * tetexpandsize - 1] + t_triexpandsize[4 * tetexpandsize - 1];
#ifdef GQM3D_DEBUG
		printf("Tri expanding size = %d\n", triexpandsize);
#endif

		numberofemptyslot = updateEmptyTriList(t_tristatus, t_emptytrislots);
		if (numberofemptyslot < triexpandsize) // dont have enough empty slots, extend lists
		{
			oldsize = t_tristatus.size();
			newsize = oldsize + triexpandsize - numberofemptyslot;
			try
			{
				t_trifacelist.resize(3 * newsize, -1);
				t_trifacecent.resize(3 * newsize);
				t_tri2tetlist.resize(2 * newsize, tethandle(-1, 11));
				t_tristatus.resize(newsize, tristatus(0));
				t_trifacepmt.resize(newsize, -1);
			}
			catch (thrust::system_error &e)
			{
				// output an error message and exit
				std::cerr << "Error: " << e.what() << std::endl;
				exit(-1);
			}
			numberofemptyslot = updateEmptyTriList(t_tristatus, t_emptytrislots);
		}

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        prepare new subface vector time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["prepare new subface vector"] += (REAL)(tv[1] - tv[0]);
		tv[0] = clock();
#endif

		numberofthreads = tetexpandsize;
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelUpdateTriAndTetStatus_Phase2 << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_emptytetslots[0]),
			thrust::raw_pointer_cast(&t_triexpandsize[0]),
			thrust::raw_pointer_cast(&t_triexpandindice[0]),
			thrust::raw_pointer_cast(&t_emptytrislots[0]),
			thrust::raw_pointer_cast(&t_trifaceipt[0]),
			thrust::raw_pointer_cast(&t_tripmtidx[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_pointtypelist[0]),
			thrust::raw_pointer_cast(&t_weightlist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_trifacecent[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_trifacepmt[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tet2trilist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			criteria->facet_angle,
			criteria->facet_size,
			criteria->facet_distance,
			numberofthreads
			);

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        kernelUpdateTriAndTetStatus_Phase2 time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["kernelUpdateTriAndTetStatus_Phase2"] += (REAL)(tv[1] - tv[0]);
		tv[0] = clock();
#endif

		// UpdateNewTetStatus
		if (insertmode == 1)
		{
			numberofthreads = tetexpandsize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelUpdateNewTetStatus << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_emptytetslots[0]),
				thrust::raw_pointer_cast(&t_domaincount[0]),
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_weightlist[0]),
				thrust::raw_pointer_cast(&t_tetlist[0]),
				thrust::raw_pointer_cast(&t_tetstatus[0]),
				criteria->cell_radius_edge_ratio,
				criteria->cell_size,
				numberofthreads
				);
		}

#ifdef GQM3D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_CHECKMEMORY
		printf("After tri and tetstatus update\n");
		cudaDeviceSynchronize();
		gpuMemoryCheck();
#endif

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        kernelUpdateNewTetStatus time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["kernelUpdateNewTetStatus"] += (REAL)(tv[1] - tv[0]);
		tv[0] = clock();
#endif
	}

	// Reset old information
	numberofthreads = updateActiveListByMarker(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        update working thread list time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["update working thread list"] += (REAL)(tv[1] - tv[0]);
	tv[0] = tv[1];
#endif

	kernelResetOldInfo << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_caveoldtettail[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

#ifdef GQM3D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_PROFILING
	cudaDeviceSynchronize();
	tv[1] = clock();
	printf("        kernelResetOldInfo time = %f\n", (REAL)(tv[1] - tv[0]));
	inserttimer["kernelResetOldInfo"] += (REAL)(tv[1] - tv[0]);
#endif

#ifdef GQM3D_CHECKMEMORY
	printf("After reset old info\n");
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

	// Update the numbers of mesh elements
	numofpoints = t_weightlist.size();
	//numofsubseg = t_segstatus.size();
	numoftriface = t_tristatus.size();
	numoftet = t_tetstatus.size();

#ifdef GQM3D_DEBUG
	int numofvalidtri, numofvalidtet;
	numofvalidtri = thrust::count_if(t_tristatus.begin(), t_tristatus.end(), isNotEmptyTri());
	numofvalidtet = thrust::count_if(t_tetstatus.begin(), t_tetstatus.end(), isNotEmptyTet());
	printf("numofpoints = %d, numoftrifaces = %d, numoftets = %d\n", 
		numofpoints, numofvalidtri, numofvalidtet);
#endif

	// Check neighbors
#ifdef GQM3D_DEBUG
	/*numberofthreads = t_pointtypelist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelCheckPointNeighbors << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_point2trilist[0]),
		thrust::raw_pointer_cast(&t_point2tetlist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		numberofthreads
		);

	numberofthreads = t_segstatus.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelCheckSubsegNeighbors << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_seg2trilist[0]),
		thrust::raw_pointer_cast(&t_seg2tetlist[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tri2seglist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		numberofthreads
		);*/

	/*numberofthreads = t_tristatus.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelCheckSubfaceNeighbors << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_seg2trilist[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tri2trilist[0]),
		thrust::raw_pointer_cast(&t_tri2seglist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		numberofthreads
		);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());*/

	//numberofthreads = t_tetstatus.size();
	//numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	//kernelCheckTetNeighbors << < numberofblocks, BLOCK_SIZE >> > (
	//	thrust::raw_pointer_cast(&t_seglist[0]),
	//	thrust::raw_pointer_cast(&t_seg2tetlist[0]),
	//	thrust::raw_pointer_cast(&t_segstatus[0]),
	//	thrust::raw_pointer_cast(&t_trifacelist[0]),
	//	thrust::raw_pointer_cast(&t_tri2tetlist[0]),
	//	thrust::raw_pointer_cast(&t_tristatus[0]),
	//	thrust::raw_pointer_cast(&t_tetlist[0]),
	//	thrust::raw_pointer_cast(&t_neighborlist[0]),
	//	thrust::raw_pointer_cast(&t_tet2trilist[0]),
	//	thrust::raw_pointer_cast(&t_tet2seglist[0]),
	//	thrust::raw_pointer_cast(&t_tetstatus[0]),
	//	numberofthreads
	//	);

	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM3D_OUTPUTMESH
	//if(iter > 300)
	//	outputAllFacets(
	//		t_pointlist,
	//		t_trifacelist,
	//		t_tristatus,
	//		t_tri2tetlist,
	//		t_tetstatus,
	//		"animation/Jacket_facet_gpu",
	//		iter
	//	);
	//outputInternalMesh(
	//	t_pointlist,
	//	t_trifacelist,
	//	t_tristatus,
	//	t_tetlist,
	//	t_tetstatus,
	//	"animation/Jacket_facet_gpu",
	//	iter
	//);
	//if(iter > 300)
	//	outputAllBadFacets(
	//		t_pointlist,
	//		t_trifacelist,
	//		t_tristatus,
	//		t_tri2tetlist,
	//		t_tetstatus,
	//		"animation/Jacket_bad_facet_gpu",
	//		iter
	//	);
#endif

	return 1;
}