#include "CudaSplitBadElement.h"
#include "CudaInitStatus.h"
#include "CudaInsertPoint.h"
#include "CudaMesh.h"
#include <time.h>

// This function splits the bad trifaces and tets iteratively
void splitBadElements(
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
	TriStatusD& t_tristatus,
	IntD& t_trifacepmt,
	TetHandleD& t_tri2tetlist,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	TetHandleD& t_neighborlist,
	TriHandleD& t_tet2trilist,
	int& numofpoints,
	int& numoftrifaces,
	int& numoftets,
	MESHCR* criteria,
	MESHIO* inputmesh,
	MESHBH* behavior
)
{
#if defined(GQM3D_PROFILING) || defined(GQM3D_ITER_PROFILING)
	clock_t tv[2];
	int npt[2];
#endif

#ifdef GQM3D_PROFLING
	inserttimer.clear();
#endif

	clock_t tvmain[2];
	tvmain[0] = clock();

	// Init status
	initTriTetQuality(
		t_pointlist,
		t_pointtypelist,
		t_weightlist,
		t_trifacelist,
		t_trifacecent,
		t_tristatus,
		t_tetlist,
		t_tetstatus,
		criteria,
		numoftrifaces,
		numoftets
	);

	cudaDeviceSynchronize();
	tvmain[1] = clock();
	behavior->times[4] = (REAL)(tvmain[1] - tvmain[0]);
	tvmain[0] = tvmain[1];

	// internal vectors
	IntD t_badelementlist;
	IntD t_badtrifacelist, t_badtetlist;
	IntD t_threadmarker;

	TetHandleD t_recordoldtetlist;
	IntD t_recordoldtetidx;

	int numberofbadelements;
	int numberofbadtrifaces, numberofbadtets;

	int code = 1;
	int iteration = 0;
	int counter;
	int insertmode;

	// Split subfaces
	printf("      Splitting subfaces......\n");
	if (behavior->R4)
		insertmode = 0; // sperate domain test for subface and tetrahedron
	else
		insertmode = 1;

#ifdef GQM3D_ITER_PROFILING
	cudaDeviceSynchronize();
	tv[0] = clock();
#endif

	while (true)
	{
#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[0] = clock();
#endif

		// Update the active bad elements list.
		// Exclude the empty ones.
		numberofbadtrifaces = updateBadTriList(t_tristatus, t_badtrifacelist, numoftrifaces);
		if (numberofbadtrifaces == 0)
			break;

		if (behavior->minbadtrifaces > 0)
		{
			if (numberofbadtrifaces <= behavior->minbadtrifaces)
			{
				code = 0;
				break;
			}
		}

		if (behavior->mintriiter != 0 && behavior->mintriiter <= iteration)
		{
			code = 0;
			break;
		}

		numberofbadelements = numberofbadtrifaces;
		if (behavior->verbose) 
			printf("      Subface Iteration #%d: number of bad subfaces = %d\n", 
				iteration, numberofbadelements);

		t_badelementlist.resize(numberofbadelements);
		thrust::copy_n(t_badtrifacelist.begin(), numberofbadtrifaces, t_badelementlist.begin());

		t_threadmarker.resize(numberofbadelements);
		thrust::fill_n(t_threadmarker.begin(), numberofbadtrifaces, 1);

#ifdef GQM3D_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("      prepare bad element list time = %f\n", (REAL)(tv[1] - tv[0]));
		inserttimer["prepare bad element list"] += (REAL)(tv[1] - tv[0]);
		tv[0] = tv[1];
#endif
#if defined(GQM3D_PROFILING) || defined(GQM3D_ITER_PROFILING)
		npt[0] = numofpoints;
#endif


		// Insert points concurrently
		//if (behavior->R3)
		if(true)
		{
			code =
				insertPoint_New(
					t_aabbnodeleft,
					t_aabbnoderight,
					t_aabbnodebbs,
					t_aabbpmcoord,
					t_aabbpmbbs,
					t_recordoldtetlist,
					t_recordoldtetidx,
					t_pointlist,
					t_weightlist,
					t_pointtypelist,
					t_pointpmt,
					t_trifacelist,
					t_trifacecent,
					t_tri2tetlist,
					t_tristatus,
					t_trifacepmt,
					t_tetlist,
					t_neighborlist,
					t_tet2trilist,
					t_tetstatus,
					t_badelementlist,
					t_threadmarker,
					numberofbadelements,
					numberofbadtrifaces,
					0,
					numofpoints,
					numoftrifaces,
					numoftets,
					criteria,
					inputmesh,
					behavior,
					insertmode,
					iteration
				);
		}
		else
		{
			code =
				insertPoint(
					t_aabbnodeleft,
					t_aabbnoderight,
					t_aabbnodebbs,
					t_aabbpmcoord,
					t_aabbpmbbs,
					t_pointlist,
					t_weightlist,
					t_pointtypelist,
					t_pointpmt,
					t_trifacelist,
					t_trifacecent,
					t_tri2tetlist,
					t_tristatus,
					t_trifacepmt,
					t_tetlist,
					t_neighborlist,
					t_tet2trilist,
					t_tetstatus,
					t_badelementlist,
					t_threadmarker,
					numberofbadelements,
					numberofbadtrifaces,
					0,
					numofpoints,
					numoftrifaces,
					numoftets,
					criteria,
					inputmesh,
					behavior,
					insertmode,
					iteration
				);
		}

		if (!code)
			break;

#if defined(GQM3D_PROFILING) || defined(GQM3D_ITER_PROFILING)
		cudaDeviceSynchronize();
		tv[1] = clock();
		npt[1] = numofpoints;
#ifdef GQM3D_PROFILING
		printf("      Subface Iteration #%d: time = %f, number of points inserted = %d\n\n", iteration, (REAL)(tv[1] - tv[0]), npt[1] - npt[0]);
#else
		printf("%d, %f, %d, %d\n", iteration, (REAL)(tv[1] - tv[0]), numberofbadelements, npt[1] - npt[0]);
#endif
#endif

		iteration++;
	}

	cudaDeviceSynchronize();
	tvmain[1] = clock();
	behavior->times[5] = (REAL)(tvmain[1] - tvmain[0]);
	printf("        Splitting subfaces time = %f\n", behavior->times[5]);
	tvmain[0] = tvmain[1];

	// Update tet status: quality and domain test
	if ((criteria->facet_angle != 0.0 || criteria->facet_size != 0.0 || criteria->facet_distance != 0.0) && insertmode == 0) // mesh has changed
	{
#if defined(GQM3D_PROFILING) || defined(GQM3D_ITER_PROFILING)
		cudaDeviceSynchronize();
		tv[0] = clock();
#endif

		printf("      Setting #%d tetrahedra status......\n", numoftets);
		if (behavior->aabbmode == 1)
		{
			initTetstatus(
				t_aabbnodeleft,
				t_aabbnoderight,
				t_aabbnodebbs,
				t_aabbpmcoord,
				t_aabbpmbbs,
				t_pointlist,
				t_weightlist,
				t_tetlist,
				t_tetstatus,
				criteria,
				inputmesh->aabb_diglen,
				numoftets
			);
		}
		else if (behavior->aabbmode == 2)
		{
			initTetstatus2(
				t_aabbnodeleft,
				t_aabbnoderight,
				t_aabbnodebbs,
				t_aabbpmcoord,
				t_aabbpmbbs,
				t_pointlist,
				t_weightlist,
				t_tetlist,
				t_tetstatus,
				criteria,
				inputmesh,
				numoftets
			);
		}

#if defined(GQM3D_PROFILING) || defined(GQM3D_ITER_PROFILING)
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("      initTetstatus time = %f\n", (REAL)(tv[1] - tv[0]));
#ifdef GQM3D_PROFILING
		inserttimer["initTetstatus"] = (REAL)(tv[1] - tv[0]);
#endif
#endif
	}

	cudaDeviceSynchronize();
	tvmain[1] = clock();
	behavior->times[6] = (REAL)(tvmain[1] - tvmain[0]);
	printf("        Set tet status time = %f\n", behavior->times[6]);
	tvmain[0] = tvmain[1];

	// Split tets and subfaces together
	if (inputmesh->aabb_closed)
	{
		printf("      Splitting tets and subfaces......\n");
		insertmode = 1;
		iteration = 0;
		if (behavior->R4)
		{
			while (true)
			{

#ifdef GQM3D_PROFILING
				cudaDeviceSynchronize();
				tv[0] = clock();
#endif

				// Update the active bad elements list.
				// Exclude the empty ones.
				numberofbadtrifaces = updateBadTriList(t_tristatus, t_badtrifacelist, numoftrifaces);
				numberofbadtets = updateBadTetList(t_tetstatus, t_badtetlist, numoftets);

				if (numberofbadtrifaces == 0 && numberofbadtets == 0)
					break;

				if (behavior->minbadtets > 0)
				{
					if (numberofbadtets <= behavior->minbadtets)
					{
						code = 0;
						break;
					}
				}

				if (behavior->mintetiter != 0 && behavior->mintetiter <= iteration)
				{
					code = 0;
					break;
				}

				//if (!behavior->R4 && numberofbadtrifaces > 0) // do not split tets and subfaces together
				//	numberofbadtets = 0;

				numberofbadelements = numberofbadtrifaces + numberofbadtets;
				if (behavior->verbose) printf("      Tet Iteration #%d: number of bad elements = %d (#%d subfaces, #%d tets)\n",
					iteration, numberofbadelements, numberofbadtrifaces, numberofbadtets);

				t_badelementlist.resize(numberofbadelements);
				thrust::copy_n(t_badtrifacelist.begin(), numberofbadtrifaces, t_badelementlist.begin());
				thrust::copy_n(t_badtetlist.begin(), numberofbadtets, t_badelementlist.begin() + numberofbadtrifaces);

				t_threadmarker.resize(numberofbadelements);
				thrust::fill_n(t_threadmarker.begin(), numberofbadtrifaces, 1);
				thrust::fill_n(t_threadmarker.begin() + numberofbadtrifaces, numberofbadtets, 2);

#ifdef GQM3D_PROFILING
				cudaDeviceSynchronize();
				tv[1] = clock();
				printf("      prepare bad element list time = %f\n", (REAL)(tv[1] - tv[0]));
				inserttimer["prepare bad element list"] += (REAL)(tv[1] - tv[0]);
				tv[0] = tv[1];
#endif

#if defined(GQM3D_PROFILING) || defined(GQM3D_ITER_PROFILING)
				npt[0] = numofpoints;
#endif

				// Insert points concurrently
				if (behavior->R3)
				{
					code =
						insertPoint_New(
							t_aabbnodeleft,
							t_aabbnoderight,
							t_aabbnodebbs,
							t_aabbpmcoord,
							t_aabbpmbbs,
							t_recordoldtetlist,
							t_recordoldtetidx,
							t_pointlist,
							t_weightlist,
							t_pointtypelist,
							t_pointpmt,
							t_trifacelist,
							t_trifacecent,
							t_tri2tetlist,
							t_tristatus,
							t_trifacepmt,
							t_tetlist,
							t_neighborlist,
							t_tet2trilist,
							t_tetstatus,
							t_badelementlist,
							t_threadmarker,
							numberofbadelements,
							numberofbadtrifaces,
							numberofbadtets,
							numofpoints,
							numoftrifaces,
							numoftets,
							criteria,
							inputmesh,
							behavior,
							1,
							iteration
						);
				}
				else
				{
					behavior->aabbmode = 2;

					code =
						insertPoint(
							t_aabbnodeleft,
							t_aabbnoderight,
							t_aabbnodebbs,
							t_aabbpmcoord,
							t_aabbpmbbs,
							t_pointlist,
							t_weightlist,
							t_pointtypelist,
							t_pointpmt,
							t_trifacelist,
							t_trifacecent,
							t_tri2tetlist,
							t_tristatus,
							t_trifacepmt,
							t_tetlist,
							t_neighborlist,
							t_tet2trilist,
							t_tetstatus,
							t_badelementlist,
							t_threadmarker,
							numberofbadelements,
							numberofbadtrifaces,
							numberofbadtets,
							numofpoints,
							numoftrifaces,
							numoftets,
							criteria,
							inputmesh,
							behavior,
							1,
							iteration
						);
				}

				if (!code)
					break;

#if defined(GQM3D_PROFILING) || defined(GQM3D_ITER_PROFILING)
				cudaDeviceSynchronize();
				tv[1] = clock();
				npt[1] = numofpoints;
#ifdef GQM3D_PROFILING
				printf("      Iteration #%d: time = %f, number of points inserted = %d\n", iteration, (REAL)(tv[1] - tv[0]), npt[1] - npt[0]);
#else
				printf("%d, %f, %d\n", iteration, (REAL)(tv[1] - tv[0]), npt[1] - npt[0]);
#endif
#endif

				iteration++;
			}
		}
		else
		{
			while (true)
			{

#ifdef GQM3D_PROFILING
				cudaDeviceSynchronize();
				tv[0] = clock();
#endif

				// Update the active bad elements list.
				// Exclude the empty ones.
				numberofbadtets = updateBadTetList(t_tetstatus, t_badtetlist, numoftets);

				if (numberofbadtets == 0)
					break;

				if (behavior->minbadtets > 0)
				{
					if (numberofbadtets <= behavior->minbadtets)
					{
						code = 0;
						break;
					}
				}

				if (behavior->mintetiter != 0 && behavior->mintetiter <= iteration)
				{
					code = 0;
					break;
				}

				numberofbadelements = numberofbadtets;
				if (behavior->verbose) printf("      Tet Iteration #%d: number of bad elements = %d (#%d subfaces, #%d tets)\n",
					iteration, numberofbadelements, 0, numberofbadtets);

				t_badelementlist.resize(numberofbadelements);
				thrust::copy_n(t_badtetlist.begin(), numberofbadtets, t_badelementlist.begin());

				t_threadmarker.resize(numberofbadelements);
				thrust::fill_n(t_threadmarker.begin(), numberofbadtets, 2);

#ifdef GQM3D_PROFILING
				cudaDeviceSynchronize();
				tv[1] = clock();
				printf("      prepare bad element list time = %f\n", (REAL)(tv[1] - tv[0]));
				inserttimer["prepare bad element list"] += (REAL)(tv[1] - tv[0]);
				tv[0] = tv[1];
#endif

#if defined(GQM3D_PROFILING) || defined(GQM3D_ITER_PROFILING)
				npt[0] = numofpoints;
#endif

				// Insert points concurrently
				if (behavior->R3)
				{
					code =
						insertPoint_New(
							t_aabbnodeleft,
							t_aabbnoderight,
							t_aabbnodebbs,
							t_aabbpmcoord,
							t_aabbpmbbs,
							t_recordoldtetlist,
							t_recordoldtetidx,
							t_pointlist,
							t_weightlist,
							t_pointtypelist,
							t_pointpmt,
							t_trifacelist,
							t_trifacecent,
							t_tri2tetlist,
							t_tristatus,
							t_trifacepmt,
							t_tetlist,
							t_neighborlist,
							t_tet2trilist,
							t_tetstatus,
							t_badelementlist,
							t_threadmarker,
							numberofbadelements,
							0,
							numberofbadtets,
							numofpoints,
							numoftrifaces,
							numoftets,
							criteria,
							inputmesh,
							behavior,
							1,
							iteration
						);
				}
				else
				{
					behavior->aabbmode = 2;

					code =
						insertPoint(
							t_aabbnodeleft,
							t_aabbnoderight,
							t_aabbnodebbs,
							t_aabbpmcoord,
							t_aabbpmbbs,
							t_pointlist,
							t_weightlist,
							t_pointtypelist,
							t_pointpmt,
							t_trifacelist,
							t_trifacecent,
							t_tri2tetlist,
							t_tristatus,
							t_trifacepmt,
							t_tetlist,
							t_neighborlist,
							t_tet2trilist,
							t_tetstatus,
							t_badelementlist,
							t_threadmarker,
							numberofbadelements,
							0,
							numberofbadtets,
							numofpoints,
							numoftrifaces,
							numoftets,
							criteria,
							inputmesh,
							behavior,
							1,
							iteration
						);
				}

				int siteration = 0;
				while (true)
				{
					// Update the active bad elements list.
					// Exclude the empty ones.
					numberofbadtrifaces = updateBadTriList(t_tristatus, t_badtrifacelist, numoftrifaces);
					if (numberofbadtrifaces == 0)
						break;

					numberofbadelements = numberofbadtrifaces;
					if (behavior->verbose)
						printf("      Subface Iteration #%d: number of bad subfaces = %d\n",
							siteration, numberofbadelements);

					t_badelementlist.resize(numberofbadelements);
					thrust::copy_n(t_badtrifacelist.begin(), numberofbadtrifaces, t_badelementlist.begin());

					t_threadmarker.resize(numberofbadelements);
					thrust::fill_n(t_threadmarker.begin(), numberofbadtrifaces, 1);

					// Insert points concurrently
					if (behavior->R3)
					{
						code =
							insertPoint_New(
								t_aabbnodeleft,
								t_aabbnoderight,
								t_aabbnodebbs,
								t_aabbpmcoord,
								t_aabbpmbbs,
								t_recordoldtetlist,
								t_recordoldtetidx,
								t_pointlist,
								t_weightlist,
								t_pointtypelist,
								t_pointpmt,
								t_trifacelist,
								t_trifacecent,
								t_tri2tetlist,
								t_tristatus,
								t_trifacepmt,
								t_tetlist,
								t_neighborlist,
								t_tet2trilist,
								t_tetstatus,
								t_badelementlist,
								t_threadmarker,
								numberofbadelements,
								numberofbadtrifaces,
								0,
								numofpoints,
								numoftrifaces,
								numoftets,
								criteria,
								inputmesh,
								behavior,
								insertmode,
								iteration
							);
					}
					else
					{
						code =
							insertPoint(
								t_aabbnodeleft,
								t_aabbnoderight,
								t_aabbnodebbs,
								t_aabbpmcoord,
								t_aabbpmbbs,
								t_pointlist,
								t_weightlist,
								t_pointtypelist,
								t_pointpmt,
								t_trifacelist,
								t_trifacecent,
								t_tri2tetlist,
								t_tristatus,
								t_trifacepmt,
								t_tetlist,
								t_neighborlist,
								t_tet2trilist,
								t_tetstatus,
								t_badelementlist,
								t_threadmarker,
								numberofbadelements,
								numberofbadtrifaces,
								0,
								numofpoints,
								numoftrifaces,
								numoftets,
								criteria,
								inputmesh,
								behavior,
								insertmode,
								iteration
							);
					}

					siteration++;
				}


				iteration++;
			}
		}
	}
	else
	{
		printf("        The input surface is not closed!\n");
	}

	cudaDeviceSynchronize();
	tvmain[1] = clock();
	behavior->times[7] = (REAL)(tvmain[1] - tvmain[0]);
	printf("        Splitting tets time = %f\n", behavior->times[7]);
#ifdef GQM3D_PROFILING
	double sum = 0;
	std::map<std::string, double>::iterator it;
	for (it = inserttimer.begin(); it != inserttimer.end(); it++)
	{
		std::cout << "Total " << it->first << " time = " << it->second << std::endl;
		sum += it->second;
	}
	std::cout << "Total time = " << sum << std::endl;
#endif

#ifdef GQM3D_LOOP_PROFILING
	double loop_sum = 0;
	std::map<std::string, double>::iterator it2;
	for (it2 = looptimer.begin(); it2 != looptimer.end(); it2++)
	{
		std::cout << "Total " << it2->first << " time = " << it2->second << std::endl;
		loop_sum += it2->second;
	}
	std::cout << "Total time = " << loop_sum << std::endl;
#endif
}