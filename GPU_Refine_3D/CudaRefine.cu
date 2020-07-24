#include <stdio.h>
#include "CudaRefine.h"
#include "MeshReconstruct.h"
#include "CudaMesh.h"
#include "CudaThrust.h"
#include "CudaSplitBadElement.h"
#include "CudaCompactMesh.h"

/*************************************************************************************/
/*																					 */
/*  GPU_Refine_3D()   Compute 3D restricted Delaunay refinement on GPU.	         */
/*                                                                                   */
/*************************************************************************************/

void GPU_Refine_3D(
	MESHCR* input_criteria,
	MESHIO* input_mesh,
	MESHBH* input_behavior
)
{
	//gpuMemoryCheck();

	/* Check input behavior */
	if (input_behavior->mode != 1 && input_behavior->mode != 2)
	{
		printf("Unknown input mode: #%d\n", input_behavior->mode);
		exit(0);
	}

	/* Set input behavior */
	if (input_behavior->R2)
	{
		//input_behavior->filtermode = 2;
		input_behavior->aabbshortcut = 2;
	}
	else
	{
		input_behavior->filtermode = 1;
		input_behavior->aabbshortcut = 1;
	}

	if (input_behavior->R3)
		input_behavior->aabbmode = 2;
	else
		input_behavior->aabbmode = 1;

	if (input_behavior->R5)
		input_behavior->cavitymode = 2;
	else
		input_behavior->cavitymode = 1;

	internalmesh* drawmesh = input_behavior->drawmesh;

	/* Set up timer */
	StopWatchInterface *inner_timer = 0;
	sdkCreateTimer(&inner_timer);

	/******************************************/
	/* 0. Reconstruct the input cdt mesh      */
	/******************************************/
	printf("   0. Reconstructing the input mesh...\n");
	
	// Reset and start timer.
	sdkResetTimer( &inner_timer );
	sdkStartTimer( &inner_timer );

	// input variables
	verttype* inpointtypelist;
	tethandle* intri2tetlist;
	int innumoftetrahedra;
	int* intetlist;
	tethandle* inneighborlist;
	trihandle* intet2trilist;

	// reconstruct mesh
	reconstructMesh(
		input_mesh,
		inpointtypelist,
		intri2tetlist,
		innumoftetrahedra,
		intetlist,
		inneighborlist,
		intet2trilist,
		false
	);

	// stop timer
	sdkStopTimer(&inner_timer);

	// print out info
	printf("      Reconstructed Mesh Size:\n");
	printf("        Number of points = %d\n", input_mesh->numofpoints);
	printf("        Number of tetrahedra = %d\n", innumoftetrahedra);
	printf("      Reconstruction time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
	input_behavior->times[0] = sdkGetTimerValue(&inner_timer);

	/******************************************/
	/* 1. Initialization				      */
	/******************************************/
	printf("   1. Initialization\n");

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	// Control variables
	int last_point = input_mesh->numofpoints;
	int last_triface = input_mesh->numoftrifaces;
	int last_tet = innumoftetrahedra;
	int last_aabbnode = input_mesh->numofaabbnodes;
	int last_aabbpms = input_mesh->numofaabbpms;

	//gpuMemoryCheck();

	// Input mesh arrays, copy from the host
	RealD t_pointlist(input_mesh->pointlist, input_mesh->pointlist + 3 * last_point);
	RealD t_weightlist(input_mesh->weightlist, input_mesh->weightlist + last_point);
	PointTypeD t_pointtypelist(inpointtypelist, inpointtypelist + last_point);
	IntD t_pointpmt(last_point, -1);
	IntD t_trifacelist(input_mesh->trifacelist, input_mesh->trifacelist + 3 * last_triface);
	RealD t_trifacecent(input_mesh->trifacecentlist, input_mesh->trifacecentlist + 3 * last_triface);
	TriStatusD t_tristatus(last_triface, tristatus(1));
	IntD t_trifacepmt(last_triface, -1);
	TetHandleD t_tri2tetlist(intri2tetlist, intri2tetlist + 2 * last_triface);
	IntD t_tetlist(intetlist, intetlist + 4 * last_tet);
	TetStatusD t_tetstatus(input_mesh->tetstatuslist, input_mesh->tetstatuslist + input_mesh->numoftets);
	t_tetstatus.resize(last_tet, tetstatus(1)); // set hull tets to non-empty only
	TetHandleD t_neighborlist(inneighborlist, inneighborlist + 4 * last_tet);
	TriHandleD t_tet2trilist(intet2trilist, intet2trilist + 4 * last_tet);

	//gpuMemoryCheck();

	// AABB tree
	IntD t_aabbnodeleft(input_mesh->aabb_nodeleftchild, input_mesh->aabb_nodeleftchild + last_aabbnode);
	IntD t_aabbnoderight(input_mesh->aabb_noderightchild, input_mesh->aabb_noderightchild + last_aabbnode);
	RealD t_aabbnodebbs(input_mesh->aabb_nodebbs, input_mesh->aabb_nodebbs + 6 * last_aabbnode);
	RealD t_aabbpmcoord(input_mesh->aabb_pmcoord, input_mesh->aabb_pmcoord + 9 * last_aabbpms);
	RealD t_aabbpmbbs(input_mesh->aabb_pmbbs, input_mesh->aabb_pmbbs + 6 * last_aabbpms);

	//gpuMemoryCheck();

	// Cuda mesh manipulation
	int xmax, xmin, ymax, ymin, zmax, zmin;
	cudamesh_inittables();
	cudamesh_initbbox(input_mesh->numofpoints, input_mesh->pointlist,
		xmax, xmin, ymax, ymin, zmax, zmin);
	cudamesh_exactinit(0, 0, 0, xmax - xmin, ymax - ymin, zmax - zmin);
	cudamesh_initkernelconstants(xmax - xmin, ymax - ymin, zmax - zmin);

	// stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&inner_timer);
	printf("      Initialization time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
	input_behavior->times[1] = sdkGetTimerValue(&inner_timer);

	printf("   2. Split bad elements\n");

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	// split encroached segments
	splitBadElements(
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
		t_tristatus,
		t_trifacepmt,
		t_tri2tetlist,
		t_tetlist,
		t_tetstatus,
		t_neighborlist,
		t_tet2trilist,
		last_point,
		last_triface,
		last_tet,
		input_criteria,
		input_mesh,
		input_behavior
	);

	// stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&inner_timer);

	// print out info
	printf("      Splitting bad elements time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
	//printf("      Number of points = %d, trifaces = %d, tets = %d\n",
	//	last_point,
	//	thrust::count_if(t_tristatus.begin(), t_tristatus.end(), isNotEmptyTri()),
	//	thrust::count_if(t_tetstatus.begin(), t_tetstatus.end(), isNotEmptyTet()));
	input_behavior->times[2] = sdkGetTimerValue(&inner_timer);

	printf("   3. Output final quality mesh\n");

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	compactMesh(
		input_mesh->out_numofpoints, input_mesh->out_pointlist, input_mesh->out_weightlist,
		t_pointlist, t_weightlist,
		input_mesh->out_numoftrifaces, input_mesh->out_trifacelist, input_mesh->out_trifacecent, 
		t_trifacelist, t_trifacecent, t_tristatus, t_tri2tetlist,
		input_mesh->out_numoftets, input_mesh->out_numoftets_indomain,
		input_mesh->out_tetlist, input_mesh->out_tetstatus,
		t_tetlist, t_tetstatus
		);

	// stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&inner_timer);
	input_behavior->times[3] = sdkGetTimerValue(&inner_timer);
}