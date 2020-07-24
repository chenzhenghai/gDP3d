#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include "MeshStructure.h"
#include "CudaThrust.h"


#define REAL double

#define EPSILON 1.0e-8
#define EPSILON2 1.0e-10

#define PI 3.141592653589793238462643383279502884197169399375105820974944592308

#define BLOCK_SIZE 256

#define MAXINT 2147483647
#define MAXUINT 0xFFFFFFFF
#define MAXULL 0xFFFFFFFFFFFFFFFF
#define MAXFLT 3.402823466e+38F

#define MAXAABBLEVEL 31

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Helpers																	 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ uint64 cudamesh_encodeUInt64Priority(int priority, int index);

__device__ int cudamesh_getUInt64PriorityIndex(uint64 priority);

__device__ int cudamesh_getUInt64Priority(uint64 priority);

__device__ bool cudamesh_isNearZero(double val);

__device__ bool cudamesh_isInvalid(double val);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Domain helpers														     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ bool cudamesh_traversal_first_intersection(REAL* s, REAL* t, 
	int* aabbnodeleft, int* aabbnoderight, REAL* aabbnodebbs, 
	REAL* aabbpmcoord, REAL* aabbpmbbs, REAL* ipt, int& pmidx);

__device__ bool cudamesh_traversal_in_domain(REAL* wc, int* aabbnodeleft, 
	int* aabbnoderight, REAL* aabbnodebbs, REAL* aabbpmcoord, REAL* aabbpmbbs);

__device__ bool cudamesh_traversal_in_domain(REAL* s, REAL* t, int* aabbnodeleft,
	int* aabbnoderight, REAL* aabbnodebbs, REAL* aabbpmcoord, REAL* aabbpmbbs);

__device__ void cudamesh_box_far_point(REAL* s, REAL* t, REAL xmin, REAL xmax,
	REAL ymin, REAL ymax, REAL zmin, REAL zmax, REAL len);

__device__ void cudamesh_primitive_fast_check(int* list, int numofaabbpms);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Basic helpers    														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
__device__ void cudamesh_swap(int& a, int& b);

__device__ void cudamesh_swap(REAL* pa, REAL* pb);

__device__ void cudamesh_copy(REAL* d, REAL* s);

__device__ int cudamesh_compare(REAL* pa, REAL* pb);

__device__ REAL cudamesh_min(REAL v1, REAL v2);

__device__ REAL cudamesh_min(REAL v1, REAL v2, REAL v3);

__device__ REAL cudamesh_max(REAL v1, REAL v2);

__device__ REAL cudamesh_max(REAL v1, REAL v2, REAL v3);

__device__ REAL cudamesh_dot(REAL* v1, REAL* v2);

__device__ REAL cudamesh_distance(REAL* p1, REAL* p2);

__device__ REAL cudamesh_squared_distance(REAL* p1, REAL* p2);

__device__ REAL cudamesh_power_distance(REAL* p1, REAL* p2, REAL w1, REAL w2);

__device__ void cudamesh_cross(REAL* v1, REAL* v2, REAL* n);

__device__ unsigned long cudamesh_randomnation(unsigned long* randomseed, unsigned int choices);

__device__ void cudamesh_random_sphere_point(unsigned long * randomseed, REAL* p);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Geometric helpers														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ bool cudamesh_lp_intersection(REAL* p, REAL* q, REAL* a, REAL* b,
	REAL* c, REAL* ipt);

__device__ bool cudamesh_ts_intersection(REAL* p, REAL* q, REAL* a, REAL* b,
	REAL* c, REAL* ipt);

__device__ bool cudamesh_ts_intersection(REAL* A, REAL* B, REAL* C,
	REAL* P, REAL* Q, int& type);

__device__ bool cudamesh_lu_decmp(REAL lu[4][4], int n, int* ps, REAL* d, int N);

__device__ void cudamesh_lu_solve(REAL lu[4][4], int n, int* ps, REAL* b, int N);

__device__ bool cudamesh_circumsphere(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL* cent, REAL* radius); // from TetGen

__device__ bool cudamesh_weightedcircumsphere(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL  aw, REAL bw, REAL cw, REAL  dw, REAL* cent, REAL* radius); // from TetGen

__device__ bool cudamesh_circumcenter(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL* cent); // from CGAL

__device__ bool cudamesh_weightedcircumcenter(REAL* pa, REAL* pb, REAL* pc,
	REAL aw, REAL bw, REAL cw, REAL* cent); // from CGAL

__device__ bool cudamesh_weightedcircumcenter(REAL* pa, REAL* pb, REAL* pc, REAL* pd, 
	REAL aw, REAL bw, REAL cw, REAL dw, REAL* cent); // from CGAL

__device__ bool cudamesh_weightedcircumcenter_equal(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL aw, REAL bw, REAL cw, REAL dw, REAL* cent);

__device__ bool cudamesh_weightedcircumcenter_perturbed(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL aw, REAL bw, REAL cw, REAL dw, REAL* cent, REAL* norm);

__device__ bool cudamesh_isDegenerateTet(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL aw, REAL bw, REAL cw, REAL dw, REAL* cent);

__device__ bool cudamesh_compute_squared_radius_smallest_orthogonal_sphere(REAL* pa, 
	REAL* pb, REAL aw, REAL bw, REAL& sradius);

__device__ bool cudamesh_compute_squared_radius_smallest_orthogonal_sphere(REAL* pa,
	REAL* pb, REAL* pc, REAL aw, REAL bw, REAL cw, REAL& sradius);

__device__ bool cudamesh_compute_squared_radius_smallest_orthogonal_sphere(REAL* pa,
	REAL* pb, REAL* pc, REAL* pd, REAL aw, REAL bw, REAL cw, REAL dw, REAL& sradius);

__device__ bool cudamesh_raydir(REAL* pa, REAL* pb, REAL* pc, REAL* dir);

__device__ void cudamesh_facenormal(REAL* pa, REAL* pb, REAL* pc, REAL *n, int pivot,
	REAL* lav);

__device__ REAL cudamesh_bbox_diglen(int nodeId, REAL* aabbnodebbs);

__device__ REAL cudamesh_triangle_squared_area(REAL* pa, REAL* pb, REAL* pc);

__device__ REAL cudamesh_tetrahedronvolume(REAL* pa, REAL* pb, REAL* pc, REAL* pd);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Geometric predicates														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ REAL cudamesh_insphere_s(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe,
	int ia, int ib, int ic, int id, int ie);

__device__ REAL cudamesh_orient4d_s(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe,
	int ia, int ib, int ic, int id, int ie, REAL aw, REAL bw, REAL cw, REAL dw, REAL ew);

__device__ REAL cudamesh_incircle3d(REAL* pa, REAL* pb, REAL* pc, REAL* pd);

__device__ bool cudamesh_is_out_bbox(REAL* pt, REAL xmin, REAL xmax, 
	REAL ymin, REAL ymax, REAL zmin, REAL zmax);

__device__ bool cudamesh_do_intersect_bbox(REAL* s, REAL* t, REAL xmin, REAL xmax,
	REAL ymin, REAL ymax, REAL zmin, REAL zmax);

__device__ bool cudamesh_is_bad_facet(REAL *pa, REAL *pb, REAL* pc,
	REAL aw, REAL bw, REAL cw, verttype at, verttype bt, verttype ct, REAL* cent,
	REAL facet_angle, REAL facet_size, REAL facet_distance);

__device__ bool cudamesh_is_bad_tet(REAL *pa, REAL *pb, REAL* pc, REAL* pd,
	REAL aw, REAL bw, REAL cw, REAL dw,
	REAL cell_radius_edge_ratio, REAL cell_size);

__device__ bool cudamesh_is_encroached_facet_splittable(REAL *pa, REAL *pb, REAL* pc,
	REAL aw, REAL bw, REAL cw);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Mesh manipulation primitives                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Init fast lookup tables */
void cudamesh_inittables();

/* Init bounding box*/
void cudamesh_initbbox(
	int numofpoints, double* pointlist,
	int& xmax, int& xmin, int& ymax, int& ymin, int& zmax, int& zmin);

/* Init Geometric primitives */
void cudamesh_exactinit(int verbose, int noexact, int nofilter, 
	REAL maxx, REAL maxy, REAL maxz);

/* Init Kernel constants */
void cudamesh_initkernelconstants(REAL maxx, REAL maxy, REAL maxz);

/* Primitives for points */
// Convert point index to pointer to pointlist
__device__ double* cudamesh_id2pointlist(int index, double* pointlist);

/* Primitives for tetrahedron */

__device__ int cudamesh_org(tethandle t, int* tetlist);
__device__ int cudamesh_dest(tethandle t, int* tetlist);
__device__ int cudamesh_apex(tethandle t, int* tetlist);
__device__ int cudamesh_oppo(tethandle t, int* tetlist);
__device__ void cudamesh_setorg(tethandle t, int p, int* tetlist);
__device__ void cudamesh_setdest(tethandle t, int p, int* tetlist);
__device__ void cudamesh_setapex(tethandle t, int p, int* tetlist);
__device__ void cudamesh_setoppo(tethandle t, int p, int* tetlist);

__device__ void cudamesh_bond(tethandle t1, tethandle t2, tethandle* neighborlist);
__device__ void cudamesh_dissolve(tethandle t, tethandle* neighborlist);

__device__ void cudamesh_esym(tethandle& t1, tethandle& t2);
__device__ void cudamesh_esymself(tethandle& t);
__device__ void cudamesh_enext(tethandle& t1, tethandle& t2);
__device__ void cudamesh_enextself(tethandle& t);
__device__ void cudamesh_eprev(tethandle& t1, tethandle& t2);
__device__ void cudamesh_eprevself(tethandle& t);
__device__ void cudamesh_enextesym(tethandle& t1, tethandle& t2);
__device__ void cudamesh_enextesymself(tethandle& t);
__device__ void cudamesh_eprevesym(tethandle& t1, tethandle& t2);
__device__ void cudamesh_eprevesymself(tethandle& t);
__device__ void cudamesh_eorgoppo(tethandle& t1, tethandle& t2);
__device__ void cudamesh_eorgoppoself(tethandle& t);
__device__ void cudamesh_edestoppo(tethandle& t1, tethandle& t2);
__device__ void cudamesh_edestoppoself(tethandle& t);

__device__ void cudamesh_fsym(tethandle& t1, tethandle& t2, tethandle* neighborlist);
__device__ void cudamesh_fsymself(tethandle& t, tethandle* neighborlist);
__device__ void cudamesh_fnext(tethandle& t1, tethandle& t2, tethandle* neigenhborlist);
__device__ void cudamesh_fnextself(tethandle& t, tethandle* neighborlist);

__device__ bool cudamesh_ishulltet(tethandle t, int* tetlist);
__device__ bool cudamesh_isdeadtet(tethandle t);

// Primitives for subfaces and subsegments.
__device__ void cudamesh_spivot(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
__device__ void cudamesh_spivotself(trihandle& s, trihandle* tri2trilist);
__device__ void cudamesh_sbond(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
__device__ void cudamesh_sbond1(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
__device__ void cudamesh_sdissolve(trihandle& s, trihandle* tri2trilist);
__device__ int cudamesh_sorg(trihandle& s, int* trilist);
__device__ int cudamesh_sdest(trihandle& s, int* trilist);
__device__ int cudamesh_sapex(trihandle& s, int* trilist);
__device__ void cudamesh_setsorg(trihandle& s, int p, int* trilist);
__device__ void cudamesh_setsdest(trihandle& s, int p, int* trilist);
__device__ void cudamesh_setsapex(trihandle& s, int p, int* trilist);
__device__ void cudamesh_sesym(trihandle& s1, trihandle& s2);
__device__ void cudamesh_sesymself(trihandle& s);
__device__ void cudamesh_senext(trihandle& s1, trihandle& s2);
__device__ void cudamesh_senextself(trihandle& s);
__device__ void cudamesh_senext2(trihandle& s1, trihandle& s2);
__device__ void cudamesh_senext2self(trihandle& s);

// Primitives for interacting tetrahedra and subfaces.
__device__ void cudamesh_tsbond(tethandle& t, trihandle& s, trihandle* tet2trilist, tethandle* tri2tetlist);
__device__ void cudamesh_tspivot(tethandle& t, trihandle& s, trihandle* tet2trilist);
__device__ void cudamesh_stpivot(trihandle& s, tethandle& t, tethandle* tri2tetlist);

// Primitives for interacting tetrahedra and segments.
__device__ void cudamesh_tsspivot1(tethandle& t, trihandle& seg, trihandle* tet2seglist);
__device__ void cudamesh_tssbond1(tethandle& t, trihandle& seg, trihandle* tet2seglist);
__device__ void cudamesh_sstbond1(trihandle& s, tethandle& t, tethandle* seg2tetlist);
__device__ void cudamesh_sstpivot1(trihandle& s, tethandle& t, tethandle* seg2tetlist);

// Primitives for interacting subfaces and segments.
__device__ void cudamesh_ssbond(trihandle& s, trihandle& edge, trihandle* tri2seglist, trihandle* seg2trilist);
__device__ void cudamesh_ssbond1(trihandle& s, trihandle& edge, trihandle* tri2seglist);
__device__ void cudamesh_sspivot(trihandle& s, trihandle& edge, trihandle* tri2seglist);
__device__ bool cudamesh_isshsubseg(trihandle&s, trihandle* tri2seglist);

/* Advanced primitives. */
__device__ void cudamesh_point2tetorg(int pa, tethandle& searchtet, tethandle* point2tetlist, int* tetlist);

/* Status initialization */
__global__ void kernelMarkInitialTrifaces(
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* trifacecount,
	REAL* trifaceipt,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int numoftets
);

__global__ void kernelAppendInitialTrifaces(
	int* trifacecount,
	int* trifaceindices,
	REAL* trifaceipt,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	REAL cr_facet_angle,
	REAL cr_facet_size,
	REAL cr_facet_distance,
	int numoftets
);

__global__ void kernelInitTetStatus(
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL cell_radius_edge_ratio,
	REAL cell_size,
	REAL aabb_diglen,
	int numoftets
);

__global__ void kernelInitTriQuality(
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	REAL cr_facet_angle,
	REAL cr_facet_size,
	REAL cr_facet_distance,
	int numoftrifaces
);

__global__ void kernelInitTetQuality(
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	int numoftets
);

__global__ void kernelInitDomainHandle_Tet(
	tethandle* d_domainhandle,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int offset,
	int numofthreads
);

__global__ void kernelInitDomainSegment_Tet(
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	REAL* d_domainsegment,
	int* d_domainthreadlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	REAL aabb_diglen,
	int numofthreads
);

__global__ void kernelDomainSegmentAndPrimitiveCheck_Tet(
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	int* d_domainnode,
	REAL* d_domainsegment,
	int* d_domaincount,
	int* d_domainthreadlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	int numofthreads
);

__global__ void kernelSetTetStatus(
	int* d_domaincount,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	int offset,
	int numofthreads
);

/* Point Insertion */
__global__ void kernelMarkAndCountInitialCavity(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int numofthreads
);

__global__ void kernelMarkAndCountInitialCavity(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int numofthreads
);

__global__ void kernelInitCavityLinklist(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	int* d_initialcavityindices,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavethreadidx,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelInitCavityLinklist(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_initialcavityindices,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_cavetetlist,
	int* d_cavetetidx,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCheckRecordOldtet(
	tethandle* d_recordoldtetlist,
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int numofbadsubface,
	int numofbadelement,
	int numofthreads
);

__global__ void kernelKeepRecordOldtet(
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSetReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int oldcaveoldtetsize,
	int numofthreads
);

__global__ void kernelCheckCavetetFromReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_neighborlist,
	int* d_cavetetexpandsize,
	uint64* d_tetmarker,
	int oldcaveoldtetsize,
	int numofthreads
);

__global__ void kernelAppendCavetetFromReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_cavetetlist,
	int* d_cavetetidx,
	tethandle* d_neighborlist,
	int* d_cavetetexpandindices,
	uint64* d_tetmarker,
	int oldcaveoldtetsize,
	int oldcavetetsize,
	int numofthreads
);

__global__ void kernelLargeCavityCheck(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_cavethreadidx,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelLargeCavityCheck(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_cavetetidx,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelMarkCavityReuse(
	int* d_insertidxlist,
	int* d_cavetetidx,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelResetCavityReuse(
	int* d_insertidxlist,
	int* d_threadlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelMarkOldtetlist(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_insertidxlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSetRecordOldtet(
	tethandle* d_recordoldtetlist,
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	int oldrecordsize,
	int numofthreads
);

__global__ void kernelMarkLargeCavityAsLoser(
	int* d_cavetetidx,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingCheck(
	int* d_cavethreadidx,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL* d_insertptlist,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingCheck(
	int* d_cavetetidx,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL* d_insertptlist,
	tethandle* d_cavetetlist,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void  kernelCorrectExpandingSize(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int numofthreads
);

__global__ void  kernelCorrectExpandingSize(
	int* d_cavetetidx,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingSetThreadidx(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int* d_caveoldtetthreadidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int numofthreads
);

__global__ void kernelCavityExpandingMarkAndAppend(
	int* d_cavethreadidx,
	tethandle* d_neighborlist,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int cavetetstartindex,
	int cavetetexpandsize,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int* d_caveoldtetthreadidx,
	int caveoldtetstartindex,
	int caveoldtetexpandsize,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int cavebdrystartindex,
	int cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingMarkAndAppend(
	int* d_cavetetidx,
	tethandle* d_neighborlist,
	tethandle* d_cavetetlist,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int cavetetstartindex,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int caveoldtetstartindex,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingUpdateListTails(
	int* d_cavethreadidx,
	int* d_cavetetnext,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int cavetetstartindex,
	int* d_caveoldtetnext,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int caveoldtetstartindex,
	int* d_cavebdrynext,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_threadmarker,
	int cavebdrystartindex,
	int numofthreads
);

__global__ void kernelMarkAdjacentCavitiesAndCountSubfaces(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	int* d_cavetetshsize,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCorrectSubfaceSizes(
	int* d_caveoldtetidx,
	int* d_cavetetshsize,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelAppendCavitySubfaces(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	int* d_cavetetshsize,
	int* d_cavetetshindices,
	uint64* d_tetmarker,
	int numofthreads
);

__global__ void kernelCheckSubfaceEncroachment_Phase1(
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_encroachmentmarker,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCheckSubfaceEncroachment_Phase2(
	int* d_cavetetshidx,
	int* d_insertidxlist,
	tetstatus* d_tetstatus,
	int* d_encroachmentmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSetCavityThreadIdx(
	int* d_cavethreadidx,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSetDuplicateThreadIdx(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int numofthreads
);

__global__ void kernelCountCavitySubfaces(
	int* d_threadlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_cavetetshsize,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelAppendCavitySubfaces(
	int* d_threadlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tethandle* d_tri2tetlist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	trihandle* d_cavetetshlist,
	int* d_cavetetshprev,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	int* d_cavetetshtail,
	int* d_cavetetshsize,
	int* d_cavetetshindices,
	uint64* d_tetmarker,
	int numofthreads
);

__global__ void kernelCheckSubfaceEncroachment(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelValidateRefinementElements(
	int* d_insertidxlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	tethandle* d_tri2tetlist,
	int* d_threadlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelRecomputeTrifaceCenter(
	int* d_insertidxlist,
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_trifacepmt,
	tethandle* d_tri2tetlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_threadlist,
	int* d_threadmarker,
	REAL aabb_diglen,
	int numofthreads
);

__global__ void kernelRecomputeTrifaceCenter(
	int* d_insertidxlist,
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_trifacepmt,
	tethandle* d_tri2tetlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_threadlist,
	int* d_threadmarker,
	REAL aabb_diglen,
	int numofsubfaces,
	int numofthreads
);

__global__ void kernelInsertNewPoints(
	int* d_threadlist,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	int* d_pointpmt,
	int* d_trifacepmt,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelInsertNewPoints(
	int* d_threadlist,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	int* d_pointpmt,
	int* d_trifacepmt,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelCountNewTets(
	int* d_threadlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_tetexpandsize,
	int numofthreads
);

__global__ void kernelCountNewTets(
	int* d_cavebdryidx,
	int* d_tetexpandsize,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelInsertNewTets(
	int* d_threadlist,
	tristatus* d_tristatus,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_tetexpandindices,
	int* d_emptytetindices,
	int* d_newtetthreadindices,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelInsertNewTets(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	tristatus* d_tristatus,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int* d_emptytetindices,
	int* d_newtetthreadindices,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewTetNeighbors(
	int* d_threadlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	uint64* d_tetmarker,
	int numofthreads
);

__global__ void kernelConnectNewTetNeighbors(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	uint64* d_tetmarker,
	int numofthreads
);

__global__ void kernelUpdateTriAndTetStatus_Phase1(
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	int* d_emptytetindices, // new tet indices
	int* d_newtetthreadindices, // new tet thread indices
	int* d_triexpandsize,
	REAL* d_trifaceipt,
	int* d_tripmtidx,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_pointpmt,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	REAL aabb_diglen,
	int numofaabbpms,
	int insertmode,
	int shortcut,
	int numofnewtets
);

__global__ void kernelUpdateTriAndTetStatus_Phase2(
	int* d_emptytetindices, // new tet indices
	int* d_triexpandsize,
	int* d_triexpandindices,
	int* d_emptytriindices,
	REAL* d_trifaceipt,
	int* d_tripmtidx,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_trifacepmt,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	REAL cr_facet_angle,
	REAL cr_facet_size,
	REAL cr_facet_distance,
	int numofnewtets
);

__global__ void kernelInitDomainHandle(
	int* d_emptytetindices, // new tet indices
	int* d_newtetthreadindices, // new tet thread indices
	tethandle* d_domainhandle,
	int* d_domaincount,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int insertmode,
	int offset,
	int numofthreads
);

__global__ void kernelInitDomainSegment(
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	REAL* d_domainsegment,
	int* d_domainthreadlist,
	int* d_triexpandsize,
	REAL* d_trifaceipt,
	int* d_tripmtidx,
	int* d_emptytetindices, // new tet indices
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_pointpmt,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL aabb_diglen,
	int numofaabbpms,
	int insertmode,
	int shortcut,
	int numofthreads
);

__global__ void kernelDomainSegmentAndBoxCheck(
	REAL* d_aabbnodebbs,
	tethandle* d_domainhandle,
	int* d_domainnode,
	REAL* d_domainsegment,
	int numofthreads
);

__global__ void kernelDomainHandleAppend(
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	tethandle* d_domainhandle,
	int* d_domainnode,
	int numofthreads
);

__global__ void kernelDomainSegmentAndPrimitiveCheck(
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	int* d_domainnode,
	REAL* d_domainsegment,
	int* d_domaincount,
	int* d_domainthreadlist,
	int* d_triexpandsize,
	int* d_emptytetindices, // new tet indices
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	uint64* d_domainmarker, // triface distance marker
	int numofthreads
);

__global__ void kernelDomainSetTriCenter(
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	int* d_domainnode,
	REAL* d_domainsegment,
	int* d_domainthreadlist,
	uint64* d_domainmarker, // triface distance marker
	REAL* d_trifaceipt,
	int* d_trifacepmt,
	int numofthreads
);

__global__ void kernelUpdateNewTetStatus(
	int* d_emptytetindices, // new tet indices
	int* d_domaincount,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	int numofthreads
);

__global__ void kernelUpdateTriStatus(
	int* d_emptytetindices, // new tet indices
	int* d_triexpandsize,
	int* d_triexpandindices,
	int* d_emptytriindices,
	REAL* d_trifaceipt,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	REAL cr_facet_angle,
	REAL cr_facet_size,
	REAL cr_facet_distance,
	int numofthreads
);

__global__ void kernelResetOldInfo(
	int* d_threadlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelResetOldInfo_Tet(
	tethandle* d_caveoldtetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int numofthreads
);

__global__ void kernelResetOldInfo_Subface(
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	trihandle* d_tet2trilist,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	int* d_threadmarker,
	int numofthreads
);

// Check mesh
__global__ void kernelCheckPointNeighbors(
	trihandle* d_point2trilist,
	tethandle* d_point2tetlist,
	verttype* d_pointtypelist,
	int* d_seglist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int numofthreads
);

__global__ void kernelCheckSubsegNeighbors(
	int* d_seglist,
	trihandle* d_seg2trilist,
	tethandle* d_seg2tetlist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	int numofthreads
);

__global__ void kernelCheckSubfaceNeighbors(
	int* d_seglist,
	trihandle* d_seg2trilist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int numofthreads
);

__global__ void kernelCheckTetNeighbors(
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int numofthreads
);

// Split bad element
__global__ void kernelCheckBadElementList(
	int* d_badeleidlist,
	int* d_threadmarker,
	int* d_segencmarker,
	int* d_subfaceencmarker,
	tetstatus* d_tetstatus,
	int numofencsegs,
	int numofencsubfaces,
	int numofbadtets,
	int numofthreads
);

__global__ void kernelComputeSteinerPointAndPriority(
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	REAL* d_priority,
	int numofthreads
);

__global__ void kernelComputePriorities(
	REAL* d_pointlist,
	int* d_trifacelist,
	int* d_tetlist,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_priority,
	int numofthreads
);

__global__ void kernelComputeSteinerPoints(
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	int numofthreads
);

__global__ void kernelModifyPriority(
	REAL* d_priorityreal,
	int* d_priorityint,
	REAL offset0,
	REAL offset1,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_insertidxlist,
	int* d_threadmarker,
	int numofbadtriface,
	int numofthreads
);

__global__ void kernelGridFiltering(
	int* d_priority,
	uint64* d_tetmarker,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	int range_left,
	int range_right,
	double step_x,
	double step_y,
	double step_z,
	double origin_x,
	double origin_y,
	double origin_z,
	int gridlength
);

__global__ void kernelLocatePoint(
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_priority,
	locateresult* d_pointlocation,
	tethandle* d_searchtet,
	int* d_insertidxlist,
	int* d_threadmarker,
	int* d_threadlist,
	REAL* d_steinerptlist,
	int numofthreads
);

__global__ void kernelCompactTriface(
	int* d_trifacelist,
	double* d_trifacecent,
	tethandle* d_tri2tetlist,
	tetstatus* d_tetstatus,
	int* d_sizes,
	int* d_indices,
	int* d_listidx,
	double* d_listpt,
	int numofthreads
);

__global__ void kernelCompactTet_Phase1(
	int* d_trifacelist,
	tetstatus* d_tetstatus,
	int* d_sizes,
	int numofthreads
);

__global__ void kernelCompactTet_Phase2(
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_sizes,
	int* d_indices,
	int* d_listidx,
	tetstatus* d_liststatus,
	int numofthreads
);

