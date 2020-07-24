#pragma once
#include <map>
#include "CudaThrust.h"

//#define GQM3D_PROFILING
//#define GQM3D_LOOP_PROFILING
//#define GQM3D_LOOP_PROFILING_VERBOSE
//#define GQM3D_ITER_PROFILING
//#define GQM3D_DEBUG
//#define GQM3D_OUTPUTMESH
//#define GQM3D_CHECKMEMORY

#ifdef GQM3D_PROFILING
	extern std::map<std::string, double> inserttimer;
#endif

#ifdef GQM3D_LOOP_PROFILING
	extern std::map<std::string, double> looptimer;
#endif

// 1: It is still possible to insert some points for current id list
// 0: It is impossible to insert any points for current id list if the mesh doesn't change
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
	IntD& t_threadmarker, // indicate insertion type: -1 failed, 0 splitsubseg, 1 splitsubface, 2 splittet
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
	int iter // iteration number for bad element splitting
);

int insertPoint_New(
	IntD& t_aabbnodeleft,
	IntD& t_aabbnoderight,
	RealD& t_aabbnodebbs,
	RealD& t_aabbpmcoord,
	RealD& t_aabbpmbbs,
	TetHandleD& t_recordoldtetlist,
	IntD& t_recordoldtetidx,
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
	IntD& t_threadmarker, // indicate insertion type: -1 failed, 0 splitsubseg, 1 splitsubface, 2 splittet
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
	int iter // iteration number for bad element splitting
);