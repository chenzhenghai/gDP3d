#pragma once
#include "CudaThrust.h"

//#define GQM3D_INIT_PROFILING
//#define GQM3D_INIT_DEBUG

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
);

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
);

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
);

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
);