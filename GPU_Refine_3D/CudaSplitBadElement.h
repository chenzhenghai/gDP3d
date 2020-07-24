#pragma once

#include "CudaThrust.h"

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
);