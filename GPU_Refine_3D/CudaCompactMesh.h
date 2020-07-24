#pragma once

#include "CudaThrust.h"

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
);