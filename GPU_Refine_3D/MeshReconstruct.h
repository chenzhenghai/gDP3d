﻿#pragma once
#include "MeshIO.h"

void reconstructMesh(
	MESHIO* input_mesh,
	verttype*& outpointtypelist,
	tethandle*& outtri2tetlist,
	int& outnumoftetrahedron,
	int*& outtetlist,
	tethandle*& outneighborlist,
	trihandle*& outtet2trilist,
	bool verbose
);

void makesegment2parentmap(
	int numofsegment,
	int* segmentlist,
	trihandle* seg2trilist,
	int*& segment2parentlist,
	int*& segmentendpointslist,
	int& numofparent
);

void makesubfacepointsmap(
	int numofpoint,
	double* pointlist,
	verttype* pointtypelist,
	int numofsubface,
	int* subfacelist,
	trihandle* subface2seglist,
	trihandle* subface2subfacelist,
	int*& subface2parentlist,
	int*& id2subfacelist,
	int*& subfacepointslist,
	int& numofparent,
	int& numofendpoints
);