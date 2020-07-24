#pragma once
#include "MeshStructure.h"

typedef struct MESHIO
{
	// input mesh
	int numofpoints;
	double* pointlist;
	double* weightlist;
	int numofedges;
	int* edgelist;
	int numoftrifaces;
	int* trifacelist;
	double* trifacecentlist;
	tristatus* tristatuslist;
	int* tri2tetlist;
	int* tri2tetverlist;
	int numoftets;
	int numoftets_indomain;
	int* tetlist;
	tetstatus* tetstatuslist;
	int* tet2trilist;
	int* tet2tetlist;
	int* tet2tetverlist;

	// input domain
	int numofaabbnodes;
	int* aabb_nodeleftchild;
	int* aabb_noderightchild;
	double* aabb_nodebbs;
	int numofaabbpms;
	double* aabb_pmcoord;
	double* aabb_pmbbs;
	double aabb_xmin;
	double aabb_ymin;
	double aabb_zmin;
	double aabb_xmax;
	double aabb_ymax;
	double aabb_zmax;
	double aabb_diglen;
	int aabb_level;
	bool aabb_closed;

	// output mesh
	int out_numofpoints;
	double* out_pointlist;
	double* out_weightlist;
	int out_numofedges;
	int* out_edgelist;
	int out_numoftrifaces;
	int* out_trifacelist;
	double* out_trifacecent;
	int out_numoftets;
	int out_numoftets_indomain;
	int* out_tetlist;
	tetstatus* out_tetstatus;

	MESHIO(void)
	{
		numofpoints = numofedges = numoftets = 0;
		pointlist = NULL;
		weightlist = NULL;
		edgelist = NULL;
		trifacelist = NULL;
		tetlist = NULL;
		tet2tetlist = NULL;
	}
} MESHIO;

typedef struct MESHCR
{
	//Mesh_criteria criteria(edge_size = 0.1,
	//	facet_angle = 25, facet_size = 0.5, facet_distance = 0.1,
	//	cell_radius_edge_ratio = 1.4, cell_size = 0.16);
	double edge_size;

	double facet_angle;
	double facet_size;
	double facet_distance;

	double cell_radius_edge_ratio;
	double cell_size;

	MESHCR(void)
	{
		edge_size = 1;
		facet_angle = 0;
		facet_size = 0;
		facet_distance = 0;
		cell_radius_edge_ratio = 0;
		cell_size = 0;
	}
} MESHCR;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for visualization													 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct internalmesh
{
	// iteration
	int iter_seg;
	int iter_subface;
	int iter_tet;
	// animation
	bool animation;
	int* anicolor;
	int framenum;
	// mesh
	int numofpoints;
	double* pointlist;
	verttype* pointtype;
	int numofsubseg;
	int* seglist;
	tristatus* segstatus;
	int numofsubface;
	int* trifacelist;
	tristatus* tristatus;
	tethandle* tri2tetlist;
	int numoftet;
	int* tetlist;
	tethandle* neighborlist;
	unsigned long long* tetmarker;
	tetstatus* tetstatus;
	tethandle* locatedtet;
	// cavity
	int numofthread;
	int *threadlist;
	int numofinsertpt;
	int insertiontype;
	int* insertidxlist;
	double* insertptlist;
	int* threadmarker;
	tethandle* cavebdrylist;
	int* cavebdrynext;
	int* cavebdryhead;
	tethandle* caveoldtetlist;
	int* caveoldtetnext;
	int* caveoldtethead;

	internalmesh(void)
	{
		iter_seg = -1;
		iter_subface = -1;
		iter_tet = -1;
		animation = false;
		framenum = 0;
	}

	internalmesh(int iseg, int isub, int itet)
	{
		iter_seg = iseg;
		iter_subface = isub;
		iter_tet = itet;
		animation = false;
		framenum = 0;
	}
} internalmesh;


typedef struct MESHBH
{
	bool R1, R2, R3, R4, R5;

	int mode;

	int filtermode;
	int filterstatus;
	int maxbadelements;

	int aabbmode;
	int aabbshortcut;
	int aabbhandlesize;
	double aabbhandlesizefac;
	int aabbwinsize;

	int cavetetsize;
	double cavetetsizefac;
	int caveoldtetsize;
	double caveoldtetsizefac;
	int cavebdrysize;
	double cavebdrysizefac;

	int cavitymode;
	int maxcavity;
	int mincavity;

	int verbose;
	int minbadtrifaces;
	int minsplittabletets;
	int minbadtets;
	int mintriiter;
	int mintetiter;
	int minthread;
	internalmesh* drawmesh;
	double times[8];

	MESHBH(void)
	{
		// Rules
		R1 = false;
		R2 = R3 = R4 = R5 = true;

		// 1: use CPU for edge protection only
		// 2: use CPU for edge protection, and angle and ratio criteria
		mode = 1;

		// 1: normal filtering
		// 2: fast filtering
		filtermode = 1;
		filterstatus = 1; // 1: not on, 2: on, 3: just off
		maxbadelements = 1000000;

		// 1: use DFS inside one kernel using inner loop
		// 2: use BFS with outer loop
		aabbmode = 2;
		// 1: no shortcut
		// 2: use shortcut
		aabbshortcut = 2;
		aabbhandlesize = 100000;
		aabbhandlesizefac = 1.1;
		aabbwinsize = -1;

		cavetetsize = 10000;
		cavetetsizefac = 1.1;
		caveoldtetsize = 10000;
		caveoldtetsizefac = 1.1;
		cavebdrysize = 10000;
		cavebdrysizefac = 1.1;

		// 1: cut cavity when it exceeds maxcavity
		// 2: record cavity when it exceeds mincavity
		cavitymode = 2;
		maxcavity = 500;
		mincavity = 30;

		verbose = 1;
		minbadtrifaces = 0;
		minsplittabletets = 0;
		minbadtets = 0;
		mintriiter = 0;
		mintetiter = 0;
		minthread = 0;
		drawmesh = NULL;
	}
} MESHBH;
