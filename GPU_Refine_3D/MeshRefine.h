#pragma once
#define CGAL_LINKED_WITH_TBB
#define CGAL_CONCURRENT_MESH_3
#define NOMINMAX
//#define CGAL_MESH_3_PROTECTION_DEBUG 1
//#define CGAL_MESH_3_VERBOSE 1
#define CGAL_MESH_3_PROFILING
#define CGAL_MESH_3_EXPORT_PERFORMANCE_DATA

#define GQM3D_WITHOUT_1D_FEATURE

#include <unordered_map>
#include <map>
#include <time.h>
#include <stdio.h>
#include "MeshIO.h"
#include "MeshStructure.h"
#include "MeshRefine.h"
#include "MeshChecker.h"

extern std::map<std::string, double> intertimer;

#ifdef CGAL_MESH_3_EXPORT_PERFORMANCE_DATA
#define CGAL_MESH_3_SET_PERFORMANCE_DATA(key, val) \
{\
	intertimer[key] = val;\
}
#endif

#define cpuErrchk(ans) { cpuAssert((ans), __FILE__, __LINE__); }
inline void cpuAssert(bool ret, const char *file, int line, bool abort = true)
{
	if (!ret)
	{
		fprintf(stderr, "CPUassert failed: %s %d\n", file, line);
		if (abort) exit(0);
	}
}

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#ifdef GQM3D_WITHOUT_1D_FEATURE
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedral_mesh_domain_3.h>
#else
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#endif

#include <CGAL/make_mesh_3.h>
#include <CGAL/Timer.h>

// Domain 
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
#ifdef GQM3D_WITHOUT_1D_FEATURE
typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K> Mesh_domain;
#else
typedef CGAL::Mesh_polyhedron_3<K>::type Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;
#endif

#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
#ifdef GQM3D_WITHOUT_1D_FEATURE
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
#else
typedef CGAL::Mesh_complex_3_in_triangulation_3<
	Tr, Mesh_domain::Corner_index, Mesh_domain::Curve_index> C3t3;
#endif

// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

void outputFacets2OFF(
	int numofpoint,
	double* pointlist,
	int numoftriface,
	int* trifacelist,
	char* filename
);

void outputTr2Medit(
	int numofpoint,
	double* pointlist,
	int numoftriface,
	int* trifacelist,
	int numoftet,
	int numoftet_indomain,
	int* tetlist,
	tetstatus* tetstatus,
	char* filename
);

int refineInputOnCGAL(
	char* infile,
	bool force_features,
	MESHCR* criteria,
	char* outmesh,
	char* outdata
);

int refineInputOnGPU(
	char* infile,
	bool force_features,
	MESHCR* criteria,
	MESHBH* behavior,
	char* outmesh,
	char* outdata
);