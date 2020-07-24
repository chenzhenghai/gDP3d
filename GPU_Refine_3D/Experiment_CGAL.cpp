#include <stdio.h>
#include "MeshRefine.h"
#include "Experiment_CGAL.h"

void experiment_surface_refine_OFFs_on_CPU()
{
	int numoffiles = 11;
	char file_name[11][20] = {
		"Accessories",
		"Arc_triomphe",
		"Aztec",
		"Hendecahedron",
		"letter-z",
		"marit",
		"romanesco",
		"Sculpture01",
		"vlamp",
		"treads_24",
		"gandhi_litho"
	};
	double facet_size[11] = {
		0.08,
		0.03,
		0.05,
		0.02,
		0.03,
		0.11,
		0.05,
		0.04,
		0.3,
		0.04,
		0.1
	};

	for (int i = 0; i < numoffiles; i++)
	{
		if (i != 4 && i != 6)
			continue;

		char* inputfn = *(file_name + i);
		double fs = facet_size[i];
		printf("Experiment: Refining off file %s, facet size = %lf.......\n",
			inputfn, fs);

		MESHCR criteria;
		criteria.edge_size = 1;
		criteria.facet_angle = 30;
		criteria.facet_size = fs;
		criteria.facet_distance = 0;
		criteria.cell_radius_edge_ratio = 0;
		criteria.cell_size = 0;

		char* inputpath = "../../../off_real/";

		char *inputfile, *outputfile;
		std::ostringstream strs;
		strs << inputpath << inputfn << ".off";
		inputfile = new char[strs.str().length() + 1];
		strcpy(inputfile, strs.str().c_str());

		std::ostringstream strs1;
		strs1 << inputpath << inputfn;
		outputfile = new char[strs1.str().length() + 1];
		strcpy(outputfile, strs1.str().c_str());

		refineInputOnCGAL(
			inputfile,
			false,
			&criteria,
			outputfile,
			outputfile
		);

		printf("\n");

		delete[] inputfile;
		delete[] outputfile;
	}
}

void experiment_volume_refine_OFFs_on_CPU()
{
	int numoffiles = 11;
	char file_name[11][20] = {
		"Accessories",
		"Arc_triomphe",
		"Aztec",
		"Hendecahedron",
		"letter-z",
		"marit",
		"romanesco",
		"Sculpture01",
		"vlamp",
		"treads_24",
		"gandhi_litho"
	};
	double cell_size[11] = {
		0.3,
		0.13,
		-1,
		0.12,
		0.09,
		0.65,
		-1,
		0.13,
		0.63,
		-1,
		-1
	};

	for (int i = 4; i < numoffiles; i++)
	{

		char* inputfn = *(file_name + i);
		double cs = cell_size[i];
		if (cs < 0) // not a closed surface
			continue;

		printf("Experiment: Refining off file %s, cell size = %lf.......\n",
			inputfn, cs);

		MESHCR criteria;
		criteria.edge_size = 1;
		criteria.facet_angle = 30;
		criteria.facet_size = 0;
		criteria.facet_distance = 0;
		criteria.cell_radius_edge_ratio = 2;
		criteria.cell_size = cs;

		char* inputpath = "../../../off_real/";

		char *inputfile, *outputfile;
		std::ostringstream strs;
		strs << inputpath << inputfn << ".off";
		inputfile = new char[strs.str().length() + 1];
		strcpy(inputfile, strs.str().c_str());

		std::ostringstream strs1;
		strs1 << inputpath << inputfn;
		outputfile = new char[strs1.str().length() + 1];
		strcpy(outputfile, strs1.str().c_str());

		refineInputOnCGAL(
			inputfile,
			false,
			&criteria,
			outputfile,
			outputfile
		);

		printf("\n");

		delete[] inputfile;
		delete[] outputfile;
	}
}