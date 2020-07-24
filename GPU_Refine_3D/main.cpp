#include <stdio.h>
#include "MeshRefine.h"
#include "Experiment_CGAL.h"

/**
* Host main routine
*/
int
main(int argc, char** argv)
{
	//experiment_volume_refine_OFFs_on_CPU();
	//return 0;

	MESHCR criteria;
	criteria.edge_size = 1;
	criteria.facet_angle = 30;
	criteria.facet_size = 0.08;
	criteria.facet_distance = 0;
	criteria.cell_radius_edge_ratio = 0;
	criteria.cell_size = 0;

	MESHBH behavior;
	behavior.verbose = 1;

	char* inputpath = "../../../off_real/";
	char* inputfn = "Accessories";
	bool run_cpu = false, run_gpu = true;
	bool no_output = true;

	char *inputfile, *outputfile;
	std::ostringstream strs;
	strs << inputpath << inputfn << ".off";
	inputfile = new char[strs.str().length() + 1];
	strcpy(inputfile, strs.str().c_str());

	std::ostringstream strs1;
	strs1 << inputpath << inputfn;
	outputfile = new char[strs1.str().length() + 1];
	strcpy(outputfile, strs1.str().c_str());

	printf("Criteria: %lf | %lf %lf %lf | %lf %lf\n",
		criteria.edge_size,
		criteria.facet_angle, criteria.facet_size, criteria.facet_distance,
		criteria.cell_radius_edge_ratio, criteria.cell_size);
	printf("Input File: %s, Output file: %s\n", inputfile, outputfile);

	if (run_cpu)
	{
		printf("Calling CPU Refinement.......\n");
		refineInputOnCGAL(
			inputfile,
			false,
			&criteria,
			no_output ? NULL : outputfile,
			no_output ? NULL : outputfile
		);
		printf("\n");
	}

	if (run_gpu)
	{
		printf("Calling GPU Refinement.......\n");
		refineInputOnGPU(
			inputfile,
			false,
			&criteria,
			&behavior,
			no_output ? NULL : outputfile,
			no_output ? NULL : outputfile
		);
	}

	return 0;
}