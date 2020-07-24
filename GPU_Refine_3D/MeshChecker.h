#pragma once

bool isBadFacet(
	double *pa, double *pb, double *pc, double aw,
	double bw, double cw, double *cent,
	double facet_angle, double facet_size, double facet_distance
);

bool isBadTet(
	double *pa, double *pb, double *pc, double *pd,
	double cell_radius_edge_ratio, double cell_size
);

void calAngles(
	double **p,
	double *allangles
);

void calDihedral(
	double **p,
	double *alldihed
);

int countBadTets(
	double* pointlist,
	int* tetlist,
	int numoftet,
	double minratio
);