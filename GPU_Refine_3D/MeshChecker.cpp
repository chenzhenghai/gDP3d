#include "Mesh.h"
#include "MeshPredicates.h"
#include "MeshReconstruct.h"
#include "MeshChecker.h"
#include <stdio.h>
#include <math.h>

bool isBadFacet(
	double *pa, double *pb, double *pc, double aw,
	double bw, double cw, double *cent,
	double facet_angle, double facet_size, double facet_distance
)
{
	if (facet_angle != 0.0)
	{
		double B = sin(PI*facet_angle / 180);
		B = B*B;

		double area = trianglesquaredarea(pa, pb, pc);
		double dab = squareddistance(pa, pb);
		double dac = squareddistance(pa, pc);
		double dbc = squareddistance(pb, pc);
		double min_dabc = min(dab, dac, dbc);
		double aspect_ratio = 4 * area * min_dabc / (dab*dac*dbc);

		if (aspect_ratio < B)
			return true;
	}

	if (facet_size != 0.0)
	{
		double B = facet_size*facet_size;
		double r = squareddistance(cent, pa);

		if (r > B)
			return true;
	}

	if (facet_distance != 0.0)
	{
		double B = facet_distance*facet_distance;
		double wc[3];
		if (weightedcircumcenter(pa, pb, pc, aw, bw, cw, wc))
		{
			double sq_dist = squareddistance(wc, cent);
			if (sq_dist > B)
				return true;
		}
	}

	return false;
}

bool isBadTet(
	double *pa, double *pb, double *pc, double *pd,
	double cell_radius_edge_ratio, double cell_size
)
{
	double cent[3];
	if (!circumcenter(pa, pb, pc, pd, cent))
		return false;
	double R = squareddistance(pa, cent);
	if (cell_radius_edge_ratio != 0.0)
	{
		double B = cell_radius_edge_ratio*cell_radius_edge_ratio;
		double min_sq_length = squareddistance(pa, pb);
		min_sq_length = min(min_sq_length, squareddistance(pa, pc));
		min_sq_length = min(min_sq_length, squareddistance(pa, pd));
		min_sq_length = min(min_sq_length, squareddistance(pb, pc));
		min_sq_length = min(min_sq_length, squareddistance(pb, pd));
		min_sq_length = min(min_sq_length, squareddistance(pc, pd));

		if (R > B*min_sq_length)
			return true;
	}

	if (cell_size != 0.0)
	{
		double B = cell_size*cell_size;
		if (R > B)
			return true;
	}

	return false;
}

void calAngles(
	double **p,
	double *allangles
)
{
	double a2, b2, c2, A, B, C, ta, tb, tc;
	a2 = squareddistance(p[0], p[1]);
	b2 = squareddistance(p[0], p[2]);
	c2 = squareddistance(p[1], p[2]);
	ta = (b2 + c2 - a2) / (2 * sqrt(b2)*sqrt(c2));
	tb = (a2 + c2 - b2) / (2 * sqrt(a2)*sqrt(c2));
	tc = (a2 + b2 - c2) / (2 * sqrt(a2)*sqrt(b2));
	A = acos(ta) * 180 / PI;
	B = acos(tb) * 180 / PI;
	C = acos(tc) * 180 / PI;
	allangles[0] = A;
	allangles[1] = B;
	allangles[2] = C;
}

void calDihedral(
	double **p,
	double *alldihed
)
{
	double A[4][4], rhs[4], D;
	double V[6][3], N[4][3], H[4]; // edge-vectors, face-normals, face-heights.
	int indx[4];

	int i, j;
	// Set the edge vectors: V[0], ..., V[5]
	for (i = 0; i < 3; i++) V[0][i] = p[0][i] - p[3][i]; // V[0]: p3->p0.
	for (i = 0; i < 3; i++) V[1][i] = p[1][i] - p[3][i]; // V[1]: p3->p1.
	for (i = 0; i < 3; i++) V[2][i] = p[2][i] - p[3][i]; // V[2]: p3->p2.
	for (i = 0; i < 3; i++) V[3][i] = p[1][i] - p[0][i]; // V[3]: p0->p1.
	for (i = 0; i < 3; i++) V[4][i] = p[2][i] - p[1][i]; // V[4]: p1->p2.
	for (i = 0; i < 3; i++) V[5][i] = p[0][i] - p[2][i]; // V[5]: p2->p0.

	// Set the matrix A = [V[0], V[1], V[2]]^T.
	for (j = 0; j < 3; j++) {
		for (i = 0; i < 3; i++) A[j][i] = V[j][i];
	}

	// Decompose A just once.
	if (meshludecmp(A, 3, indx, &D, 0)) {
		// Get the three faces normals.
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) rhs[i] = 0.0;
			rhs[j] = 1.0;  // Positive means the inside direction
			meshlusolve(A, 3, indx, rhs, 0);
			for (i = 0; i < 3; i++) N[j][i] = rhs[i];
		}
		// Get the fourth face normal by summing up the first three.
		for (i = 0; i < 3; i++) N[3][i] = -N[0][i] - N[1][i] - N[2][i];
		// Get the radius of the circumsphere.
		for (i = 0; i < 3; i++) rhs[i] = 0.5 * meshdot(V[i], V[i]);
		meshlusolve(A, 3, indx, rhs, 0);
		// Normalize the face normals.
		for (i = 0; i < 4; i++) {
			// H[i] is the inverse of height of its corresponding face.
			H[i] = sqrt(meshdot(N[i], N[i]));
			for (j = 0; j < 3; j++) N[i][j] /= H[i];
		}
	}
	else {
		// Calculate the four face normals.
		meshfacenormal(p[2], p[1], p[3], N[0], 1, NULL);
		meshfacenormal(p[0], p[2], p[3], N[1], 1, NULL);
		meshfacenormal(p[1], p[0], p[3], N[2], 1, NULL);
		meshfacenormal(p[0], p[1], p[2], N[3], 1, NULL);
		// Normalize the face normals.
		for (i = 0; i < 4; i++) {
			// H[i] is the twice of the area of the face.
			H[i] = sqrt(meshdot(N[i], N[i]));
			for (j = 0; j < 3; j++) N[i][j] /= H[i];
		}
	}

	// Get the dihedrals (in degree) at each edges.
	j = 0;
	for (i = 1; i < 4; i++) {
		alldihed[j] = -meshdot(N[0], N[i]); // Edge cd, bd, bc.
		if (alldihed[j] < -1.0) alldihed[j] = -1; // Rounding.
		else if (alldihed[j] > 1.0) alldihed[j] = 1;
		alldihed[j] = acos(alldihed[j]) / PI * 180.0;
		j++;
	}
	for (i = 2; i < 4; i++) {
		alldihed[j] = -meshdot(N[1], N[i]); // Edge ad, ac.
		if (alldihed[j] < -1.0) alldihed[j] = -1; // Rounding.
		else if (alldihed[j] > 1.0) alldihed[j] = 1;
		alldihed[j] = acos(alldihed[j]) / PI * 180.0;
		j++;
	}
	alldihed[j] = -meshdot(N[2], N[3]); // Edge ab.
	if (alldihed[j] < -1.0) alldihed[j] = -1; // Rounding.
	else if (alldihed[j] > 1.0) alldihed[j] = 1;
	alldihed[j] = acos(alldihed[j]) / PI * 180.0;
}

int countBadTets(
	double* pointlist,
	int* tetlist,
	int numoftet,
	double minratio
)
{
	int ip[4];
	REAL* p[4];
	int count = 0;
	/*int i, j;
	for (i = 0; i < numoftet; i++)
	{
		for (j = 0; j < 4; j++)
		{
			ip[j] = tetlist[4 * i + j];
			p[j] = id2pointlist(ip[j], pointlist);
		}
		if (isBadTet(p[0], p[1], p[2], p[3], minratio))
			count++;
	}*/

	return count;
}