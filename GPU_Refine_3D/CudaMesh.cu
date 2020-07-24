// This file is adapted from TetGen

#include "CudaMesh.h"
#include "CudaPredicates.h"
#include <thrust/device_ptr.h>
#include <stdio.h>
#include <assert.h>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Variables			                                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Kernel constants */

__constant__ REAL raw_kernelconstants[2];

REAL host_kernelconstants[2];

/* Helpers */
__device__ uint64 cudamesh_encodeUInt64Priority(int priority, int index)
{
	return (((uint64)priority) << 32) + index;
}

__device__ int cudamesh_getUInt64PriorityIndex(uint64 priority)
{
	return (priority & 0xFFFFFFFF);
}

__device__ int cudamesh_getUInt64Priority(uint64 priority)
{
	return (priority >> 32);
}

__device__ bool cudamesh_isNearZero(double val)
{
	if (val > -EPSILON && val < EPSILON)
		return true;
	else
		return false;
}

__device__ bool cudamesh_isInvalid(double val)
{
	if (val > 10000000.0 || val < -10000000.0)
		return true;
	else
		return false;
}

/* Initialize fast lookup tables for mesh maniplulation primitives. */

__constant__ int raw_bondtbl[144];
__constant__ int raw_fsymtbl[144];
__constant__ int raw_enexttbl[12];
__constant__ int raw_eprevtbl[12];
__constant__ int raw_enextesymtbl[12];
__constant__ int raw_eprevesymtbl[12];
__constant__ int raw_eorgoppotbl[12];
__constant__ int raw_edestoppotbl[12];
__constant__ int raw_facepivot1[12];
__constant__ int raw_facepivot2[144];
__constant__ int raw_tsbondtbl[72];
__constant__ int raw_stbondtbl[72];
__constant__ int raw_tspivottbl[72];
__constant__ int raw_stpivottbl[72];

int host_bondtbl[144] = { 0, };
int host_fsymtbl[144] = { 0, };
int host_enexttbl[12] = { 0, };
int host_eprevtbl[12] = { 0, };
int host_enextesymtbl[12] = { 0, };
int host_eprevesymtbl[12] = { 0, };
int host_eorgoppotbl[12] = { 0, };
int host_edestoppotbl[12] = { 0, };
int host_facepivot1[12] = { 0, };
int host_facepivot2[144] = { 0, };
int host_tsbondtbl[72] = { 0, };
int host_stbondtbl[72] = { 0, };
int host_tspivottbl[72] = { 0, };
int host_stpivottbl[72] = { 0, };

// Table 'esymtbl' takes an directed edge (version) as input, returns the
//   inversed edge (version) of it.

__constant__ int raw_esymtbl[12];

int host_esymtbl[12] = { 9, 6, 11, 4, 3, 7, 1, 5, 10, 0, 8, 2 };

// The following four tables give the 12 permutations of the set {0,1,2,3}.

__constant__ int raw_orgpivot[12];
__constant__ int raw_destpivot[12];
__constant__ int raw_apexpivot[12];
__constant__ int raw_oppopivot[12];

int host_orgpivot[12] = { 3, 3, 1, 1, 2, 0, 0, 2, 1, 2, 3, 0 };
int host_destpivot[12] = { 2, 0, 0, 2, 1, 2, 3, 0, 3, 3, 1, 1 };
int host_apexpivot[12] = { 1, 2, 3, 0, 3, 3, 1, 1, 2, 0, 0, 2 };
int host_oppopivot[12] = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };

// The twelve versions correspond to six undirected edges. The following two
//   tables map a version to an undirected edge and vice versa.

__constant__ int raw_ver2edge[12];
__constant__ int raw_edge2ver[6];

int host_ver2edge[12] = { 0, 1, 2, 3, 3, 5, 1, 5, 4, 0, 4, 2 };
int host_edge2ver[6] = { 0, 1, 2, 3, 8, 5 };

// Edge versions whose apex or opposite may be dummypoint.

__constant__ int raw_epivot[12];

int host_epivot[12] = { 4, 5, 2, 11, 4, 5, 2, 11, 4, 5, 2, 11 };

// Table 'snextpivot' takes an edge version as input, returns the next edge
//   version in the same edge ring.

__constant__ int raw_snextpivot[6];

int host_snextpivot[6] = { 2, 5, 4, 1, 0, 3 };

// The following three tables give the 6 permutations of the set {0,1,2}.
//   An offset 3 is added to each element for a direct access of the points
//   in the triangle data structure.

__constant__ int raw_sorgpivot[6];
__constant__ int raw_sdestpivot[6];
__constant__ int raw_sapexpivot[6];


int host_sorgpivot[6] = { 0, 1, 1, 2, 2, 0 };
int host_sdestpivot[6] = { 1, 0, 2, 1, 0, 2 };
int host_sapexpivot[6] = { 2, 2, 0, 0, 1, 1 };

/* Initialize Geometric Predicates arrays*/

REAL host_constData[17];
int host_constOptions[2];

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Domain helpers														     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ bool cudamesh_traversal_first_intersection(REAL* s, REAL* t,
	int* aabbnodeleft, int* aabbnoderight, REAL* aabbnodebbs,
	REAL* aabbpmcoord, REAL* aabbpmbbs, REAL* ipt, int& pmidx)
{
	// stack variables
	int stack[MAXAABBLEVEL]; // can support up to 2^MAXAABBLEVEL polygons
	int curpos = 0;
	stack[curpos] = 0; // push root node
	
	// traversal
	int curidx, left, right;
	REAL bxmin, bxmax, bymin, bymax, bzmin, bzmax;
	while (curpos >= 0)
	{
		curidx = stack[curpos];
		if (curidx >= 0) // a node
		{
			bxmin = aabbnodebbs[6 * curidx + 0]; bxmax = aabbnodebbs[6 * curidx + 1];
			bymin = aabbnodebbs[6 * curidx + 2]; bymax = aabbnodebbs[6 * curidx + 3];
			bzmin = aabbnodebbs[6 * curidx + 4]; bzmax = aabbnodebbs[6 * curidx + 5];
			if (cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
			{
				left = aabbnodeleft[curidx];
				right = aabbnoderight[curidx];
				stack[curpos++] = right;
				stack[curpos] = left;
			}
			else
			{
				curpos--;
			}
		}
		else // a leaf
		{
			curidx = -curidx - 1;
			bxmin = aabbpmbbs[6 * curidx + 0]; bxmax = aabbpmbbs[6 * curidx + 1];
			bymin = aabbpmbbs[6 * curidx + 2]; bymax = aabbpmbbs[6 * curidx + 3];
			bzmin = aabbpmbbs[6 * curidx + 4]; bzmax = aabbpmbbs[6 * curidx + 5];
			if (cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
			{
				REAL* p[3];
				p[0] = aabbpmcoord + 9 * curidx + 0;
				p[1] = aabbpmcoord + 9 * curidx + 3;
				p[2] = aabbpmcoord + 9 * curidx + 6;

				if (cudamesh_ts_intersection(s, t, p[0], p[1], p[2], ipt))
				{
					pmidx = curidx;
					return true;
				}
			}
			curpos--;
		}
	}
	return false;
}

__device__ bool cudamesh_traversal_in_domain(REAL* s, int* aabbnodeleft,
	int* aabbnoderight, REAL* aabbnodebbs, REAL* aabbpmcoord, REAL* aabbpmbbs)
{
	// Stack variables
	int stack[MAXAABBLEVEL]; // can support up to 2^MAXAABBLEVEL polygons
	int curpos;

	// Calculate the digonal len of the largest bounding box to create long segment
	REAL dlen = cudamesh_bbox_diglen(0, aabbnodebbs);
	REAL t[3], v[3];
	unsigned long randomseed = 0;

	// Traversal
	bool uncertain;
	int ucounter = 0; // count the time of uncertain
	int icounter = 0; // count the time of valid intersection
	int res, type; // res: 0 - Out, 1 - In, 2 - On, -1 - Uninitialized
	int curidx, left, right;
	REAL bxmin, bxmax, bymin, bymax, bzmin, bzmax;
	REAL* p[3];

	// reset variables
	curpos = 0; stack[curpos] = 0;
	icounter = 0; res = -1;
	// generate random ray direction
	cudamesh_random_sphere_point(&randomseed, v);
	t[0] = s[0] + v[0] * 1.5*dlen;
	t[1] = s[1] + v[1] * 1.5*dlen;
	t[2] = s[2] + v[2] * 1.5*dlen;
	// start traversal
	while (curpos >= 0)
	{
		curidx = stack[curpos];
		if (curidx >= 0) // a node
		{
			bxmin = aabbnodebbs[6 * curidx + 0]; bxmax = aabbnodebbs[6 * curidx + 1];
			bymin = aabbnodebbs[6 * curidx + 2]; bymax = aabbnodebbs[6 * curidx + 3];
			bzmin = aabbnodebbs[6 * curidx + 4]; bzmax = aabbnodebbs[6 * curidx + 5];
			if (cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
			{
				left = aabbnodeleft[curidx];
				right = aabbnoderight[curidx];
				stack[curpos++] = right;
				stack[curpos] = left;
			}
			else
			{
				curpos--;
			}
		}
		else // a leaf
		{
			curidx = -curidx - 1;
			bxmin = aabbpmbbs[6 * curidx + 0]; bxmax = aabbpmbbs[6 * curidx + 1];
			bymin = aabbpmbbs[6 * curidx + 2]; bymax = aabbpmbbs[6 * curidx + 3];
			bzmin = aabbpmbbs[6 * curidx + 4]; bzmax = aabbpmbbs[6 * curidx + 5];
			if (cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
			{
				p[0] = aabbpmcoord + 9 * curidx + 0;
				p[1] = aabbpmcoord + 9 * curidx + 3;
				p[2] = aabbpmcoord + 9 * curidx + 6;
				if (cudamesh_ts_intersection(p[0], p[1], p[2], s, t, type))
				{
					if (type == (int)UNKNOWNINTER || type == (int)ACROSSEDGE
						|| type == (int)ACROSSVERT || type == (int)COPLANAR)
					{
						// uncertain
						return false;
					}
					else if (type == (int)TOUCHEDGE || type == (int)TOUCHFACE
						|| type == (int)SHAREVERT)
					{
						// on boundary
						return true;
					}
					else if (type == (int)ACROSSFACE)
					{
						icounter++;
					}
				}
			}
			curpos--;
		}
	}
	// analyze result of this shooting
	res = (icounter & 1) == 1 ? 1 : 0;
	return (res > 0);
}

__device__ bool cudamesh_traversal_in_domain(REAL* s, REAL* t, int* aabbnodeleft,
	int* aabbnoderight, REAL* aabbnodebbs, REAL* aabbpmcoord, REAL* aabbpmbbs)
{
	// Stack variables
	int stack[MAXAABBLEVEL]; // can support up to 2^MAXAABBLEVEL polygons
	int curpos;

	// Traversal
	int icounter = 0; // count the time of valid intersection
	int res, type; // res: 0 - Out, 1 - In, 2 - On, -1 - Uninitialized
	int curidx, left, right;
	REAL bxmin, bxmax, bymin, bymax, bzmin, bzmax;
	REAL* p[3];

	// reset variables
	curpos = 0; stack[curpos] = 0;
	icounter = 0; res = -1;
	// start traversal
	while (curpos >= 0)
	{
		curidx = stack[curpos];
		if (curidx >= 0) // a node
		{
			bxmin = aabbnodebbs[6 * curidx + 0]; bxmax = aabbnodebbs[6 * curidx + 1];
			bymin = aabbnodebbs[6 * curidx + 2]; bymax = aabbnodebbs[6 * curidx + 3];
			bzmin = aabbnodebbs[6 * curidx + 4]; bzmax = aabbnodebbs[6 * curidx + 5];
			if (cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
			{
				left = aabbnodeleft[curidx];
				right = aabbnoderight[curidx];
				stack[curpos++] = right;
				stack[curpos] = left;
			}
			else
			{
				curpos--;
			}
		}
		else // a leaf
		{
			curidx = -curidx - 1;
			bxmin = aabbpmbbs[6 * curidx + 0]; bxmax = aabbpmbbs[6 * curidx + 1];
			bymin = aabbpmbbs[6 * curidx + 2]; bymax = aabbpmbbs[6 * curidx + 3];
			bzmin = aabbpmbbs[6 * curidx + 4]; bzmax = aabbpmbbs[6 * curidx + 5];
			if (cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
			{
				p[0] = aabbpmcoord + 9 * curidx + 0;
				p[1] = aabbpmcoord + 9 * curidx + 3;
				p[2] = aabbpmcoord + 9 * curidx + 6;
				if (cudamesh_ts_intersection(p[0], p[1], p[2], s, t, type))
				{
					if (type == (int)UNKNOWNINTER || type == (int)ACROSSEDGE
						|| type == (int)ACROSSVERT || type == (int)COPLANAR)
					{
						// uncertain
						return false;
					}
					else if (type == (int)TOUCHEDGE || type == (int)TOUCHFACE
						|| type == (int)SHAREVERT)
					{
						// on boundary
						return true;
					}
					else if (type == (int)ACROSSFACE)
					{
						icounter++;
					}
				}
			}
			curpos--;
		}
	}
	// analyze result of this shooting
	res = (icounter & 1) == 1 ? 1 : 0;
	return (res > 0);
}

__device__ void cudamesh_box_far_point(REAL* s, REAL* t, REAL xmin, REAL xmax,
	REAL ymin, REAL ymax, REAL zmin, REAL zmax, REAL diglen)
{
	REAL boxcenter[3], dir[3], len;
	boxcenter[0] = (xmin + xmax) / 2;
	boxcenter[1] = (ymin + ymax) / 2;
	boxcenter[2] = (zmin + zmax) / 2;
	dir[0] = s[0] - boxcenter[0];
	dir[1] = s[1] - boxcenter[1];
	dir[2] = s[2] - boxcenter[2];
	len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
	if (len == 0.0)
	{
		dir[0] = 1; dir[1] = 0; dir[2] = 0;
	}
	else
	{
		dir[0] /= len; dir[1] /= len; dir[2] /= len;
	}
	t[0] = s[0] + dir[0] * 1.5*diglen;
	t[1] = s[1] + dir[1] * 1.5*diglen;
	t[2] = s[2] + dir[2] * 1.5*diglen;
}

__device__ void cudamesh_primitive_fast_check(int* list, int numofaabbpms)
{

}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Basic helpers    														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
__device__ void cudamesh_swap(int& a, int& b)
{
	int tmp;
	tmp = a;
	a = b;
	b = tmp;
}

__device__ void cudamesh_swap(REAL& a, REAL& b)
{
	REAL tmp;
	tmp = a;
	a = b;
	b = tmp;
}

__device__ void cudamesh_swap(REAL* pa, REAL* pb)
{
	REAL tmp[3];
	for (int i = 0; i < 3; i++)
	{
		tmp[i] = pa[i];
		pa[i] = pb[i];
		pb[i] = tmp[i];
	}
}

__device__ void cudamesh_copy(REAL* d, REAL* s)
{
	d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
}

__device__ int cudamesh_compare(REAL* pa, REAL* pb)
{
	if (pa[0] < pb[0])
		return -1;
	if (pa[0] > pb[0])
		return 1;

	if (pa[1] < pb[1])
		return -1;
	if (pa[1] > pb[1])
		return 1;

	if (pa[2] < pb[2])
		return -1;
	if (pa[2] > pb[2])
		return 1;

	return 0;
}

__device__ REAL cudamesh_min(REAL v1, REAL v2)
{
	REAL min = v1;
	if (v2 < min)
		min = v2;
	return min;
}

__device__ REAL cudamesh_min(REAL v1, REAL v2, REAL v3)
{
	REAL min = v1;
	if (v2 < min)
		min = v2;
	if (v3 < min)
		min = v3;
	return min;
}

__device__ REAL cudamesh_max(REAL v1, REAL v2)
{
	REAL max = v1;
	if (v2 > max)
		max = v2;
	return max;
}

__device__ REAL cudamesh_max(REAL v1, REAL v2, REAL v3)
{
	REAL max = v1;
	if (v2 > max)
		max = v2;
	if (v3 > max)
		max = v3;
	return max;
}

// dot() returns the dot product: v1 dot v2.
__device__ REAL cudamesh_dot(REAL* v1, REAL* v2)
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// distance() computes the Euclidean distance between two points.
__device__ REAL cudamesh_distance(REAL* p1, REAL* p2)
{
	return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) +
		(p2[1] - p1[1]) * (p2[1] - p1[1]) +
		(p2[2] - p1[2]) * (p2[2] - p1[2]));
}

__device__ REAL cudamesh_squared_distance(REAL* p1, REAL* p2)
{
	return (p2[0] - p1[0]) * (p2[0] - p1[0]) +
		(p2[1] - p1[1]) * (p2[1] - p1[1]) +
		(p2[2] - p1[2]) * (p2[2] - p1[2]);
}

__device__ REAL cudamesh_power_distance(REAL* p1, REAL* p2, REAL w1, REAL w2)
{
	return cudamesh_squared_distance(p1, p2) - w1 - w2;
}

// cross() computes the cross product: n = v1 cross v2.
__device__ void cudamesh_cross(REAL* v1, REAL* v2, REAL* n)
{
	n[0] = v1[1] * v2[2] - v2[1] * v1[2];
	n[1] = -(v1[0] * v2[2] - v2[0] * v1[2]);
	n[2] = v1[0] * v2[1] - v2[0] * v1[1];
}

// randomnation()    Generate a random number between 0 and 'choices' - 1.
__device__ unsigned long cudamesh_randomnation(unsigned long * randomseed, unsigned int choices)
{
	unsigned long newrandom;

	if (choices >= 714025l) {
		newrandom = (*randomseed * 1366l + 150889l) % 714025l;
		*randomseed = (newrandom * 1366l + 150889l) % 714025l;
		newrandom = newrandom * (choices / 714025l) + *randomseed;
		if (newrandom >= choices) {
			return newrandom - choices;
		}
		else {
			return newrandom;
		}
	}
	else {

		*randomseed = (*randomseed * 1366l + 150889l) % 714025l;
		return *randomseed % choices;
	}
}

// Generate a random point on the sphere
__device__ void cudamesh_random_sphere_point(unsigned long * randomseed, REAL* p)
{
	int choices = 1001;
	int offset = choices / 2;
	int x, y, z;
	REAL len;
	do {
		x = cudamesh_randomnation(randomseed, choices);
		y = cudamesh_randomnation(randomseed, choices);
		z = cudamesh_randomnation(randomseed, choices);
		x -= offset; y -= offset; z -= offset;
		len = sqrt((x*x + y*y + z*z)*1.0);
	} while (len == 0.0);
	REAL X = x*1.0 / len;
	REAL Y = y*1.0 / len;
	REAL Z = z*1.0 / len;
	p[0] = X; p[1] = Y; p[2] = Z;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Geometric helpers														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ bool cudamesh_lp_intersection(REAL* p, REAL* q, REAL* a, REAL* b,
	REAL* c, REAL* ipt)
{
	REAL ap[3], ab[3], ac[3], qp[3], abac[3];
	ap[0] = p[0] - a[0]; ap[1] = p[1] - a[1]; ap[2] = p[2] - a[2];
	ab[0] = b[0] - a[0]; ab[1] = b[1] - a[1]; ab[2] = b[2] - a[2];
	ac[0] = c[0] - a[0]; ac[1] = c[1] - a[1]; ac[2] = c[2] - a[2];
	qp[0] = q[0] - p[0]; qp[1] = q[1] - p[1]; qp[2] = q[2] - p[2];
	abac[0] = ab[1] * ac[2] - ab[2] * ac[1];
	abac[1] = ab[2] * ac[0] - ab[0] * ac[2];
	abac[2] = ab[0] * ac[1] - ab[1] * ac[0];
	
	REAL den = qp[0] * abac[0] + qp[1] * abac[1] + qp[2] * abac[2];
	// [Note: den is equal to det(qp, ab, ac).]

	// If the denominator is certainly not 0...
	if (den != 0.0) {
		// ...then compute the intersection point...
		REAL fac = (ap[0] * abac[0] + ap[1] * abac[1] + ap[2] * abac[2]) / den;
		//                       [Note: ap * abac is equal to det(ap, ab, ac).]
		ipt[0] = p[0] - qp[0] * fac;
		ipt[1] = p[1] - qp[1] * fac;
		ipt[2] = p[2] - qp[2] * fac;

		return true;
	}
	return false;
}

__device__ bool cudamesh_ts_intersection(REAL* A, REAL* B, REAL* C,
	REAL* P, REAL* Q, int& type)
{
#define SETVECTOR3(V, a0, a1, a2) (V)[0] = (a0); (V)[1] = (a1); (V)[2] = (a2)
	REAL sP, sQ;

	REAL* R = NULL; // no use
	// Test the locations of P and Q with respect to ABC.
	sP = cuda_orient3dfast(A, B, C, P);
	sQ = cuda_orient3dfast(A, B, C, Q);

	REAL *U[3], *V[3]; //, Ptmp;
	int pu[3], pv[3]; //, itmp;
	REAL s1, s2, s3;
	int z1;


	if (sP < 0) {
		if (sQ < 0) { // (--) disjoint
			return false;
		}
		else {
			if (sQ > 0) { // (-+)
				SETVECTOR3(U, A, B, C);
				SETVECTOR3(V, P, Q, R);
				SETVECTOR3(pu, 0, 1, 2);
				SETVECTOR3(pv, 0, 1, 2);
				z1 = 0;
			}
			else { // (-0)
				SETVECTOR3(U, A, B, C);
				SETVECTOR3(V, P, Q, R);
				SETVECTOR3(pu, 0, 1, 2);
				SETVECTOR3(pv, 0, 1, 2);
				z1 = 1;
			}
		}
	}
	else {
		if (sP > 0) { // (+-)
			if (sQ < 0) {
				SETVECTOR3(U, A, B, C);
				SETVECTOR3(V, Q, P, R);  // P and Q are flipped.
				SETVECTOR3(pu, 0, 1, 2);
				SETVECTOR3(pv, 1, 0, 2);
				z1 = 0;
			}
			else {
				if (sQ > 0) { // (++) disjoint
					return false;
				}
				else { // (+0)
					SETVECTOR3(U, B, A, C); // A and B are flipped.
					SETVECTOR3(V, P, Q, R);
					SETVECTOR3(pu, 1, 0, 2);
					SETVECTOR3(pv, 0, 1, 2);
					z1 = 1;
				}
			}
		}
		else { // sP == 0
			if (sQ < 0) { // (0-)
				SETVECTOR3(U, A, B, C);
				SETVECTOR3(V, Q, P, R);  // P and Q are flipped.
				SETVECTOR3(pu, 0, 1, 2);
				SETVECTOR3(pv, 1, 0, 2);
				z1 = 1;
			}
			else {
				if (sQ > 0) { // (0+)
					SETVECTOR3(U, B, A, C);  // A and B are flipped.
					SETVECTOR3(V, Q, P, R);  // P and Q are flipped.
					SETVECTOR3(pu, 1, 0, 2);
					SETVECTOR3(pv, 1, 0, 2);
					z1 = 1;
				}
				else { // (00)
					   // A, B, C, P, and Q are coplanar.
					z1 = 2;
				}
			}
		}
	}

	if (z1 == 2) {
		// The triangle and the edge are coplanar.
		type = COPLANAR;
		return true;
	}

	s1 = cuda_orient3dfast(U[0], U[1], V[0], V[1]);
	if (s1 < 0) {
		return false;
	}

	s2 = cuda_orient3dfast(U[1], U[2], V[0], V[1]);
	if (s2 < 0) {
		return false;
	}

	s3 = cuda_orient3dfast(U[2], U[0], V[0], V[1]);
	if (s3 < 0) {
		return false;
	}

	if (z1 == 0) {
		if (s1 > 0) {
			if (s2 > 0) {
				if (s3 > 0) { // (+++)
							  // [P, Q] passes interior of [A, B, C].
					type = (int)ACROSSFACE;
				}
				else { // s3 == 0 (++0)
					   // [P, Q] intersects [C, A].
					type = (int)ACROSSEDGE;
				}
			}
			else { // s2 == 0
				if (s3 > 0) { // (+0+)
							  // [P, Q] intersects [B, C].
					type = (int)ACROSSEDGE;
				}
				else { // s3 == 0 (+00)
					   // [P, Q] passes C.
					type = (int)ACROSSVERT;
				}
			}
		}
		else { // s1 == 0
			if (s2 > 0) {
				if (s3 > 0) { // (0++)
							  // [P, Q] intersects [A, B].
					type = (int)ACROSSEDGE;
				}
				else { // s3 == 0 (0+0)
					   // [P, Q] passes A.
					type = (int)ACROSSVERT;
				}
			}
			else { // s2 == 0
				if (s3 > 0) { // (00+)
							  // [P, Q] passes B.
					type = (int)ACROSSVERT;
				}
				else { // s3 == 0 (000)
					   // Impossible.
					type = (int)UNKNOWNINTER;
				}
			}
		}
	}
	else { // z1 == 1
		if (s1 > 0) {
			if (s2 > 0) {
				if (s3 > 0) { // (+++)
							  // Q lies in [A, B, C].
					type = (int)TOUCHFACE;
				}
				else { // s3 == 0 (++0)
					   // Q lies on [C, A].
					type = (int)TOUCHEDGE;
				}
			}
			else { // s2 == 0
				if (s3 > 0) { // (+0+)
							  // Q lies on [B, C].
					type = (int)TOUCHEDGE;
				}
				else { // s3 == 0 (+00)
					   // Q = C.
					type = (int)SHAREVERT;
				}
			}
		}
		else { // s1 == 0
			if (s2 > 0) {
				if (s3 > 0) { // (0++)
							  // Q lies on [A, B].
					type = (int)TOUCHEDGE;
				}
				else { // s3 == 0 (0+0)
					   // Q = A.
					type = (int)SHAREVERT;
				}
			}
			else { // s2 == 0
				if (s3 > 0) { // (00+)
							  // Q = B.
					type = (int)SHAREVERT;
				}
				else { // s3 == 0 (000)
					   // Impossible.
					type = (int)UNKNOWNINTER;
				}
			}
		}
	}

	// T and E intersect in a single point.
	return true;

#undef SETVECTOR3
}

__device__ bool cudamesh_ts_intersection(REAL* p, REAL* q, REAL* a, REAL* b,
	REAL* c, REAL* ipt)
{
	// Adapted from CGAL-4.13\include\CGAL\Mesh_3\Robust_intersection_traits_3.h
	double ret;
	int abcp, abcq;
	ret = cuda_orient3dfast(a, b, c, p);
	if (ret > 0.0)
		abcp = 1;
	else if (ret < 0.0)
		abcp = -1;
	else
		abcp = 0;
	ret = cuda_orient3dfast(a, b, c, q);
	if (ret > 0.0)
		abcq = 1;
	else if (ret < 0.0)
		abcq = -1;
	else
		abcq = 0;

	switch (abcp) {
	case -1:
		switch (abcq) {
		case -1:
			// the segment lies in the above open halfspaces defined by the
			// triangle's supporting plane
			return false;

		case 1:
			// p sees the triangle in counterclockwise order
			if (cuda_orient3dfast(p, q, a, b) >= 0.0
				&& cuda_orient3dfast(p, q, b, c) >= 0.0
				&& cuda_orient3dfast(p, q, c, a) >= 0.0)
			{
				// The intersection is a point
				return cudamesh_lp_intersection(p, q, a, b, c, ipt);
			}
			else
				return false;

		default: // coplanar
				 // q belongs to the triangle's supporting plane
				 // p sees the triangle in counterclockwise order
			if (cuda_orient3dfast(p, q, a, b) >= 0.0
				&& cuda_orient3dfast(p, q, b, c) >= 0.0
				&& cuda_orient3dfast(p, q, c, a) >= 0.0)
			{
				cudamesh_copy(ipt, q);
				return true;
			}
			else return false;
		}
	case 1:
		switch (abcq) {
		case -1:
			// q sees the triangle in counterclockwise order
			if (cuda_orient3dfast(q, p, a, b) >= 0.0
				&& cuda_orient3dfast(q, p, b, c) >= 0.0
				&& cuda_orient3dfast(q, p, c, a) >= 0.0)
			{
				// The intersection is a point
				return cudamesh_lp_intersection(p, q, a, b, c, ipt);
			}
			else
				return false;

		case 1:
			// the segment lies in the below open halfspaces defined by the
			// triangle's supporting plane
			return false;

		default: // coplanar
				 // q belongs to the triangle's supporting plane
				 // p sees the triangle in clockwise order
			if (cuda_orient3dfast(q, p, a, b) >= 0.0
				&& cuda_orient3dfast(q, p, b, c) >= 0.0
				&& cuda_orient3dfast(q, p, c, a) >= 0.0)
			{
				cudamesh_copy(ipt, q);
				return true;
			}
			else return false;
		}
	default: // coplanar
		switch (abcq) {
		case -1:
			// q sees the triangle in counterclockwise order
			if (cuda_orient3dfast(q, p, a, b) >= 0.0
				&& cuda_orient3dfast(q, p, b, c) >= 0.0
				&& cuda_orient3dfast(q, p, c, a) >= 0.0)
			{
				cudamesh_copy(ipt, p);
				return true;
			}
			else
				return false;
		case 1:
			// q sees the triangle in clockwise order
			if (cuda_orient3dfast(p, q, a, b) >= 0.0
				&& cuda_orient3dfast(p, q, b, c) >= 0.0
				&& cuda_orient3dfast(p, q, c, a) >= 0.0)
			{
				cudamesh_copy(ipt, p);
				return true;
			}
			else
				return false;
		case 0:
			// the segment is coplanar with the triangle's supporting plane
			// we test whether the segment intersects the triangle in the common
			// supporting plane
			//
			// Laurent Rineau, 2016/10/10: this case is purposely ignored by
			// Mesh_3, because if the intersection is not a point, it is
			// ignored anyway.
			return false;
		default: // should not happen.
			return false;
		}
	}
}

__device__ bool cudamesh_lu_decmp(REAL lu[4][4], int n, int* ps, REAL* d, int N)
{
	REAL scales[4];
	REAL pivot, biggest, mult, tempf;
	int pivotindex = 0;
	int i, j, k;

	*d = 1.0;                                      // No row interchanges yet.

	for (i = N; i < n + N; i++) {                             // For each row.
															  // Find the largest element in each row for row equilibration
		biggest = 0.0;
		for (j = N; j < n + N; j++)
			if (biggest < (tempf = fabs(lu[i][j])))
				biggest = tempf;
		if (biggest != 0.0)
			scales[i] = 1.0 / biggest;
		else {
			scales[i] = 0.0;
			return false;                            // Zero row: singular matrix.
		}
		ps[i] = i;                                 // Initialize pivot sequence.
	}

	for (k = N; k < n + N - 1; k++) {                      // For each column.
														   // Find the largest element in each column to pivot around.
		biggest = 0.0;
		for (i = k; i < n + N; i++) {
			if (biggest < (tempf = fabs(lu[ps[i]][k]) * scales[ps[i]])) {
				biggest = tempf;
				pivotindex = i;
			}
		}
		if (biggest == 0.0) {
			return false;                         // Zero column: singular matrix.
		}
		if (pivotindex != k) {                         // Update pivot sequence.
			j = ps[k];
			ps[k] = ps[pivotindex];
			ps[pivotindex] = j;
			*d = -(*d);                          // ...and change the parity of d.
		}

		// Pivot, eliminating an extra variable  each time
		pivot = lu[ps[k]][k];
		for (i = k + 1; i < n + N; i++) {
			lu[ps[i]][k] = mult = lu[ps[i]][k] / pivot;
			if (mult != 0.0) {
				for (j = k + 1; j < n + N; j++)
					lu[ps[i]][j] -= mult * lu[ps[k]][j];
			}
		}
	}

	// (lu[ps[n + N - 1]][n + N - 1] == 0.0) ==> A is singular.
	return lu[ps[n + N - 1]][n + N - 1] != 0.0;
}

__device__ void cudamesh_lu_solve(REAL lu[4][4], int n, int* ps, REAL* b, int N)
{
	int i, j;
	REAL X[4], dot;

	for (i = N; i < n + N; i++) X[i] = 0.0;

	// Vector reduction using U triangular matrix.
	for (i = N; i < n + N; i++) {
		dot = 0.0;
		for (j = N; j < i + N; j++)
			dot += lu[ps[i]][j] * X[j];
		X[i] = b[ps[i]] - dot;
	}

	// Back substitution, in L triangular matrix.
	for (i = n + N - 1; i >= N; i--) {
		dot = 0.0;
		for (j = i + 1; j < n + N; j++)
			dot += lu[ps[i]][j] * X[j];
		X[i] = (X[i] - dot) / lu[ps[i]][i];
	}

	for (i = N; i < n + N; i++) b[i] = X[i];
}

__device__ bool cudamesh_circumsphere(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL* cent, REAL* radius)
{
	REAL A[4][4], rhs[4], D;
	int indx[4];

	// Compute the coefficient matrix A (3x3).
	A[0][0] = pb[0] - pa[0];
	A[0][1] = pb[1] - pa[1];
	A[0][2] = pb[2] - pa[2];
	A[1][0] = pc[0] - pa[0];
	A[1][1] = pc[1] - pa[1];
	A[1][2] = pc[2] - pa[2];
	if (pd != NULL) {
		A[2][0] = pd[0] - pa[0];
		A[2][1] = pd[1] - pa[1];
		A[2][2] = pd[2] - pa[2];
	}
	else {
		cudamesh_cross(A[0], A[1], A[2]);
	}

	// Compute the right hand side vector b (3x1).
	rhs[0] = 0.5 * cudamesh_dot(A[0], A[0]);
	rhs[1] = 0.5 * cudamesh_dot(A[1], A[1]);
	if (pd != NULL) {
		rhs[2] = 0.5 * cudamesh_dot(A[2], A[2]);
	}
	else {
		rhs[2] = 0.0;
	}

	// Solve the 3 by 3 equations use LU decomposition with partial pivoting
	//   and backward and forward substitute..
	if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
		if (radius != (REAL *)NULL) *radius = 0.0;
		return false;
	}
	cudamesh_lu_solve(A, 3, indx, rhs, 0);
	if (cent != (REAL *)NULL) {
		cent[0] = pa[0] + rhs[0];
		cent[1] = pa[1] + rhs[1];
		cent[2] = pa[2] + rhs[2];
	}
	if (radius != (REAL *)NULL) {
		*radius = sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);
	}
	return true;
}

__device__ bool cudamesh_weightedcircumsphere(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL  aw, REAL bw, REAL cw, REAL  dw, REAL* cent, REAL* radius)
{
	bool unweighted = (aw == 0.0) && (bw == 0.0) && (cw == 0.0) && (dw == 0.0);

	if (unweighted)
	{
		return cudamesh_circumsphere(pa, pb, pc, pd, cent, radius);
	}
	else
	{
		REAL A[4][4], rhs[4], D;
		int indx[4];
		REAL aheight, bheight, cheight, dheight;
		aheight = pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2] - aw;
		bheight = pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2] - bw;
		cheight = pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2] - cw;
		dheight = pd[0] * pd[0] + pd[1] * pd[1] + pd[2] * pd[2] - dw;

		// Set the coefficient matrix A (4 x 4).
		A[0][0] = 1.0; A[0][1] = pa[0]; A[0][2] = pa[1]; A[0][3] = pa[2];
		A[1][0] = 1.0; A[1][1] = pb[0]; A[1][2] = pb[1]; A[1][3] = pb[2];
		A[2][0] = 1.0; A[2][1] = pc[0]; A[2][2] = pc[1]; A[2][3] = pc[2];
		A[3][0] = 1.0; A[3][1] = pd[0]; A[3][2] = pd[1]; A[3][3] = pd[2];

		// Set the right hand side vector (4 x 1).
		rhs[0] = 0.5 * aheight;
		rhs[1] = 0.5 * bheight;
		rhs[2] = 0.5 * cheight;
		rhs[3] = 0.5 * dheight;

		// Solve the 4 by 4 equations use LU decomposition with partial pivoting
		//   and backward and forward substitute..
		if (!cudamesh_lu_decmp(A, 4, indx, &D, 0)) {
			if (radius != (REAL *)NULL) *radius = 0.0;
			return false;
		}
		cudamesh_lu_solve(A, 4, indx, rhs, 0);

		if (cent != (REAL *)NULL) {
			cent[0] = rhs[1];
			cent[1] = rhs[2];
			cent[2] = rhs[3];
		}
		if (radius != (REAL *)NULL) {
			// rhs[0] = - rheight / 2;
			// rheight  = - 2 * rhs[0];
			//          =  r[0]^2 + r[1]^2 + r[2]^2 - radius^2
			// radius^2 = r[0]^2 + r[1]^2 + r[2]^2 -rheight
			//          = r[0]^2 + r[1]^2 + r[2]^2 + 2 * rhs[0]
			*radius = sqrt(rhs[1] * rhs[1] + rhs[2] * rhs[2] + rhs[3] * rhs[3]
				+ 2.0 * rhs[0]);
		}
		return true;
	}
}

__device__ REAL cudamesh_determinant(REAL a00, REAL a01, REAL a10, REAL a11)
{
	// First compute the det2x2
	const REAL m01 = a00*a11 - a10*a01;
	return m01;
}

__device__ REAL cudamesh_determinant(REAL a00, REAL a01, REAL a02,
	REAL a10, REAL a11, REAL a12, REAL a20, REAL a21, REAL a22)
{
	// First compute the det2x2
	const REAL m01 = a00*a11 - a10*a01;
	const REAL m02 = a00*a21 - a20*a01;
	const REAL m12 = a10*a21 - a20*a11;
	// Now compute the minors of rank 3
	const REAL m012 = m01*a22 - m02*a12 + m12*a02;
	return m012;
}

__device__ bool cudamesh_circumcenter(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL* cent
)
{
	//if (cuda_orient3d(pa, pb, pc, pd) >= 0) // may be a degenerate tet
	//	return false;

	REAL num_x, num_y, num_z, den;

	REAL bax = pb[0] - pa[0];
	REAL bay = pb[1] - pa[1];
	REAL baz = pb[2] - pa[2];
	REAL ba2 = bax*bax + bay*bay + baz*baz;

	REAL cax = pc[0] - pa[0];
	REAL cay = pc[1] - pa[1];
	REAL caz = pc[2] - pa[2];
	REAL ca2 = cax*cax + cay*cay + caz*caz;

	REAL dax = pd[0] - pa[0];
	REAL day = pd[1] - pa[1];
	REAL daz = pd[2] - pa[2];
	REAL da2 = dax*dax + day*day + daz*daz;

	num_x = cudamesh_determinant(bay, baz, ba2,
		cay, caz, ca2, day, daz, da2);
	num_y = cudamesh_determinant(bax, baz, ba2,
		cax, caz, ca2, dax, daz, da2);
	num_z = cudamesh_determinant(bax, bay, ba2,
		cax, cay, ca2, dax, day, da2);
	den = cudamesh_determinant(bax, bay, baz,
		cax, cay, caz, dax, day, daz);

	if (den == 0.0)
		return false;

	REAL inv = 1 / (2 * den);
	cent[0] = pa[0] + num_x*inv;
	cent[1] = pa[1] - num_y*inv;
	cent[2] = pa[2] + num_z*inv;
	return true;
}

__device__ bool cudamesh_weightedcircumcenter(REAL* pa, REAL* pb, REAL* pc,
	REAL aw, REAL bw, REAL cw, REAL* cent
)
{

	REAL num_x, num_y, num_z, den;

	REAL bax = pb[0] - pa[0];
	REAL bay = pb[1] - pa[1];
	REAL baz = pb[2] - pa[2];
	REAL ba2 = bax*bax + bay*bay + baz*baz - bw + aw;

	REAL cax = pc[0] - pa[0];
	REAL cay = pc[1] - pa[1];
	REAL caz = pc[2] - pa[2];
	REAL ca2 = cax*cax + cay*cay + caz*caz - cw + aw;

	REAL sx = bay*caz - baz*cay;
	REAL sy = baz*cax - bax*caz;
	REAL sz = bax*cay - bay*cax;

	num_x = ba2 * cudamesh_determinant(cay, caz, sy, sz)
		- ca2*cudamesh_determinant(bay, baz, sy, sz);
	num_y = ba2 * cudamesh_determinant(cax, caz, sx, sz)
		- ca2*cudamesh_determinant(bax, baz, sx, sz);
	num_z = ba2 * cudamesh_determinant(cax, cay, sx, sy)
		- ca2*cudamesh_determinant(bax, bay, sx, sy);
	den = cudamesh_determinant(bax, bay, baz,
		cax, cay, caz, sx, sy, sz);

	if (den == 0.0)
		return false;

	REAL inv = 1 / (2 * den);
	cent[0] = pa[0] + num_x*inv;
	cent[1] = pa[1] - num_y*inv;
	cent[2] = pa[2] + num_z*inv;
	return true;
}

__device__ bool cudamesh_weightedcircumcenter(REAL* pa, REAL* pb, REAL* pc, REAL* pd, 
	REAL aw, REAL bw, REAL cw, REAL dw, REAL* cent
)
{
	//if (cuda_orient3d(pa, pb, pc, pd) >= 0) // may be a degenerate tet
	//	return false;
	bool unweighted = (aw == 0.0) && (bw == 0.0) && (cw == 0.0) && (dw == 0.0);

	if (unweighted)
	{
		return cudamesh_circumcenter(pa, pb, pc, pd, cent);
	}
	else
	{
		REAL num_x, num_y, num_z, den;

		REAL bax = pb[0] - pa[0];
		REAL bay = pb[1] - pa[1];
		REAL baz = pb[2] - pa[2];
		REAL ba2 = bax*bax + bay*bay + baz*baz - bw + aw;

		REAL cax = pc[0] - pa[0];
		REAL cay = pc[1] - pa[1];
		REAL caz = pc[2] - pa[2];
		REAL ca2 = cax*cax + cay*cay + caz*caz - cw + aw;

		REAL dax = pd[0] - pa[0];
		REAL day = pd[1] - pa[1];
		REAL daz = pd[2] - pa[2];
		REAL da2 = dax*dax + day*day + daz*daz - dw + aw;

		num_x = cudamesh_determinant(bay, baz, ba2,
			cay, caz, ca2, day, daz, da2);
		num_y = cudamesh_determinant(bax, baz, ba2,
			cax, caz, ca2, dax, daz, da2);
		num_z = cudamesh_determinant(bax, bay, ba2,
			cax, cay, ca2, dax, day, da2);
		den = cudamesh_determinant(bax, bay, baz,
			cax, cay, caz, dax, day, daz);

		if (den == 0.0)
			return false;

		//printf(" num_x = %g, num_y = %g, num_z = %g, den = %g ",
		//	num_x, num_y, num_z, den);

		REAL inv = 1 / (2 * den);
		cent[0] = pa[0] + num_x*inv;
		cent[1] = pa[1] - num_y*inv;
		cent[2] = pa[2] + num_z*inv;
		return true;
	}
}

__device__ bool cudamesh_weightedcircumcenter_equal(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL aw, REAL bw, REAL cw, REAL dw, REAL* cent)
{
	REAL mycent[3];
	cudamesh_weightedcircumsphere(pa, pb, pc, pd, aw, bw, cw, dw, mycent, NULL); // Use method from TetGen
	cudamesh_weightedcircumcenter(pa, pb, pc, pd, aw, bw, cw, dw, cent); // Use method from CGAL
	bool equal = cudamesh_squared_distance(mycent, cent) < EPSILON;
	return equal;
}

__device__ bool cudamesh_weightedcircumcenter_perturbed(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL aw, REAL bw, REAL cw, REAL dw, REAL* cent, REAL* norm)
{
	// pertube the oppo
	REAL factor = 0.1;
	REAL pd_pertubed[3];
	pd_pertubed[0] = pd[0] + factor*norm[0]; 
	pd_pertubed[1] = pd[1] + factor*norm[1]; 
	pd_pertubed[2] = pd[2] + factor*norm[2];

	return cudamesh_weightedcircumcenter_equal(pa, pb, pc, pd_pertubed, aw, bw, cw, dw, cent);
}

__device__ bool cudamesh_isDegenerateTet(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL aw, REAL bw, REAL cw, REAL dw, REAL* cent)
{
	bool degenerate = !cudamesh_weightedcircumcenter_equal(pa, pb, pc, pd, aw, bw, cw, dw, cent);

	return false;

	if (!degenerate)
	{
		degenerate = cudamesh_tetrahedronvolume(pa, pb, pc, pd) == 0.0;
	}

	//if (!degenerate)
	//	degenerate = cudamesh_isInvalid(cent[0]) ||
	//		cudamesh_isInvalid(cent[1]) ||
	//		cudamesh_isInvalid(cent[2]);

	return degenerate;
}

__device__ bool cudamesh_compute_squared_radius_smallest_orthogonal_sphere(REAL* pa,
	REAL* pb, REAL aw, REAL bw, REAL& sradius)
{
	REAL abx = pa[0] - pb[0];
	REAL aby = pa[1] - pb[1];
	REAL abz = pa[2] - pb[2];
	REAL ab2 = abx*abx + aby*aby + abz*abz;
	
	if (ab2 != 0.0)
	{
		REAL inv = 1.0 / (2.0*ab2);
		REAL alpha = 1.0 / 2 + (aw - bw)*inv;

		sradius = alpha*alpha*ab2 - aw;
		return true;
	}

	return false;
}

__device__ bool cudamesh_compute_squared_radius_smallest_orthogonal_sphere(REAL* pa,
	REAL* pb, REAL* pc, REAL aw, REAL bw, REAL cw, REAL& sradius)
{
	REAL num_x, num_y, num_z, den;

	REAL bax = pb[0] - pa[0];
	REAL bay = pb[1] - pa[1];
	REAL baz = pb[2] - pa[2];
	REAL ba2 = bax*bax + bay*bay + baz*baz - bw + aw;

	REAL cax = pc[0] - pa[0];
	REAL cay = pc[1] - pa[1];
	REAL caz = pc[2] - pa[2];
	REAL ca2 = cax*cax + cay*cay + caz*caz - cw + aw;

	REAL sx = bay*caz - baz*cay;
	REAL sy = baz*cax - bax*caz;
	REAL sz = bax*cay - bay*cax;

	num_x = ba2 * cudamesh_determinant(cay, caz, sy, sz)
		- ca2*cudamesh_determinant(bay, baz, sy, sz);
	num_y = ba2 * cudamesh_determinant(cax, caz, sx, sz)
		- ca2*cudamesh_determinant(bax, baz, sx, sz);
	num_z = ba2 * cudamesh_determinant(cax, cay, sx, sy)
		- ca2*cudamesh_determinant(bax, bay, sx, sy);
	den = cudamesh_determinant(bax, bay, baz,
		cax, cay, caz, sx, sy, sz);

	if (den != 0.0)
	{
		REAL inv = 1.0 / (2 * den);
		sradius = (num_x*num_x + num_y*num_y + num_z*num_z) * inv*inv - aw;
		return true;
	}

	return false;
}

__device__ bool cudamesh_compute_squared_radius_smallest_orthogonal_sphere(REAL* pa,
	REAL* pb, REAL* pc, REAL* pd, REAL aw, REAL bw, REAL cw, REAL dw, REAL& sradius)
{
	REAL num_x, num_y, num_z, den;

	REAL bax = pb[0] - pa[0];
	REAL bay = pb[1] - pa[1];
	REAL baz = pb[2] - pa[2];
	REAL ba2 = bax*bax + bay*bay + baz*baz - bw + aw;

	REAL cax = pc[0] - pa[0];
	REAL cay = pc[1] - pa[1];
	REAL caz = pc[2] - pa[2];
	REAL ca2 = cax*cax + cay*cay + caz*caz - cw + aw;

	REAL dax = pd[0] - pa[0];
	REAL day = pd[1] - pa[1];
	REAL daz = pd[2] - pa[2];
	REAL da2 = dax*dax + day*day + daz*daz - dw + aw;

	num_x = cudamesh_determinant(bay, baz, ba2, cay, caz, ca2, day, daz, da2);
	num_y = cudamesh_determinant(bax, baz, ba2, cax, caz, ca2, dax, daz, da2);
	num_z = cudamesh_determinant(bax, bay, ba2, cax, cay, ca2, dax, day, da2);
	den = cudamesh_determinant(bax, bay, baz, cax, cay, caz, dax, day, daz);

	if (den != 0.0)
	{
		REAL inv = 1.0 / (2 * den);
		sradius = (num_x*num_x + num_y*num_y + num_z*num_z) * inv*inv - aw;
		return true;
	}

	return false;
}

__device__ bool cudamesh_raydir(REAL* pa, REAL* pb, REAL* pc, REAL* dir)
{
	// the direction should be opposite pa->pb X pa->pc
	REAL bax = pb[0] - pa[0];
	REAL bay = pb[1] - pa[1];
	REAL baz = pb[2] - pa[2];

	REAL cax = pc[0] - pa[0];
	REAL cay = pc[1] - pa[1];
	REAL caz = pc[2] - pa[2];

	REAL sx = bay*caz - baz*cay;
	REAL sy = baz*cax - bax*caz;
	REAL sz = bax*cay - bay*cax;
	// Inverse the direction
	sx = -sx; sy = -sy; sz = -sz;
	dir[0] = sx; dir[1] = sy; dir[2] = sz;
}

__device__ void cudamesh_facenormal(REAL* pa, REAL* pb, REAL* pc, REAL *n, int pivot,
	REAL* lav)
{
	REAL v1[3], v2[3], v3[3], *pv1, *pv2;
	REAL L1, L2, L3;

	v1[0] = pb[0] - pa[0];  // edge vector v1: a->b
	v1[1] = pb[1] - pa[1];
	v1[2] = pb[2] - pa[2];
	v2[0] = pa[0] - pc[0];  // edge vector v2: c->a
	v2[1] = pa[1] - pc[1];
	v2[2] = pa[2] - pc[2];

	// Default, normal is calculated by: v1 x (-v2) (see Fig. fnormal).
	if (pivot > 0) {
		// Choose edge vectors by Burdakov's algorithm.
		v3[0] = pc[0] - pb[0];  // edge vector v3: b->c
		v3[1] = pc[1] - pb[1];
		v3[2] = pc[2] - pb[2];
		L1 = cudamesh_dot(v1, v1);
		L2 = cudamesh_dot(v2, v2);
		L3 = cudamesh_dot(v3, v3);
		// Sort the three edge lengths.
		if (L1 < L2) {
			if (L2 < L3) {
				pv1 = v1; pv2 = v2; // n = v1 x (-v2).
			}
			else {
				pv1 = v3; pv2 = v1; // n = v3 x (-v1).
			}
		}
		else {
			if (L1 < L3) {
				pv1 = v1; pv2 = v2; // n = v1 x (-v2).
			}
			else {
				pv1 = v2; pv2 = v3; // n = v2 x (-v3).
			}
		}
		if (lav) {
			// return the average edge length.
			*lav = (sqrt(L1) + sqrt(L2) + sqrt(L3)) / 3.0;
		}
	}
	else {
		pv1 = v1; pv2 = v2; // n = v1 x (-v2).
	}

	// Calculate the face normal.
	cudamesh_cross(pv1, pv2, n);
	// Inverse the direction;
	n[0] = -n[0];
	n[1] = -n[1];
	n[2] = -n[2];
}

__device__ void cudamesh_calculateabovepoint4(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* abovept)
{
	REAL n1[3], n2[3], *norm;
	REAL len, len1, len2;

	// Select a base.
	cudamesh_facenormal(pa, pb, pc, n1, 1, NULL);
	len1 = sqrt(cudamesh_dot(n1, n1));
	cudamesh_facenormal(pa, pb, pd, n2, 1, NULL);
	len2 = sqrt(cudamesh_dot(n2, n2));
	if (len1 > len2) {
		norm = n1;
		len = len1;
	}
	else {
		norm = n2;
		len = len2;
	}
	assert(len > 0);
	norm[0] /= len;
	norm[1] /= len;
	norm[2] /= len;
	len = cudamesh_distance(pa, pb);
	abovept[0] = pa[0] + len * norm[0];
	abovept[1] = pa[1] + len * norm[1];
	abovept[2] = pa[2] + len * norm[2];
}

__device__ REAL cudamesh_bbox_diglen(int nodeId, REAL* aabbnodebbs)
{
	REAL bb[3], bt[3], dvec[3], dlen;
	bb[0] = aabbnodebbs[6 * nodeId + 0]; bb[1] = aabbnodebbs[6 * nodeId + 2]; bb[2] = aabbnodebbs[6 * nodeId + 4];
	bt[0] = aabbnodebbs[6 * nodeId + 1]; bt[1] = aabbnodebbs[6 * nodeId + 3]; bt[2] = aabbnodebbs[6 * nodeId + 5];
	dvec[0] = bt[0] - bb[0]; dvec[1] = bt[1] - bb[1]; dvec[2] = bt[2] - bb[2];
	dlen = sqrt(dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]);
	return dlen;
}

__device__ REAL cudamesh_triangle_squared_area(
	REAL* pa, REAL* pb, REAL* pc
)
{
	// Compute the area of this 3D triangle
	REAL AB[3], AC[3];
	int i;
	for (i = 0; i < 3; i++)
	{
		AB[i] = pb[i] - pa[i];
		AC[i] = pc[i] - pa[i];
	}

	REAL sarea =
		((AB[1] * AC[2] - AB[2] * AC[1])*(AB[1] * AC[2] - AB[2] * AC[1]) +
		(AB[2] * AC[0] - AB[0] * AC[2])*(AB[2] * AC[0] - AB[0] * AC[2]) +
		(AB[0] * AC[1] - AB[1] * AC[0])*(AB[0] * AC[1] - AB[1] * AC[0])) / 4;

	return sarea;
}

__device__ REAL cudamesh_tetrahedronvolume(
	REAL* pa, REAL* pb, REAL* pc, REAL* pd
)
{
	REAL vda[3], vdb[3], vdc[3];
	REAL vab[3], vbc[3], vca[3];
	REAL elen[6];

	int i;
	// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
	for (i = 0; i < 3; i++) vda[i] = pa[i] - pd[i];
	for (i = 0; i < 3; i++) vdb[i] = pb[i] - pd[i];
	for (i = 0; i < 3; i++) vdc[i] = pc[i] - pd[i];

	// Get the other edge vectors.
	for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
	for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
	for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

	elen[0] = cudamesh_dot(vda, vda);
	elen[1] = cudamesh_dot(vdb, vdb);
	elen[2] = cudamesh_dot(vdc, vdc);
	elen[3] = cudamesh_dot(vab, vab);
	elen[4] = cudamesh_dot(vbc, vbc);
	elen[5] = cudamesh_dot(vca, vca);

	// Use heron-type formula to compute the volume of a tetrahedron
	// https://en.wikipedia.org/wiki/Heron%27s_formula
	REAL U, V, W, u, v, w; // first three form a triangle; u opposite to U and so on
	REAL X, x, Y, y, Z, z;
	REAL a, b, c, d;
	U = sqrt(elen[3]); //ab
	V = sqrt(elen[4]); //bc
	W = sqrt(elen[5]); //ca
	u = sqrt(elen[2]); //dc
	v = sqrt(elen[0]); //da
	w = sqrt(elen[1]); //db

	X = (w - U + v)*(U + v + w);
	x = (U - v + w)*(v - w + U);
	Y = (u - V + w)*(V + w + u);
	y = (V - w + u)*(w - u + V);
	Z = (v - W + u)*(W + u + v);
	z = (W - u + v)*(u - v + W);

	a = sqrt(x*Y*Z);
	b = sqrt(y*Z*X);
	c = sqrt(z*X*Y);
	d = sqrt(x*y*z);

	REAL vol, val1, val2;
	val1 = (-a + b + c + d)*(a - b + c + d)*(a + b - c + d)*(a + b + c - d);
	val2 = 192 * u*v*w;
	if (val1 < 0.0 || val2 == 0.0)
		vol = 0.0;
	else
		vol = sqrt(val1) / val2;
	return vol;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Geometric predicates with symbolic perturbation							 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ REAL cudamesh_insphere_s(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe, 
	int ia, int ib, int ic, int id, int ie)
{
	REAL sign;
	sign = cuda_inspherefast(pa, pb, pc, pd, pe);
	//if (fabs(sign) < EPSILON)
	//	sign = cuda_insphereexact(pa, pb, pc, pd, pe);

	if (sign != 0.0) {
		return sign;
	}

	// Symbolic perturbation.
	REAL* pt[5], *swappt;
	int idx[5], swapidx;
	REAL oriA, oriB;
	int swaps, count;
	int n, i;

	pt[0] = pa;
	pt[1] = pb;
	pt[2] = pc;
	pt[3] = pd;
	pt[4] = pe;

	idx[0] = ia;
	idx[1] = ib;
	idx[2] = ic;
	idx[3] = id;
	idx[4] = ie;

	// Sort the five points such that their indices are in the increasing
	//   order. An optimized bubble sort algorithm is used, i.e., it has
	//   the worst case O(n^2) runtime, but it is usually much faster.
	swaps = 0; // Record the total number of swaps.
	n = 5;
	do {
		count = 0;
		n = n - 1;
		for (i = 0; i < n; i++) {
			if (idx[i] > idx[i + 1]) {
				swappt = pt[i]; pt[i] = pt[i + 1]; pt[i + 1] = swappt;
				swapidx = idx[i]; idx[i] = idx[i + 1]; idx[i + 1] = swapidx;
				count++;
			}
		}
		swaps += count;
		break;
	} while (count > 0); // Continue if some points are swapped.

	oriA = cuda_orient3d(pt[1], pt[2], pt[3], pt[4]);
	if (oriA != 0.0) {
		// Flip the sign if there are odd number of swaps.
		if ((swaps % 2) != 0) oriA = -oriA;
		return oriA;
	}

	oriB = -cuda_orient3d(pt[0], pt[2], pt[3], pt[4]);
	assert(oriB != 0.0); // SELF_CHECK
						 // Flip the sign if there are odd number of swaps.
	if ((swaps % 2) != 0) oriB = -oriB;
	return oriB;
}

__device__ REAL cudamesh_orient4d_s(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe,
	int ia, int ib, int ic, int id, int ie, REAL aw, REAL bw, REAL cw, REAL dw, REAL ew)
{
	REAL sign;
	REAL aheight, bheight, cheight, dheight, eheight;
	aheight = pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2] - aw;
	bheight = pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2] - bw;
	cheight = pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2] - cw;
	dheight = pd[0] * pd[0] + pd[1] * pd[1] + pd[2] * pd[2] - dw;
	eheight = pe[0] * pe[0] + pe[1] * pe[1] + pe[2] * pe[2] - ew;

	sign = cuda_orient4dfast(pa, pb, pc, pd, pe,
		aheight, bheight, cheight, dheight, eheight);
	if (sign != 0.0) {
		return sign;
	}

	// Symbolic perturbation.
	REAL* pt[5], *swappt;
	int idx[5], swapidx;
	REAL oriA, oriB;
	int swaps, count;
	int n, i;

	pt[0] = pa;
	pt[1] = pb;
	pt[2] = pc;
	pt[3] = pd;
	pt[4] = pe;

	idx[0] = ia;
	idx[1] = ib;
	idx[2] = ic;
	idx[3] = id;
	idx[4] = ie;

	// Sort the five points such that their indices are in the increasing
	//   order. An optimized bubble sort algorithm is used, i.e., it has
	//   the worst case O(n^2) runtime, but it is usually much faster.
	swaps = 0; // Record the total number of swaps.
	n = 5;
	do {
		count = 0;
		n = n - 1;
		for (i = 0; i < n; i++) {
			if (idx[i] > idx[i + 1]) {
				swappt = pt[i]; pt[i] = pt[i + 1]; pt[i + 1] = swappt;
				swapidx = idx[i]; idx[i] = idx[i + 1]; idx[i + 1] = swapidx;
				count++;
			}
		}
		swaps += count;
	} while (count > 0); // Continue if some points are swapped.

	oriA = cuda_orient3dfast(pt[1], pt[2], pt[3], pt[4]);
	if (oriA != 0.0) {
		// Flip the sign if there are odd number of swaps.
		if ((swaps % 2) != 0) oriA = -oriA;
		return oriA;
	}

	oriB = -cuda_orient3dfast(pt[0], pt[2], pt[3], pt[4]);
	assert(oriB != 0.0); // SELF_CHECK
						 // Flip the sign if there are odd number of swaps.
	if ((swaps % 2) != 0) oriB = -oriB;
	return oriB;
}

__device__ REAL cudamesh_incircle3d(REAL* pa, REAL* pb, REAL* pc, REAL* pd)
{
	REAL area2[2], n1[3], n2[3], c[3];
	REAL sign, r, d;

	// Calculate the areas of the two triangles [a, b, c] and [b, a, d].
	cudamesh_facenormal(pa, pb, pc, n1, 1, NULL);
	area2[0] = cudamesh_dot(n1, n1);
	cudamesh_facenormal(pb, pa, pd, n2, 1, NULL);
	area2[1] = cudamesh_dot(n2, n2);

	if (area2[0] > area2[1]) {
		// Choose [a, b, c] as the base triangle.
		cudamesh_circumsphere(pa, pb, pc, NULL, c, &r);
		d = cudamesh_distance(c, pd);
	}
	else {
		// Choose [b, a, d] as the base triangle.
		if (area2[1] > 0) {
			cudamesh_circumsphere(pb, pa, pd, NULL, c, &r);
			d = cudamesh_distance(c, pc);
		}
		else {
			// The four points are collinear. This case only happens on the boundary.
			return 0; // Return "not inside".
		}
	}

	sign = d - r;
	if (fabs(sign) / r < EPSILON) {
		sign = 0;
	}

	return sign;
}

__device__ bool cudamesh_is_out_bbox(REAL* pt, REAL xmin, REAL xmax,
	REAL ymin, REAL ymax, REAL zmin, REAL zmax)
{
	if (pt[0] < xmin || pt[0] > xmax
		|| pt[1] < ymin || pt[1] > ymax
		|| pt[2] < zmin || pt[2] > zmax)
		return true;
	else
		return false;
}

__device__ bool cudamesh_do_intersect_bbox(REAL* s, REAL* t, REAL bxmin, REAL bxmax,
	REAL bymin, REAL bymax, REAL bzmin, REAL bzmax)
{
	// Fast segment and box intersection test
	// Adapted from http://www.3dkingdoms.com/weekly/weekly.php?a=21

	// Set up bounding box
	REAL bboxCenter[3], bboxExtend[3];
	bboxCenter[0] = (bxmin + bxmax)*0.5; bboxExtend[0] = (bxmax - bxmin)*0.5;
	bboxCenter[1] = (bymin + bymax)*0.5; bboxExtend[1] = (bymax - bymin)*0.5;
	bboxCenter[2] = (bzmin + bzmax)*0.5; bboxExtend[2] = (bzmax - bzmin)*0.5;

	// Put segment in box space
	REAL sa[3], sb[3], smid[3], sl[3], sext[3];
	sa[0] = s[0] - bboxCenter[0]; sb[0] = t[0] - bboxCenter[0];
	sa[1] = s[1] - bboxCenter[1]; sb[1] = t[1] - bboxCenter[1];
	sa[2] = s[2] - bboxCenter[2]; sb[2] = t[2] - bboxCenter[2];
	smid[0] = (sa[0] + sb[0])*0.5; smid[1] = (sa[1] + sb[1])*0.5; smid[2] = (sa[2] + sb[2])*0.5;
	sl[0] = sa[0] - smid[0]; sl[1] = sa[1] - smid[1]; sl[2] = sa[2] - smid[2];
	sext[0] = fabs(sl[0]); sext[1] = fabs(sl[1]); sext[2] = fabs(sl[2]);

	// Use Separating Axis Test
	// Separation vector from box center to segment center is smid, since the segment is in box space
	if (fabs(smid[0]) > bboxExtend[0] + sext[0]) return false;
	if (fabs(smid[1]) > bboxExtend[1] + sext[1]) return false;
	if (fabs(smid[2]) > bboxExtend[2] + sext[2]) return false;
	// Crossproducts of segment and each axis
	if (fabs(smid[1] * sl[2] - smid[2] * sl[1])  >  (bboxExtend[1] * sext[2] + bboxExtend[2] * sext[1])) return false;
	if (fabs(smid[0] * sl[2] - smid[2] * sl[0])  >  (bboxExtend[0] * sext[2] + bboxExtend[2] * sext[0])) return false;
	if (fabs(smid[0] * sl[1] - smid[1] * sl[0])  >  (bboxExtend[0] * sext[1] + bboxExtend[1] * sext[0])) return false;
	// No separating axis, the segment and box intersect
	return true;
}

__device__ bool cudamesh_is_bad_aspect_ratio(REAL angle_bound, REAL* pa, REAL* pb, REAL* pc)
{
	REAL B = sin(PI*angle_bound / 180);
	B = B*B;

	REAL area = cudamesh_triangle_squared_area(pa, pb, pc);
	REAL dab = cudamesh_squared_distance(pa, pb);
	REAL dac = cudamesh_squared_distance(pa, pc);
	REAL dbc = cudamesh_squared_distance(pb, pc);
	REAL min_dabc = cudamesh_min(dab, dac, dbc);
	REAL aspect_ratio = 4 * area * min_dabc / (dab*dac*dbc);
	
	if (aspect_ratio < B)
		return true;
	else
		return false;
}

__device__ bool cudamesh_is_bad_radius_edge_ratio(REAL ratio_bound, REAL* p, REAL* q, REAL* r, REAL* s,
	REAL sradius)
{
	REAL B = ratio_bound*ratio_bound;
	REAL R = sradius;
	REAL min_sq_length = cudamesh_squared_distance(p, q);
	min_sq_length = cudamesh_min(min_sq_length, cudamesh_squared_distance(p, r));
	min_sq_length = cudamesh_min(min_sq_length, cudamesh_squared_distance(p, s));
	min_sq_length = cudamesh_min(min_sq_length, cudamesh_squared_distance(q, r));
	min_sq_length = cudamesh_min(min_sq_length, cudamesh_squared_distance(q, s));
	min_sq_length = cudamesh_min(min_sq_length, cudamesh_squared_distance(r, s));

	if (R > B*min_sq_length)
		return true;
	else
		return false;
}

__device__ bool cudamesh_is_bad_radius_size(REAL size_bound, REAL* cent, REAL* pt)
{
	REAL B = size_bound*size_bound;
	REAL r = cudamesh_squared_distance(cent, pt);

	if (r > B)
		return true;
	else
		return false;
}

__device__ bool cudamesh_is_bad_radius_size(REAL size_bound, REAL sradius)
{
	REAL B = size_bound*size_bound;
	REAL R = sradius;
	if (R > B)
		return true;
	else
		return false;
}

__device__ bool cudamesh_is_bad_distance(REAL distance_bound, REAL* cent, REAL* pa, REAL* pb, REAL* pc,
	REAL aw, REAL bw, REAL cw)
{
	REAL B = distance_bound*distance_bound;
	REAL wc[3];
	if (!cudamesh_weightedcircumcenter(pa, pb, pc, aw, bw, cw, wc))
		return false; // Ignore a degenerated facet
	REAL sq_dist = cudamesh_squared_distance(wc, cent);

	if (sq_dist > B)
		return true;
	else
		return false;
}

__device__ bool cudamesh_is_facet_on_surface(verttype at, verttype bt, verttype ct)
{
	if ((at != RIDGEVERTEX && at != FREESEGVERTEX && at != FACETVERTEX && at != FREEFACETVERTEX) ||
		(bt != RIDGEVERTEX && bt != FREESEGVERTEX && bt != FACETVERTEX && bt != FREEFACETVERTEX) ||
		(ct != RIDGEVERTEX && ct != FREESEGVERTEX && ct != FACETVERTEX && ct != FREEFACETVERTEX))
	{
		return false;
	}
	else
	{
		return true;
	}
}

__device__ bool cudamesh_is_bad_facet(REAL *pa, REAL *pb, REAL* pc,
	REAL aw, REAL bw, REAL cw, verttype at, verttype bt, verttype ct, REAL* cent,
	REAL facet_angle, REAL facet_size, REAL facet_distance)
{
	if (cudamesh_triangle_squared_area(pa, pb, pc) == 0.0) // ignore a degenerate triface
		return false;

	REAL p1[3], p2[3], p3[3];
	cudamesh_copy(p1, pa);
	cudamesh_copy(p2, pb);
	cudamesh_copy(p3, pc);
	REAL wpk1 = aw;
	REAL wpk2 = bw;
	REAL wpk3 = cw;

	// Get number of weighted points, and ensure that they will be accessible
	// using k1...ki, if i is the number of weighted points.
	int wp_nb = 0;
	if (wpk1 > 0.0)
	{
		++wp_nb;
	}

	if (wpk2 > 0.0)
	{
		if (wp_nb == 0)
		{
			cudamesh_swap(p1, p2);
			cudamesh_swap(wpk1, wpk2);
		}
		++wp_nb;
	}

	if (wpk3 > 0.0)
	{
		if (wp_nb == 0)
		{
			cudamesh_swap(p1, p3);
			cudamesh_swap(wpk1, wpk3);
		}
		if (wp_nb == 1)
		{
			cudamesh_swap(p2, p3);
			cudamesh_swap(wpk2, wpk3);
		}
		++wp_nb;
	}

	bool do_spheres_intersect = false;
	REAL ratio, approx_ratio, angle_ratio, size_ratio;
	ratio = 0.0;
	approx_ratio = 0.1*0.1*4;
	angle_ratio = 0.5*0.5*4;
	size_ratio = 0.4*0.4*4;

	REAL sr1, sr2;
	bool ret;

	// Check ratio
	switch (wp_nb)
	{
	case 1:
	{
		if (!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, wpk1, wpk2, sr1) ||
			!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p3, wpk1, wpk3, sr2))
			return false;
		REAL r = cudamesh_max(sr1, sr2);
		ratio = r / wpk1;
		break;
	}
	case 2:
	{
		ret = cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p3, wpk1, wpk3, sr1);
		if (!ret)
			return false;
		ret = cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p2, p3, wpk2, wpk3, sr2);
		if (!ret)
			return false;
		REAL r13 = sr1 / wpk1;
		REAL r23 = sr2 / wpk2;
		ratio = cudamesh_max(r13, r23);

		ret = cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, wpk1, wpk2, sr1);
		if (!ret)
			return false;

		do_spheres_intersect = sr1 <= 0.0;
		break;
	}
	case 3:
	{
		ret = cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, p3, wpk1, wpk2, wpk3, sr1);
		if (!ret)
			return false;

		do_spheres_intersect = sr1 <= 0.0;
		break;
	}

	default: break;
	}

	bool bad = false;
	if (facet_angle != 0.0)
	{
		if (ratio < angle_ratio && (do_spheres_intersect || wp_nb == 1))
		{ }
		else
			bad = cudamesh_is_bad_aspect_ratio(facet_angle, pa, pb, pc);
	}

	if (facet_size != 0.0 && !bad)
	{
		if(ratio < size_ratio && (do_spheres_intersect || wp_nb == 1))
		{ }
		else
			bad = cudamesh_is_bad_radius_size(facet_size, cent, pa);
	}

	if (facet_distance != 0.0 && !bad)
	{
		if (ratio < approx_ratio && (do_spheres_intersect || wp_nb == 1))
		{ }
		else
			bad = cudamesh_is_bad_distance(facet_distance, cent, pa, pb, pc, aw, bw, cw);
	}

	// topology check
	if (do_spheres_intersect && wp_nb == 3)
	{ }
	else if(!bad)
		bad = !cudamesh_is_facet_on_surface(at, bt, ct);

	return bad;
}

__device__ bool cudamesh_is_bad_tet(REAL *pa, REAL *pb, REAL* pc, REAL* pd,
	REAL aw, REAL bw, REAL cw, REAL dw,
	REAL cell_radius_edge_ratio, REAL cell_size)
{
	if (cudamesh_tetrahedronvolume(pa, pb, pc, pd) == 0.0) // ignore a degenerate tet
		return false;

	REAL p1[3], p2[3], p3[3], p4[3];
	cudamesh_copy(p1, pa);
	cudamesh_copy(p2, pb);
	cudamesh_copy(p3, pc);
	cudamesh_copy(p4, pd);
	REAL wpk1 = aw;
	REAL wpk2 = bw;
	REAL wpk3 = cw;
	REAL wpk4 = dw;

	// Get number of weighted points, and ensure that they will be accessible
	// using k1...ki, if i is the number of weighted points.
	int wp_nb = 0;
	if (wpk1 > 0.0)
	{
		++wp_nb;
	}

	if (wpk2 > 0.0)
	{
		if (wp_nb == 0)
		{
			cudamesh_swap(p1, p2);
			cudamesh_swap(wpk1, wpk2);
		}
		++wp_nb;
	}

	if (wpk3 > 0.0)
	{
		if (wp_nb == 0)
		{
			cudamesh_swap(p1, p3);
			cudamesh_swap(wpk1, wpk3);
		}
		if (wp_nb == 1)
		{
			cudamesh_swap(p2, p3);
			cudamesh_swap(wpk2, wpk3);
		}
		++wp_nb;
	}

	if (wpk4 > 0.0)
	{
		if (wp_nb == 0)
		{
			cudamesh_swap(p1, p4);
			cudamesh_swap(wpk1, wpk4);
		}
		if (wp_nb == 1)
		{
			cudamesh_swap(p2, p4);
			cudamesh_swap(wpk2, wpk4);
		}
		if (wp_nb == 2)
		{
			cudamesh_swap(p3, p4);
			cudamesh_swap(wpk3, wpk4);
		}
		++wp_nb;
	}

	bool do_spheres_intersect = false;
	REAL ratio, size_ratio;
	ratio = 0.0;
	size_ratio = 0.5*0.5 * 4;

	REAL sr1, sr2, sr3;

	// Check ratio
	switch (wp_nb)
	{
	case 1:
	{
		if (!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, wpk1, wpk2, sr1) ||
			!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p3, wpk1, wpk3, sr2) ||
			!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p4, wpk1, wpk4, sr3))
			return false;
		REAL r = cudamesh_max(sr1, sr2, sr3);
		ratio = r / wpk1;
		break;
	}
	case 2:
	{
		if(!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p3, wpk1, wpk3, sr1) ||
		   !cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p4, wpk1, wpk4, sr2))
			return false;
		REAL r1 = cudamesh_max(sr1, sr2) / wpk1;
		if (!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p2, p3, wpk2, wpk3, sr1) ||
			!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p2, p4, wpk2, wpk4, sr2))
			return false;
		REAL r2 = cudamesh_max(sr1, sr2) / wpk2;
		ratio = cudamesh_max(r1, r2);

		if(!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, wpk1, wpk2, sr1))
			return false;

		do_spheres_intersect = sr1 <= 0.0;
		break;
	}
	case 3:
	{
		if (!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p4, wpk1, wpk4, sr1) ||
			!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p2, p4, wpk2, wpk4, sr2) ||
			!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p3, p4, wpk3, wpk4, sr3))
			return false;
		REAL r1 = sr1 / wpk1;
		REAL r2 = sr2 / wpk2;
		REAL r3 = sr3 / wpk3;
		ratio = cudamesh_max(r1, r2, r3);

		if (!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, p3, wpk1, wpk2, wpk3, sr1))
			return false;

		do_spheres_intersect = sr1 <= 0.0;
		break;
	}
	case 4:
	{
		if (!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, p3, p4, wpk1, wpk2, wpk3, wpk4, sr1))
			return false;

		do_spheres_intersect = sr1 <= 0.0;
		break;
	}

	default: break;
	}

	bool bad = false;
	REAL sradius, cent[3];
	if (!cudamesh_circumcenter(pa, pb, pc, pd, cent))
		return false;
	sradius = cudamesh_squared_distance(cent, p1);

	if (cell_radius_edge_ratio != 0.0)
	{
		if ((wp_nb >= 2 && do_spheres_intersect) || wp_nb == 1)
		{
		}
		else
			bad = cudamesh_is_bad_radius_edge_ratio(cell_radius_edge_ratio, pa, pb, pc, pd, sradius);
	}

	if (cell_size != 0.0 && !bad)
	{
		if (ratio < size_ratio && (do_spheres_intersect || wp_nb == 1))
		{
		}
		else
			bad = cudamesh_is_bad_radius_size(cell_size, sradius);
	}
	
	return bad;
}

__device__ bool cudamesh_is_encroached_facet_splittable(REAL *pa, REAL *pb, REAL* pc,
	REAL aw, REAL bw, REAL cw)
{
	REAL p1[3], p2[3], p3[3];
	cudamesh_copy(p1, pa);
	cudamesh_copy(p2, pb);
	cudamesh_copy(p3, pc);
	REAL wpk1 = aw;
	REAL wpk2 = bw;
	REAL wpk3 = cw;

	// Get number of weighted points, and ensure that they will be accessible
	// using k1...ki, if i is the number of weighted points.
	int wp_nb = 0;
	if (wpk1 > 0.0)
	{
		++wp_nb;
	}

	if (wpk2 > 0.0)
	{
		if (wp_nb == 0)
		{
			cudamesh_swap(p1, p2);
			cudamesh_swap(wpk1, wpk2);
		}
		++wp_nb;
	}

	if (wpk3 > 0.0)
	{
		if (wp_nb == 0)
		{
			cudamesh_swap(p1, p3);
			cudamesh_swap(wpk1, wpk3);
		}
		if (wp_nb == 1)
		{
			cudamesh_swap(p2, p3);
			cudamesh_swap(wpk2, wpk3);
		}
		++wp_nb;
	}

	REAL min_ratio = 0.16; // (0.2 * 2) ^ 2
	REAL sr1, sr2;
	bool ret;
	
	// Check ratio
	switch (wp_nb)
	{
		case 1:
		{
			if (!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, wpk1, wpk2, sr1) ||
				!cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p3, wpk1, wpk3, sr2))
				return false;
			REAL r = cudamesh_max(sr1, sr2);
			if (r < min_ratio*wpk1)
				return false;
			break;
		}
		case 2:
		{
			ret = cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, wpk1, wpk2, sr1);
			if (!ret)
				return false;
			if (sr1 <= 0.0) // spheres intersect
			{
				ret = cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p3, wpk1, wpk3, sr1);
				if (!ret)
					return false;
				ret = cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p2, p3, wpk2, wpk3, sr2);
				if (!ret)
					return false;
				REAL r13 = sr1 / wpk1;
				REAL r23 = sr2 / wpk1;
				REAL r = cudamesh_max(r13, r23);

				if (r < min_ratio)
					return false;
			}
			break;
		}
		case 3:
		{
			ret = cudamesh_compute_squared_radius_smallest_orthogonal_sphere(p1, p2, p3, wpk1, wpk2, wpk3, sr1);
			if (!ret)
				return false;
			if (sr1 <= 0.0)
				return false;
			break;
		}

		default: break;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Mesh manipulation primitives                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Initialize tables */
void cudamesh_inittables()
{
	// init arrays
	int i, j;

	cudaMemcpyToSymbol(raw_esymtbl, host_esymtbl, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_orgpivot, host_orgpivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_destpivot, host_destpivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_apexpivot, host_apexpivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_oppopivot, host_oppopivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_ver2edge, host_ver2edge, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_edge2ver, host_edge2ver, 6 * sizeof(int));

	cudaMemcpyToSymbol(raw_epivot, host_epivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_snextpivot, host_snextpivot, 6 * sizeof(int));

	cudaMemcpyToSymbol(raw_sorgpivot, host_sorgpivot, 6 * sizeof(int));

	cudaMemcpyToSymbol(raw_sdestpivot, host_sdestpivot, 6 * sizeof(int));

	cudaMemcpyToSymbol(raw_sapexpivot, host_sapexpivot, 6 * sizeof(int));

	// i = t1.ver; j = t2.ver;
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			host_bondtbl[12* i + j] = (j & 3) + (((i & 12) + (j & 12)) % 12);
		}
	}
	cudaMemcpyToSymbol(raw_bondtbl, host_bondtbl, 144 * sizeof(int));

	// i = t1.ver; j = t2.ver
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			host_fsymtbl[12 * i + j] = (j + 12 - (i & 12)) % 12;
		}
	}
	cudaMemcpyToSymbol(raw_fsymtbl, host_fsymtbl, 144 * sizeof(int));

	for (i = 0; i < 12; i++) {
		host_facepivot1[i] = (host_esymtbl[i] & 3);
	}
	cudaMemcpyToSymbol(raw_facepivot1, host_facepivot1, 12 * sizeof(int));

	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			host_facepivot2[12 * i + j] = host_fsymtbl[12 * host_esymtbl[i] + j];
		}
	}
	cudaMemcpyToSymbol(raw_facepivot2, host_facepivot2, 144 * sizeof(int));

	for (i = 0; i < 12; i++) {
		host_enexttbl[i] = (i + 4) % 12;
		host_eprevtbl[i] = (i + 8) % 12;
	}
	cudaMemcpyToSymbol(raw_enexttbl, host_enexttbl, 12 * sizeof(int));
	cudaMemcpyToSymbol(raw_eprevtbl, host_eprevtbl, 12 * sizeof(int));

	for (i = 0; i < 12; i++) {
		host_enextesymtbl[i] = host_esymtbl[host_enexttbl[i]];
		host_eprevesymtbl[i] = host_esymtbl[host_eprevtbl[i]];
	}
	cudaMemcpyToSymbol(raw_enextesymtbl, host_enextesymtbl, 12 * sizeof(int));
	cudaMemcpyToSymbol(raw_eprevesymtbl, host_eprevesymtbl, 12 * sizeof(int));

	for (i = 0; i < 12; i++) {
		host_eorgoppotbl[i] = host_eprevtbl[host_esymtbl[host_enexttbl[i]]];
		host_edestoppotbl[i] = host_enexttbl[host_esymtbl[host_eprevtbl[i]]];
	}
	cudaMemcpyToSymbol(raw_eorgoppotbl, host_eorgoppotbl, 12 * sizeof(int));
	cudaMemcpyToSymbol(raw_edestoppotbl, host_edestoppotbl, 12 * sizeof(int));

	int soffset, toffset;

	// i = t.ver, j = s.shver
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 6; j++) {
			if ((j & 1) == 0) {
				soffset = (6 - ((i & 12) >> 1)) % 6;
				toffset = (12 - ((j & 6) << 1)) % 12;
			}
			else {
				soffset = (i & 12) >> 1;
				toffset = (j & 6) << 1;
			}
			host_tsbondtbl[6 * i + j] = (j & 1) + (((j & 6) + soffset) % 6);
			host_stbondtbl[6 * i + j] = (i & 3) + (((i & 12) + toffset) % 12);
		}
	}
	cudaMemcpyToSymbol(raw_tsbondtbl, host_tsbondtbl, 72 * sizeof(int));
	cudaMemcpyToSymbol(raw_stbondtbl, host_stbondtbl, 72 * sizeof(int));

	// i = t.ver, j = s.shver
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 6; j++) {
			if ((j & 1) == 0) {
				soffset = (i & 12) >> 1;
				toffset = (j & 6) << 1;
			}
			else {
				soffset = (6 - ((i & 12) >> 1)) % 6;
				toffset = (12 - ((j & 6) << 1)) % 12;
			}
			host_tspivottbl[6 * i + j] = (j & 1) + (((j & 6) + soffset) % 6);
			host_stpivottbl[6 * i + j] = (i & 3) + (((i & 12) + toffset) % 12);
		}
	}
	cudaMemcpyToSymbol(raw_tspivottbl, host_tspivottbl, 72 * sizeof(int));
	cudaMemcpyToSymbol(raw_stpivottbl, host_stpivottbl, 72 * sizeof(int));
}


/* Init bounding box*/
void cudamesh_initbbox(
	int numofpoints, double* pointlist,
	int& xmax, int& xmin, int& ymax, int& ymin, int& zmax, int& zmin)
{
	int i;
	double x, y, z;
	for (i = 0; i < numofpoints; i++)
	{
		x = pointlist[3 * i];
		y = pointlist[3 * i + 1];
		z = pointlist[3 * i + 2];
		if (i == 0) 
		{
			xmin = xmax = x;
			ymin = ymax = y;
			zmin = zmax = z;
		}
		else
		{
			xmin = (x < xmin) ? x : xmin;
			xmax = (x > xmax) ? x : xmax;
			ymin = (y < ymin) ? y : ymin;
			ymax = (y > ymax) ? y : ymax;
			zmin = (z < zmin) ? z : zmin;
			zmax = (z > zmax) ? z : zmax;
		}
	}
}

/* Initialize Geometric primitives */
void cudamesh_exactinit(int verbose, int noexact, int nofilter,
	REAL maxx, REAL maxy, REAL maxz)
{
	REAL half;
	REAL check, lastcheck;
	int every_other;

	every_other = 1;
	half = 0.5;
	host_constData[1] /*epsilon*/ = 1.0;
	host_constData[0] /*splitter*/ = 1.0;
	check = 1.0;
	/* Repeatedly divide `epsilon' by two until it is too small to add to    */
	/*   one without causing roundoff.  (Also check if the sum is equal to   */
	/*   the previous sum, for machines that round up instead of using exact */
	/*   rounding.  Not that this library will work on such machines anyway. */
	do {
		lastcheck = check;
		host_constData[1] /*epsilon*/ *= half;
		if (every_other) {
			host_constData[0] /*splitter*/ *= 2.0;
		}
		every_other = !every_other;
		check = 1.0 +  host_constData[1] /*epsilon*/;
	} while ((check != 1.0) && (check != lastcheck));
	host_constData[0] /*splitter*/ += 1.0;

	/* Error bounds for orientation and incircle tests. */
	host_constData[2] /*resulterrbound*/ = (3.0 + 8.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[3] /*ccwerrboundA*/ = (3.0 + 16.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[4] /*ccwerrboundB*/ = (2.0 + 12.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[5] /*ccwerrboundC*/ = (9.0 + 64.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	host_constData[6] /*o3derrboundA*/ = (7.0 + 56.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[7] /*o3derrboundB*/ = (3.0 + 28.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[8] /*o3derrboundC*/ = (26.0 + 288.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	host_constData[9] /*iccerrboundA*/ = (10.0 + 96.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[10] /*iccerrboundB*/ = (4.0 + 48.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[11] /*iccerrboundC*/ = (44.0 + 576.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	host_constData[12] /*isperrboundA*/ = (16.0 + 224.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[13] /*isperrboundB*/ = (5.0 + 72.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[14] /*isperrboundC*/ = (71.0 + 1408.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;

	// Set TetGen options.  Added by H. Si, 2012-08-23.
	host_constOptions[0] /*_use_inexact_arith*/ = noexact;
	host_constOptions[1] /*_use_static_filter*/ = !nofilter;

	// Calculate the two static filters for orient3d() and insphere() tests.
	// Added by H. Si, 2012-08-23.

	// Sort maxx < maxy < maxz. Re-use 'half' for swapping.
	assert(maxx > 0);
	assert(maxy > 0);
	assert(maxz > 0);

	if (maxx > maxz) {
		half = maxx; maxx = maxz; maxz = half;
	}
	if (maxy > maxz) {
		half = maxy; maxy = maxz; maxz = half;
	}
	else if (maxy < maxx) {
		half = maxy; maxy = maxx; maxx = half;
	}

	host_constData[15] /*o3dstaticfilter*/ = 5.1107127829973299e-15 * maxx * maxy * maxz;
	host_constData[16] /*ispstaticfilter*/ = 1.2466136531027298e-13 * maxx * maxy * maxz * (maxz * maxz);

	// Copy to const memory
	cudaMemcpyToSymbol(raw_constData, host_constData, 17 * sizeof(REAL));
	cudaMemcpyToSymbol(raw_constOptions, host_constOptions, 2 * sizeof(int));

	//for (int i = 0; i<17; i++)
	//	printf("host_constData[%d] = %g\n", i, host_constData[i]);
	//for (int i = 0; i < 2; i++)
	//	printf("host_constOptions[%d] = %d\n", i, host_constOptions[i]);
}

/* Init Kernel constants */
void cudamesh_initkernelconstants(REAL maxx, REAL maxy, REAL maxz)
{
	REAL longest = sqrt(maxx*maxx + maxy*maxy + maxz*maxz);
	REAL minedgelength = longest*EPSILON;
	host_kernelconstants[0] = minedgelength;

	cudaMemcpyToSymbol(raw_kernelconstants, host_kernelconstants, sizeof(REAL));
}

/* Primitives for points */

// Convert point index to pointer to pointlist
__device__ double* cudamesh_id2pointlist(int index, double* pointlist)
{
	return (pointlist + 3 * index);
}


/* Primitives for tetrahedron */

// The following primtives get or set the origin, destination, face apex,
//   or face opposite of an ordered tetrahedron.

__device__ int cudamesh_org(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + raw_orgpivot[t.ver]];
}

__device__ int cudamesh_dest(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + raw_destpivot[t.ver]];
}

__device__ int cudamesh_apex(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + raw_apexpivot[t.ver]];
}

__device__ int cudamesh_oppo(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + raw_oppopivot[t.ver]];
}

__device__ void cudamesh_setorg(tethandle t, int p, int* tetlist)
{
	tetlist[4 * t.id + raw_orgpivot[t.ver]] = p;
}

__device__ void cudamesh_setdest(tethandle t, int p, int* tetlist)
{
	tetlist[4 * t.id + raw_destpivot[t.ver]] = p;
}

__device__ void cudamesh_setapex(tethandle t, int p, int* tetlist)
{
	tetlist[4 * t.id + raw_apexpivot[t.ver]] = p;
}

__device__ void cudamesh_setoppo(tethandle t, int p, int* tetlist)
{
	tetlist[4 * t.id + raw_oppopivot[t.ver]] = p;
}

// bond()  connects two tetrahedra together. (t1,v1) and (t2,v2) must 
//   refer to the same face and the same edge.

__device__ void cudamesh_bond(tethandle t1, tethandle t2, tethandle* neighborlist)
{
	neighborlist[4 * t1.id + (t1.ver & 3)] = tethandle(t2.id, raw_bondtbl[12 * t1.ver + t2.ver]);
	neighborlist[4 * t2.id + (t2.ver & 3)] = tethandle(t1.id, raw_bondtbl[12 * t2.ver + t1.ver]);
}

// dissolve()  a bond (from one side).

__device__ void cudamesh_dissolve(tethandle t, tethandle* neighborlist)
{
	neighborlist[4 * t.id + (t.ver & 3)] = tethandle(-1, 11); // empty handle
}

// esym()  finds the reversed edge.  It is in the other face of the
//   same tetrahedron.

__device__ void cudamesh_esym(tethandle& t1, tethandle& t2)
{
	(t2).id = (t1).id;
	(t2).ver = raw_esymtbl[(t1).ver];
}
__device__ void cudamesh_esymself(tethandle& t)
{
	(t).ver = raw_esymtbl[(t).ver];
}

// enext()  finds the next edge (counterclockwise) in the same face.

__device__ void cudamesh_enext(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = raw_enexttbl[t1.ver];
}
__device__ void cudamesh_enextself(tethandle& t)
{
	t.ver = raw_enexttbl[t.ver];
}

// eprev()   finds the next edge (clockwise) in the same face.

__device__ void cudamesh_eprev(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = raw_eprevtbl[t1.ver];
}
__device__ void cudamesh_eprevself(tethandle& t)
{
	t.ver = raw_eprevtbl[t.ver];
}

// enextesym()  finds the reversed edge of the next edge. It is in the other
//   face of the same tetrahedron. It is the combination esym() * enext(). 

__device__ void cudamesh_enextesym(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = raw_enextesymtbl[t1.ver];
}

__device__ void cudamesh_enextesymself(tethandle& t) {
	t.ver = raw_enextesymtbl[t.ver];
}

// eprevesym()  finds the reversed edge of the previous edge.

__device__ void cudamesh_eprevesym(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = raw_eprevesymtbl[t1.ver];
}

__device__ void cudamesh_eprevesymself(tethandle& t) {
	t.ver = raw_eprevesymtbl[t.ver];
}

// eorgoppo()    Finds the opposite face of the origin of the current edge.
//               Return the opposite edge of the current edge.

__device__ void cudamesh_eorgoppo(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = raw_eorgoppotbl[t1.ver];
}

__device__ void cudamesh_eorgoppoself(tethandle& t) {
	t.ver = raw_eorgoppotbl[t.ver];
}

// edestoppo()    Finds the opposite face of the destination of the current 
//                edge. Return the opposite edge of the current edge.

__device__ void cudamesh_edestoppo(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = raw_edestoppotbl[t1.ver];
}

__device__ void cudamesh_edestoppoself(tethandle& t) {
	t.ver = raw_edestoppotbl[t.ver];
}

// fsym()  finds the adjacent tetrahedron at the same face and the same edge.

__device__ void cudamesh_fsym(tethandle& t1, tethandle& t2, tethandle* neighborlist)
{
	t2 = neighborlist[4 * t1.id + (t1.ver & 3)];
	t2.ver = raw_fsymtbl[12 * t1.ver + t2.ver];
}

__device__ void cudamesh_fsymself(tethandle& t, tethandle* neighborlist)
{
	char t1ver = t.ver;
	t = neighborlist[4 * t.id + (t.ver & 3)];
	t.ver = raw_fsymtbl[12 * t1ver + t.ver];
}

// fnext()  finds the next face while rotating about an edge according to
//   a right-hand rule. The face is in the adjacent tetrahedron.  It is
//   the combination: fsym() * esym().

__device__ void cudamesh_fnext(tethandle& t1, tethandle& t2, tethandle* neighborlist)
{
	t2 = neighborlist[4 * t1.id + raw_facepivot1[t1.ver]];
	t2.ver = raw_facepivot2[12 * t1.ver + t2.ver];
}

__device__ void cudamesh_fnextself(tethandle& t, tethandle* neighborlist)
{
	char t1ver = t.ver;
	t = neighborlist[4 * t.id + raw_facepivot1[t.ver]];
	t.ver = raw_facepivot2[12 * t1ver + t.ver];
}

// ishulltet()  tests if t is a hull tetrahedron.

__device__ bool cudamesh_ishulltet(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + 3] == -1;
}

// isdeadtet()  tests if t is a tetrahedron is dead.

__device__ bool cudamesh_isdeadtet(tethandle t)
{
	return (t.id == -1);
}

/* Primitives for subfaces and subsegments. */

// spivot() finds the adjacent subface (s2) for a given subface (s1).
//   s1 and s2 share at the same edge.

__device__ void cudamesh_spivot(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	s2 = tri2trilist[3 * s1.id + (s1.shver >> 1)];
}

__device__ void cudamesh_spivotself(trihandle& s, trihandle* tri2trilist)
{
	s = tri2trilist[3 * s.id + (s.shver >> 1)];
}

// sbond() bonds two subfaces (s1) and (s2) together. s1 and s2 must refer
//   to the same edge. No requirement is needed on their orientations.

__device__ void cudamesh_sbond(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	tri2trilist[3 * s1.id + (s1.shver >> 1)] = s2;
	tri2trilist[3 * s2.id + (s2.shver >> 1)] = s1;
}

// sbond1() bonds s1 <== s2, i.e., after bonding, s1 is pointing to s2,
//   but s2 is not pointing to s1.  s1 and s2 must refer to the same edge.
//   No requirement is needed on their orientations.

__device__ void cudamesh_sbond1(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	tri2trilist[3 * s1.id + (s1.shver >> 1)] = s2;
}

// Dissolve a subface bond (from one side).  Note that the other subface
//   will still think it's connected to this subface.
__device__ void cudamesh_sdissolve(trihandle& s, trihandle* tri2trilist)
{
	tri2trilist[3 * s.id + (s.shver >> 1)] = trihandle(-1, 0);
}

// These primitives determine or set the origin, destination, or apex
//   of a subface with respect to the edge version.

__device__ int cudamesh_sorg(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + raw_sorgpivot[s.shver]];
}

__device__ int cudamesh_sdest(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + raw_sdestpivot[s.shver]];
}

__device__ int cudamesh_sapex(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + raw_sapexpivot[s.shver]];
}

__device__ void cudamesh_setsorg(trihandle& s, int p, int* trilist)
{
	trilist[3 * s.id + raw_sorgpivot[s.shver]] = p;
}

__device__ void cudamesh_setsdest(trihandle& s, int p, int* trilist)
{
	trilist[3 * s.id + raw_sdestpivot[s.shver]] = p;
}

__device__ void cudamesh_setsapex(trihandle& s, int p, int* trilist)
{
	trilist[3 * s.id + raw_sapexpivot[s.shver]] = p;
}

// sesym()  reserves the direction of the lead edge.

__device__ void cudamesh_sesym(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = (s1.shver ^ 1);  // Inverse the last bit.
}

__device__ void cudamesh_sesymself(trihandle& s)
{
	s.shver ^= 1;
}

// senext()  finds the next edge (counterclockwise) in the same orientation
//   of this face.

__device__ void cudamesh_senext(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = raw_snextpivot[s1.shver];
}

__device__ void cudamesh_senextself(trihandle& s)
{
	s.shver = raw_snextpivot[s.shver];
}

__device__ void cudamesh_senext2(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = raw_snextpivot[raw_snextpivot[s1.shver]];
}

__device__ void cudamesh_senext2self(trihandle& s)
{
	s.shver = raw_snextpivot[raw_snextpivot[s.shver]];
}


/* Primitives for interacting tetrahedra and subfaces. */

// tsbond() bond a tetrahedron (t) and a subface (s) together.
// Note that t and s must be the same face and the same edge. Moreover,
//   t and s have the same orientation. 
// Since the edge number in t and in s can be any number in {0,1,2}. We bond
//   the edge in s which corresponds to t's 0th edge, and vice versa.

__device__ void cudamesh_tsbond(tethandle& t, trihandle& s, trihandle* tet2trilist, tethandle* tri2tetlist)
{
	// Bond t <== s.
	tet2trilist[4 * t.id + (t.ver & 3)] = trihandle(s.id, raw_tsbondtbl[6 * t.ver + s.shver]);
	// Bond s <== t.
	tri2tetlist[2 * s.id + (s.shver & 1)] = tethandle(t.id, raw_stbondtbl[6 * t.ver + s.shver]);
}

// tspivot() finds a subface (s) abutting on the given tetrahdera (t).
//   Return s.id = -1 if there is no subface at t. Otherwise, return
//   the subface s, and s and t must be at the same edge wth the same
//   orientation.
__device__ void cudamesh_tspivot(tethandle& t, trihandle& s, trihandle* tet2trilist)
{
	// Get the attached subface s.
	s = tet2trilist[4 * t.id + (t.ver & 3)];
	if (s.id == -1)
		return;
	(s).shver = raw_tspivottbl[6 * t.ver + s.shver];
}

// stpivot() finds a tetrahedron (t) abutting a given subface (s).
//   Return the t (if it exists) with the same edge and the same
//   orientation of s.
__device__ void cudamesh_stpivot(trihandle& s, tethandle& t, tethandle* tri2tetlist)
{
	t = tri2tetlist[2 * s.id + (s.shver & 1)];
	if (t.id == -1) {
		return;
	}
	(t).ver = raw_stpivottbl[6 * t.ver + s.shver];
}

/* Primitives for interacting between tetrahedra and segments */

__device__ void cudamesh_tsspivot1(tethandle& t, trihandle& seg, trihandle* tet2seglist)
{
	seg = tet2seglist[6 * t.id + raw_ver2edge[t.ver]];
}

__device__ void cudamesh_tssbond1(tethandle& t, trihandle& seg, trihandle* tet2seglist)
{
	tet2seglist[6 * t.id + raw_ver2edge[t.ver]] = seg;
}

__device__ void cudamesh_sstbond1(trihandle& s, tethandle& t, tethandle* seg2tetlist)
{
	seg2tetlist[s.id + 0] = t;
}

__device__ void cudamesh_sstpivot1(trihandle& s, tethandle& t, tethandle* seg2tetlist)
{
	t = seg2tetlist[s.id];
}

/* Primitives for interacting between subfaces and segments */

__device__ void cudamesh_ssbond(trihandle& s, trihandle& edge, trihandle* tri2seglist, trihandle* seg2trilist)
{
	tri2seglist[3 * s.id + (s.shver >> 1)] = edge;
	seg2trilist[3 * edge.id + 0] = s;
}

__device__ void cudamesh_ssbond1(trihandle& s, trihandle& edge, trihandle* tri2seglist)
{
	tri2seglist[3 * s.id + (s.shver >> 1)] = edge;
}

__device__ void cudamesh_sspivot(trihandle& s, trihandle& edge, trihandle* tri2seglist)
{
	edge = tri2seglist[3 * s.id + (s.shver >> 1)];
}

__device__ bool cudamesh_isshsubseg(trihandle&s, trihandle* tri2seglist)
{
	return (tri2seglist[3 * s.id + (s.shver >> 1)].id != -1);
}



/* Advanced primitives. */

__device__ void cudamesh_point2tetorg(int pa, tethandle& searchtet, tethandle* point2tetlist, int* tetlist)
{
	searchtet = point2tetlist[pa];
	if (tetlist[4 * searchtet.id + 0] == pa) {
		searchtet.ver = 11;
	}
	else if (tetlist[4 * searchtet.id + 1] == pa) {
		searchtet.ver = 3;
	}
	else if (tetlist[4 * searchtet.id + 2] == pa) {
		searchtet.ver = 7;
	}
	else {
		assert(tetlist[4 * searchtet.id + 3] == pa); // SELF_CHECK
		searchtet.ver = 0;
	}
}

/* Status initialization */
__global__ void kernelMarkInitialTrifaces(
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* trifacecount,
	REAL* trifaceipt,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int numoftets
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numoftets)
		return;

	tethandle checktet(pos, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return;

	int i, ret;
	int pi[4];
	REAL *pc[4], w[4], wc[3];
	for (i = 0; i < 4; i++)
	{
		pi[i] = d_tetlist[4 * checktet.id + i];
		pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
		w[i] = d_weightlist[pi[i]];
	}

	if(!cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], pc[3],
		w[0], w[1], w[2], w[3], wc)) // is degenerate tet
		return;

	// Calculate the length of the largest bounding box diagonal
	REAL dlen = cudamesh_bbox_diglen(0, d_aabbnodebbs);

	// Check 4 neighbors
	tethandle neightet;
	bool larger, nhtet, ndeg, found;
	int npi[4], pmidx;
	REAL *npc[4], nw[4], cwc[3], nwc[3], ipt[3], dir[3], vdir[3], fwc[3], vec1[3], vec2[3], len;
	for (checktet.ver = 0; checktet.ver < 4; checktet.ver++)
	{
		cudamesh_fsym(checktet, neightet, d_neighborlist);
		larger = (checktet.id > neightet.id);
		nhtet = cudamesh_ishulltet(neightet, d_tetlist);

		if (larger && !nhtet) // let neighbor handle it
		{
			continue;
		}

		// calculate the voronoi dual of the facet
		if (!nhtet) // the dual is a segment
		{
			for (i = 0; i < 4; i++)
			{
				npi[i] = d_tetlist[4 * neightet.id + i];
				npc[i] = cudamesh_id2pointlist(npi[i], d_pointlist);
				nw[i] = d_weightlist[npi[i]];
			}
			if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], npc[3],
				nw[0], nw[1], nw[2], nw[3], nwc))
				continue;
		}
		else // the dual is a ray
		{
			// get boundary face points
			npi[0] = cudamesh_org(checktet, d_tetlist);
			npc[0] = cudamesh_id2pointlist(npi[0], d_pointlist);
			nw[0] = d_weightlist[npi[0]];
			npi[1] = cudamesh_dest(checktet, d_tetlist);
			npc[1] = cudamesh_id2pointlist(npi[1], d_pointlist);
			nw[1] = d_weightlist[npi[1]];
			npi[2] = cudamesh_apex(checktet, d_tetlist);
			npc[2] = cudamesh_id2pointlist(npi[2], d_pointlist);
			nw[2] = d_weightlist[npi[2]];
			// get oppo point
			npi[3] = cudamesh_oppo(checktet, d_tetlist);
			npc[3] = cudamesh_id2pointlist(npi[3], d_pointlist);
			// caculate the weighted center perpendicular vector of boundary face
			if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], nw[0], nw[1], nw[2], fwc))
				continue; // degenerate boundary face
			cudamesh_raydir(npc[0], npc[1], npc[2], dir);
			vdir[0] = npc[3][0] - fwc[0]; vdir[1] = npc[3][1] - fwc[1]; vdir[2] = npc[3][2] - fwc[2];
			if (dir[0] * vdir[0] + dir[1] * vdir[1] + dir[2] * vdir[2] >= 0.0)
				continue; // degenerate ray
			// calculate a point outside the bounding box
			len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
			if (len == 0.0)
				continue;
			dir[0] /= len; dir[1] /= len; dir[2] /= len;
			nwc[0] = wc[0] + dir[0] * 1.2*dlen;
			nwc[1] = wc[1] + dir[1] * 1.2*dlen;
			nwc[2] = wc[2] + dir[2] * 1.2*dlen;
		}
		cudamesh_copy(cwc, wc); // to avoid wc being changed
		ret = cudamesh_compare(cwc, nwc);
		if (ret == 0) // degenerate segment
			continue;
		else if (ret == 1) // make canonical vector
			cudamesh_swap(cwc, nwc);

		// Try to find any intersections with input polygons
		found = cudamesh_traversal_first_intersection(cwc, nwc, d_aabbnodeleft, d_aabbnoderight, d_aabbnodebbs,
					d_aabbpmcoord, d_aabbpmbbs, ipt, pmidx);

		if (!found)
			continue;

		// Mark and record this facet
		trifacecount[4 * checktet.id + checktet.ver] = 1;
		trifaceipt[3 * (4 * checktet.id + checktet.ver) + 0] = ipt[0];
		trifaceipt[3 * (4 * checktet.id + checktet.ver) + 1] = ipt[1];
		trifaceipt[3 * (4 * checktet.id + checktet.ver) + 2] = ipt[2];
	}
}

__global__ void kernelAppendInitialTrifaces(
	int* trifacecount,
	int* trifaceindices,
	REAL* trifaceipt,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	REAL cr_facet_angle,
	REAL cr_facet_size,
	REAL cr_facet_distance,
	int numoftets
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numoftets)
		return;

	tethandle checktet(pos, 11);
	tethandle neightet;
	int pi[3], index;
	REAL *pc[3], pw[3];
	verttype pt[3];
	trihandle checksh;
	REAL* ipt;
	for (checktet.ver = 0; checktet.ver < 4; checktet.ver++)
	{
		if (trifacecount[4 * checktet.id + checktet.ver] == 1) //  a facet
		{
			index = trifaceindices[4 * checktet.id + checktet.ver];
			// triface endpoints
			pi[0] = cudamesh_org(checktet, d_tetlist);
			pi[1] = cudamesh_dest(checktet, d_tetlist);
			pi[2] = cudamesh_apex(checktet, d_tetlist);
			checksh.id = index; checksh.shver = 0;
			cudamesh_setsorg(checksh, pi[0], d_trifacelist);
			cudamesh_setsdest(checksh, pi[1], d_trifacelist);
			cudamesh_setsapex(checksh, pi[2], d_trifacelist);
			// bond triface and tetrahedron together
			cudamesh_tsbond(checktet, checksh, d_tet2trilist, d_tri2tetlist);
			cudamesh_fsym(checktet, neightet, d_neighborlist);
			cudamesh_sesymself(checksh);
			cudamesh_tsbond(neightet, checksh, d_tet2trilist, d_tri2tetlist);
			// triface corresponding intersection point
			ipt = trifaceipt + 3 * (4 * checktet.id + checktet.ver);
			d_trifacecent[3 * index + 0] = ipt[0];
			d_trifacecent[3 * index + 1] = ipt[1];
			d_trifacecent[3 * index + 2] = ipt[2];
			// triface criteria
			pc[0] = cudamesh_id2pointlist(pi[0], d_pointlist);
			pc[1] = cudamesh_id2pointlist(pi[1], d_pointlist);
			pc[2] = cudamesh_id2pointlist(pi[2], d_pointlist);
			pw[0] = d_weightlist[pi[0]]; pt[0] = d_pointtypelist[pi[0]];
			pw[1] = d_weightlist[pi[1]]; pt[1] = d_pointtypelist[pi[1]];
			pw[2] = d_weightlist[pi[2]]; pt[2] = d_pointtypelist[pi[2]];
			if (cudamesh_is_bad_facet(pc[0], pc[1], pc[2], pw[0], pw[1], pw[2],
				pt[0], pt[1], pt[2], ipt,
				cr_facet_angle, cr_facet_size, cr_facet_distance))
				d_tristatus[index].setBad(true);
		}
	}
}

__global__ void kernelInitTetStatus(
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	REAL aabb_diglen,
	int numoftets
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numoftets)
		return;

	if (d_tetstatus[pos].isEmpty())
		return;

	tethandle checktet(pos, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return; // a hull tet, ignore it

	int i;
	int pi[4];
	REAL *pc[4], w[4], wc[3];
	for (i = 0; i < 4; i++)
	{
		pi[i] = d_tetlist[4 * checktet.id + i];
		pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
		w[i] = d_weightlist[pi[i]];
	}
	
	if (!cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], pc[3], w[0], w[1], w[2], w[3], wc))
		return; // a degenerate tet, ignore it

	// Fast check
	REAL xmin, xmax, ymin, ymax, zmin, zmax;
	xmin = d_aabbnodebbs[0]; xmax = d_aabbnodebbs[1];
	ymin = d_aabbnodebbs[2]; ymax = d_aabbnodebbs[3];
	zmin = d_aabbnodebbs[4]; zmax = d_aabbnodebbs[5];
	if (cudamesh_is_out_bbox(wc, xmin, xmax, ymin, ymax, zmin, zmax))
		return; // outside bounding box, ignore it
	
	REAL t[3];
	cudamesh_box_far_point(wc, t, xmin, xmax, ymin, ymax, zmin, zmax, aabb_diglen);
	// Travsersal
	if (cudamesh_traversal_in_domain(wc, t, d_aabbnodeleft, d_aabbnoderight, d_aabbnodebbs, d_aabbpmcoord, d_aabbpmbbs))
	{
		d_tetstatus[pos].setInDomain(true);
		if (cudamesh_is_bad_tet(pc[0], pc[1], pc[2], pc[3], w[0], w[1], w[2], w[3],
			cr_cell_radius_edge_ratio, cr_cell_size))
			d_tetstatus[pos].setBad(true);
	}
}

__global__ void kernelInitTriQuality(
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	REAL cr_facet_angle,
	REAL cr_facet_size,
	REAL cr_facet_distance,
	int numoftrifaces
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numoftrifaces)
		return;

	int pi[3];
	REAL *pc[3], pw[3], *center;
	verttype pt[3];
	trihandle checksh(pos, 0);

	pi[0] = cudamesh_sorg(checksh, d_trifacelist);
	pi[1] = cudamesh_sdest(checksh, d_trifacelist);
	pi[2] = cudamesh_sapex(checksh, d_trifacelist);
	pc[0] = cudamesh_id2pointlist(pi[0], d_pointlist);
	pc[1] = cudamesh_id2pointlist(pi[1], d_pointlist);
	pc[2] = cudamesh_id2pointlist(pi[2], d_pointlist);
	pw[0] = d_weightlist[pi[0]]; pt[0] = d_pointtypelist[pi[0]];
	pw[1] = d_weightlist[pi[1]]; pt[1] = d_pointtypelist[pi[1]];
	pw[2] = d_weightlist[pi[2]]; pt[2] = d_pointtypelist[pi[2]];
	center = d_trifacecent + 3 * pos;
	if (cudamesh_is_bad_facet(pc[0], pc[1], pc[2], pw[0], pw[1], pw[2],
		pt[0], pt[1], pt[2], center,
		cr_facet_angle, cr_facet_size, cr_facet_distance))
		d_tristatus[pos].setBad(true);
}

__global__ void kernelInitTetQuality(
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	int numoftets
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numoftets)
		return;

	tethandle checktet(pos, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return; // a hull tet, ignore it

	int i;
	int pi[4];
	REAL *pc[4], w[4], wc[3];
	for (i = 0; i < 4; i++)
	{
		pi[i] = d_tetlist[4 * checktet.id + i];
		pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
		w[i] = d_weightlist[pi[i]];
	}

	if (!cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], pc[3], w[0], w[1], w[2], w[3], wc))
	{
		d_tetstatus[pos] = tetstatus(1); // reset
		return; // a degenerate tet, ignore it
	}

	if (d_tetstatus[pos].isInDomain())
	{
		if (cudamesh_is_bad_tet(pc[0], pc[1], pc[2], pc[3], w[0], w[1], w[2], w[3],
			cr_cell_radius_edge_ratio, cr_cell_size))
			d_tetstatus[pos].setBad(true);
	}
}

__global__ void kernelInitDomainHandle_Tet(
	tethandle* d_domainhandle,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int offset,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int tetidx = pos + offset;
	if (d_tetstatus[tetidx].isEmpty())
		return;

	tethandle checktet(tetidx, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return;

	d_domainhandle[pos] = tethandle(tetidx, 11);
}

__global__ void kernelInitDomainSegment_Tet(
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	REAL* d_domainsegment,
	int* d_domainthreadlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	REAL aabb_diglen,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	tethandle checkhandle = d_domainhandle[pos];

	int i, j;
	int pi[4];
	REAL *pc[4], w[4], wc[3];
	for (i = 0; i < 4; i++)
	{
		pi[i] = d_tetlist[4 * checkhandle.id + i];
		pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
		w[i] = d_weightlist[pi[i]];
	}

	if (!cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], pc[3],
		w[0], w[1], w[2], w[3], wc)) // is degenerate tet
	{
		d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
		return;
	}

	REAL xmin, xmax, ymin, ymax, zmin, zmax;
	xmin = d_aabbnodebbs[0]; xmax = d_aabbnodebbs[1];
	ymin = d_aabbnodebbs[2]; ymax = d_aabbnodebbs[3];
	zmin = d_aabbnodebbs[4]; zmax = d_aabbnodebbs[5];
	if (cudamesh_is_out_bbox(wc, xmin, xmax, ymin, ymax, zmin, zmax))
	{
		d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
		return;
	}
	else
	{
		//REAL t[3], v[3];
		//unsigned long randomseed = 0;
		//// generate random ray direction
		//cudamesh_random_sphere_point(&randomseed, v);
		//t[0] = wc[0] + v[0] * 1.5*aabb_diglen;
		//t[1] = wc[1] + v[1] * 1.5*aabb_diglen;
		//t[2] = wc[2] + v[2] * 1.5*aabb_diglen;

		REAL t[3];
		cudamesh_box_far_point(wc, t, xmin, xmax, ymin, ymax, zmin, zmax, aabb_diglen);

		d_domainsegment[6 * pos + 0] = wc[0];
		d_domainsegment[6 * pos + 1] = wc[1];
		d_domainsegment[6 * pos + 2] = wc[2];
		d_domainsegment[6 * pos + 3] = t[0];
		d_domainsegment[6 * pos + 4] = t[1];
		d_domainsegment[6 * pos + 5] = t[2];
	}

	// swap new tet thread index and domain thread (segment) index
	d_domainthreadlist[pos] = checkhandle.id;
	checkhandle.id = pos;
	d_domainhandle[pos] = checkhandle;
}

__global__ void kernelDomainSegmentAndPrimitiveCheck_Tet(
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	int* d_domainnode,
	REAL* d_domainsegment,
	int* d_domaincount,
	int* d_domainthreadlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int pmidx = d_domainnode[pos];
	if (pmidx >= 0) // this should not happend
		return;

	REAL s[3], t[3];
	REAL bxmin, bxmax, bymin, bymax, bzmin, bzmax;
	tethandle checkhandle = d_domainhandle[pos];
	s[0] = d_domainsegment[6 * checkhandle.id + 0];
	s[1] = d_domainsegment[6 * checkhandle.id + 1];
	s[2] = d_domainsegment[6 * checkhandle.id + 2];
	t[0] = d_domainsegment[6 * checkhandle.id + 3];
	t[1] = d_domainsegment[6 * checkhandle.id + 4];
	t[2] = d_domainsegment[6 * checkhandle.id + 5];
	pmidx = -pmidx - 1; // shift back to true primitive index
	bxmin = d_aabbpmbbs[6 * pmidx + 0]; bxmax = d_aabbpmbbs[6 * pmidx + 1];
	bymin = d_aabbpmbbs[6 * pmidx + 2]; bymax = d_aabbpmbbs[6 * pmidx + 3];
	bzmin = d_aabbpmbbs[6 * pmidx + 4]; bzmax = d_aabbpmbbs[6 * pmidx + 5];
	if (cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
	{
		int type, tetidx, counter = 0;
		REAL* p[3];
		p[0] = d_aabbpmcoord + 9 * pmidx + 0;
		p[1] = d_aabbpmcoord + 9 * pmidx + 3;
		p[2] = d_aabbpmcoord + 9 * pmidx + 6;
		tetidx = d_domainthreadlist[checkhandle.id];
		if (cudamesh_ts_intersection(p[0], p[1], p[2], s, t, type))
		{
			if (type == (int)UNKNOWNINTER || type == (int)ACROSSEDGE
				|| type == (int)ACROSSVERT || type == (int)COPLANAR)
			{
				counter = -100000000;
			}
			else if (type == (int)TOUCHEDGE || type == (int)TOUCHFACE
				|| type == (int)SHAREVERT)
			{
				counter = 100000000;
			}
			else if (type == (int)ACROSSFACE)
			{
				counter = 1;
			}
			atomicAdd(d_domaincount + tetidx, counter);
		}
	}
}

__global__ void kernelSetTetStatus(
	int* d_domaincount,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	int offset,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int tetidx = pos + offset;
	if (d_tetstatus[tetidx].isEmpty()) // skip unused empty slot
		return;

	tethandle checktet(tetidx, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return;

	int counter = d_domaincount[tetidx];
	bool indomain = false;
	if (counter >= 100000000)
		indomain = true;
	else if (counter > 0)
		indomain = (counter & 1) == 1 ? true : false;

	d_tetstatus[tetidx].setInDomain(indomain);
	if (indomain)
	{
		int pi[4];
		REAL *pc[4], w[4];
		for (int i = 0; i < 4; i++)
		{
			pi[i] = d_tetlist[4 * checktet.id + i];
			pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
			w[i] = d_weightlist[pi[i]];
		}
		if (cudamesh_is_bad_tet(pc[0], pc[1], pc[2], pc[3], w[0], w[1], w[2], w[3],
			cr_cell_radius_edge_ratio, cr_cell_size))
		{
			d_tetstatus[tetidx].setBad(true);
		}
	}
}

// Check mesh
__global__ void kernelCheckPointNeighbors(
	trihandle* d_point2trilist,
	tethandle* d_point2tetlist,
	verttype* d_pointtypelist,
	int* d_seglist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int i, p;
	bool flag = false;
	trihandle neighseg, neighsh;
	tethandle neightet;

	verttype pointtype = d_pointtypelist[pos];

	if (pointtype == FREESEGVERTEX)
	{
		neighseg = d_point2trilist[pos];
		if (neighseg.id != -1)
		{
			if (d_segstatus[neighseg.id].isEmpty())
			{
				printf("Point #%d: Empty subseg neighbor #%d\n", pos, neighseg.id);
			}
			else
			{
				for (i = 0; i < 3; i++)
				{
					p = d_seglist[3 * neighseg.id + i];
					if (i == 2 && p != -1)
					{
						printf("Point #%d: Wrong point type (on subseg) or neighbor type (subseg) #%d - %d, %d, %d\n", pos,
							neighseg.id, d_seglist[3 * neighseg.id + 0], d_seglist[3 * neighseg.id + 1], d_seglist[3 * neighseg.id + 2]);
					}
					if (p == pos)
					{
						flag = true;
						break;
					}
				}
				if (!flag)
					printf("Point #%d: Wrong subface neighbor #%d - %d, %d, %d\n", pos,
						neighseg.id, d_seglist[3 * neighseg.id + 0], d_seglist[3 * neighseg.id + 1], d_seglist[3 * neighseg.id + 2]);
			}
		}
		else
		{
			printf("Point #%d: Missing segment neighbor\n");
		}
	}
	else if (pointtype == FREEFACETVERTEX)
	{
		neighsh = d_point2trilist[pos];
		if (neighsh.id != -1)
		{
			if (d_tristatus[neighsh.id].isEmpty())
			{
				printf("Point #%d: Empty subface neighbor #%d\n", pos, neighsh.id);
			}
			else
			{
				for (i = 0; i < 3; i++)
				{
					p = d_trifacelist[3 * neighsh.id + i];
					if (p == -1)
					{
						printf("Point #%d: Wrong point type (on subface) or neighbor type (subface) #%d - %d, %d, %d\n",pos,
							neighsh.id, d_trifacelist[3 * neighsh.id + 0], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
					}
					if (p == pos)
					{
						flag = true;
						break;
					}
				}
				if (!flag)
					printf("Point #%d: Wrong subface neighbor #%d - %d, %d, %d\n", pos,
						neighsh.id, d_trifacelist[3 * neighsh.id + 0], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
			}
		}
		else
		{
			printf("Point #%d: Missing subface neighbor\n");
		}
	}

	neightet = d_point2tetlist[pos];
	if (neightet.id != -1)
	{
		//printf("%d ", neightet.id);
		if (d_tetstatus[neightet.id].isEmpty())
		{
			printf("Point #%d: Empty tet neighbor #%d\n", pos, neightet.id);
		}
		else
		{
			for (i = 0; i < 4; i++)
			{
				p = d_tetlist[4 * neightet.id + i];
				if (p == pos)
				{
					flag = true;
					break;
				}
			}
			if (!flag)
				printf("Point #%d: Wrong tet neighbor #%d - %d, %d, %d, %d\n", pos,
					neightet.id,
					d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3]);
		}
	}
}

__global__ void kernelCheckSubsegNeighbors(
	int* d_seglist,
	trihandle* d_seg2trilist,
	tethandle* d_seg2tetlist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_segstatus[pos].isEmpty())
		return;

	trihandle checkseg(pos, 0), neighsh, neighseg, prevseg, nextseg;
	int pa, pb, pc, pd;

	cudamesh_spivot(checkseg, neighsh, d_seg2trilist);
	if (neighsh.id != -1)
	{
		if (d_tristatus[neighsh.id].isEmpty())
		{
			printf("Subseg #%d: Empty subface neighbor #%d\n", checkseg.id, neighsh.id);
		}
		else
		{
			if (d_trifacelist[3 * neighsh.id + 2] == -1)
			{
				printf("Subseg #%d: Wrong neighbor type (Should be subface) #%d\n", checkseg.id, neighsh.id);
			}
			else
			{
				cudamesh_sspivot(neighsh, neighseg, d_tri2seglist);
				if (neighseg.id != checkseg.id)
					printf("Subseg #%d: Wrong subface neighbor #%d - %d, %d, %d\n", checkseg.id,
						neighsh.id, d_tri2seglist[3 * neighsh.id + 0].id, d_tri2seglist[3 * neighsh.id + 1].id, d_tri2seglist[3 * neighsh.id + 2].id);
				else
				{
					pa = cudamesh_sorg(checkseg, d_seglist);
					pb = cudamesh_sdest(checkseg, d_seglist);
					pc = cudamesh_sorg(neighsh, d_trifacelist);
					pd = cudamesh_sdest(neighsh, d_trifacelist);
					if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
					{

					}
					else
					{
						printf("Subseg #%d - %d, %d: Wrong subface neighbor endpoints #%d - %d, %d, %d\n", checkseg.id,
							d_seglist[3 * checkseg.id], d_seglist[3 * checkseg.id + 1],
							neighsh.id, d_trifacelist[3 * neighsh.id], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
					}
				}
			}
		}
	}

	cudamesh_senextself(checkseg);
	cudamesh_spivot(checkseg, prevseg, d_seg2trilist);
	if (prevseg.id != -1)
	{
		if (d_segstatus[prevseg.id].isEmpty())
		{
			printf("Subseg #%d: Empty subseg neighbor #%d\n", checkseg.id, prevseg.id);
		}
		else
		{
			if (d_seglist[3 * prevseg.id + 2] != -1)
			{
				printf("Subseg #%d: Wrong neighbor type (Should be subseg) #%d\n", checkseg.id, prevseg.id);
			}
			else
			{
				cudamesh_spivot(prevseg, neighseg, d_seg2trilist);
				if(neighseg.id != checkseg.id)
					printf("Subseg #%d: Wrong subseg neighbor #%d - %d, %d, %d\n", checkseg.id,
						prevseg.id, d_seg2trilist[3 * prevseg.id + 0].id, d_seg2trilist[3 * prevseg.id + 1].id, d_seg2trilist[3 * prevseg.id + 2].id);
			}
		}
	}

	cudamesh_senextself(checkseg);
	cudamesh_spivot(checkseg, nextseg, d_seg2trilist);
	if (nextseg.id != -1)
	{
		if (d_segstatus[nextseg.id].isEmpty())
		{
			printf("Subseg #%d: Empty subseg neighbor #%d\n", checkseg.id, prevseg.id);
		}
		else
		{
			if (d_seglist[3 * nextseg.id + 2] != -1)
			{
				printf("Subseg #%d: Wrong neighbor type (Should be subseg) #%d\n", checkseg.id, nextseg.id);
			}
			else
			{
				cudamesh_spivot(nextseg, neighseg, d_seg2trilist);
				if (neighseg.id != checkseg.id)
					printf("Subseg #%d: Wrong subseg neighbor #%d - %d, %d, %d\n", checkseg.id,
						nextseg.id, d_seg2trilist[3 * nextseg.id + 0].id, d_seg2trilist[3 * nextseg.id + 1].id, d_seg2trilist[3 * nextseg.id + 2].id);
			}
		}
	}

	tethandle neightet;
	checkseg.shver = 0;
	cudamesh_sstpivot1(checkseg, neightet, d_seg2tetlist);
	if (neightet.id != -1)
	{
		if (d_tetstatus[neightet.id].isEmpty())
		{
			printf("Subseg #%d: Empty tet neighbor #%d\n", checkseg.id, neightet.id);
		}
		else
		{
			cudamesh_tsspivot1(neightet, neighseg, d_tet2seglist);
			if (neighseg.id != checkseg.id)
				printf("Subseg #%d: Wrong tet neighbor #%d - %d, %d, %d, %d, %d, %d\n", checkseg.id,
					neightet.id, d_tet2seglist[6 * neightet.id + 0].id, d_tet2seglist[6 * neightet.id + 1].id, d_tet2seglist[6 * neightet.id + 2].id,
					d_tet2seglist[6 * neightet.id + 3].id, d_tet2seglist[6 * neightet.id + 4].id, d_tet2seglist[6 * neightet.id + 5].id);
			else
			{
				pa = cudamesh_sorg(checkseg, d_seglist);
				pb = cudamesh_sdest(checkseg, d_seglist);
				pc = cudamesh_org(neightet, d_tetlist);
				pd = cudamesh_dest(neightet, d_tetlist);
				if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
				{

				}
				else
				{
					printf("Subseg #%d - %d, %d: Wrong tet neighbor endpoints #%d(%d) - %d, %d, %d, %d\n", checkseg.id,
						d_seglist[3 * checkseg.id], d_seglist[3 * checkseg.id + 1],
						neightet.id, neightet.ver,
						d_tetlist[4 * neightet.id], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3]);
				}
			}
		}
	}
}

__global__ void kernelCheckSubfaceNeighbors(
	int* d_seglist,
	trihandle* d_seg2trilist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	//if (d_tristatus[pos].isEmpty())
	//	return;

	trihandle checksh(pos, 0), neighseg, neighsh, neineighsh;
	tethandle neightet;
	int i, pa, pb, pc, pd, pe, pf;

	//for (i = 0; i < 3; i++)
	//{
	//	cudamesh_senextself(checksh);
	//	cudamesh_sspivot(checksh, neighseg, d_tri2seglist);
	//	if (neighseg.id != -1)
	//	{
	//		if (d_segstatus[neighseg.id].isEmpty())
	//			printf("Subface #%d: Empty subseg neighbor #%d\n", checksh.id, neighseg.id);
	//		else
	//		{
	//			cudamesh_spivot(neighseg, neighsh, d_seg2trilist);
	//			if (neighsh.id == -1)
	//			{
	//				printf("Subface #%d: Wrong subseg neighbor, Subface #%d - %d, %d, %d, Subseg #%d - (-1)\n",
	//					checksh.id, d_tri2seglist[3 * checksh.id + 0].id, d_tri2seglist[3 * checksh.id + 1].id, d_tri2seglist[3 * checksh.id + 2].id,
	//					neighseg.id);
	//			}
	//			else
	//			{
	//				//printf("%d ", neighsh.id);
	//				bool found = false;
	//				cudamesh_spivot(neighsh, neineighsh, d_tri2trilist);
	//				if (neighsh.id == checksh.id)
	//					found = true;
	//				if (neineighsh.id == -1) // this only happen when neighsh is a single subface
	//				{
	//					if(checksh.id != neighsh.id)
	//						printf("Subface: Wrong single subface neighbor - Checksh #%d, Neighseg #%d, Neighsh #%d\n", checksh.id, neighseg.id, neighsh.id);
	//				}
	//				else
	//				{
	//					if (neighsh.id == neineighsh.id)
	//					{
	//						if (checksh.id != neighsh.id)
	//							printf("Subface: Wrong single subface neighbor - Checksh #%d, Neighsh #%d, neineighsh #%d\n", checksh.id, neighsh.id, neineighsh.id);
	//					}
	//					else
	//					{
	//						while (neineighsh.id != neighsh.id)
	//						{
	//							if (neineighsh.id == checksh.id)
	//							{
	//								found = true;
	//								break;
	//							}
	//							cudamesh_spivotself(neineighsh, d_tri2trilist);
	//						}
	//					}
	//				}
	//				if (!found)
	//					printf("Subface #%d: Wrong subseg neighbor #%d, missing in loop\n",
	//						checksh.id, neighseg.id);
	//				else
	//				{
	//					pa = cudamesh_sorg(checksh, d_trifacelist);
	//					pb = cudamesh_sdest(checksh, d_trifacelist);
	//					pc = cudamesh_sorg(neighseg, d_seglist);
	//					pd = cudamesh_sdest(neighseg, d_seglist);
	//					if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
	//					{

	//					}
	//					else
	//					{
	//						printf("Subface #%d - %d, %d, %d: Wrong subseg neighbor endpoints #%d - %d, %d, %d\n",
	//							checksh.id, d_trifacelist[3 * checksh.id + 0], d_trifacelist[3 * checksh.id + 1], d_trifacelist[3 * checksh.id + 2],
	//							neighseg.id, d_seglist[3 * neighseg.id + 0], d_seglist[3 * neighseg.id + 1], d_seglist[3 * neighseg.id + 2]);
	//					}
	//				}
	//			}
	//		}
	//	}
	//}

	/*for (i = 0; i < 3; i++)
	{
		cudamesh_senextself(checksh);
		cudamesh_spivot(checksh, neighsh, d_tri2trilist);
		if (neighsh.id != -1)
		{
			while (neighsh.id != checksh.id)
			{
				if (d_tristatus[neighsh.id].isEmpty())
				{
					printf("Subface #%d - %d, %d, %d - %d, %d, %d: Empty subface neighbor #%d - %d, %d, %d - %d, %d, %d\n",
						checksh.id, d_tri2trilist[3 * checksh.id + 0].id, d_tri2trilist[3 * checksh.id + 1].id, d_tri2trilist[3 * checksh.id + 2].id,
						d_trifacelist[3 * checksh.id + 0], d_trifacelist[3 * checksh.id + 1], d_trifacelist[3 * checksh.id + 2],
						neighsh.id, d_tri2trilist[3 * neighsh.id + 0].id, d_tri2trilist[3 * neighsh.id + 1].id, d_tri2trilist[3 * neighsh.id + 2].id,
						d_trifacelist[3 * neighsh.id + 0], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
					break;
				}
				cudamesh_spivotself(neighsh, d_tri2trilist);
			}
		}
	}*/

	for (i = 0; i < 2; i++)
	{
		cudamesh_sesymself(checksh);
		cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
		if(d_tristatus[pos].isEmpty())
		{
			if(neightet.id != -1)
				printf("Subface #%d (old): Did not reset tet neighbor #%d\n", checksh.id, neightet.id);
		}
		else
		{
			if (neightet.id != -1)
			{
				if (d_tetstatus[neightet.id].isEmpty())
				{
					printf("Subface #%d: Empty tet neighbor #%d\n", checksh.id, neightet.id);
				}
				else
				{
					cudamesh_tspivot(neightet, neighsh, d_tet2trilist);
					if (neighsh.id != checksh.id)
						printf("Subface #%d: Wrong tet neighbor #%d - %d, %d, %d, %d\n", checksh.id,
							neightet.id, d_tet2trilist[4 * neightet.id + 0].id, d_tet2trilist[4 * neightet.id + 1].id, d_tet2trilist[4 * neightet.id + 2].id, d_tet2trilist[4 * neightet.id + 3].id);
					else
					{
						pa = cudamesh_sorg(checksh, d_trifacelist);
						pb = cudamesh_sdest(checksh, d_trifacelist);
						pc = cudamesh_sapex(checksh, d_trifacelist);
						pd = cudamesh_org(neightet, d_tetlist);
						pe = cudamesh_dest(neightet, d_tetlist);
						pf = cudamesh_apex(neightet, d_tetlist);
						if (pa == pd && pb == pe && pc == pf)
						{

						}
						else
						{
							printf("Subface #%d - %d, %d, %d: Wrong tet neighbor endpoints #%d - %d, %d, %d, %d\n",
								checksh.id, d_trifacelist[3 * checksh.id + 0], d_trifacelist[3 * checksh.id + 1], d_trifacelist[3 * checksh.id + 2],
								neightet.id, d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3]);
						}
					}
				}
			}
			else
				printf("Subface #%d: Empty tet neighbor #%d\n", checksh.id, neightet.id);
		}
	}
}

__global__ void kernelCheckTetNeighbors(
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	//if (d_tetstatus[pos].isEmpty())
	//	return;

	tethandle checktet, neightet, neineightet;
	trihandle neighsh, neighseg;
	int i, pa, pb, pc, pd, pe, pf;

	for (i = 0; i < 4; i++)
	{
		neightet = d_neighborlist[4 * pos + i];
		if (d_tetstatus[pos].isEmpty())
		{
			if(neightet.id != -1)
				printf("Tet #%d (old) - %d, %d, %d, %d: Did not reset tet neighbor #%d - %d, %d, %d, %d\n",
					pos, d_neighborlist[4 * pos].id, d_neighborlist[4 * pos + 1].id, d_neighborlist[4 * pos + 2].id, d_neighborlist[4 * pos + 3].id,
					neightet.id, d_neighborlist[4 * neightet.id].id, d_neighborlist[4 * neightet.id + 1].id, d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
		}
		else
		{
			if (neightet.id != -1)
			{
				if(neightet.id == pos)
					printf("Tet #%d - %d, %d, %d, %d: Self tet neighbor #%d - %d, %d, %d, %d\n",
						pos, d_neighborlist[4 * pos].id, d_neighborlist[4 * pos + 1].id, d_neighborlist[4 * pos + 2].id, d_neighborlist[4 * pos + 3].id,
						neightet.id, d_neighborlist[4 * neightet.id].id, d_neighborlist[4 * neightet.id + 1].id, d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
				if (i == 3)
				{
					checktet.id = pos; checktet.ver = 11;
					if(cudamesh_ishulltet(checktet, d_tetlist) &&
						cudamesh_ishulltet(neightet, d_tetlist))
						printf("Tet #%d - %d, %d, %d, %d - %d, %d, %d, %d: Wrong hull tet neighbor #%d - %d, %d, %d, %d - %d, %d, %d, %d\n",
							pos,
							d_tetlist[4 * pos], d_tetlist[4 * pos + 1], d_tetlist[4 * pos + 2], d_tetlist[4 * pos + 3],
							d_neighborlist[4 * pos].id, d_neighborlist[4 * pos + 1].id, d_neighborlist[4 * pos + 2].id, d_neighborlist[4 * pos + 3].id,
							neightet.id,
							d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3],
							d_neighborlist[4 * neightet.id].id, d_neighborlist[4 * neightet.id + 1].id, d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
				}

				if (d_tetstatus[neightet.id].isEmpty())
				{
					printf("Tet #%d - %d, %d, %d, %d: Empty tet neighbor #%d - %d, %d, %d, %d\n",
						pos, d_neighborlist[4 * pos].id, d_neighborlist[4 * pos + 1].id, d_neighborlist[4 * pos + 2].id, d_neighborlist[4 * pos + 3].id,
						neightet.id, d_neighborlist[4 * neightet.id].id, d_neighborlist[4 * neightet.id + 1].id, d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
				}
				else
				{
					cudamesh_fsym(neightet, neineightet, d_neighborlist);
					if (neineightet.id != pos)
						printf("Tet #%d: Wrong tet neighbor #%d - %d, %d, %d, %d\n", pos,
							neightet.id, d_neighborlist[4 * neightet.id + 0].id, d_neighborlist[4 * neightet.id + 1].id,
							d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
					else
					{
						pa = cudamesh_org(neightet, d_tetlist);
						pb = cudamesh_dest(neightet, d_tetlist);
						pc = cudamesh_apex(neightet, d_tetlist);
						pd = cudamesh_org(neineightet, d_tetlist);
						pe = cudamesh_dest(neineightet, d_tetlist);
						pf = cudamesh_apex(neineightet, d_tetlist);
						if (pa == pe && pb == pd && pc == pf)
						{

						}
						else
						{
							printf("Tet #%d - %d, %d, %d, %d: Wrong tet neighbor endpoints #%d - %d, %d, %d, %d\n",
								pos, d_tetlist[4 * pos], d_tetlist[4 * pos + 1], d_tetlist[4 * pos + 2], d_tetlist[4 * pos + 3],
								neightet.id, d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3]);
						}
					}
				}
			}
			else
			{
				printf("Tet #%d - %d, %d, %d, %d: Empty tet neighbor #%d - %d, %d, %d, %d\n",
					pos, d_neighborlist[4 * pos].id, d_neighborlist[4 * pos + 1].id, d_neighborlist[4 * pos + 2].id, d_neighborlist[4 * pos + 3].id,
					neightet.id, d_neighborlist[4 * neightet.id].id, d_neighborlist[4 * neightet.id + 1].id, d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
			}
		}
	}

	for (i = 0; i < 4; i++)
	{
		neighsh = d_tet2trilist[4 * pos + i];
		if (d_tetstatus[pos].isEmpty())
		{
			if(neighsh.id != -1)
				printf("Tet #%d (old) - %d, %d, %d, %d: Did not reset subface neighbor #%d - %d, %d\n",
					pos, d_tet2trilist[4 * pos].id, d_tet2trilist[4 * pos + 1].id, d_tet2trilist[4 * pos + 2].id, d_tet2trilist[4 * pos + 3].id,
					neighsh.id, d_tri2tetlist[2 * neighsh.id].id, d_tri2tetlist[2 * neighsh.id + 1].id);
		}
		else
		{
			if (neighsh.id != -1)
			{
				if (d_tristatus[neighsh.id].isEmpty())
				{
					printf("Tet #%d - %d, %d, %d, %d: Empty subface neighbor #%d - %d, %d\n",
						pos, d_tet2trilist[4 * pos].id, d_tet2trilist[4 * pos + 1].id, d_tet2trilist[4 * pos + 2].id, d_tet2trilist[4 * pos + 3].id,
						neighsh.id, d_tri2tetlist[2 * neighsh.id].id, d_tri2tetlist[2 * neighsh.id + 1].id);
				}
				else
				{
					cudamesh_stpivot(neighsh, neightet, d_tri2tetlist);
					if (neightet.id != pos)
					{
						cudamesh_sesymself(neighsh);
						cudamesh_stpivot(neighsh, neightet, d_tri2tetlist);
						if (neightet.id != pos)
							printf("Tet #%d: Wrong subface neighbor #%d - %d, %d\n", pos,
								neighsh.id, d_tri2tetlist[2 * neighsh.id + 0].id, d_tri2tetlist[2 * neighsh.id + 1].id);
					}
					if (neightet.id == pos)
					{
						pa = cudamesh_sorg(neighsh, d_trifacelist);
						pb = cudamesh_sdest(neighsh, d_trifacelist);
						pc = cudamesh_sapex(neighsh, d_trifacelist);
						pd = cudamesh_org(neightet, d_tetlist);
						pe = cudamesh_dest(neightet, d_tetlist);
						pf = cudamesh_apex(neightet, d_tetlist);
						if (pa == pd && pb == pe && pc == pf)
						{

						}
						else
						{
							printf("Tet #%d - %d, %d, %d, %d: Wrong subface neighbor endpoints #%d - %d, %d, %d\n",
								pos, d_tetlist[4 * pos], d_tetlist[4 * pos + 1], d_tetlist[4 * pos + 2], d_tetlist[4 * pos + 3],
								neighsh.id, d_trifacelist[3 * neighsh.id + 0], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
						}
					}
				}
			}
		}
	}

	/*for (i = 0; i < 6; i++)
	{
		neighseg = d_tet2seglist[6 * pos + i];
		if (neighseg.id != -1)
		{
			if(d_segstatus[neighseg.id].isEmpty())
			{
				printf("Tet #%d - %d, %d, %d, %d, %d, %d: Empty subseg neighbor #%d - %d\n",
					pos, d_tet2seglist[6 * pos].id, d_tet2seglist[6 * pos + 1].id, d_tet2seglist[6 * pos + 2].id, 
					d_tet2seglist[6 * pos + 3].id, d_tet2seglist[6 * pos + 4].id, d_tet2seglist[6 * pos + 5].id,
					neighseg.id, d_seg2tetlist[neighseg.id].id);
			}
			else
			{
				cudamesh_sstpivot1(neighseg, neightet, d_seg2tetlist);
				if (neightet.id == -1)
					printf("Tet #%d - Incident Subseg #%d has empty tet neighbor\n",
						pos, neighseg.id);
				else
				{
					pa = cudamesh_sorg(neighseg, d_seglist);
					pb = cudamesh_sdest(neighseg, d_seglist);
					pc = cudamesh_org(neightet, d_tetlist);
					pd = cudamesh_dest(neightet, d_tetlist);
					if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
					{

					}
					else
					{
						printf("pa = %d, pb = %d, pc = %d, pd = %d\n", pa, pb, pc, pd);
						printf("Tet #%d(%d) - %d, %d, %d, %d: Wrong subseg neighbor endpoints #%d - %d, %d, %d\n",
							neightet.id, neightet.ver,
							d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3],
							neighseg.id, d_seglist[3 * neighseg.id], d_seglist[3 * neighseg.id + 1], d_seglist[3 * neighseg.id + 2]);
					}
				}
			}
		}
	}*/
}

// Split bad elements
__global__ void kernelCheckBadElementList(
	int* d_badeleidlist,
	int* d_threadmarker,
	int* d_segencmarker,
	int* d_subfaceencmarker,
	tetstatus* d_tetstatus,
	int numofencsegs,
	int numofencsubfaces,
	int numofbadtets,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (pos < numofencsegs)
	{
		if(d_threadmarker[pos] != 0)
			printf("threadId #%d - seg #%d - wrong thread marker %d\n", pos, d_badeleidlist[pos], d_threadmarker[pos]);
		else if(d_segencmarker[d_badeleidlist[pos]] < 0)
			printf("threadId #%d - seg #%d - wrong encroachement marker %d\n", pos, d_badeleidlist[pos], d_segencmarker[d_badeleidlist[pos]]);
	}
	else if (pos < numofencsubfaces + numofencsegs)
	{	
		if (d_threadmarker[pos] != 1)
			printf("threadId #%d - subface #%d - wrong thread marker %d\n", pos, d_badeleidlist[pos], d_threadmarker[pos]);
		else if (d_subfaceencmarker[d_badeleidlist[pos]] < 0)
			printf("threadId #%d - subface #%d - wrong encroachement marker %d\n", pos, d_badeleidlist[pos], d_subfaceencmarker[d_badeleidlist[pos]]);
	}
	else
	{
		if (d_threadmarker[pos] != 2)
			printf("threadId #%d - tet #%d - wrong thread marker %d\n", pos, d_badeleidlist[pos], d_threadmarker[pos]);
		else if (!d_tetstatus[d_badeleidlist[pos]].isBad() || d_tetstatus[d_badeleidlist[pos]].isEmpty())
			printf("threadId #%d - tet #%d - wrong tet status\n", pos, d_badeleidlist[pos]);
	}

}

/* Point Insertion */
__global__ void kernelComputeSteinerPointAndPriority(
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	REAL* d_priority,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int eleidx = d_insertidxlist[pos];
	int threadmarker = d_threadmarker[pos];
	REAL* steinpt = cudamesh_id2pointlist(pos, d_steinerptlist);

	if (threadmarker == 1) // is a subface
	{
		REAL *pa, *pb, *pc;
		REAL sarea;

		trihandle chkfac(eleidx, 0);
		steinpt[0] = d_trifacecent[3 * eleidx + 0];
		steinpt[1] = d_trifacecent[3 * eleidx + 1];
		steinpt[2] = d_trifacecent[3 * eleidx + 2];

		pa = cudamesh_id2pointlist(cudamesh_sorg(chkfac, d_trifacelist), d_pointlist);
		pb = cudamesh_id2pointlist(cudamesh_sdest(chkfac, d_trifacelist), d_pointlist);
		pc = cudamesh_id2pointlist(cudamesh_sapex(chkfac, d_trifacelist), d_pointlist);
		sarea = cudamesh_triangle_squared_area(pa, pb, pc);
		d_priority[pos] = sarea;
	}
	else if(threadmarker == 2) // is a tetrahedron
	{
		int tetid = eleidx;

		int ipa, ipb, ipc, ipd;
		REAL *pa, *pb, *pc, *pd;
		REAL wc[3], aw, bw, cw, dw;

		ipd = d_tetlist[4 * tetid + 3];
		if (ipd == -1) {
			// This should not happend
			return;
		}

		ipa = d_tetlist[4 * tetid + 0];
		ipb = d_tetlist[4 * tetid + 1];
		ipc = d_tetlist[4 * tetid + 2];

		pa = cudamesh_id2pointlist(ipa, d_pointlist);
		pb = cudamesh_id2pointlist(ipb, d_pointlist);
		pc = cudamesh_id2pointlist(ipc, d_pointlist);
		pd = cudamesh_id2pointlist(ipd, d_pointlist);
		aw = d_weightlist[ipa];
		bw = d_weightlist[ipb];
		cw = d_weightlist[ipc];
		dw = d_weightlist[ipd];

		// Calculate the weighted circumcenter
		if (!cudamesh_weightedcircumsphere(pa, pb, pc, pd, aw, bw, cw, dw, wc, NULL))
		{
			// This should not happen because degenerate tets have been ignored
			d_tetstatus[tetid].setUnsplittable(true);
			d_tetstatus[tetid].setBad(false);
			d_threadmarker[pos] = -1;
			return;
		}

		steinpt[0] = wc[0];
		steinpt[1] = wc[1];
		steinpt[2] = wc[2];

		REAL vol = cudamesh_tetrahedronvolume(pa, pb, pc, pd);
		d_priority[pos] = vol;
	}
}

__global__ void kernelComputePriorities(
	REAL* d_pointlist,
	int* d_trifacelist,
	int* d_tetlist,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_priority,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int eleidx = d_insertidxlist[pos];
	int threadmarker = d_threadmarker[pos];

	if (threadmarker == 1) // is a subface
	{
		REAL *pa, *pb, *pc;
		REAL sarea;

		trihandle chkfac(eleidx, 0);

		pa = cudamesh_id2pointlist(cudamesh_sorg(chkfac, d_trifacelist), d_pointlist);
		pb = cudamesh_id2pointlist(cudamesh_sdest(chkfac, d_trifacelist), d_pointlist);
		pc = cudamesh_id2pointlist(cudamesh_sapex(chkfac, d_trifacelist), d_pointlist);
		sarea = cudamesh_triangle_squared_area(pa, pb, pc);
		d_priority[pos] = sarea;
	}
	else if (threadmarker == 2) // is a tetrahedron
	{
		int tetid = eleidx;

		int ipa, ipb, ipc, ipd;
		REAL *pa, *pb, *pc, *pd;

		ipd = d_tetlist[4 * tetid + 3];
		if (ipd == -1) {
			// This should not happend
			return;
		}

		ipa = d_tetlist[4 * tetid + 0];
		ipb = d_tetlist[4 * tetid + 1];
		ipc = d_tetlist[4 * tetid + 2];

		pa = cudamesh_id2pointlist(ipa, d_pointlist);
		pb = cudamesh_id2pointlist(ipb, d_pointlist);
		pc = cudamesh_id2pointlist(ipc, d_pointlist);
		pd = cudamesh_id2pointlist(ipd, d_pointlist);

		REAL vol = cudamesh_tetrahedronvolume(pa, pb, pc, pd);
		d_priority[pos] = vol;
	}
}

__global__ void kernelComputeSteinerPoints(
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int eleidx = d_insertidxlist[pos];
	int threadmarker = d_threadmarker[pos];
	if (threadmarker < 0)
		return;

	REAL* steinpt = cudamesh_id2pointlist(pos, d_steinerptlist);

	if (threadmarker == 1) // is a subface
	{
		REAL *pa, *pb, *pc;
		REAL sarea;

		trihandle chkfac(eleidx, 0);
		steinpt[0] = d_trifacecent[3 * eleidx + 0];
		steinpt[1] = d_trifacecent[3 * eleidx + 1];
		steinpt[2] = d_trifacecent[3 * eleidx + 2];
	}
	else if (threadmarker == 2) // is a tetrahedron
	{
		int tetid = eleidx;

		int ipa, ipb, ipc, ipd;
		REAL *pa, *pb, *pc, *pd;
		REAL wc[3], aw, bw, cw, dw;

		ipd = d_tetlist[4 * tetid + 3];
		if (ipd == -1) {
			// This should not happend
			return;
		}

		ipa = d_tetlist[4 * tetid + 0];
		ipb = d_tetlist[4 * tetid + 1];
		ipc = d_tetlist[4 * tetid + 2];

		pa = cudamesh_id2pointlist(ipa, d_pointlist);
		pb = cudamesh_id2pointlist(ipb, d_pointlist);
		pc = cudamesh_id2pointlist(ipc, d_pointlist);
		pd = cudamesh_id2pointlist(ipd, d_pointlist);
		aw = d_weightlist[ipa];
		bw = d_weightlist[ipb];
		cw = d_weightlist[ipc];
		dw = d_weightlist[ipd];

		// Calculate the weighted circumcenter
		if (!cudamesh_weightedcircumsphere(pa, pb, pc, pd, aw, bw, cw, dw, wc, NULL))
		{
			// This should not happen because degenerate tets have been ignored
			d_tetstatus[tetid].setUnsplittable(true);
			d_tetstatus[tetid].setBad(false);
			d_threadmarker[pos] = -1;
			return;
		}

		steinpt[0] = wc[0];
		steinpt[1] = wc[1];
		steinpt[2] = wc[2];
	}
}

__global__ void kernelModifyPriority(
	REAL* d_priorityreal,
	int* d_priorityint,
	REAL offset0,
	REAL offset1,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_insertidxlist,
	int* d_threadmarker,
	int numofbadtriface,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	// threadmarker might be -1, but we still modify it
	int threadmarker = d_threadmarker[pos];
	int eleidx = d_insertidxlist[pos];

	REAL offset;
	// Exclude unsplittable elements
	if (pos < numofbadtriface)
	{
		if (d_tristatus[eleidx].isUnsplittable())
		{
			d_threadmarker[pos] = -1;
		}

		offset = offset0;
	}
	else
	{
		if (d_tetstatus[eleidx].isUnsplittable())
		{
			d_threadmarker[pos] = -1;
		}

		offset = offset1;
	}

	REAL priority = d_priorityreal[pos] + offset;
	d_priorityreal[pos] = priority;
	d_priorityint[pos] = __float_as_int((float)priority);
}

__global__ void kernelGridFiltering(
	int* d_priority,
	uint64* d_tetmarker,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	int range_left,
	int range_right,
	double step_x,
	double step_y,
	double step_z,
	double origin_x,
	double origin_y,
	double origin_z,
	int gridlength
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= range_right || pos < range_left)
		return;

	int threadmarker = d_threadmarker[pos];
	if (threadmarker < 0) // lost already
		return;

	uint64 marker, oldmarker;
	// here we use threadId + 1, 0 thread index is reserved for default marker
	marker = cudamesh_encodeUInt64Priority(d_priority[pos], pos + 1);
	REAL* searchpt = d_steinerptlist + 3 * pos;

	int gridIdx = (searchpt[0] - origin_x) / step_x;
	if (gridIdx >= gridlength)
		gridIdx = gridlength - 1;

	int gridIdy = (searchpt[1] - origin_y) / step_y;
	if (gridIdy >= gridlength)
		gridIdy = gridlength - 1;

	int gridIdz = (searchpt[2] - origin_z) / step_z;
	if (gridIdz >= gridlength)
		gridIdz = gridlength - 1;

	int gridIndex = gridIdx + gridIdy * (gridlength - 1) + 
		gridIdz * (gridlength - 1) * (gridlength - 1);

	// marking competition
	int old;
	oldmarker = atomicMax(d_tetmarker + gridIndex, marker);
	if (marker > oldmarker) // winned
	{
		old = cudamesh_getUInt64PriorityIndex(oldmarker);
		if (old != 0) // marked by others
		{
			d_threadmarker[old - 1] = -1;
		}
	}
	else // lost
	{
		d_threadmarker[pos] = -1;
	}
}

__global__ void kernelLocatePoint(
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_priority,
	locateresult* d_pointlocation,
	tethandle* d_searchtet,
	int* d_insertidxlist,
	int* d_threadmarker,
	int* d_threadlist,
	REAL* d_steinerptlist,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker < 0)
		return;

	int eleidx = d_insertidxlist[threadId];

	if (threadmarker == 1 || threadmarker == 2)
	{
		tethandle* searchtet = d_searchtet + threadId;
		REAL* searchpt = d_steinerptlist + 3 * threadId;
		unsigned long randomseed = 1;

		REAL *torg, *tdest, *tapex, *toppo;
		enum { ORGMOVE, DESTMOVE, APEXMOVE } nextmove;
		REAL ori, oriorg, oridest, oriapex;
		enum locateresult loc = OUTSIDE;
		int t1ver;
		int s;
		int step = 0;

		// Init searchtet
		if (threadmarker == 1)
		{
			trihandle searchsh(eleidx, 0);
			cudamesh_stpivot(searchsh, *searchtet, d_tri2tetlist);
		}
		else
		{
			searchtet->id = eleidx;
			searchtet->ver = 11;
		}

		// Check if we are in the outside of the convex hull.
		if (cudamesh_ishulltet(*searchtet, d_tetlist)) {
			// Get its adjacent tet (inside the hull).
			searchtet->ver = 3;
			cudamesh_fsymself(*searchtet, d_neighborlist);
		}

		// Let searchtet be the face such that 'searchpt' lies above to it.
		for (searchtet->ver = 0; searchtet->ver < 4; searchtet->ver++) {
			torg = cudamesh_id2pointlist(cudamesh_org(*searchtet, d_tetlist), d_pointlist);
			tdest = cudamesh_id2pointlist(cudamesh_dest(*searchtet, d_tetlist), d_pointlist);
			tapex = cudamesh_id2pointlist(cudamesh_apex(*searchtet, d_tetlist), d_pointlist);
			ori = cuda_orient3dfast(torg, tdest, tapex, searchpt);
			if (ori < 0.0) break;
		}
		//assert(searchtet->ver != 4);
		//if (searchtet->ver == 4)
		//{
		//	if (threadmarker == 1)
		//	{
		//		d_tristatus[eleidx].setUnsplittable(true);
		//		d_tristatus[eleidx].setBad(false);
		//	}
		//	else
		//	{
		//		d_tetstatus[eleidx].setUnsplittable(true);
		//		d_tetstatus[eleidx].setBad(false);
		//	}
		//	d_threadmarker[threadId] = -1;
		//	return;
		//}

		// Walk through tetrahedra to locate the point.
		while (true) {
			toppo = cudamesh_id2pointlist(cudamesh_oppo(*searchtet, d_tetlist), d_pointlist);

			// Check if the vertex is we seek.
			if (toppo[0] == searchpt[0] && toppo[1] == searchpt[1] && toppo[2] == searchpt[2]) {
				// Adjust the origin of searchtet to be searchpt.
				cudamesh_esymself(*searchtet);
				cudamesh_eprevself(*searchtet);
				loc = ONVERTEX; // return ONVERTEX;
				break;
			}

			// We enter from one of serarchtet's faces, which face do we exit?
			oriorg = cuda_orient3dfast(tdest, tapex, toppo, searchpt);
			oridest = cuda_orient3dfast(tapex, torg, toppo, searchpt);
			oriapex = cuda_orient3dfast(torg, tdest, toppo, searchpt);

			// Now decide which face to move. It is possible there are more than one
			//   faces are viable moves. If so, randomly choose one.
			if (oriorg < 0) {
				if (oridest < 0) {
					if (oriapex < 0) {
						// All three faces are possible.
						s = cudamesh_randomnation(&randomseed, 3); // 's' is in {0,1,2}.
						if (s == 0) {
							nextmove = ORGMOVE;
						}
						else if (s == 1) {
							nextmove = DESTMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Two faces, opposite to origin and destination, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(&randomseed, 2)) {
							nextmove = ORGMOVE;
						}
						else {
							nextmove = DESTMOVE;
						}
					}
				}
				else {
					if (oriapex < 0) {
						// Two faces, opposite to origin and apex, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(&randomseed, 2)) {
							nextmove = ORGMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Only the face opposite to origin is viable.
						nextmove = ORGMOVE;
					}
				}
			}
			else {
				if (oridest < 0) {
					if (oriapex < 0) {
						// Two faces, opposite to destination and apex, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(&randomseed, 2)) {
							nextmove = DESTMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Only the face opposite to destination is viable.
						nextmove = DESTMOVE;
					}
				}
				else {
					if (oriapex < 0) {
						// Only the face opposite to apex is viable.
						nextmove = APEXMOVE;
					}
					else {
						// The point we seek must be on the boundary of or inside this
						//   tetrahedron. Check for boundary cases.
						if (oriorg == 0) {
							// Go to the face opposite to origin.
							cudamesh_enextesymself(*searchtet);
							if (oridest == 0) {
								cudamesh_eprevself(*searchtet); // edge oppo->apex
								if (oriapex == 0) {
									// oppo is duplicated with p.
									loc = ONVERTEX; // return ONVERTEX;
									break;
								}
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							if (oriapex == 0) {
								cudamesh_enextself(*searchtet); // edge dest->oppo
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							loc = ONFACE; // return ONFACE;
							break;
						}
						if (oridest == 0) {
							// Go to the face opposite to destination.
							cudamesh_eprevesymself(*searchtet);
							if (oriapex == 0) {
								cudamesh_eprevself(*searchtet); // edge oppo->org
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							loc = ONFACE; // return ONFACE;
							break;
						}
						if (oriapex == 0) {
							// Go to the face opposite to apex
							cudamesh_esymself(*searchtet);
							loc = ONFACE; // return ONFACE;
							break;
						}
						loc = INTETRAHEDRON; // return INTETRAHEDRON;
						break;
					}
				}
			}

			// Move to the selected face.
			if (nextmove == ORGMOVE) {
				cudamesh_enextesymself(*searchtet);
			}
			else if (nextmove == DESTMOVE) {
				cudamesh_eprevesymself(*searchtet);
			}
			else {
				cudamesh_esymself(*searchtet);
			}
			// Move to the adjacent tetrahedron (maybe a hull tetrahedron).
			cudamesh_fsymself(*searchtet, d_neighborlist);
			if (cudamesh_oppo(*searchtet, d_tetlist) == -1) {
				loc = OUTSIDE; // return OUTSIDE;
				break;
			}

			// Retreat the three vertices of the base face.
			torg = cudamesh_id2pointlist(cudamesh_org(*searchtet, d_tetlist), d_pointlist);
			tdest = cudamesh_id2pointlist(cudamesh_dest(*searchtet, d_tetlist), d_pointlist);
			tapex = cudamesh_id2pointlist(cudamesh_apex(*searchtet, d_tetlist), d_pointlist);

			step++;
			if (step > 5000)
			{
				if (threadmarker == 1)
				{
					d_tristatus[eleidx].setUnsplittable(true);
					d_tristatus[eleidx].setBad(false);
				}
				else
				{
					d_tetstatus[eleidx].setUnsplittable(true);
					d_tetstatus[eleidx].setBad(false);
				}
				d_threadmarker[threadId] = -1;
				return;
			}
		} // while (true)

		d_pointlocation[threadId] = loc;

		// duplicate vertex
		if (loc == ONVERTEX || loc == UNKNOWN)
		{
			if (threadmarker == 1)
			{
				d_tristatus[eleidx].setUnsplittable(true);
				d_tristatus[eleidx].setBad(false);
			}
			else
			{
				d_tetstatus[eleidx].setUnsplittable(true);
				d_tetstatus[eleidx].setBad(false);
			}
			d_threadmarker[threadId] = -1;
			return;
		}

		// regular point
		if (loc != OUTSIDE)
		{
			REAL *pts[4], wts[4], sign;
			bool invalid = false;
			int idx[4];
			for (int i = 0; i < 4; i++)
			{
				idx[i] = d_tetlist[4 * (*searchtet).id + i];
				if (idx[i] != -1)
				{
					pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
					wts[i] = d_weightlist[idx[i]];
				}
				else
					invalid = true;
			}
			if (!invalid)
			{
				sign = cudamesh_orient4d_s(pts[0], pts[1], pts[2], pts[3], searchpt,
					idx[0], idx[1], idx[2], idx[3], MAXINT,
					wts[0], wts[1], wts[2], wts[3], 0.0);
				if (sign > 0.0)
				{
					if (threadmarker == 1)
					{
						d_tristatus[eleidx].setUnsplittable(true);
						d_tristatus[eleidx].setBad(false);
					}
					else
					{
						d_tetstatus[eleidx].setUnsplittable(true);
						d_tetstatus[eleidx].setBad(false);
					}
					d_threadmarker[threadId] = -1;
				}
			}
		}
	}
}

__global__ void kernelMarkAndCountInitialCavity(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	tethandle searchtet = d_searchtet[threadId];
	tethandle spintet, neightet;
	locateresult loc = d_pointlocation[threadId];

	// initial cavity
	// mark all tets intersecting the splitting point
	int count = 0, i;
	int old;
	uint64 marker, oldmarker;
	// here we use threadId + 1, 0 thread index is reserved for default marker
	marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);

	if (loc == ONEDGE)
	{
		spintet = searchtet;
		while (1) {
			// check if already lost
			if (d_threadmarker[threadId] == -1) // lost because of other threads
			{
				count = 0;
				break;
			}

			// marking competition
			oldmarker = atomicMax(d_tetmarker + spintet.id, marker);
			if (marker > oldmarker) // winned
			{
				old = cudamesh_getUInt64PriorityIndex(oldmarker);
				if (old != 0) // marked by others
				{
					d_threadmarker[old - 1] = -1;
					atomicMin(d_initialcavitysize + old - 1, 0);
				}
			}
			else // lost
			{
				d_threadmarker[threadId] = -1;
				count = 0;
				break;
			}

			count++;
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id) break;
		} // while (1)
	}
	else if (loc == ONFACE)
	{
		// check if already lost
		if (d_threadmarker[threadId] == -1) // lost because of other threads
		{
			count = 0;
		}
		else // mark two adjacent tets on the face 
		{
			spintet = searchtet;
			for (i = 0; i < 2; i++)
			{
				// marking competition
				oldmarker = atomicMax(d_tetmarker + spintet.id, marker);
				if (marker > oldmarker) // winned
				{
					old = cudamesh_getUInt64PriorityIndex(oldmarker);
					if (old != 0)
					{
						d_threadmarker[old - 1] = -1;
						atomicMin(d_initialcavitysize + old - 1, 0);
					}
				}
				else // lost
				{
					d_threadmarker[threadId] = -1;
					count = 0;
					break;
				}
				count++;
				spintet = d_neighborlist[4 * searchtet.id + (searchtet.ver & 3)];
			}
		}
	}
	else if (loc == INTETRAHEDRON || loc == OUTSIDE)
	{
		// check if already lost
		if (d_threadmarker[threadId] == -1) // lost because of other threads
		{
			count = 0;
		}
		else // mark this tet
		{
			// marking competition
			oldmarker = atomicMax(d_tetmarker + searchtet.id, marker);
			if (marker > oldmarker) // winned
			{
				count = 1;
				old = cudamesh_getUInt64PriorityIndex(oldmarker);
				if (old != 0)
				{
					d_threadmarker[old - 1] = -1;
					atomicMin(d_initialcavitysize + old - 1, 0);
				}
			}
			else // lost
			{
				d_threadmarker[threadId] = -1;
				count = 0;
			}
		}
	}

	atomicMin(d_initialcavitysize + threadId, count);
}

__global__ void kernelMarkAndCountInitialCavity(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	int eleidx = d_insertidxlist[threadId];
	if (threadmarker == 1)
	{
		if (d_tristatus[eleidx].isCavityReuse())
		{
			atomicMin(d_initialcavitysize + threadId, 0);
			return;
		}
	}
	else if (threadmarker == 2)
	{
		if (d_tetstatus[eleidx].isCavityReuse())
		{
			atomicMin(d_initialcavitysize + threadId, 0);
			return;
		}
	}

	tethandle searchtet = d_searchtet[threadId];
	tethandle spintet, neightet;
	locateresult loc = d_pointlocation[threadId];

	// initial cavity
	// mark all tets intersecting the splitting point
	int count = 0, i;
	int old;
	uint64 marker, oldmarker;
	// here we use threadId + 1, 0 thread index is reserved for default marker
	marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);

	if (loc == ONEDGE)
	{
		spintet = searchtet;
		while (1) {
			// check if already lost
			if (d_threadmarker[threadId] == -1) // lost because of other threads
			{
				count = 0;
				break;
			}

			// marking competition
			oldmarker = atomicMax(d_tetmarker + spintet.id, marker);
			if (marker > oldmarker) // winned
			{
				old = cudamesh_getUInt64PriorityIndex(oldmarker);
				if (old != 0) // marked by others
				{
					d_threadmarker[old - 1] = -1;
					atomicMin(d_initialcavitysize + old - 1, 0);
				}
			}
			else // lost
			{
				d_threadmarker[threadId] = -1;
				count = 0;
				break;
			}

			count++;
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id) break;
		} // while (1)
	}
	else if (loc == ONFACE)
	{
		// check if already lost
		if (d_threadmarker[threadId] == -1) // lost because of other threads
		{
			count = 0;
		}
		else // mark two adjacent tets on the face 
		{
			spintet = searchtet;
			for (i = 0; i < 2; i++)
			{
				// marking competition
				oldmarker = atomicMax(d_tetmarker + spintet.id, marker);
				if (marker > oldmarker) // winned
				{
					old = cudamesh_getUInt64PriorityIndex(oldmarker);
					if (old != 0)
					{
						d_threadmarker[old - 1] = -1;
						atomicMin(d_initialcavitysize + old - 1, 0);
					}
				}
				else // lost
				{
					d_threadmarker[threadId] = -1;
					count = 0;
					break;
				}
				count++;
				spintet = d_neighborlist[4 * searchtet.id + (searchtet.ver & 3)];
			}
		}
	}
	else if (loc == INTETRAHEDRON || loc == OUTSIDE)
	{
		// check if already lost
		if (d_threadmarker[threadId] == -1) // lost because of other threads
		{
			count = 0;
		}
		else // mark this tet
		{
			// marking competition
			oldmarker = atomicMax(d_tetmarker + searchtet.id, marker);
			if (marker > oldmarker) // winned
			{
				count = 1;
				old = cudamesh_getUInt64PriorityIndex(oldmarker);
				if (old != 0)
				{
					d_threadmarker[old - 1] = -1;
					atomicMin(d_initialcavitysize + old - 1, 0);
				}
			}
			else // lost
			{
				d_threadmarker[threadId] = -1;
				count = 0;
			}
		}
	}

	atomicMin(d_initialcavitysize + threadId, count);
}

__global__ void kernelInitCavityLinklist(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	int* d_initialcavityindices,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavethreadidx,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	tethandle searchtet = d_searchtet[threadId];
	tethandle spintet, neightet;
	locateresult loc = d_pointlocation[threadId];

	// cavities
	int icsindex = d_initialcavityindices[threadId];
	int count = 0;
	int prev = -1;
	int icavityIdx;
	int tetidxfactor = 4;

	if (loc == ONEDGE)
	{
		spintet = searchtet;
		while (1) {
			// initial cavity index
			icavityIdx = icsindex + count;

			// add to tet list
			cudamesh_eorgoppo(spintet, neightet);
			neightet = d_neighborlist[4 * neightet.id + (neightet.ver & 3)];
			neightet.ver = raw_epivot[neightet.ver];
			d_cavetetlist[tetidxfactor * icavityIdx] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx] = (prev == -1) ? -1 : tetidxfactor * prev + 1;
			d_cavetetnext[tetidxfactor * icavityIdx] = tetidxfactor * icavityIdx + 1;
			if (prev != -1)
				d_cavetetnext[tetidxfactor * prev + 1] = tetidxfactor * icavityIdx;
			d_cavethreadidx[tetidxfactor * icavityIdx] = threadId;

			cudamesh_edestoppo(spintet, neightet);
			neightet = d_neighborlist[4 * neightet.id + (neightet.ver & 3)];
			neightet.ver = raw_epivot[neightet.ver];
			d_cavetetlist[tetidxfactor * icavityIdx + 1] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx + 1] = tetidxfactor * icavityIdx;
			d_cavetetnext[tetidxfactor * icavityIdx + 1] = -1;
			d_cavethreadidx[tetidxfactor * icavityIdx + 1] = threadId;

			// add to old tet list
			d_caveoldtetlist[icavityIdx] = spintet; // current tet
			d_caveoldtetprev[icavityIdx] = prev; // previous
			d_caveoldtetnext[icavityIdx] = -1; // next, set to -1 first
			if (prev != -1)
				d_caveoldtetnext[prev] = icavityIdx; // previous next, set to me

			if (count == 0)
			{
				d_caveoldtethead[threadId] = icavityIdx;
				d_cavetethead[threadId] = tetidxfactor * icavityIdx;
			}

			// next iteration
			prev = icavityIdx;
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id)
			{
				d_caveoldtettail[threadId] = icavityIdx;
				d_cavetettail[threadId] = tetidxfactor * icavityIdx + 1;
				break;
			}
			count++;
		} // while (1)
	}
	else if (loc == ONFACE)
	{
		int i, j;
		// initial cavity index
		icavityIdx = icsindex;

		// add to tet and old tet list
		j = (searchtet.ver & 3);
		for (i = 1; i < 4; i++)
		{
			neightet = d_neighborlist[4 * searchtet.id + (j + i) % 4];
			d_cavetetlist[tetidxfactor * icavityIdx + i - 1] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx + i - 1] = (i == 1) ? -1 : tetidxfactor * icavityIdx + i - 2;
			d_cavetetnext[tetidxfactor * icavityIdx + i - 1] = tetidxfactor * icavityIdx + i;
			d_cavethreadidx[tetidxfactor * icavityIdx + i - 1] = threadId;
		}
		d_cavetethead[threadId] = tetidxfactor * icavityIdx;

		d_caveoldtetlist[icavityIdx] = searchtet;
		d_caveoldtetprev[icavityIdx] = -1;
		d_caveoldtetnext[icavityIdx] = icavityIdx + 1;
		d_caveoldtethead[threadId] = icavityIdx;

		icavityIdx++;
		spintet = d_neighborlist[4 * searchtet.id + j];
		j = (spintet.ver & 3);
		for (i = 1; i < 4; i++)
		{
			neightet = d_neighborlist[4 * spintet.id + (j + i) % 4];
			d_cavetetlist[tetidxfactor * icavityIdx + i - 1] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx + i - 1] = tetidxfactor * icavityIdx + i - 2;
			d_cavetetnext[tetidxfactor * icavityIdx + i - 1] = (i == 3) ? -1 : tetidxfactor * icavityIdx + i;
			d_cavethreadidx[tetidxfactor * icavityIdx + i - 1] = threadId;
		}
		d_cavetettail[threadId] = tetidxfactor * icavityIdx + 2;

		d_caveoldtetlist[icavityIdx] = spintet;
		d_caveoldtetprev[icavityIdx] = icavityIdx - 1;
		d_caveoldtetnext[icavityIdx] = -1;
		d_caveoldtettail[threadId] = icavityIdx;
	}
	else if (loc == INTETRAHEDRON || loc == OUTSIDE)
	{
		int i;
		// initial cavity index
		icavityIdx = icsindex;

		// add to tet and old tet list
		for (i = 0; i < 4; i++)
		{
			neightet = d_neighborlist[4 * searchtet.id + i];
			d_cavetetlist[tetidxfactor * icavityIdx + i] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx + i] = (i == 0) ? -1 : tetidxfactor * icavityIdx + i - 1;
			d_cavetetnext[tetidxfactor * icavityIdx + i] = (i == 3) ? -1 : tetidxfactor * icavityIdx + i + 1;
			d_cavethreadidx[tetidxfactor * icavityIdx + i] = threadId;
		}
		d_cavetethead[threadId] = tetidxfactor * icavityIdx;
		d_cavetettail[threadId] = tetidxfactor * icavityIdx + 3;

		d_caveoldtetlist[icavityIdx] = searchtet;
		d_caveoldtetprev[icavityIdx] = -1;
		d_caveoldtetnext[icavityIdx] = -1;
		d_caveoldtethead[threadId] = icavityIdx;
		d_caveoldtettail[threadId] = icavityIdx;
	}
}

__global__ void kernelInitCavityLinklist(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_initialcavityindices,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_cavetetlist,
	int* d_cavetetidx,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	int eleidx = d_insertidxlist[threadId];
	if (threadmarker == 1)
	{
		if (d_tristatus[eleidx].isCavityReuse())
		{
			return;
		}
	}
	else if (threadmarker == 2)
	{
		if (d_tetstatus[eleidx].isCavityReuse())
		{
			return;
		}
	}

	tethandle searchtet = d_searchtet[threadId];
	tethandle spintet, neightet;
	locateresult loc = d_pointlocation[threadId];

	// cavities
	int icsindex = d_initialcavityindices[threadId];
	int count = 0;
	int prev = -1;
	int icavityIdx;
	int tetidxfactor = 4;

	if (loc == ONEDGE)
	{
		spintet = searchtet;
		while (1) {
			// initial cavity index
			icavityIdx = icsindex + count;

			// add to tet list
			cudamesh_eorgoppo(spintet, neightet);
			neightet = d_neighborlist[4 * neightet.id + (neightet.ver & 3)];
			neightet.ver = raw_epivot[neightet.ver];
			d_cavetetlist[tetidxfactor * icavityIdx] = neightet;
			d_cavetetidx[tetidxfactor * icavityIdx] = threadId;

			cudamesh_edestoppo(spintet, neightet);
			neightet = d_neighborlist[4 * neightet.id + (neightet.ver & 3)];
			neightet.ver = raw_epivot[neightet.ver];
			d_cavetetlist[tetidxfactor * icavityIdx + 1] = neightet;
			d_cavetetidx[tetidxfactor * icavityIdx + 1] = threadId;

			// add to old tet list
			d_caveoldtetlist[icavityIdx] = spintet; // current tet
			d_caveoldtetidx[icavityIdx] = threadId;

			// next iteration
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id)
			{
				break;
			}
			count++;
		} // while (1)
	}
	else if (loc == ONFACE)
	{
		int i, j;
		// initial cavity index
		icavityIdx = icsindex;

		// add to tet and old tet list
		j = (searchtet.ver & 3);
		for (i = 1; i < 4; i++)
		{
			neightet = d_neighborlist[4 * searchtet.id + (j + i) % 4];
			d_cavetetlist[tetidxfactor * icavityIdx + i - 1] = neightet;
			d_cavetetidx[tetidxfactor * icavityIdx + i - 1] = threadId;
		}

		d_caveoldtetlist[icavityIdx] = searchtet;
		d_caveoldtetidx[icavityIdx] = threadId;

		icavityIdx++;
		spintet = d_neighborlist[4 * searchtet.id + j];
		j = (spintet.ver & 3);
		for (i = 1; i < 4; i++)
		{
			neightet = d_neighborlist[4 * spintet.id + (j + i) % 4];
			d_cavetetlist[tetidxfactor * icavityIdx + i - 1] = neightet;
			d_cavetetidx[tetidxfactor * icavityIdx + i - 1] = threadId;
		}

		d_caveoldtetlist[icavityIdx] = spintet;
		d_caveoldtetidx[icavityIdx] = threadId;
	}
	else if (loc == INTETRAHEDRON || loc == OUTSIDE)
	{
		int i;
		// initial cavity index
		icavityIdx = icsindex;

		// add to tet and old tet list
		for (i = 0; i < 4; i++)
		{
			neightet = d_neighborlist[4 * searchtet.id + i];
			d_cavetetlist[tetidxfactor * icavityIdx + i] = neightet;
			d_cavetetidx[tetidxfactor * icavityIdx + i] = threadId;
		}

		d_caveoldtetlist[icavityIdx] = searchtet;
		d_caveoldtetidx[icavityIdx] = threadId;
	}
}

__device__ int elementId2threadId(
	int* d_insertidxlist,
	int eleidx,
	int eletype,
	int numofbadsubface,
	int numofbadelement
)
{
	int low, high;
	if (eletype == 0) // subface
	{
		low = 0;
		high = numofbadsubface - 1;
	}
	else if (eletype == 1) // tet
	{
		low = numofbadsubface;
		high = numofbadelement - 1;
	}

	int middle, val;
	while (high >= low)
	{
		middle = (low + high) / 2;
		val = d_insertidxlist[middle];
		if (val == eleidx)
			return middle;
		else if(val < eleidx)
			low = middle + 1;
		else
			high = middle - 1;
	}
	
	return -1;
}

__global__ void kernelCheckRecordOldtet(
	tethandle* d_recordoldtetlist,
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int numofbadsubface,
	int numofbadelement,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	tethandle checktet = d_recordoldtetlist[pos];
	int eleidx = d_recordoldtetidx[pos];

	if (checktet.id < 0)
		printf("kernelCheckRecordOldtet: checktet.id = %d\n", checktet.id);

	if (eleidx < 0)
		printf("kernelCheckRecordOldtet: eleidx = %d\n", eleidx);

	// use binary search to find its threadId
	int threadId = elementId2threadId(d_insertidxlist,
		eleidx, checktet.ver, numofbadsubface, numofbadelement);

	if(threadId < -1 || threadId >= numofbadelement)
		printf("kernelCheckRecordOldtet: threadId = %d\n", threadId);
	
	// check if the element is deleted or not
	if (checktet.ver == 0) // subface
	{
		if (d_tristatus[eleidx].isEmpty() || !d_tristatus[eleidx].isCavityReuse())
		{
			d_recordoldtetidx[pos] = -1;
			if (threadId != -1)
				d_threadmarker[threadId] = -1;
			return;
		}
	}
	else if (checktet.ver == 1) // tet
	{
		if (d_tetstatus[eleidx].isEmpty() || !d_tetstatus[eleidx].isCavityReuse())
		{
			d_recordoldtetidx[pos] = -1;
			if (threadId != -1)
				d_threadmarker[threadId] = -1;
			return;
		}
	}

	// check if the tetrahedron is deleted or not
	if (d_tetstatus[checktet.id].isEmpty())
	{
		d_recordoldtetidx[pos] = -1;
		return;
	}

	if (threadId == -1) // this element is not going to be inserted in this iteration
		return;

	if (d_threadmarker[threadId] < 0) // lost already
		return;

	// this element is going to be inserted, check if all tetrahedra are still in its cavity
	REAL* insertpt = d_insertptlist + 3 * threadId;
	int old;
	uint64 marker, oldmarker;
	// here we use threadId + 1, 0 thread index is reserved for default marker
	marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);

	bool enqflag = false;
	double sign, ori;
	// Get four endpoints of cavetet
	REAL *pts[4], wts[4];
	int idx[4];
	for (int i = 0; i < 4; i++)
	{
		idx[i] = d_tetlist[4 * checktet.id + i];
		if (idx[i] != -1)
		{
			pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
			wts[i] = d_weightlist[idx[i]];
		}
		else
			pts[i] = NULL;
	}
	// Test if cavetet is included in the (enlarged) cavity
	if (idx[3] != -1)
	{
		sign = cudamesh_orient4d_s(pts[0], pts[1], pts[2], pts[3], insertpt,
			idx[0], idx[1], idx[2], idx[3], MAXINT,
			wts[0], wts[1], wts[2], wts[3], 0.0);
		enqflag = (sign < 0.0);
	}
	else // a hull tet
	{
		// Test if this hull face visible by the new point. 
		ori = cuda_orient3d(pts[0], pts[1], pts[2], insertpt);
		if (ori < -EPSILON2) {
			// A visible hull face. 
			// Include it in the cavity. The convex hull will be enlarged.
			enqflag = true;
		}
		else if (ori <= EPSILON2 && ori >= -EPSILON2)
		{
			// A coplanar hull face. We need to test if this hull face is
			//   Delaunay or not. We test if the adjacent tet (not faked)
			//   of this hull face is Delaunay or not.
			tethandle neineitet = d_neighborlist[4 * checktet.id + 3];
			if (d_tetmarker[neineitet.id] != marker) // need to check
			{
				// Get four endpoints of neineitet
				for (int i = 0; i < 4; i++)
				{
					idx[i] = d_tetlist[4 * neineitet.id + i];
					if (idx[i] != -1)
					{
						pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
						wts[i] = d_weightlist[idx[i]];
					}
					else
						pts[i] = NULL;
				}
				//assert(idx[3] != -1);
				if (idx[3] == -1)
				{
					enqflag = false;
				}
				else
				{
					sign = cudamesh_orient4d_s(pts[0], pts[1], pts[2], pts[3], insertpt,
						idx[0], idx[1], idx[2], idx[3], MAXINT,
						wts[0], wts[1], wts[2], wts[3], 0.0);
					enqflag = (sign < 0.0);
				}
			}
			else
			{
				//The adjacent tet is non-Delaunay. The hull face is non-
				//  Delaunay as well. Include it in the cavity.
				enqflag = true;
			}

		}
	}
	// Count size
	if (enqflag)
	{
		oldmarker = atomicMax(d_tetmarker + checktet.id, marker);
		if (marker > oldmarker) // I winned
		{
			old = cudamesh_getUInt64PriorityIndex(oldmarker);
			if (old != 0)
			{
				d_threadmarker[old - 1] = -1;
				d_initialcavitysize[old - 1] = 0;
			}
		}
		else if (marker < oldmarker) // I lost
		{
			d_threadmarker[threadId] = -1;
		}
		d_recordoldtetidx[pos] = -(threadId + 2);
	}
	else
	{
		d_recordoldtetidx[pos] = -1;
	}
}

__global__ void kernelKeepRecordOldtet(
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_recordoldtetidx[pos];
	if (threadId >= -1) // invalid or do not participate
		return;

	threadId = -threadId - 2;
	if (d_threadmarker[threadId] < 0)
	{
		d_recordoldtetidx[pos] = d_insertidxlist[threadId];
	}
	//else
	//	printf("kernelKeepRecordOldtet: %d\n", threadId);
}

__global__ void kernelSetReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int oldcaveoldtetsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos + oldcaveoldtetsize];
	threadId = -threadId - 2;
	d_caveoldtetidx[pos + oldcaveoldtetsize] = threadId;
}

__global__ void kernelCheckCavetetFromReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_neighborlist,
	int* d_cavetetexpandsize,
	uint64* d_tetmarker,
	int oldcaveoldtetsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos + oldcaveoldtetsize];

	int cavetetexpandsize = 0;
	uint64 marker;
	int ownerId;
	bool todo;
	tethandle cavetet, neightet;

	cavetet = d_caveoldtetlist[pos + oldcaveoldtetsize];
	for (int j = 0; j < 4; j++)
	{
		// check neighbor
		cavetet.ver = j;
		cudamesh_fsym(cavetet, neightet, d_neighborlist);
		marker = d_tetmarker[neightet.id];
		ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;

		if (ownerId != threadId) // boundary face
		{
			cavetetexpandsize++;
		}
	}
	d_cavetetexpandsize[pos] = cavetetexpandsize;
}

__global__ void kernelAppendCavetetFromReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_cavetetlist,
	int* d_cavetetidx,
	tethandle* d_neighborlist,
	int* d_cavetetexpandindices,
	uint64* d_tetmarker,
	int oldcaveoldtetsize,
	int oldcavetetsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos + oldcaveoldtetsize];

	int startindex = d_cavetetexpandindices[pos];
	uint64 marker;
	int ownerId;
	tethandle cavetet, neightet;

	cavetet = d_caveoldtetlist[pos + oldcaveoldtetsize];
	for (int j = 0; j < 4; j++)
	{
		// check neighbor
		cavetet.ver = j;
		cudamesh_fsym(cavetet, neightet, d_neighborlist);
		marker = d_tetmarker[neightet.id];
		ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;

		if (ownerId != threadId) // boundary face
		{
			d_cavetetlist[oldcavetetsize + startindex] = neightet;
			d_cavetetidx[oldcavetetsize + startindex] = threadId;
			startindex++;
		}
	}
}

__global__ void kernelLargeCavityCheck(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_cavethreadidx,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId != -1) // owners of large cavities
	{
		int threadmarker = d_threadmarker[threadId];
		if (threadmarker != -1)
		{
			int eleidx = d_insertidxlist[threadId];
			if (threadmarker == 1)
			{
				d_tristatus[eleidx].setUnsplittable(true);
				d_tristatus[eleidx].setBad(false);
			}
			else if (threadmarker == 2)
			{
				d_tetstatus[eleidx].setUnsplittable(true);
				d_tetstatus[eleidx].setBad(false);
			}
			d_threadmarker[threadId] = -1;
		}
	}
}

__global__ void kernelLargeCavityCheck(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_cavetetidx,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetidx[pos + cavetetcurstartindex];
	if (threadId != -1) // owners of large cavities
	{
		int threadmarker = d_threadmarker[threadId];
		if (threadmarker != -1)
		{
			int eleidx = d_insertidxlist[threadId];
			if (threadmarker == 1)
			{
				d_tristatus[eleidx].setUnsplittable(true);
				d_tristatus[eleidx].setBad(false);
			}
			else if (threadmarker == 2)
			{
				d_tetstatus[eleidx].setUnsplittable(true);
				d_tetstatus[eleidx].setBad(false);
			}
			d_threadmarker[threadId] = -1;
		}
	}
}

__global__ void kernelMarkCavityReuse(
	int* d_insertidxlist,
	int* d_cavetetidx,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetidx[pos + cavetetcurstartindex];
	if (threadId != -1) // owners of large cavities
	{
		int threadmarker = d_threadmarker[threadId];
		if (threadmarker != -1)
		{
			int eleidx = d_insertidxlist[threadId];
			if (threadmarker == 1 || threadmarker == -2)
			{
				d_tristatus[eleidx].setCavityReuse(true);
				d_threadmarker[threadId] = -2;
			}
			else if (threadmarker == 2 || threadmarker == -3)
			{
				d_tetstatus[eleidx].setCavityReuse(true);
				d_threadmarker[threadId] = -3;
			}
		}
	}
}

__global__ void kernelResetCavityReuse(
	int* d_insertidxlist,
	int* d_threadlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int eleidx = d_insertidxlist[threadId];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 1)
	{
		if (d_tristatus[eleidx].isCavityReuse())
			d_tristatus[eleidx].setCavityReuse(false);
	}
	else if (threadmarker == 2)
	{
		if (d_tetstatus[eleidx].isCavityReuse())
			d_tetstatus[eleidx].setCavityReuse(false);
	}
}

__global__ void kernelMarkOldtetlist(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_insertidxlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == -1)
		return;

	int eleidx = d_insertidxlist[threadId];
	tethandle checktet;
	if (threadmarker == -2) // a subface whose cavity to reuse
	{
		//if (d_tristatus[eleidx].isCavityReuse())
		{
			checktet = d_caveoldtetlist[pos];
			checktet.id = -(checktet.id + 1);
			checktet.ver = 0; // indicate it is a subface
			d_caveoldtetlist[pos] = checktet;
		}
	}
	else if (threadmarker == -3) // a tetrahedron whose cavity to reuse
	{
		//if (d_tetstatus[eleidx].isCavityReuse())
		{
			checktet = d_caveoldtetlist[pos];
			checktet.id = -(checktet.id + 1);
			checktet.ver = 1; // indicate it is a tet
			d_caveoldtetlist[pos] = checktet;
		}
	}
}

__global__ void kernelSetRecordOldtet(
	tethandle* d_recordoldtetlist,
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	int oldrecordsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	tethandle checktet = d_recordoldtetlist[pos + oldrecordsize];
	checktet.id = -checktet.id - 1;
	d_recordoldtetlist[pos + oldrecordsize] = checktet;

	int threadId = d_recordoldtetidx[pos + oldrecordsize];
	int eleidx = d_insertidxlist[threadId];
	d_recordoldtetidx[pos + oldrecordsize] = eleidx;
}

__global__ void kernelMarkLargeCavityAsLoser(
	int* d_cavetetidx,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetidx[pos + cavetetcurstartindex];
	if (threadId != -1) // owners of large cavities
	{
		int threadmarker = d_threadmarker[threadId];
		if (threadmarker != -1)
		{
			d_threadmarker[threadId] = -1;
		}
	}
}

__global__ void kernelCavityExpandingCheck(
	int* d_cavethreadidx,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL* d_insertptlist,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int tetexpandsize = 0;
	int oldtetexpandsize = 0;
	int bdryexpandsize = 0;

	int threadId = d_cavethreadidx[pos];

	if (threadId != -1) // threadId is -1 in the unused slot
	{
		REAL* insertpt = d_insertptlist + 3 * threadId;
		int cur = cavetetcurstartindex + pos;
		tethandle cavetet = d_cavetetlist[cur];

		if (d_threadmarker[threadId] != -1) // avoid to expand loser
		{
			uint64 marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);
			if (d_tetmarker[cavetet.id] != marker) // need to check
			{
				bool enqflag = false;
				double sign, ori;
				// Get four endpoints of cavetet
				REAL *pts[4], wts[4];
				int idx[4];
				for (int i = 0; i < 4; i++)
				{
					idx[i] = d_tetlist[4 * cavetet.id + i];
					if (idx[i] != -1)
					{
						pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
						wts[i] = d_weightlist[idx[i]];
					}
					else
						pts[i] = NULL;
				}
				// Test if cavetet is included in the (enlarged) cavity
				if (idx[3] != -1)
				{
					sign = cudamesh_orient4d_s(pts[0], pts[1], pts[2], pts[3], insertpt,
						idx[0], idx[1], idx[2], idx[3], MAXINT,
						wts[0], wts[1], wts[2], wts[3], 0.0);
					enqflag = (sign < 0.0);
				}
				else // a hull tet
				{
					// Test if this hull face visible by the new point. 
					ori = cuda_orient3d(pts[0], pts[1], pts[2], insertpt);
					if (ori < -EPSILON2) {
						// A visible hull face. 
						// Include it in the cavity. The convex hull will be enlarged.
						enqflag = true;
					}
					else if (ori <= EPSILON2 && ori >= -EPSILON2)
					{
						// A coplanar hull face. We need to test if this hull face is
						//   Delaunay or not. We test if the adjacent tet (not faked)
						//   of this hull face is Delaunay or not.
						tethandle neineitet = d_neighborlist[4 * cavetet.id + 3];
						if (d_tetmarker[neineitet.id] != marker) // need to check
						{
							// Get four endpoints of neineitet
							for (int i = 0; i < 4; i++)
							{
								idx[i] = d_tetlist[4 * neineitet.id + i];
								if (idx[i] != -1)
								{
									pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
									wts[i] = d_weightlist[idx[i]];
								}
								else
									pts[i] = NULL;
							}
							//assert(idx[3] != -1);
							if (idx[3] == -1)
							{
								enqflag = false;
								//printf("Debug: Tet #%d - %d, %d, %d, %d - %d, %d, %d, %d, Tet #%d - %d, %d, %d, %d - %d, %d, %d, %d\n",
								//	cavetet.id,
								//	d_tetlist[4 * cavetet.id + 0], d_tetlist[4 * cavetet.id + 1],
								//	d_tetlist[4 * cavetet.id + 2], d_tetlist[4 * cavetet.id + 3],
								//	d_neighborlist[4 * cavetet.id + 0].id, d_neighborlist[4 * cavetet.id + 1].id,
								//	d_neighborlist[4 * cavetet.id + 2].id, d_neighborlist[4 * cavetet.id + 3].id,
								//	neineitet.id,
								//	d_tetlist[4 * neineitet.id + 0], d_tetlist[4 * neineitet.id + 1],
								//	d_tetlist[4 * neineitet.id + 2], d_tetlist[4 * neineitet.id + 3],
								//	d_neighborlist[4 * neineitet.id + 0].id, d_neighborlist[4 * neineitet.id + 1].id,
								//	d_neighborlist[4 * neineitet.id + 2].id, d_neighborlist[4 * neineitet.id + 3].id);
							}
							else
							{
								sign = cudamesh_orient4d_s(pts[0], pts[1], pts[2], pts[3], insertpt,
									idx[0], idx[1], idx[2], idx[3], MAXINT,
									wts[0], wts[1], wts[2], wts[3], 0.0);
								enqflag = (sign < 0.0);
							}
						}
						else
						{
							//The adjacent tet is non-Delaunay. The hull face is non-
							//  Delaunay as well. Include it in the cavity.
							enqflag = true;
						}

					}
				}
				// Count size
				if (enqflag)
				{
					uint64 oldmarker = atomicMax(d_tetmarker + cavetet.id, marker);
					if (marker > oldmarker) // I winned
					{
						tetexpandsize = 3;
						oldtetexpandsize = 1;
						int old = cudamesh_getUInt64PriorityIndex(oldmarker);
						if (old != 0)
						{
							d_threadmarker[old - 1] = -1;
						}
					}
					else if (marker < oldmarker) // I lost
					{
						d_threadmarker[threadId] = -1;
					}
				}
				else
				{
					bdryexpandsize = 1;
				}
			}
		}
	}

	d_cavetetexpandsize[pos] = tetexpandsize;
	d_caveoldtetexpandsize[pos] = oldtetexpandsize;
	d_cavebdryexpandsize[pos] = bdryexpandsize;
}

__global__ void kernelCavityExpandingCheck(
	int* d_cavetetidx,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL* d_insertptlist,
	tethandle* d_cavetetlist,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int tetexpandsize = 0;
	int oldtetexpandsize = 0;
	int bdryexpandsize = 0;

	int cur = cavetetcurstartindex + pos;
	int threadId = d_cavetetidx[cur];

	if (threadId != -1) // threadId is -1 in the unused slot
	{
		REAL* insertpt = d_insertptlist + 3 * threadId;
		tethandle cavetet = d_cavetetlist[cur];

		if (d_threadmarker[threadId] != -1) // avoid to expand loser
		{
			uint64 marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);
			if (d_tetmarker[cavetet.id] != marker) // need to check
			{
				bool enqflag = false;
				double sign, ori;
				// Get four endpoints of cavetet
				REAL *pts[4], wts[4];
				int idx[4];
				for (int i = 0; i < 4; i++)
				{
					idx[i] = d_tetlist[4 * cavetet.id + i];
					if (idx[i] != -1)
					{
						pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
						wts[i] = d_weightlist[idx[i]];
					}
					else
						pts[i] = NULL;
				}
				// Test if cavetet is included in the (enlarged) cavity
				if (idx[3] != -1)
				{
					sign = cudamesh_orient4d_s(pts[0], pts[1], pts[2], pts[3], insertpt,
						idx[0], idx[1], idx[2], idx[3], MAXINT,
						wts[0], wts[1], wts[2], wts[3], 0.0);
					enqflag = (sign < 0.0);
				}
				else // a hull tet
				{
					// Test if this hull face visible by the new point. 
					ori = cuda_orient3d(pts[0], pts[1], pts[2], insertpt);
					if (ori < -EPSILON2) {
						// A visible hull face. 
						// Include it in the cavity. The convex hull will be enlarged.
						enqflag = true;
					}
					else if (ori <= EPSILON2 && ori >= -EPSILON2)
					{
						// A coplanar hull face. We need to test if this hull face is
						//   Delaunay or not. We test if the adjacent tet (not faked)
						//   of this hull face is Delaunay or not.
						tethandle neineitet = d_neighborlist[4 * cavetet.id + 3];
						if (d_tetmarker[neineitet.id] != marker) // need to check
						{
							// Get four endpoints of neineitet
							for (int i = 0; i < 4; i++)
							{
								idx[i] = d_tetlist[4 * neineitet.id + i];
								if (idx[i] != -1)
								{
									pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
									wts[i] = d_weightlist[idx[i]];
								}
								else
									pts[i] = NULL;
							}
							//assert(idx[3] != -1);
							if (idx[3] == -1)
							{
								enqflag = false;
								//printf("Debug: Tet #%d - %d, %d, %d, %d - %d, %d, %d, %d, Tet #%d - %d, %d, %d, %d - %d, %d, %d, %d\n",
								//	cavetet.id,
								//	d_tetlist[4 * cavetet.id + 0], d_tetlist[4 * cavetet.id + 1],
								//	d_tetlist[4 * cavetet.id + 2], d_tetlist[4 * cavetet.id + 3],
								//	d_neighborlist[4 * cavetet.id + 0].id, d_neighborlist[4 * cavetet.id + 1].id,
								//	d_neighborlist[4 * cavetet.id + 2].id, d_neighborlist[4 * cavetet.id + 3].id,
								//	neineitet.id,
								//	d_tetlist[4 * neineitet.id + 0], d_tetlist[4 * neineitet.id + 1],
								//	d_tetlist[4 * neineitet.id + 2], d_tetlist[4 * neineitet.id + 3],
								//	d_neighborlist[4 * neineitet.id + 0].id, d_neighborlist[4 * neineitet.id + 1].id,
								//	d_neighborlist[4 * neineitet.id + 2].id, d_neighborlist[4 * neineitet.id + 3].id);
							}
							else
							{
								sign = cudamesh_orient4d_s(pts[0], pts[1], pts[2], pts[3], insertpt,
									idx[0], idx[1], idx[2], idx[3], MAXINT,
									wts[0], wts[1], wts[2], wts[3], 0.0);
								enqflag = (sign < 0.0);
							}
						}
						else
						{
							//The adjacent tet is non-Delaunay. The hull face is non-
							//  Delaunay as well. Include it in the cavity.
							enqflag = true;
						}

					}
				}
				// Count size
				if (enqflag)
				{
					uint64 oldmarker = atomicMax(d_tetmarker + cavetet.id, marker);
					if (marker > oldmarker) // I winned
					{
						tetexpandsize = 3;
						oldtetexpandsize = 1;
						int old = cudamesh_getUInt64PriorityIndex(oldmarker);
						if (old != 0)
						{
							d_threadmarker[old - 1] = -1;
						}
					}
					else if (marker < oldmarker) // I lost
					{
						d_threadmarker[threadId] = -1;
					}
				}
				else
				{
					bdryexpandsize = 1;
				}
			}
		}
	}

	d_cavetetexpandsize[pos] = tetexpandsize;
	d_caveoldtetexpandsize[pos] = oldtetexpandsize;
	d_cavebdryexpandsize[pos] = bdryexpandsize;
}

__global__ void  kernelCorrectExpandingSize(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId != -1 && d_threadmarker[threadId] == -1)
	{
		d_cavetetexpandsize[pos] = 0;
		d_caveoldtetexpandsize[pos] = 0;
		d_cavebdryexpandsize[pos] = 0;
	}
}

__global__ void  kernelCorrectExpandingSize(
	int* d_cavetetidx,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetidx[pos + cavetetcurstartindex];
	if (threadId != -1 && d_threadmarker[threadId] == -1)
	{
		d_cavetetexpandsize[pos] = 0;
		d_caveoldtetexpandsize[pos] = 0;
		d_cavebdryexpandsize[pos] = 0;
	}
}

__global__ void kernelCavityExpandingSetThreadidx(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int* d_caveoldtetthreadidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId == -1)
		return;

	int eindex;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		for (int j = 0; j < 3; j++) {
			d_cavetetthreadidx[eindex + j] = threadId;
		}
	}

	if (d_caveoldtetexpandsize[pos] != 0)
	{
		eindex = d_caveoldtetexpandindices[pos];
		d_caveoldtetthreadidx[eindex] = threadId;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		d_cavebdrythreadidx[eindex] = threadId;
	}
}

__global__ void kernelCavityExpandingMarkAndAppend(
	int* d_cavethreadidx,
	tethandle* d_neighborlist,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int cavetetstartindex,
	int cavetetexpandsize,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int* d_caveoldtetthreadidx,
	int caveoldtetstartindex,
	int caveoldtetexpandsize,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int cavebdrystartindex,
	int cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId == -1)
		return;
	if (d_threadmarker[threadId] == -1)
		return;

	int cur = cavetetcurstartindex + pos;
	tethandle cavetet = d_cavetetlist[cur];

	int sindex, eindex, prev;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		sindex = cavetetstartindex + eindex;

		// Append cavetetlist and mark current tet
		int k = (cavetet.ver & 3); // The current face number
		tethandle neightet;
		int newid;

		if (eindex == 0 || d_cavetetthreadidx[eindex - 1] != threadId)
		{
			prev = d_cavetettail[threadId];
			d_cavetetnext[prev] = sindex; // prev must not be -1
		}
		else
			prev = sindex - 1;

		for (int j = 1; j < 4; j++) {
			neightet = d_neighborlist[4 * cavetet.id + (j + k) % 4];
			newid = sindex + j - 1;
			d_cavetetlist[newid] = neightet;
			d_cavetetprev[newid] = prev;
			d_cavetetnext[newid] = newid + 1; // set to next one first
			prev = newid;
		}

		if (eindex + 2 == cavetetexpandsize - 1 || d_cavetetthreadidx[eindex + 3] != threadId)
			d_cavetetnext[newid] = -1;
	}

	if (d_caveoldtetexpandsize[pos] != 0)
	{
		eindex = d_caveoldtetexpandindices[pos];
		sindex = caveoldtetstartindex + eindex;

		if (eindex == 0 || d_caveoldtetthreadidx[eindex - 1] != threadId)
		{
			prev = d_caveoldtettail[threadId];
			d_caveoldtetnext[prev] = sindex; // prev must not be -1
		}
		else
			prev = sindex - 1;

		d_caveoldtetlist[sindex] = cavetet;
		d_caveoldtetprev[sindex] = prev;
		d_caveoldtetnext[sindex] = sindex + 1;

		if (eindex == caveoldtetexpandsize - 1 || d_caveoldtetthreadidx[eindex + 1] != threadId)
			d_caveoldtetnext[sindex] = -1;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		sindex = cavebdrystartindex + eindex;

		if (eindex == 0 || d_cavebdrythreadidx[eindex - 1] != threadId)
		{
			prev = d_cavebdrytail[threadId];
			if (prev != -1)
				d_cavebdrynext[prev] = sindex; // prev must not be -1
			if (d_cavebdryhead[threadId] == -1) // initialize cavebdry list header
				d_cavebdryhead[threadId] = sindex;
		}
		else
			prev = sindex - 1;

		cavetet.ver = raw_epivot[cavetet.ver];
		d_cavebdrylist[sindex] = cavetet;
		d_cavebdryprev[sindex] = prev;
		d_cavebdrynext[sindex] = sindex + 1;

		if (eindex == cavebdryexpandsize - 1 || d_cavebdrythreadidx[eindex + 1] != threadId)
			d_cavebdrynext[sindex] = -1;
	}
}

__global__ void kernelCavityExpandingMarkAndAppend(
	int* d_cavetetidx,
	tethandle* d_neighborlist,
	tethandle* d_cavetetlist,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int cavetetstartindex,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int caveoldtetstartindex,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int cur = cavetetcurstartindex + pos;

	int threadId = d_cavetetidx[cur];
	if (threadId == -1)
		return;
	if (d_threadmarker[threadId] == -1)
		return;

	tethandle cavetet = d_cavetetlist[cur];

	int sindex, eindex, prev;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		sindex = cavetetstartindex + eindex;

		// Append cavetetlist and mark current tet
		int k = (cavetet.ver & 3); // The current face number
		tethandle neightet;
		int newid;

		for (int j = 1; j < 4; j++) {
			neightet = d_neighborlist[4 * cavetet.id + (j + k) % 4];
			newid = sindex + j - 1;
			d_cavetetlist[newid] = neightet;
			d_cavetetidx[newid] = threadId;
		}
	}

	if (d_caveoldtetexpandsize[pos] != 0)
	{
		eindex = d_caveoldtetexpandindices[pos];
		sindex = caveoldtetstartindex + eindex;

		d_caveoldtetlist[sindex] = cavetet;
		d_caveoldtetidx[sindex] = threadId;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		sindex = cavebdrystartindex + eindex;

		cavetet.ver = raw_epivot[cavetet.ver];
		d_cavebdrylist[sindex] = cavetet;
		d_cavebdryidx[sindex] = threadId;
	}
}

__global__ void kernelCavityExpandingUpdateListTails(
	int* d_cavethreadidx,
	int* d_cavetetnext,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int cavetetstartindex,
	int* d_caveoldtetnext,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int caveoldtetstartindex,
	int* d_cavebdrynext,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_threadmarker,
	int cavebdrystartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId == -1)
		return;
	if (d_threadmarker[threadId] == -1)
		return;

	int sindex, eindex, prev;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		sindex = cavetetstartindex + eindex + 2;
		if (d_cavetetnext[sindex] == -1)
			d_cavetettail[threadId] = sindex;
	}

	if (d_caveoldtetexpandsize[pos] != 0)
	{
		eindex = d_caveoldtetexpandindices[pos];
		sindex = caveoldtetstartindex + eindex;
		if (d_caveoldtetnext[sindex] == -1)
			d_caveoldtettail[threadId] = sindex;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		sindex = cavebdrystartindex + eindex;
		if (d_cavebdrynext[sindex] == -1)
			d_cavebdrytail[threadId] = sindex;
	}
}

__global__ void kernelMarkAdjacentCavitiesAndCountSubfaces(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	int* d_cavetetshsize,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	//if (threadId < 0 || threadId > 500000)
	//	printf("kernelMarkAdjacentCavitiesAndCountSubfaces: threadId = %d\n", threadId);
	if (d_threadmarker[threadId] == -1)
		return;

	int cavetetshsize = 0;
	uint64 marker;
	int ownerId;
	bool todo;
	tethandle cavetet, neightet;
	trihandle checksh;

	cavetet = d_caveoldtetlist[pos];
	if(cavetet.id < 0)
		printf("kernelMarkAdjacentCavitiesAndCountSubfaces: cavetet.id = %d, threadId = %d\n", cavetet.id, threadId);
	for (int j = 0; j < 4; j++)
	{
		// check neighbor
		cavetet.ver = j;
		cudamesh_fsym(cavetet, neightet, d_neighborlist);
		marker = d_tetmarker[neightet.id];
		ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;

		todo = false;
		if (ownerId != threadId) // boundary face
		{
			if (ownerId != -1 && d_threadmarker[ownerId] >= 0
				&& (d_priority[threadId] < d_priority[ownerId] ||
					(d_priority[threadId] == d_priority[ownerId] && threadId > ownerId))) // I lost
			{
				d_threadmarker[threadId] = -1;
				d_cavetetshsize[pos] = 0;
				return;
			}
			else
				todo = true;
		}
		else // interior face
		{
			if (cavetet.id < neightet.id) // I should check
				todo = true;
		}
		// counting
		if (todo)
		{
			checksh = d_tet2trilist[4 * cavetet.id + j];
			if (checksh.id != -1)
			{
				cavetetshsize++;
			}
		}
	}
	d_cavetetshsize[pos] = cavetetshsize;
}

__global__ void kernelCorrectSubfaceSizes(
	int* d_caveoldtetidx,
	int* d_cavetetshsize,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	if (d_threadmarker[threadId] == -1)
		d_cavetetshsize[pos] = 0;
}

__global__ void kernelAppendCavitySubfaces(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	int* d_cavetetshsize,
	int* d_cavetetshindices,
	uint64* d_tetmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	if (d_cavetetshsize[pos] == 0)
		return;

	int sindex = d_cavetetshindices[pos];

	uint64 marker;
	int ownerId;
	bool todo;
	tethandle cavetet, neightet;
	trihandle checksh;
	int index, count = 0;

	cavetet = d_caveoldtetlist[pos];
	for (int j = 0; j < 4; j++)
	{
		// check neighbor
		cavetet.ver = j;
		cudamesh_fsym(cavetet, neightet, d_neighborlist);
		marker = d_tetmarker[neightet.id];
		ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;

		todo = false;
		if (ownerId != threadId) // boundary face
			todo = true;
		else // interior face
		{
			if (cavetet.id < neightet.id) // I should check
				todo = true;
		}
		// append
		if (todo)
		{
			checksh = d_tet2trilist[4 * cavetet.id + j];
			if (checksh.id != -1)
			{
				index = sindex + count;
				d_cavetetshlist[index] = checksh;
				d_cavetetshidx[index] = threadId;
				count++;
			}
		}
	}
}

__global__ void kernelCheckSubfaceEncroachment_Phase1(
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_encroachmentmarker,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 2) // not a tetrahedron
		return;

	REAL *insertpt = d_insertptlist + 3 * threadId;
	REAL *p[3], w[3], center[3];
	trihandle checksh;
	tethandle neightet;

	bool encroached, internalfacet;
	encroached = false;
	internalfacet = false;
	// get subface
	checksh = d_cavetetshlist[pos];
	for (int j = 0; j < 3; j++)
	{
		int pidx = d_trifacelist[3 * checksh.id + j];
		p[j] = cudamesh_id2pointlist(pidx, d_pointlist);
		w[j] = d_weightlist[pidx];
	}
	center[0] = d_trifacecent[3 * checksh.id + 0];
	center[1] = d_trifacecent[3 * checksh.id + 1];
	center[2] = d_trifacecent[3 * checksh.id + 2];
	// test if encroached
	cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
	if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
	{
		cudamesh_sesymself(checksh);
		cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
			internalfacet = true;
	}
	if (internalfacet)
		encroached = true; // interior subface always encroached
	else // boundary subface
	{
		if (cudamesh_power_distance(insertpt, center, 0.0, 0.0) <=
			cudamesh_power_distance(p[0], center, w[0], 0.0))
			encroached = true;
	}
	// test if splittable
	if (encroached) // encroached
	{
		if (!d_tristatus[checksh.id].isUnsplittable())
		{
			if (cudamesh_is_encroached_facet_splittable(p[0], p[1], p[2], w[0], w[1], w[2]))
			{
				d_encroachmentmarker[pos] = 1;
				int oldmarker = atomicMin(d_threadmarker + threadId, -1);
				if(oldmarker >= 0)
					d_tristatus[checksh.id].setBad(true);
			}
			else
			{
				d_tristatus[checksh.id].setUnsplittable(true);
				d_tristatus[checksh.id].setBad(false);
			}
		}
	}
}

__global__ void kernelCheckSubfaceEncroachment_Phase2(
	int* d_cavetetshidx,
	int* d_insertidxlist,
	tetstatus* d_tetstatus,
	int* d_encroachmentmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 2) // not a tetrahedron
		return;

	int encmarker = d_encroachmentmarker[pos];
	if (encmarker && d_threadmarker[threadId] != -1)
	{
		// subfaces encroached are all unsplittable
		int insertidx = d_insertidxlist[threadId];
		d_tetstatus[insertidx].setUnsplittable(true);
		d_tetstatus[insertidx].setBad(false);
		d_threadmarker[threadId] = -1;
	}
}

__global__ void kernelSetCavityThreadIdx(
	int* d_cavethreadidx,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (d_threadmarker[threadId] == -1)
		d_cavethreadidx[pos] = -1;
}

__global__ void kernelSetDuplicateThreadIdx(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;
	if (pos == 0)
		return;

	int left_pos = pos - 1;
	tethandle left_tet = d_cavebdrylist[left_pos];
	int left_idx = d_cavebdryidx[left_pos];
	tethandle tet = d_cavebdrylist[pos];
	int idx = d_cavebdryidx[pos];
	if (left_idx == idx)
	{
		if (left_tet.id == tet.id && left_tet.ver == tet.ver) // duplicate face
		{
			d_cavebdryidx[pos] = -1;
		}
	}
}

__global__ void kernelCountCavitySubfaces(
	int* d_threadlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_cavetetshsize,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int cavetetshsize = 0;

	if (d_threadmarker[threadId] != -1)
	{
		int i = d_caveoldtethead[threadId];
		uint64 marker;
		int ownerId;
		bool todo;
		tethandle cavetet, neightet;
		trihandle checksh;
		while (i != -1)
		{
			cavetet = d_caveoldtetlist[i];
			for (int j = 0; j < 4; j++)
			{
				// check neighbor
				cavetet.ver = j;
				cudamesh_fsym(cavetet, neightet, d_neighborlist);
				marker = d_tetmarker[neightet.id];
				ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;
				
				todo = false;
				if (ownerId != threadId) // boundary face
				{
					if (ownerId != -1 && d_threadmarker[ownerId] >= 0
						&& threadId > ownerId) // I lost
					{
						d_threadmarker[threadId] = -1;
						d_cavetetshsize[pos] = 0;
						return;
					}
					else
						todo = true;
				}
				else // interior face
				{
					if (cavetet.id < neightet.id) // I should check
						todo = true;
				}
				// counting
				if (todo)
				{
					checksh = d_tet2trilist[4 * cavetet.id + j];
					if (checksh.id != -1)
					{
						cavetetshsize++;
					}
				}
			}
			i = d_caveoldtetnext[i];
		}
	}
	d_cavetetshsize[pos] = cavetetshsize;
}

__global__ void kernelAppendCavitySubfaces(
	int* d_threadlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tethandle* d_tri2tetlist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	trihandle* d_cavetetshlist,
	int* d_cavetetshprev,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	int* d_cavetetshtail,
	int* d_cavetetshsize,
	int* d_cavetetshindices,
	uint64* d_tetmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	if (d_cavetetshsize[pos] == 0)
		return;

	int sindex = d_cavetetshindices[pos];
	d_cavetetshhead[threadId] = sindex;

	int i = d_caveoldtethead[threadId];
	uint64 marker;
	int ownerId;
	bool todo;
	tethandle cavetet, neightet;
	trihandle checksh;
	int index, count = 0, prev = -1;
	while (i != -1)
	{
		cavetet = d_caveoldtetlist[i];
		for (int j = 0; j < 4; j++)
		{
			// check neighbor
			cavetet.ver = j;
			cudamesh_fsym(cavetet, neightet, d_neighborlist);
			marker = d_tetmarker[neightet.id];
			ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;

			todo = false;
			if (ownerId != threadId) // boundary face
				todo = true;
			else // interior face
			{
				if (cavetet.id < neightet.id) // I should check
					todo = true;
			}
			// append
			if (todo)
			{
				checksh = d_tet2trilist[4 * cavetet.id + j];
				if (checksh.id != -1)
				{
					index = sindex + count;
					d_cavetetshlist[index] = checksh;
					d_cavetetshprev[index] = prev;
					d_cavetetshnext[index] = -1;
					if (prev != -1)
						d_cavetetshnext[prev] = index;
					count++;
					prev = index;
				}
			}
		}
		i = d_caveoldtetnext[i];
		if (i == -1) // reached the end
		{
			d_cavetetshtail[threadId] = index;
		}
	}
}

__global__ void kernelCheckSubfaceEncroachment(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 2) // not a tetrahedron
		return;

	REAL *insertpt = d_insertptlist + 3 * threadId;
	REAL *p[3], w[3], center[3];
	trihandle checksh;
	tethandle neightet;

	bool flag = false, encroached, internalfacet;
	int i = d_cavetetshhead[threadId];
	while (i != -1)
	{
		encroached = false;
		internalfacet = false;
		// get subface
		checksh = d_cavetetshlist[i];
		for (int j = 0; j < 3; j++)
		{
			int pidx = d_trifacelist[3 * checksh.id + j];
			p[j] = cudamesh_id2pointlist(pidx, d_pointlist);
			w[j] = d_weightlist[pidx];
		}
		center[0] = d_trifacecent[3 * checksh.id + 0];
		center[1] = d_trifacecent[3 * checksh.id + 1];
		center[2] = d_trifacecent[3 * checksh.id + 2];
		// test if encroached
		cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
		{
			cudamesh_sesymself(checksh);
			cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
				internalfacet = true;
		}
		if (internalfacet)
			encroached = true; // interior subface always encroached
		else // boundary subface
		{
			if (cudamesh_power_distance(insertpt, center, 0.0, 0.0) <=
				cudamesh_power_distance(p[0], center, w[0], 0.0))
				encroached = true;
		}
		// test if splittable
		if (encroached) // encroached
		{
			flag = true;
			if (!d_tristatus[checksh.id].isUnsplittable())
			{
				if (cudamesh_is_encroached_facet_splittable(p[0], p[1], p[2], w[0], w[1], w[2]))
				{
					d_tristatus[checksh.id].setBad(true);
					d_threadmarker[threadId] = -1;
					break;
				}
				else
				{
					d_tristatus[checksh.id].setUnsplittable(true);
					d_tristatus[checksh.id].setBad(false);
				}
			}
		}
		i = d_cavetetshnext[i];
	}

	if (flag && d_threadmarker[threadId] != -1)
	{
		// subfaces encroached are all unsplittable
		int insertidx = d_insertidxlist[threadId];
		d_tetstatus[insertidx].setUnsplittable(true);
		d_tetstatus[insertidx].setBad(false);
		d_threadmarker[threadId] = -1;
	}
}

__global__ void kernelValidateRefinementElements(
	int* d_insertidxlist,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	tethandle* d_tri2tetlist,
	int* d_threadlist,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];

	int insertidx = d_insertidxlist[threadId];
	if (threadmarker == 1)
	{
		// check if located tet is marked
		int tetid = d_searchtet[threadId].id;
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[tetid]) != threadId + 1)
		{
			if (!d_tristatus[insertidx].isNew()) // no need to re-compute center
			{
				d_tristatus[insertidx].setUnsplittable(true);
				d_tristatus[insertidx].setBad(false);
			}
			else
			{
				d_tristatus[insertidx].setToCheck(true);
			}
			d_threadmarker[threadId] = -1;
			return;
		}

		// check if subface's neighbors are marked
		tethandle neightet, neineitet;
		trihandle checksh(insertidx, 0);
		cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
		cudamesh_fsym(neightet, neineitet, d_neighborlist);
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) != threadId + 1 &&
			cudamesh_getUInt64PriorityIndex(d_tetmarker[neineitet.id]) != threadId + 1)
		{
			if (!d_tristatus[insertidx].isNew()) // no need to re-compute center
			{
				d_tristatus[insertidx].setUnsplittable(true);
				d_tristatus[insertidx].setBad(false);
			}
			else
			{
				d_tristatus[insertidx].setToCheck(true);
			}
			d_threadmarker[threadId] = -1;
			return;
		}
	}
	else if (threadmarker == 2)
	{
		// check if located tet is marked
		int tetid = d_searchtet[threadId].id;
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[tetid]) != threadId + 1)
		{
			d_tetstatus[insertidx].setUnsplittable(true);
			d_tristatus[insertidx].setBad(false);
			d_threadmarker[threadId] = -1;
			return;
		}
		// check if bad tet itself is marked
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[insertidx]) != threadId + 1)
		{
			d_tetstatus[insertidx].setUnsplittable(true);
			d_tristatus[insertidx].setBad(false);
			d_threadmarker[threadId] = -1;
			return;
		}
	}
}

__global__ void kernelRecomputeTrifaceCenter(
	int* d_insertidxlist,
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_trifacepmt,
	tethandle* d_tri2tetlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_threadlist,
	int* d_threadmarker,
	REAL aabb_diglen,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int subfaceidx = d_insertidxlist[threadId];
	if (!d_tristatus[subfaceidx].isToCheck())
		return;

	d_tristatus[subfaceidx].setNew(false);
	d_tristatus[subfaceidx].setToCheck(false);
	trihandle checksh(subfaceidx, 0);
	tethandle checktet, neightet, tmptet;
	cudamesh_stpivot(checksh, checktet, d_tri2tetlist);
	cudamesh_sesymself(checksh);
	cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
	bool hull, nhull;
	hull = cudamesh_ishulltet(checktet, d_tetlist);
	nhull = cudamesh_ishulltet(neightet, d_tetlist);
	if (hull && nhull)
	{
		d_tristatus[subfaceidx].setUnsplittable(true);
		d_tristatus[subfaceidx].setBad(false);
		return;
	}

	if (hull)
	{
		tmptet = checktet;
		checktet = neightet;
		neightet = tmptet;
		nhull = true;
	}

	int i, j, ret;
	int pi[4];
	REAL *pc[4], w[4], wc[3];
	for (i = 0; i < 4; i++)
	{
		pi[i] = d_tetlist[4 * checktet.id + i];
		pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
		w[i] = d_weightlist[pi[i]];
	}

	if (!cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], pc[3],
		w[0], w[1], w[2], w[3], wc)) // is degenerate tet
	{
		d_tristatus[subfaceidx].setUnsplittable(true);
		d_tristatus[subfaceidx].setBad(false);
		return;
	}

	int npi[4], pmidx;
	REAL *npc[4], nw[4], nwc[3], ipt[3], dir[3], vdir[3], fwc[3], vec1[3], vec2[3], len;
	// calculate the voronoi dual of the facet
	if (!nhull) // the dual is a segment
	{
		for (i = 0; i < 4; i++)
		{
			npi[i] = d_tetlist[4 * neightet.id + i];
			npc[i] = cudamesh_id2pointlist(npi[i], d_pointlist);
			nw[i] = d_weightlist[npi[i]];
		}
		if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], npc[3],
			nw[0], nw[1], nw[2], nw[3], nwc))
		{
			d_tristatus[subfaceidx].setUnsplittable(true);
			d_tristatus[subfaceidx].setBad(false);
			return;
		}
	}
	else // the dual is a ray
	{
		// get boundary face points
		npi[0] = cudamesh_org(checktet, d_tetlist);
		npc[0] = cudamesh_id2pointlist(npi[0], d_pointlist);
		nw[0] = d_weightlist[npi[0]];
		npi[1] = cudamesh_dest(checktet, d_tetlist);
		npc[1] = cudamesh_id2pointlist(npi[1], d_pointlist);
		nw[1] = d_weightlist[npi[1]];
		npi[2] = cudamesh_apex(checktet, d_tetlist);
		npc[2] = cudamesh_id2pointlist(npi[2], d_pointlist);
		nw[2] = d_weightlist[npi[2]];
		// get oppo point
		npi[3] = cudamesh_oppo(checktet, d_tetlist);
		npc[3] = cudamesh_id2pointlist(npi[3], d_pointlist);
		// caculate the weighted center perpendicular vector of boundary face
		if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], nw[0], nw[1], nw[2], fwc))
		{
			// degenerate boundary face
			d_tristatus[subfaceidx].setUnsplittable(true);
			d_tristatus[subfaceidx].setBad(false);
			return;
		}
		cudamesh_raydir(npc[0], npc[1], npc[2], dir);
		vdir[0] = npc[3][0] - fwc[0]; vdir[1] = npc[3][1] - fwc[1]; vdir[2] = npc[3][2] - fwc[2];
		if (dir[0] * vdir[0] + dir[1] * vdir[1] + dir[2] * vdir[2] >= 0.0)
		{
			// degenerate ray
			d_tristatus[subfaceidx].setUnsplittable(true);
			d_tristatus[subfaceidx].setBad(false);
			return;
		}
		// calculate a point outside the bounding box
		len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
		if (len == 0.0)
		{
			d_tristatus[subfaceidx].setUnsplittable(true);
			d_tristatus[subfaceidx].setBad(false);
			return;
		}
		dir[0] /= len; dir[1] /= len; dir[2] /= len;
		nwc[0] = wc[0] + dir[0] * 1.2*aabb_diglen;
		nwc[1] = wc[1] + dir[1] * 1.2*aabb_diglen;
		nwc[2] = wc[2] + dir[2] * 1.2*aabb_diglen;
	}

	ret = cudamesh_compare(wc, nwc);
	if (ret == 0) // degenerate segment
	{
		d_tristatus[subfaceidx].setUnsplittable(true);
		d_tristatus[subfaceidx].setBad(false);
		return;
	}
	else if (ret == 1) // make canonical vector
		cudamesh_swap(wc, nwc);

	bool found = cudamesh_traversal_first_intersection(wc, nwc, d_aabbnodeleft, d_aabbnoderight, d_aabbnodebbs,
		d_aabbpmcoord, d_aabbpmbbs, ipt, pmidx);

	if (found)
	{
		d_trifacecent[3 * subfaceidx + 0] = ipt[0];
		d_trifacecent[3 * subfaceidx + 1] = ipt[1];
		d_trifacecent[3 * subfaceidx + 2] = ipt[2];
		d_trifacepmt[subfaceidx] = pmidx;
	}
	else // this should not happen
	{
		d_tristatus[subfaceidx].setUnsplittable(true);
		d_tristatus[subfaceidx].setBad(false);
		return;
	}
}

__global__ void kernelRecomputeTrifaceCenter(
	int* d_insertidxlist,
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_trifacepmt,
	tethandle* d_tri2tetlist,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_threadlist,
	int* d_threadmarker,
	REAL aabb_diglen,
	int numofsubfaces,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int subfaceidx = d_insertidxlist[threadId];
	if (subfaceidx >= numofsubfaces || !d_tristatus[subfaceidx].isToCheck())
		return;

	d_tristatus[subfaceidx].setNew(false);
	d_tristatus[subfaceidx].setToCheck(false);
	trihandle checksh(subfaceidx, 0);
	tethandle checktet, neightet, tmptet;
	cudamesh_stpivot(checksh, checktet, d_tri2tetlist);
	cudamesh_sesymself(checksh);
	cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
	bool hull, nhull;
	hull = cudamesh_ishulltet(checktet, d_tetlist);
	nhull = cudamesh_ishulltet(neightet, d_tetlist);
	if (hull && nhull)
	{
		d_tristatus[subfaceidx].setUnsplittable(true);
		d_tristatus[subfaceidx].setBad(false);
		return;
	}

	if (hull)
	{
		tmptet = checktet;
		checktet = neightet;
		neightet = tmptet;
		nhull = true;
	}

	int i, j, ret;
	int pi[4];
	REAL *pc[4], w[4], wc[3];
	for (i = 0; i < 4; i++)
	{
		pi[i] = d_tetlist[4 * checktet.id + i];
		pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
		w[i] = d_weightlist[pi[i]];
	}

	if (!cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], pc[3],
		w[0], w[1], w[2], w[3], wc)) // is degenerate tet
	{
		d_tristatus[subfaceidx].setUnsplittable(true);
		d_tristatus[subfaceidx].setBad(false);
		return;
	}

	int npi[4], pmidx;
	REAL *npc[4], nw[4], nwc[3], ipt[3], dir[3], vdir[3], fwc[3], vec1[3], vec2[3], len;
	// calculate the voronoi dual of the facet
	if (!nhull) // the dual is a segment
	{
		for (i = 0; i < 4; i++)
		{
			npi[i] = d_tetlist[4 * neightet.id + i];
			npc[i] = cudamesh_id2pointlist(npi[i], d_pointlist);
			nw[i] = d_weightlist[npi[i]];
		}
		if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], npc[3],
			nw[0], nw[1], nw[2], nw[3], nwc))
		{
			d_tristatus[subfaceidx].setUnsplittable(true);
			d_tristatus[subfaceidx].setBad(false);
			return;
		}
	}
	else // the dual is a ray
	{
		// get boundary face points
		npi[0] = cudamesh_org(checktet, d_tetlist);
		npc[0] = cudamesh_id2pointlist(npi[0], d_pointlist);
		nw[0] = d_weightlist[npi[0]];
		npi[1] = cudamesh_dest(checktet, d_tetlist);
		npc[1] = cudamesh_id2pointlist(npi[1], d_pointlist);
		nw[1] = d_weightlist[npi[1]];
		npi[2] = cudamesh_apex(checktet, d_tetlist);
		npc[2] = cudamesh_id2pointlist(npi[2], d_pointlist);
		nw[2] = d_weightlist[npi[2]];
		// get oppo point
		npi[3] = cudamesh_oppo(checktet, d_tetlist);
		npc[3] = cudamesh_id2pointlist(npi[3], d_pointlist);
		// caculate the weighted center perpendicular vector of boundary face
		if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], nw[0], nw[1], nw[2], fwc))
		{
			// degenerate boundary face
			d_tristatus[subfaceidx].setUnsplittable(true);
			d_tristatus[subfaceidx].setBad(false);
			return;
		}
		cudamesh_raydir(npc[0], npc[1], npc[2], dir);
		vdir[0] = npc[3][0] - fwc[0]; vdir[1] = npc[3][1] - fwc[1]; vdir[2] = npc[3][2] - fwc[2];
		if (dir[0] * vdir[0] + dir[1] * vdir[1] + dir[2] * vdir[2] >= 0.0)
		{
			// degenerate ray
			d_tristatus[subfaceidx].setUnsplittable(true);
			d_tristatus[subfaceidx].setBad(false);
			return;
		}
		// calculate a point outside the bounding box
		len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
		if (len == 0.0)
		{
			d_tristatus[subfaceidx].setUnsplittable(true);
			d_tristatus[subfaceidx].setBad(false);
			return;
		}
		dir[0] /= len; dir[1] /= len; dir[2] /= len;
		nwc[0] = wc[0] + dir[0] * 1.2*aabb_diglen;
		nwc[1] = wc[1] + dir[1] * 1.2*aabb_diglen;
		nwc[2] = wc[2] + dir[2] * 1.2*aabb_diglen;
	}

	ret = cudamesh_compare(wc, nwc);
	if (ret == 0) // degenerate segment
	{
		d_tristatus[subfaceidx].setUnsplittable(true);
		d_tristatus[subfaceidx].setBad(false);
		return;
	}
	else if (ret == 1) // make canonical vector
		cudamesh_swap(wc, nwc);

	bool found = cudamesh_traversal_first_intersection(wc, nwc, d_aabbnodeleft, d_aabbnoderight, d_aabbnodebbs,
		d_aabbpmcoord, d_aabbpmbbs, ipt, pmidx);

	if (found)
	{
		d_trifacecent[3 * subfaceidx + 0] = ipt[0];
		d_trifacecent[3 * subfaceidx + 1] = ipt[1];
		d_trifacecent[3 * subfaceidx + 2] = ipt[2];
		d_trifacepmt[subfaceidx] = pmidx;
	}
	else // this should not happen
	{
		d_tristatus[subfaceidx].setUnsplittable(true);
		d_tristatus[subfaceidx].setBad(false);
		return;
	}
}

__global__ void kernelInsertNewPoints(
	int* d_threadlist,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	int* d_pointpmt,
	int* d_trifacepmt,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];

	int newidx = oldpointsize + pos;
	if (threadmarker == 1)
	{
		d_pointtypelist[newidx] = FACETVERTEX;
		int trifaceidx = d_insertidxlist[threadId];
		d_pointpmt[newidx] = d_trifacepmt[trifaceidx];
	}
	else
		d_pointtypelist[newidx] = VOLVERTEX;

	newidx *= 3;
	REAL* insertpt = d_insertptlist + 3 * threadId;
	d_pointlist[newidx++] = insertpt[0];
	d_pointlist[newidx++] = insertpt[1];
	d_pointlist[newidx++] = insertpt[2];
}

__global__ void kernelInsertNewPoints(
	int* d_threadlist,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	int* d_pointpmt,
	int* d_trifacepmt,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	d_threadpos[threadId] = pos;

	int newidx = oldpointsize + pos;
	if (threadmarker == 1)
	{
		d_pointtypelist[newidx] = FACETVERTEX;
		int trifaceidx = d_insertidxlist[threadId];
		d_pointpmt[newidx] = d_trifacepmt[trifaceidx];
	}
	else
		d_pointtypelist[newidx] = VOLVERTEX;

	newidx *= 3;
	REAL* insertpt = d_insertptlist + 3 * threadId;
	d_pointlist[newidx++] = insertpt[0];
	d_pointlist[newidx++] = insertpt[1];
	d_pointlist[newidx++] = insertpt[2];
}

__global__ void kernelCountNewTets(
	int* d_threadlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_tetexpandsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int expandsize = 0;
	int i = d_cavebdryhead[threadId];
	while (i != -1)
	{
		expandsize++;
		i = d_cavebdrynext[i];
	}
	d_tetexpandsize[pos] = expandsize;
}

__global__ void kernelCountNewTets(
	int* d_cavebdryidx,
	int* d_tetexpandsize,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavebdryidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 1)
		d_tetexpandsize[pos] = 1;
}

__global__ void kernelInsertNewTets(
	int* d_threadlist,
	tristatus* d_tristatus,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_tetexpandindices,
	int* d_emptytetindices,
	int* d_newtetthreadindices,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	int startidx = d_tetexpandindices[pos], newtetidx;
	int newptidx = oldpointsize + pos;
	tethandle neightet, oldtet, newtet;

	int i = d_cavebdryhead[threadId];
	while (i != -1)
	{
		d_newtetthreadindices[startidx] = threadId;
		newtetidx = d_emptytetindices[startidx++];
		neightet = d_cavebdrylist[i];
		cudamesh_fsym(neightet, oldtet, d_neighborlist); // Get the oldtet (inside the cavity).

		// There might be duplicate elements in cavebdrylist.
		// In that case, oldtet will be newtet. Check to avoid
		if (!d_tetstatus[oldtet.id].isEmpty())
		{
			if (cudamesh_apex(neightet, d_tetlist) != -1)
			{
				// Create a new tet in the cavity
				newtet.id = newtetidx;
				newtet.ver = 11;
				cudamesh_setorg(newtet, cudamesh_dest(neightet, d_tetlist), d_tetlist);
				cudamesh_setdest(newtet, cudamesh_org(neightet, d_tetlist), d_tetlist);
				cudamesh_setapex(newtet, cudamesh_apex(neightet, d_tetlist), d_tetlist);
				cudamesh_setoppo(newtet, newptidx, d_tetlist);
			}
			else
			{
				// Create a new hull tet
				newtet.id = newtetidx;
				newtet.ver = 11;
				cudamesh_setorg(newtet, cudamesh_org(neightet, d_tetlist), d_tetlist);
				cudamesh_setdest(newtet, cudamesh_dest(neightet, d_tetlist), d_tetlist);
				cudamesh_setapex(newtet, newptidx, d_tetlist);
				cudamesh_setoppo(newtet, -1, d_tetlist); // It must opposite to face 3.
														 // Adjust back to the cavity bounday face.
				cudamesh_esymself(newtet);
			}
			// Connect newtet <==> neightet, this also disconnect the old bond.
			cudamesh_bond(newtet, neightet, d_neighborlist);
			// Oldtet still connects to neightet
			d_cavebdrylist[i] = oldtet;
			// connect newtet <==> old subface
			if (threadmarker == 2)
			{
				trihandle checksh;
				cudamesh_tspivot(neightet, checksh, d_tet2trilist);
				if (checksh.id != -1)
				{
					cudamesh_sesymself(checksh);
					cudamesh_tsbond(newtet, checksh, d_tet2trilist, d_tri2tetlist);
					d_tristatus[checksh.id].setNew(true);
				}
			}
		}
		else // duplicate elements cause fake oldtet
		{
			d_cavebdrylist[i] = tethandle(-1, 11);
		}

		i = d_cavebdrynext[i];
	}
}

__global__ void kernelInsertNewTets(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	tristatus* d_tristatus,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int* d_emptytetindices,
	int* d_newtetthreadindices,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavebdryidx[pos];
	int threadmarker = d_threadmarker[threadId];
	int newtetidx = d_emptytetindices[pos];
	int newptidx = oldpointsize + d_threadpos[threadId];
	tethandle neightet, oldtet, newtet;

	d_newtetthreadindices[pos] = threadId; 
	neightet = d_cavebdrylist[pos];
	cudamesh_fsym(neightet, oldtet, d_neighborlist); // Get the oldtet (inside the cavity).

	if (cudamesh_apex(neightet, d_tetlist) != -1)
	{
		// Create a new tet in the cavity
		newtet.id = newtetidx;
		newtet.ver = 11;
		cudamesh_setorg(newtet, cudamesh_dest(neightet, d_tetlist), d_tetlist);
		cudamesh_setdest(newtet, cudamesh_org(neightet, d_tetlist), d_tetlist);
		cudamesh_setapex(newtet, cudamesh_apex(neightet, d_tetlist), d_tetlist);
		cudamesh_setoppo(newtet, newptidx, d_tetlist);
	}
	else
	{
		// Create a new hull tet
		newtet.id = newtetidx;
		newtet.ver = 11;
		cudamesh_setorg(newtet, cudamesh_org(neightet, d_tetlist), d_tetlist);
		cudamesh_setdest(newtet, cudamesh_dest(neightet, d_tetlist), d_tetlist);
		cudamesh_setapex(newtet, newptidx, d_tetlist);
		cudamesh_setoppo(newtet, -1, d_tetlist); // It must opposite to face 3.
													// Adjust back to the cavity bounday face.
		cudamesh_esymself(newtet);
	}
	// Connect newtet <==> neightet, this also disconnect the old bond.
	cudamesh_bond(newtet, neightet, d_neighborlist);
	// Oldtet still connects to neightet
	d_cavebdrylist[pos] = oldtet;
	// connect newtet <==> old subface
	if (threadmarker == 2)
	{
		trihandle checksh;
		cudamesh_tspivot(neightet, checksh, d_tet2trilist);
		if (checksh.id != -1)
		{
			cudamesh_sesymself(checksh);
			cudamesh_tsbond(newtet, checksh, d_tet2trilist, d_tri2tetlist);
			d_tristatus[checksh.id].setNew(true);
		}
	}

	//printf("kernelInsertNewTets: oldtet.id = %d, ver = %d\n", oldtet.id, oldtet.ver);
}

__global__ void kernelConnectNewTetNeighbors(
	int* d_threadlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	uint64* d_tetmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	tethandle oldtet, neightet, newtet, newneitet, spintet;
	int orgidx;

	int i = d_cavebdryhead[threadId], j;

	while (i != -1)
	{
		// Get the newtet and oldtet at the same face.
		oldtet = d_cavebdrylist[i];
		if (oldtet.id != -1) // not fake one
		{
			cudamesh_fsym(oldtet, neightet, d_neighborlist);
			cudamesh_fsym(neightet, newtet, d_neighborlist);

			// Comment: oldtet and newtet must be at the same directed edge.
			// Connect the three other faces of this newtet.
			for (j = 0; j < 3; j++)
			{
				cudamesh_esym(newtet, neightet); // Go to the face
												 // Do not have neighbor yet
				if (d_neighborlist[4 * neightet.id + (neightet.ver & 3)].id == -1)
				{
					// Find the adjacent face of this new tet
					spintet = oldtet;
					while (1)
					{
						cudamesh_fnextself(spintet, d_neighborlist);
						if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId + 1)
							break;
					}
					cudamesh_fsym(spintet, newneitet, d_neighborlist);
					cudamesh_esymself(newneitet);
					cudamesh_bond(neightet, newneitet, d_neighborlist);
				}
				orgidx = cudamesh_org(newtet, d_tetlist);
				cudamesh_enextself(newtet);
				cudamesh_enextself(oldtet);
			}
			d_cavebdrylist[i] = newtet; // Save the new tet

			// Update tetstatus
			d_tetstatus[oldtet.id].clear();
			d_tetstatus[newtet.id].setEmpty(false);
			d_tetstatus[newtet.id].setNew(true);
		}

		i = d_cavebdrynext[i];
	}

	// Check neighbor
	/*i = d_cavebdryhead[threadId];
	while (i != -1)
	{
		newtet = d_cavebdrylist[i];
		if (newtet.id != -1)
		{
			for (j = 0; j < 4; j++)
			{
				newtet.ver = j;
				neightet = d_neighborlist[4 * newtet.id + (newtet.ver & 3)];
				if (d_neighborlist[4 * neightet.id + (neightet.ver & 3)].id != newtet.id)
					printf("Wrong neighbor(%d): Tet#%d - %d, %d, %d, %d, Tet#%d - %d, %d, %d, %d\n",
						threadId,
						newtet.id,
						d_neighborlist[4 * newtet.id + 0].id, d_neighborlist[4 * newtet.id + 1].id,
						d_neighborlist[4 * newtet.id + 2].id, d_neighborlist[4 * newtet.id + 3].id,
						neightet.id,
						d_neighborlist[4 * neightet.id + 0].id, d_neighborlist[4 * neightet.id + 1].id,
						d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
			}
		}
		i = d_cavebdrynext[i];
	}*/
}

__global__ void kernelConnectNewTetNeighbors(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	uint64* d_tetmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavebdryidx[pos];
	tethandle oldtet, neightet, newtet, newneitet, spintet;
	int j;

	// Get the newtet and oldtet at the same face.
	oldtet = d_cavebdrylist[pos];
	cudamesh_fsym(oldtet, neightet, d_neighborlist);
	cudamesh_fsym(neightet, newtet, d_neighborlist);

	// Comment: oldtet and newtet must be at the same directed edge.
	// Connect the three other faces of this newtet.
	for (j = 0; j < 3; j++)
	{
		cudamesh_esym(newtet, neightet); // Go to the face
		// Do not have neighbor yet
		if (d_neighborlist[4 * neightet.id + (neightet.ver & 3)].id == -1)
		{
			// Find the adjacent face of this new tet
			spintet = oldtet;
			while (1)
			{
				cudamesh_fnextself(spintet, d_neighborlist);
				if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId + 1)
					break;
			}
			cudamesh_fsym(spintet, newneitet, d_neighborlist);
			cudamesh_esymself(newneitet);
			cudamesh_bond(neightet, newneitet, d_neighborlist);
		}
		cudamesh_enextself(newtet);
		cudamesh_enextself(oldtet);
	}
	d_cavebdrylist[pos] = newtet; // Save the new tet

	// Update tetstatus
	d_tetstatus[oldtet.id].clear();
	d_tetstatus[newtet.id].setEmpty(false);
	d_tetstatus[newtet.id].setNew(true);

	//printf("kernelConnectNewTetNeighbors: oldtet.id = %d, ver = %d\n", oldtet.id, oldtet.ver);

	// Check neighbor
	/*newtet = d_cavebdrylist[pos];
	for (j = 0; j < 4; j++)
	{
		newtet.ver = j;
		neightet = d_neighborlist[4 * newtet.id + (newtet.ver & 3)];
		if (d_neighborlist[4 * neightet.id + (neightet.ver & 3)].id != newtet.id)
			printf("Wrong neighbor(%d): Tet#%d - %d, %d, %d, %d, Tet#%d - %d, %d, %d, %d\n",
				threadId,
				newtet.id,
				d_neighborlist[4 * newtet.id + 0].id, d_neighborlist[4 * newtet.id + 1].id,
				d_neighborlist[4 * newtet.id + 2].id, d_neighborlist[4 * newtet.id + 3].id,
				neightet.id,
				d_neighborlist[4 * neightet.id + 0].id, d_neighborlist[4 * neightet.id + 1].id,
				d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
	}*/
}

__global__ void kernelUpdateTriAndTetStatus_Phase1(
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	int* d_emptytetindices, // new tet indices
	int* d_newtetthreadindices, // new tet thread indices
	int* d_triexpandsize,
	REAL* d_trifaceipt,
	int* d_tripmtidx,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_pointpmt,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	REAL aabb_diglen,
	int numofaabbpms,
	int insertmode,
	int shortcut,
	int numofnewtets
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofnewtets)
		return;

	int newtetidx = d_emptytetindices[pos];
	if (d_tetstatus[newtetidx].isEmpty()) // skip unused empty slot
		return;
	
	tethandle checktet(newtetidx, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return;

	int threadId = d_newtetthreadindices[pos];
	int threadmarker = d_threadmarker[threadId];

	int i, j, ret;
	int pi[4];
	REAL *pc[4], w[4], wc[3];
	int pmidxlist[5] = { -1, -1, -1, -1, -1 };
	for (i = 0; i < 4; i++)
	{
		pi[i] = d_tetlist[4 * checktet.id + i];
		pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
		w[i] = d_weightlist[pi[i]];
		if(shortcut == 2)
			pmidxlist[i] = d_pointpmt[pi[i]];
	}

	if (!cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], pc[3],
		w[0], w[1], w[2], w[3], wc)) // is degenerate tet
		return;

	// Get bounding box information
	REAL xmin, xmax, ymin, ymax, zmin, zmax;
	xmin = d_aabbnodebbs[0]; xmax = d_aabbnodebbs[1];
	ymin = d_aabbnodebbs[2]; ymax = d_aabbnodebbs[3];
	zmin = d_aabbnodebbs[4]; zmax = d_aabbnodebbs[5];

	// indomain test
	bool indomain = false;
	if (insertmode == 1)
	{
		if (threadmarker == 2) // tet splitting
			indomain = true; // new finite tets are always in domain
		else // threadmarker == 1, facet splitting
		{
			// may generate new tets that are outside domain, hence need to check
			if (cudamesh_is_out_bbox(wc, xmin, xmax, ymin, ymax, zmin, zmax))
				indomain = false;
			else
			{
				REAL t[3];
				cudamesh_box_far_point(wc, t, xmin, xmax, ymin, ymax, zmin, zmax, aabb_diglen);
				indomain =
					cudamesh_traversal_in_domain(wc, t, d_aabbnodeleft, d_aabbnoderight,
						d_aabbnodebbs, d_aabbpmcoord, d_aabbpmbbs);
			}
		}

		d_tetstatus[newtetidx].setInDomain(indomain);
		if (indomain &&
			cudamesh_is_bad_tet(pc[0], pc[1], pc[2], pc[3], w[0], w[1], w[2], w[3],
				cr_cell_radius_edge_ratio, cr_cell_size))
		{
			d_tetstatus[newtetidx].setBad(true);
		}
	}

	// Check 4 neighbors
	tethandle neightet;
	bool larger, nnewtet, nhtet, found;
	int npi[4], pmidx = -1;
	REAL *npc[4], nw[4], cwc[3], nwc[3], ipt[3], dir[3], vdir[3], fwc[3], vec1[3], vec2[3], len;
	REAL bxmin, bxmax, bymin, bymax, bzmin, bzmax;
	for (checktet.ver = 0; checktet.ver < 4; checktet.ver++)
	{
		cudamesh_fsym(checktet, neightet, d_neighborlist);
		larger = (checktet.id > neightet.id);
		nhtet = cudamesh_ishulltet(neightet, d_tetlist);
		nnewtet = d_tetstatus[neightet.id].isNew();  // is neighbor a new tet ?

		if (larger && nnewtet && !nhtet) // let neighbor handle it (must be a new tet)
		{
			continue;
		}

		// calculate the voronoi dual of the facet
		if (!nhtet) // the dual is a segment
		{
			for (i = 0; i < 4; i++)
			{
				npi[i] = d_tetlist[4 * neightet.id + i];
				npc[i] = cudamesh_id2pointlist(npi[i], d_pointlist);
				nw[i] = d_weightlist[npi[i]];
				if (shortcut == 2 && i == (neightet.ver & 3))
					pmidxlist[4] = d_pointpmt[npi[i]];
			}
			if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], npc[3],
				nw[0], nw[1], nw[2], nw[3], nwc))
				continue;
		}
		else // the dual is a ray
		{
			// get boundary face points
			npi[0] = cudamesh_org(checktet, d_tetlist);
			npc[0] = cudamesh_id2pointlist(npi[0], d_pointlist);
			nw[0] = d_weightlist[npi[0]];
			npi[1] = cudamesh_dest(checktet, d_tetlist);
			npc[1] = cudamesh_id2pointlist(npi[1], d_pointlist);
			nw[1] = d_weightlist[npi[1]];
			npi[2] = cudamesh_apex(checktet, d_tetlist);
			npc[2] = cudamesh_id2pointlist(npi[2], d_pointlist);
			nw[2] = d_weightlist[npi[2]];
			// get oppo point
			npi[3] = cudamesh_oppo(checktet, d_tetlist);
			npc[3] = cudamesh_id2pointlist(npi[3], d_pointlist);
			// set up pmidxlist for fast check
			if(shortcut == 2)
				pmidxlist[4] = d_pointpmt[npi[3]];
			// caculate the weighted center perpendicular vector of boundary face
			if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], nw[0], nw[1], nw[2], fwc))
				continue; // degenerate boundary face
			cudamesh_raydir(npc[0], npc[1], npc[2], dir);
			vdir[0] = npc[3][0] - fwc[0]; vdir[1] = npc[3][1] - fwc[1]; vdir[2] = npc[3][2] - fwc[2];
			if (dir[0] * vdir[0] + dir[1] * vdir[1] + dir[2] * vdir[2] >= 0.0)
				continue; // degenerate ray
						  // calculate a point outside the bounding box
			len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
			if (len == 0.0)
				continue;
			dir[0] /= len; dir[1] /= len; dir[2] /= len;
			nwc[0] = wc[0] + dir[0] * 1.2*aabb_diglen;
			nwc[1] = wc[1] + dir[1] * 1.2*aabb_diglen;
			nwc[2] = wc[2] + dir[2] * 1.2*aabb_diglen;
		}
		cudamesh_copy(cwc, wc); // to avoid wc being changed
		ret = cudamesh_compare(cwc, nwc);
		if (ret == 0) // degenerate segment
			continue;
		else if (ret == 1) // make canonical vector
			cudamesh_swap(cwc, nwc);

		if (shortcut == 2)
		{
			//remove duplicate primitive indice from pmidxlist
			for (i = 0; i < 4; i++)
			{
				if (pmidxlist[i] == -1)
					continue;
				for (j = i + 1; j < 5; j++)
				{
					if (pmidxlist[i] == pmidxlist[j])
						pmidxlist[j] = -1;
				}
			}
		}

		// Try to find any intersections with input polygons
		found = false;

		if (shortcut == 2)
		{
			for (i = 0; i < 5; i++) // fast check
			{
				pmidx = pmidxlist[i];
				if (pmidx != -1)
				{
					bxmin = d_aabbpmbbs[6 * pmidx + 0]; bxmax = d_aabbpmbbs[6 * pmidx + 1];
					bymin = d_aabbpmbbs[6 * pmidx + 2]; bymax = d_aabbpmbbs[6 * pmidx + 3];
					bzmin = d_aabbpmbbs[6 * pmidx + 4]; bzmax = d_aabbpmbbs[6 * pmidx + 5];
					if (cudamesh_do_intersect_bbox(wc, nwc, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
					{
						REAL* p[3];
						p[0] = d_aabbpmcoord + 9 * pmidx + 0;
						p[1] = d_aabbpmcoord + 9 * pmidx + 3;
						p[2] = d_aabbpmcoord + 9 * pmidx + 6;
						if (cudamesh_ts_intersection(wc, nwc, p[0], p[1], p[2], ipt))
						{
							found = true;
							break;
						}
					}
				}
			}
		}
		
		if (!found)
			found = cudamesh_traversal_first_intersection(cwc, nwc, d_aabbnodeleft, d_aabbnoderight, d_aabbnodebbs,
				d_aabbpmcoord, d_aabbpmbbs, ipt, pmidx);

		if (!found)
			continue;

		// Mark and record this facet
		d_triexpandsize[4 * pos + checktet.ver] = 1;
		d_trifaceipt[3 * (4 * pos + checktet.ver) + 0] = ipt[0];
		d_trifaceipt[3 * (4 * pos + checktet.ver) + 1] = ipt[1];
		d_trifaceipt[3 * (4 * pos + checktet.ver) + 2] = ipt[2];
		d_tripmtidx[4 * pos + checktet.ver] = pmidx;
	}
}

__global__ void kernelUpdateTriAndTetStatus_Phase2(
	int* d_emptytetindices, // new tet indices
	int* d_triexpandsize,
	int* d_triexpandindices,
	int* d_emptytriindices,
	REAL* d_trifaceipt,
	int* d_tripmtidx,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	int* d_trifacepmt,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	REAL cr_facet_angle,
	REAL cr_facet_size,
	REAL cr_facet_distance,
	int numofnewtets
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofnewtets)
		return;

	int newtetidx = d_emptytetindices[pos];
	d_tetstatus[newtetidx].setNew(false); // reset all to old
	if (d_tetstatus[newtetidx].isEmpty()) // skip unused empty slot
		return;

	tethandle checktet(newtetidx, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return;

	tethandle neightet;
	int pi[3], index;
	REAL *pc[3], pw[3];
	verttype pt[3];
	trihandle checksh;
	REAL* ipt;
	for (checktet.ver = 0; checktet.ver < 4; checktet.ver++)
	{
		if (d_triexpandsize[4 * pos + checktet.ver] == 1) //  a facet
		{
			index = d_triexpandindices[4 * pos + checktet.ver];
			index = d_emptytriindices[index];
			// triface endpoints
			pi[0] = cudamesh_org(checktet, d_tetlist);
			pi[1] = cudamesh_dest(checktet, d_tetlist);
			pi[2] = cudamesh_apex(checktet, d_tetlist);
			checksh.id = index; checksh.shver = 0;
			cudamesh_setsorg(checksh, pi[0], d_trifacelist);
			cudamesh_setsdest(checksh, pi[1], d_trifacelist);
			cudamesh_setsapex(checksh, pi[2], d_trifacelist);
			// bond triface and tetrahedron together
			cudamesh_tsbond(checktet, checksh, d_tet2trilist, d_tri2tetlist);
			cudamesh_fsym(checktet, neightet, d_neighborlist);
			cudamesh_sesymself(checksh);
			cudamesh_tsbond(neightet, checksh, d_tet2trilist, d_tri2tetlist);
			// triface corresponding intersection point
			ipt = d_trifaceipt + 3 * (4 * pos + checktet.ver);
			d_trifacecent[3 * index + 0] = ipt[0];
			d_trifacecent[3 * index + 1] = ipt[1];
			d_trifacecent[3 * index + 2] = ipt[2];
			// triface primitive id
			d_trifacepmt[index] = d_tripmtidx[4 * pos + checktet.ver];
			// update triface status
			d_tristatus[index].setEmpty(false);
			pc[0] = cudamesh_id2pointlist(pi[0], d_pointlist);
			pc[1] = cudamesh_id2pointlist(pi[1], d_pointlist);
			pc[2] = cudamesh_id2pointlist(pi[2], d_pointlist);
			pw[0] = d_weightlist[pi[0]]; pt[0] = d_pointtypelist[pi[0]];
			pw[1] = d_weightlist[pi[1]]; pt[1] = d_pointtypelist[pi[1]];
			pw[2] = d_weightlist[pi[2]]; pt[2] = d_pointtypelist[pi[2]];
			if (cudamesh_is_bad_facet(pc[0], pc[1], pc[2], pw[0], pw[1], pw[2],
				pt[0], pt[1], pt[2], ipt,
				cr_facet_angle, cr_facet_size, cr_facet_distance))
				d_tristatus[index].setBad(true);
		}
	}
}

__global__ void kernelInitDomainHandle(
	int* d_emptytetindices, // new tet indices
	int* d_newtetthreadindices, // new tet thread indices
	tethandle* d_domainhandle,
	int* d_domaincount,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int insertmode,
	int offset,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int newtetidx = d_emptytetindices[pos + offset];

	if (d_tetstatus[newtetidx].isEmpty())
		return;

	tethandle checktet(newtetidx, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return;

	int threadId = d_newtetthreadindices[pos + offset];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2) // tet splitting
	{
		d_domaincount[pos + offset] = 1; // always inside domain
		return;
	}

	// subface splitting
	tethandle neightet;
	bool larger, nnewtet, nhtet;

	for (checktet.ver = 0; checktet.ver < 4; checktet.ver++)
	{
		cudamesh_fsym(checktet, neightet, d_neighborlist);
		larger = (checktet.id > neightet.id);
		nhtet = cudamesh_ishulltet(neightet, d_tetlist);
		nnewtet = d_tetstatus[neightet.id].isNew();

		if (larger && nnewtet && !nhtet) // let neighbor handle it
			continue;

		d_domainhandle[5 * pos + checktet.ver] = tethandle(pos + offset, checktet.ver);
	}

	if(insertmode == 1)
		d_domainhandle[5 * pos + checktet.ver] = tethandle(pos + offset, checktet.ver); // need to check
}

__global__ void kernelInitDomainSegment(
	REAL* d_aabbnodebbs,
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	REAL* d_domainsegment,
	int* d_domainthreadlist,
	int* d_triexpandsize,
	REAL* d_trifaceipt,
	int* d_tripmtidx,
	int* d_emptytetindices, // new tet indices
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_pointpmt,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL aabb_diglen,
	int numofaabbpms,
	int insertmode,
	int shortcut,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	tethandle checkhandle = d_domainhandle[pos];
	int newtetidx = d_emptytetindices[checkhandle.id];
	tethandle checktet(newtetidx, checkhandle.ver);

	int i, j;
	int pi[4];
	REAL *pc[4], w[4], wc[3];
	int pmidxlist[5] = { -1, -1, -1, -1, -1 };
	for (i = 0; i < 4; i++)
	{
		pi[i] = d_tetlist[4 * checktet.id + i];
		pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
		w[i] = d_weightlist[pi[i]];
		if(shortcut == 2)
			pmidxlist[i] = d_pointpmt[pi[i]];
	}

	if (!cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], pc[3],
		w[0], w[1], w[2], w[3], wc)) // is degenerate tet
	{
		d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
		return;
	}

	if (checktet.ver == 4) // random ray for domain test
	{
		if (insertmode == 0) // subface only, lazy processing
		{
			d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
			return;
		}
		else
		{
			REAL xmin, xmax, ymin, ymax, zmin, zmax;
			xmin = d_aabbnodebbs[0]; xmax = d_aabbnodebbs[1];
			ymin = d_aabbnodebbs[2]; ymax = d_aabbnodebbs[3];
			zmin = d_aabbnodebbs[4]; zmax = d_aabbnodebbs[5];
			if (cudamesh_is_out_bbox(wc, xmin, xmax, ymin, ymax, zmin, zmax))
			{
				d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
				return;
			}
			else
			{
				//REAL t[3], v[3];
				//unsigned long randomseed = 0;
				//// generate random ray direction
				//cudamesh_random_sphere_point(&randomseed, v);
				//t[0] = wc[0] + v[0] * 1.5*aabb_diglen;
				//t[1] = wc[1] + v[1] * 1.5*aabb_diglen;
				//t[2] = wc[2] + v[2] * 1.5*aabb_diglen;
				REAL t[3];
				cudamesh_box_far_point(wc, t, xmin, xmax, ymin, ymax, zmin, zmax, aabb_diglen);
				d_domainsegment[6 * pos + 0] = wc[0];
				d_domainsegment[6 * pos + 1] = wc[1];
				d_domainsegment[6 * pos + 2] = wc[2];
				d_domainsegment[6 * pos + 3] = t[0];
				d_domainsegment[6 * pos + 4] = t[1];
				d_domainsegment[6 * pos + 5] = t[2];
			}
		}
	}
	else // dual ray or segment
	{
		tethandle neightet;
		bool nhtet;
		int npi[4], pmidx;
		REAL *npc[4], nw[4], nwc[3], ipt[3], dir[3], vdir[3], fwc[3], len;
		REAL bxmin, bxmax, bymin, bymax, bzmin, bzmax;
		cudamesh_fsym(checktet, neightet, d_neighborlist);
		nhtet = cudamesh_ishulltet(neightet, d_tetlist);
		if (!nhtet) // the dual is a segment
		{
			for (i = 0; i < 4; i++)
			{
				npi[i] = d_tetlist[4 * neightet.id + i];
				npc[i] = cudamesh_id2pointlist(npi[i], d_pointlist);
				nw[i] = d_weightlist[npi[i]];
				if (shortcut == 2 && i == (neightet.ver & 3))
					pmidxlist[4] = d_pointpmt[npi[i]];
			}
			if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], npc[3],
				nw[0], nw[1], nw[2], nw[3], nwc))
			{
				d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
				return;
			}
		}
		else
		{
			// get boundary face points
			npi[0] = cudamesh_org(checktet, d_tetlist);
			npc[0] = cudamesh_id2pointlist(npi[0], d_pointlist);
			nw[0] = d_weightlist[npi[0]];
			npi[1] = cudamesh_dest(checktet, d_tetlist);
			npc[1] = cudamesh_id2pointlist(npi[1], d_pointlist);
			nw[1] = d_weightlist[npi[1]];
			npi[2] = cudamesh_apex(checktet, d_tetlist);
			npc[2] = cudamesh_id2pointlist(npi[2], d_pointlist);
			nw[2] = d_weightlist[npi[2]];
			// get oppo point
			npi[3] = cudamesh_oppo(checktet, d_tetlist);
			npc[3] = cudamesh_id2pointlist(npi[3], d_pointlist);
			// set up pmidxlist for fast check
			if(shortcut == 2)
				pmidxlist[4] = d_pointpmt[npi[3]];
			// caculate the weighted center perpendicular vector of boundary face
			if (!cudamesh_weightedcircumcenter(npc[0], npc[1], npc[2], nw[0], nw[1], nw[2], fwc))
			{
				d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
				return;
			}
			cudamesh_raydir(npc[0], npc[1], npc[2], dir);
			vdir[0] = npc[3][0] - fwc[0]; vdir[1] = npc[3][1] - fwc[1]; vdir[2] = npc[3][2] - fwc[2];
			if (dir[0] * vdir[0] + dir[1] * vdir[1] + dir[2] * vdir[2] >= 0.0)
			{
				// degenerate ray
				d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
				return;
			}
			// calculate a point outside the bounding box
			len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
			if (len == 0.0)
			{
				d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
				return;
			}
			dir[0] /= len; dir[1] /= len; dir[2] /= len;
			nwc[0] = wc[0] + dir[0] * 1.2*aabb_diglen;
			nwc[1] = wc[1] + dir[1] * 1.2*aabb_diglen;
			nwc[2] = wc[2] + dir[2] * 1.2*aabb_diglen;
		}

		int ret = cudamesh_compare(wc, nwc);
		if (ret == 0) // degenerate segment
		{
			d_domainhandle[pos] = tethandle(-1, 11); // set to invalid
			return;
		}
		else if (ret == 1) // make canonical vector
			cudamesh_swap(wc, nwc);
		d_domainsegment[6 * pos + 0] = wc[0];
		d_domainsegment[6 * pos + 1] = wc[1];
		d_domainsegment[6 * pos + 2] = wc[2];
		d_domainsegment[6 * pos + 3] = nwc[0];
		d_domainsegment[6 * pos + 4] = nwc[1];
		d_domainsegment[6 * pos + 5] = nwc[2];

		if (shortcut == 2)
		{
			//remove duplicate primitive indice from pmidxlist
			for (i = 0; i < 4; i++)
			{
				if (pmidxlist[i] == -1)
					continue;
				for (j = i + 1; j < 5; j++)
				{
					if (pmidxlist[i] == pmidxlist[j])
						pmidxlist[j] = -1;
				}
			}

			for (i = 0; i < 5; i++) // fast check
			{
				pmidx = pmidxlist[i];
				if (pmidx != -1)
				{
					bxmin = d_aabbpmbbs[6 * pmidx + 0]; bxmax = d_aabbpmbbs[6 * pmidx + 1];
					bymin = d_aabbpmbbs[6 * pmidx + 2]; bymax = d_aabbpmbbs[6 * pmidx + 3];
					bzmin = d_aabbpmbbs[6 * pmidx + 4]; bzmax = d_aabbpmbbs[6 * pmidx + 5];
					if (cudamesh_do_intersect_bbox(wc, nwc, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
					{
						REAL* p[3];
						p[0] = d_aabbpmcoord + 9 * pmidx + 0;
						p[1] = d_aabbpmcoord + 9 * pmidx + 3;
						p[2] = d_aabbpmcoord + 9 * pmidx + 6;
						if (cudamesh_ts_intersection(wc, nwc, p[0], p[1], p[2], ipt))
						{
							d_triexpandsize[4 * checkhandle.id + checkhandle.ver] = 1;
							REAL* pt;
							pt = d_trifaceipt + 3 * (4 * checkhandle.id + checkhandle.ver);
							pt[0] = ipt[0];
							pt[1] = ipt[1];
							pt[2] = ipt[2];
							d_tripmtidx[4 * checkhandle.id + checkhandle.ver] = pmidx;

							d_domainhandle[pos] = tethandle(-1, 11); // no need to travel aabb tree
							return;
						}
					}
				}
			}
		}
	}

	// swap new tet thread index and domain thread (segment) index
	d_domainthreadlist[pos] = checkhandle.id;
	checkhandle.id = pos;
	d_domainhandle[pos] = checkhandle;
}

__global__ void kernelDomainSegmentAndBoxCheck(
	REAL* d_aabbnodebbs,
	tethandle* d_domainhandle,
	int* d_domainnode,
	REAL* d_domainsegment,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int nodeidx = d_domainnode[pos];
	if (nodeidx <= 0) // invalid or primitive
		return;

	REAL s[3], t[3];
	REAL bxmin, bxmax, bymin, bymax, bzmin, bzmax;
	tethandle checkhandle = d_domainhandle[pos];
	s[0] = d_domainsegment[6 * checkhandle.id + 0];
	s[1] = d_domainsegment[6 * checkhandle.id + 1];
	s[2] = d_domainsegment[6 * checkhandle.id + 2];
	t[0] = d_domainsegment[6 * checkhandle.id + 3];
	t[1] = d_domainsegment[6 * checkhandle.id + 4];
	t[2] = d_domainsegment[6 * checkhandle.id + 5];
	nodeidx--; // shift back to true node index
	bxmin = d_aabbnodebbs[6 * nodeidx + 0]; bxmax = d_aabbnodebbs[6 * nodeidx + 1];
	bymin = d_aabbnodebbs[6 * nodeidx + 2]; bymax = d_aabbnodebbs[6 * nodeidx + 3];
	bzmin = d_aabbnodebbs[6 * nodeidx + 4]; bzmax = d_aabbnodebbs[6 * nodeidx + 5];
	if (!cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
	{
		// set to invalid
		d_domainhandle[pos] = tethandle(-1, 11);
		d_domainnode[pos] = 0;
	}
}

__global__ void  kernelDomainHandleAppend(
	int* d_aabbnodeleft,
	int* d_aabbnoderight,
	tethandle* d_domainhandle,
	int* d_domainnode,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int nodeidx = d_domainnode[pos];
	if (nodeidx <= 0) // invalid or primitive
		return;

	nodeidx--; // shift back to true node index
	int left, right;
	left = d_aabbnodeleft[nodeidx];
	right = d_aabbnoderight[nodeidx];
	d_domainhandle[pos + numofthreads] = d_domainhandle[pos];
	d_domainnode[pos] = (left < 0 ? left : left + 1);
	d_domainnode[pos + numofthreads] = (right < 0 ? right : right + 1);
}

__global__ void kernelDomainSegmentAndPrimitiveCheck(
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	int* d_domainnode,
	REAL* d_domainsegment,
	int* d_domaincount,
	int* d_domainthreadlist,
	int* d_triexpandsize,
	int* d_emptytetindices, // new tet indices
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	uint64* d_domainmarker, // triface distance marker
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int pmidx = d_domainnode[pos];
	if (pmidx >= 0) // this should not happend
		return;

	REAL s[3], t[3];
	REAL bxmin, bxmax, bymin, bymax, bzmin, bzmax;
	tethandle checkhandle = d_domainhandle[pos];
	s[0] = d_domainsegment[6 * checkhandle.id + 0];
	s[1] = d_domainsegment[6 * checkhandle.id + 1];
	s[2] = d_domainsegment[6 * checkhandle.id + 2];
	t[0] = d_domainsegment[6 * checkhandle.id + 3];
	t[1] = d_domainsegment[6 * checkhandle.id + 4];
	t[2] = d_domainsegment[6 * checkhandle.id + 5];
	pmidx = -pmidx - 1; // shift back to true primitive index
	bxmin = d_aabbpmbbs[6 * pmidx + 0]; bxmax = d_aabbpmbbs[6 * pmidx + 1];
	bymin = d_aabbpmbbs[6 * pmidx + 2]; bymax = d_aabbpmbbs[6 * pmidx + 3];
	bzmin = d_aabbpmbbs[6 * pmidx + 4]; bzmax = d_aabbpmbbs[6 * pmidx + 5];
	if (cudamesh_do_intersect_bbox(s, t, bxmin, bxmax, bymin, bymax, bzmin, bzmax))
	{
		int type, emptytetthreadIdx, counter = 0;
		REAL* p[3];
		p[0] = d_aabbpmcoord + 9 * pmidx + 0;
		p[1] = d_aabbpmcoord + 9 * pmidx + 3;
		p[2] = d_aabbpmcoord + 9 * pmidx + 6;
		emptytetthreadIdx = d_domainthreadlist[checkhandle.id];
		if (checkhandle.ver == 4) // domain test
		{
			if (cudamesh_ts_intersection(p[0], p[1], p[2], s, t, type))
			{
				if (type == (int)UNKNOWNINTER || type == (int)ACROSSEDGE
					|| type == (int)ACROSSVERT || type == (int)COPLANAR)
				{
					counter = -10000000;
				}
				else if (type == (int)TOUCHEDGE || type == (int)TOUCHFACE
					|| type == (int)SHAREVERT)
				{
					counter = 10000000;
				}
				else if (type == (int)ACROSSFACE)
				{
					counter = 1;
				}
				atomicAdd(d_domaincount + emptytetthreadIdx, counter);
			}
		}
		else // subface test, need to ignore duplicate one
		{
			REAL ipt[3], *pt;
			if (cudamesh_ts_intersection(s, t, p[0], p[1], p[2], ipt))
			{
				int pi[4], newtetidx;
				REAL *pc[4], w[4], wc[3];
				newtetidx = d_emptytetindices[emptytetthreadIdx];
				tethandle checktet(newtetidx, checkhandle.ver);
				pi[0] = cudamesh_org(checktet, d_tetlist);
				pc[0] = cudamesh_id2pointlist(pi[0], d_pointlist);
				w[0] = d_weightlist[pi[0]];
				pi[1] = cudamesh_dest(checktet, d_tetlist);
				pc[1] = cudamesh_id2pointlist(pi[1], d_pointlist);
				w[1] = d_weightlist[pi[1]];
				pi[2] = cudamesh_apex(checktet, d_tetlist);
				pc[2] = cudamesh_id2pointlist(pi[2], d_pointlist);
				w[2] = d_weightlist[pi[2]];
				cudamesh_weightedcircumcenter(pc[0], pc[1], pc[2], w[0], w[1], w[2], wc);
				REAL sdis = cudamesh_squared_distance(wc, ipt);
				uint64 marker = cudamesh_encodeUInt64Priority(__float_as_int(sdis), pos);
				uint64 oldmarker = atomicMin(d_domainmarker + 4 * emptytetthreadIdx + checkhandle.ver, marker);
				d_triexpandsize[4 * emptytetthreadIdx + checkhandle.ver] = 1;
			}
		}
	}
}

__global__ void kernelDomainSetTriCenter(
	REAL* d_aabbpmcoord,
	REAL* d_aabbpmbbs,
	tethandle* d_domainhandle,
	int* d_domainnode,
	REAL* d_domainsegment,
	int* d_domainthreadlist,
	uint64* d_domainmarker, // triface distance marker
	REAL* d_trifaceipt,
	int* d_tripmtidx,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int pmidx = d_domainnode[pos];
	if (pmidx >= 0) // this should not happend
		return;

	tethandle checkhandle = d_domainhandle[pos];
	int emptytetthreadIdx = d_domainthreadlist[checkhandle.id];
	uint64 marker = d_domainmarker[4 * emptytetthreadIdx + checkhandle.ver];
	int winneridx = cudamesh_getUInt64PriorityIndex(marker);
	if (marker == MAXULL || winneridx != pos)
		return;

	REAL s[3], t[3];
	s[0] = d_domainsegment[6 * checkhandle.id + 0];
	s[1] = d_domainsegment[6 * checkhandle.id + 1];
	s[2] = d_domainsegment[6 * checkhandle.id + 2];
	t[0] = d_domainsegment[6 * checkhandle.id + 3];
	t[1] = d_domainsegment[6 * checkhandle.id + 4];
	t[2] = d_domainsegment[6 * checkhandle.id + 5];
	pmidx = -pmidx - 1; // shift back to true primitive index
	REAL *p[3], ipt[3];
	p[0] = d_aabbpmcoord + 9 * pmidx + 0;
	p[1] = d_aabbpmcoord + 9 * pmidx + 3;
	p[2] = d_aabbpmcoord + 9 * pmidx + 6;
	cudamesh_ts_intersection(s, t, p[0], p[1], p[2], ipt);

	REAL* pt;
	pt = d_trifaceipt + 3 * (4 * emptytetthreadIdx + checkhandle.ver);
	pt[0] = ipt[0];
	pt[1] = ipt[1];
	pt[2] = ipt[2];

	d_tripmtidx[4 * emptytetthreadIdx + checkhandle.ver] = pmidx;
}

__global__ void kernelUpdateNewTetStatus(
	int* d_emptytetindices, // new tet indices
	int* d_domaincount,
	REAL* d_pointlist,
	REAL* d_weightlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL cr_cell_radius_edge_ratio,
	REAL cr_cell_size,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int newtetidx = d_emptytetindices[pos];
	if (d_tetstatus[newtetidx].isEmpty()) // skip unused empty slot
		return;

	tethandle checktet(newtetidx, 11);
	if (cudamesh_ishulltet(checktet, d_tetlist))
		return;

	int counter = d_domaincount[pos];
	bool indomain = false;
	if (counter >= 10000000)
		indomain = true;
	else if (counter > 0)
		indomain = (counter & 1) == 1 ? true : false;

	d_tetstatus[newtetidx].setInDomain(indomain);
	if (indomain)
	{
		int pi[4];
		REAL *pc[4], w[4];
		for (int i = 0; i < 4; i++)
		{
			pi[i] = d_tetlist[4 * checktet.id + i];
			pc[i] = cudamesh_id2pointlist(pi[i], d_pointlist);
			w[i] = d_weightlist[pi[i]];
		}
		if (cudamesh_is_bad_tet(pc[0], pc[1], pc[2], pc[3], w[0], w[1], w[2], w[3],
			cr_cell_radius_edge_ratio, cr_cell_size))
		{
			d_tetstatus[newtetidx].setBad(true);
		}
	}
}

__global__ void kernelUpdateTriStatus(
	int* d_emptytetindices, // new tet indices
	int* d_triexpandsize,
	int* d_triexpandindices,
	int* d_emptytriindices,
	REAL* d_trifaceipt,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_weightlist,
	int* d_trifacelist,
	REAL* d_trifacecent,
	tristatus* d_tristatus,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	REAL cr_facet_angle,
	REAL cr_facet_size,
	REAL cr_facet_distance,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_triexpandsize[pos] == 0)
		return;

	int newtetidx = d_emptytetindices[pos / 4];
	d_tetstatus[newtetidx].setNew(false); // reset all to old
	tethandle checktet(newtetidx, pos % 4);

	tethandle neightet;
	int pi[3], index;
	REAL *pc[3], pw[3];
	verttype pt[3];
	trihandle checksh;
	REAL* ipt;

	index = d_triexpandindices[pos];
	index = d_emptytriindices[index];
	// triface endpoints
	pi[0] = cudamesh_org(checktet, d_tetlist);
	pi[1] = cudamesh_dest(checktet, d_tetlist);
	pi[2] = cudamesh_apex(checktet, d_tetlist);
	checksh.id = index; checksh.shver = 0;
	cudamesh_setsorg(checksh, pi[0], d_trifacelist);
	cudamesh_setsdest(checksh, pi[1], d_trifacelist);
	cudamesh_setsapex(checksh, pi[2], d_trifacelist);
	// bond triface and tetrahedron together
	cudamesh_tsbond(checktet, checksh, d_tet2trilist, d_tri2tetlist);
	cudamesh_fsym(checktet, neightet, d_neighborlist);
	cudamesh_sesymself(checksh);
	cudamesh_tsbond(neightet, checksh, d_tet2trilist, d_tri2tetlist);
	// triface corresponding intersection point
	ipt = d_trifaceipt + 3 * pos;
	d_trifacecent[3 * index + 0] = ipt[0];
	d_trifacecent[3 * index + 1] = ipt[1];
	d_trifacecent[3 * index + 2] = ipt[2];
	// update triface status
	d_tristatus[index].setEmpty(false);
	pc[0] = cudamesh_id2pointlist(pi[0], d_pointlist);
	pc[1] = cudamesh_id2pointlist(pi[1], d_pointlist);
	pc[2] = cudamesh_id2pointlist(pi[2], d_pointlist);
	pw[0] = d_weightlist[pi[0]]; pt[0] = d_pointtypelist[pi[0]];
	pw[1] = d_weightlist[pi[1]]; pt[1] = d_pointtypelist[pi[1]];
	pw[2] = d_weightlist[pi[2]]; pt[2] = d_pointtypelist[pi[2]];
	if (cudamesh_is_bad_facet(pc[0], pc[1], pc[2], pw[0], pw[1], pw[2],
		pt[0], pt[1], pt[2], ipt,
		cr_facet_angle, cr_facet_size, cr_facet_distance))
		d_tristatus[index].setBad(true);
}

__global__ void kernelResetOldInfo(
	int* d_threadlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos], j;
	int threadmarker = d_threadmarker[threadId];

	// Clear old tets
	tethandle checktet;
	int i = d_caveoldtethead[threadId];
	while (i != -1)
	{
		checktet = d_caveoldtetlist[i];
		d_tetstatus[checktet.id].clear();
		for (j = 0; j < 4; j++)
		{
			d_neighborlist[4 * checktet.id + j] = tethandle(-1, 11); // reset neighbor to empty
		}
		for (j = 0; j < 4; j++)
		{
			d_tet2trilist[4 * checktet.id + j] = trihandle(-1, 0); // reset subface to empty
		}

		i = d_caveoldtetnext[i];
	}

	// Clear old trifaces
	if (threadmarker == 1) // subface splitting
	{
		trihandle checksh;
		i = d_cavetetshhead[threadId];
		while (i != -1)
		{
			checksh = d_cavetetshlist[i];
			d_tristatus[checksh.id].clear(); // delete all old trifaces
			for (j = 0; j < 2; j++)
			{
				checktet = d_tri2tetlist[2 * checksh.id + j];
				// some tets outside cavities still remember the old trifaces, reset their neighbors as well
				if (d_tet2trilist[4 * checktet.id + (checktet.ver & 3)].id == checksh.id)
					d_tet2trilist[4 * checktet.id + (checktet.ver & 3)] = trihandle(-1, 0);
				d_tri2tetlist[2 * checksh.id + j] = tethandle(-1, 11); // reset neighbor to empty
			}

			i = d_cavetetshnext[i];
		}
	}
}

__global__ void kernelResetOldInfo_Tet(
	tethandle* d_caveoldtetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int j;

	// Clear old tets
	tethandle checktet;
	checktet = d_caveoldtetlist[pos];
	d_tetstatus[checktet.id].clear();
	for (j = 0; j < 4; j++)
	{
		d_neighborlist[4 * checktet.id + j] = tethandle(-1, 11); // reset neighbor to empty
	}
	for (j = 0; j < 4; j++)
	{
		d_tet2trilist[4 * checktet.id + j] = trihandle(-1, 0); // reset subface to empty
	}
}

__global__ void kernelResetOldInfo_Subface(
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	trihandle* d_tet2trilist,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos], j;
	int threadmarker = d_threadmarker[threadId];

	tethandle checktet;
	// Clear old trifaces
	if (threadmarker == 1) // subface splitting
	{
		trihandle checksh;
		checksh = d_cavetetshlist[pos];
		d_tristatus[checksh.id].clear(); // delete all old trifaces
		for (j = 0; j < 2; j++)
		{
			checktet = d_tri2tetlist[2 * checksh.id + j];
			// some tets outside cavities still remember the old trifaces, reset their neighbors as well
			if (d_tet2trilist[4 * checktet.id + (checktet.ver & 3)].id == checksh.id)
				d_tet2trilist[4 * checktet.id + (checktet.ver & 3)] = trihandle(-1, 0);
			d_tri2tetlist[2 * checksh.id + j] = tethandle(-1, 11); // reset neighbor to empty
		}
	}
}

// Compact mesh
__global__ void kernelCompactTriface(
	int* d_trifacelist,
	double* d_trifacecent,
	tethandle* d_tri2tetlist,
	tetstatus* d_tetstatus,
	int* d_sizes,
	int* d_indices,
	int* d_listidx,
	double* d_listpt,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_sizes[pos] == 0)
		return;

	int index = d_indices[pos];
	tethandle neitet = d_tri2tetlist[2 * pos + 0];
	if(!d_tetstatus[neitet.id].isInDomain())
	{
		d_listidx[3 * index + 0] = d_trifacelist[3 * pos + 0];
		d_listidx[3 * index + 1] = d_trifacelist[3 * pos + 1];
	}
	else
	{
		d_listidx[3 * index + 1] = d_trifacelist[3 * pos + 0];
		d_listidx[3 * index + 0] = d_trifacelist[3 * pos + 1];
	}
	d_listidx[3 * index + 2] = d_trifacelist[3 * pos + 2];

	d_listpt[3 * index + 0] = d_trifacecent[3 * pos + 0];
	d_listpt[3 * index + 1] = d_trifacecent[3 * pos + 1];
	d_listpt[3 * index + 2] = d_trifacecent[3 * pos + 2];
}

__global__ void kernelCompactTet_Phase1(
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_sizes,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_tetstatus[pos].isEmpty())
		d_sizes[pos] = 0;

	if (d_tetlist[4 * pos + 3] == -1)
		d_sizes[pos] = 0;
}

__global__ void kernelCompactTet_Phase2(
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_sizes,
	int* d_indices,
	int* d_listidx,
	tetstatus* d_liststatus,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_sizes[pos] == 0)
		return;

	int index = d_indices[pos];
	d_listidx[4 * index + 0] = d_tetlist[4 * pos + 0];
	d_listidx[4 * index + 1] = d_tetlist[4 * pos + 1];
	d_listidx[4 * index + 2] = d_tetlist[4 * pos + 2];
	d_listidx[4 * index + 3] = d_tetlist[4 * pos + 3];

	d_liststatus[index] = d_tetstatus[pos];
}