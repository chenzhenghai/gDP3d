#include "Mesh.h"
#include "MeshPredicates.h"
#include "MeshReconstruct.h"
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <time.h>

#define myErrchk(ans) { myAssert((ans), __FILE__, __LINE__); }
inline void myAssert(bool ret, const char *file, int line, bool abort = true)
{
	if (!ret)
	{
		fprintf(stderr, "assert failed: %s %d\n", file, line);
		if (abort) exit(0);
	}
}

// helper functions
bool tetshareface(int t1, int t2, int* tetlist)
{
	std::vector<int> list;
	for (int i = 0; i < 4; i++)
	{
		int p[2];
		p[0] = tetlist[4 * t1 + i];
		p[1] = tetlist[4 * t2 + i];
		for (int j = 0; j < 2; j++)
		{
			if (std::find(list.begin(), list.end(), p[j]) == list.end())
				list.push_back(p[j]);
		}
	}
	return (list.size() == 5);
}

bool trishareedge(int s1, int s2, int* trilist)
{
	std::vector<int> list;
	for (int i = 0; i < 3; i++)
	{
		int p[2];
		p[0] = trilist[3 * s1 + i];
		p[1] = trilist[3 * s2 + i];
		for (int j = 0; j < 2; j++)
		{
			if (std::find(list.begin(), list.end(), p[j]) == list.end())
				list.push_back(p[j]);
		}
	}
	return (list.size() == 4);
}

bool isDegenerateTet(double* pa, double *pb, double *pc, double* pd)
{
	double ret = orient3d(pa, pb, pc, pd);
	if (ret < 0.001 && ret > -0.001) // nearly degenerate
		return true;
	else
		return false;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// facenormal()    Calculate the normal of the face.                         //
//                                                                           //
// The normal of the face abc can be calculated by the cross product of 2 of //
// its 3 edge vectors.  A better choice of two edge vectors will reduce the  //
// numerical error during the calculation.  Burdakov proved that the optimal //
// basis problem is equivalent to the minimum spanning tree problem with the //
// edge length be the functional, see Burdakov, "A greedy algorithm for the  //
// optimal basis problem", BIT 37:3 (1997), 591-599. If 'pivot' > 0, the two //
// short edges in abc are chosen for the calculation.                        //
//                                                                           //
// If 'lav' is not NULL and if 'pivot' is set, the average edge length of    //
// the edges of the face [a,b,c] is returned.                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

void mymeshfacenormal(REAL* pa, REAL* pb, REAL* pc, REAL *n, int pivot,
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
		L1 = meshdot(v1, v1);
		L2 = meshdot(v2, v2);
		L3 = meshdot(v3, v3);
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
	meshcross(pv1, pv2, n);
	// Inverse the direction;
	n[0] = -n[0];
	n[1] = -n[1];
	n[2] = -n[2];
}

//////////////////////////////////////////////////////////////////////////////
//                                                                           //
// facedihedral()    Return the dihedral angle (in radian) between two       //
//                   adjoining faces.                                        //
//                                                                           //
// 'pa', 'pb' are the shared edge of these two faces, 'pc1', and 'pc2' are   //
// apexes of these two faces.  Return the angle (between 0 to 2*pi) between  //
// the normal of face (pa, pb, pc1) and normal of face (pa, pb, pc2).        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

REAL facedihedral(REAL* pa, REAL* pb, REAL* pc1, REAL* pc2)
{
	REAL n1[3], n2[3];
	REAL n1len, n2len;
	REAL costheta, ori;
	REAL theta;

	mymeshfacenormal(pa, pb, pc1, n1, 1, NULL);
	mymeshfacenormal(pa, pb, pc2, n2, 1, NULL);
	n1len = sqrt(meshdot(n1, n1));
	n2len = sqrt(meshdot(n2, n2));
	costheta = meshdot(n1, n2) / (n1len * n2len);
	// Be careful rounding error!
	if (costheta > 1.0) {
		costheta = 1.0;
	}
	else if (costheta < -1.0) {
		costheta = -1.0;
	}
	theta = acos(costheta);
	ori = orient3d(pa, pb, pc1, pc2);
	if (ori > 0.0) {
		theta = 2 * PI - theta;
	}

	return theta;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// makepoint2submap()    Create a map from vertex to subfaces incident at it.  //
//                                                                           //
// The map is returned in two arrays 'idx2faclist' and 'facperverlist'.  All //
// subfaces incident at i-th vertex (i is counted from 0) are found in the   //
// array facperverlist[j], where idx2faclist[i] <= j < idx2faclist[i + 1].   //
// Each entry in facperverlist[j] is a subface whose origin is the vertex.   //
//                                                                           //
// NOTE: These two arrays will be created inside this routine, don't forget  //
// to free them after using.                                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

void makepoint2submap(int* trilist, int*& idx2faclist,
	trihandle*& facperverlist, int numoftriface, int numofpoint)
{
	trihandle shloop;
	int i, j, k;
	//printf("  Making a map from points to subfaces.\n");

	// Initialize 'idx2faclist'.
	idx2faclist = new int[numofpoint + 1];
	for (i = 0; i < numofpoint + 1; i++) idx2faclist[i] = 0;

	// Loop all subfaces, counter the number of subfaces incident at a vertex.
	int count = 0;
	while (count < numoftriface) {
		trihandle shloop(count, 0);
		// Increment the number of incident subfaces for each vertex.
		j = trilist[3 * shloop.id];
		idx2faclist[j]++;
		j = trilist[3 * shloop.id + 1];
		idx2faclist[j]++;
		// Skip the third corner if it is a segment.
		if (trilist[3 * shloop.id + 2] != -1) {
			j = trilist[3 * shloop.id + 2];
			idx2faclist[j]++;
		}
		count++;
	}

	// Calculate the total length of array 'facperverlist'.
	j = idx2faclist[0];
	idx2faclist[0] = 0;  // Array starts from 0 element.
	for (i = 0; i < numofpoint; i++) {
		k = idx2faclist[i + 1];
		idx2faclist[i + 1] = idx2faclist[i] + j;
		j = k;
	}

	// The total length is in the last unit of idx2faclist.
	facperverlist = new trihandle[idx2faclist[i]];

	// Loop all subfaces again, remember the subfaces at each vertex.
	count = 0;
	while (count < numoftriface) {
		trihandle shloop(count, 0);
		j = trilist[3 * shloop.id];
		shloop.shver = 0; // save the origin.
		facperverlist[idx2faclist[j]] = shloop;
		idx2faclist[j]++;
		// Is it a subface or a subsegment?
		if (trilist[3 * shloop.id + 2] != -1) {
			j = trilist[3 * shloop.id + 1];
			shloop.shver = 2; // save the origin.
			facperverlist[idx2faclist[j]] = shloop;
			idx2faclist[j]++;
			j = trilist[3 * shloop.id + 2];
			shloop.shver = 4; // save the origin.
			facperverlist[idx2faclist[j]] = shloop;
			idx2faclist[j]++;
		}
		else {
			j = trilist[3 * shloop.id + 1];
			shloop.shver = 1; // save the origin.
			facperverlist[idx2faclist[j]] = shloop;
			idx2faclist[j]++;
		}
		count++;
	}

	// Contents in 'idx2faclist' are shifted, now shift them back.
	for (i = numofpoint - 1; i >= 0; i--) {
		idx2faclist[i + 1] = idx2faclist[i];
	}
	idx2faclist[0] = 0;
}

// reconstructMesh

void reconstructMesh(
	MESHIO* input_mesh,
	verttype*& outpointtypelist,
	tethandle*& outtri2tetlist,
	int& outnumoftetrahedron,
	int*& outtetlist,
	tethandle*& outneighborlist,
	trihandle*& outtet2trilist,
	bool verbose
)
{
	assert(input_mesh != NULL);
	int numofpoint = input_mesh->numofpoints;
	double* pointlist = input_mesh->pointlist;
	int numoftet = input_mesh->numoftets;
	int* tetlist = input_mesh->tetlist;
	int* tet2tetlist = input_mesh->tet2tetlist;
	int* tet2tetverlist = input_mesh->tet2tetverlist;
	int numoftriface = input_mesh->numoftrifaces;
	int* trifacelist = input_mesh->trifacelist;
	int* tri2tetlist = input_mesh->tri2tetlist;
	int* tri2tetverlist = input_mesh->tri2tetverlist;

	// Initialization
	inittables();

	std::vector<int> outtetvector(4 * numoftet);
	memcpy(outtetvector.data(), tetlist, 4 * numoftet * sizeof(int));
	verttype * pointtype = new verttype[numofpoint];
	std::vector<tethandle> outneighborvector(4 * numoftet, tethandle(-1, 11));

	// Initialize point type
	for (int i = 0; i < numofpoint; i++)
	{
		pointtype[i] = RIDGEVERTEX; // may need to change later
	}

	clock_t tv[2];

	// Create the tetrahedra and connect those that share a common face
	if (verbose)
	{
		printf("1. Create neighbor information.\n");
		tv[0] = clock();
	}
	
	int p[4], q[4], neighidx;
	bool found = false;
	tethandle tetloop, checktet;
	for (int i = 0; i < numoftet; i++)
	{
		tetloop.id = i;
		for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++)
		{
			p[3] = oppo(tetloop, outtetvector.data()); // d

			if (outneighborvector[4 * tetloop.id + tetloop.ver].id != -1)
				continue;

			neighidx = tet2tetlist[4 * tetloop.id + tetloop.ver];
			if (neighidx == -1)
				continue;

			p[0] = org(tetloop, outtetvector.data()); // a
			p[1] = dest(tetloop, outtetvector.data()); // b
			p[2] = apex(tetloop, outtetvector.data()); // c

			// let checktet oppo become 'c'
			checktet.id = neighidx;
			checktet.ver = tet2tetverlist[4 * tetloop.id + tetloop.ver];

			// let checktet share the edge org->dest with tetloop
			for (int j = 0; j < 3; j++)
			{
				q[0] = org(checktet, outtetvector.data());
				q[1] = dest(checktet, outtetvector.data());
				if (q[0] == p[1] && q[1] == p[0]) // found the edge
				{
					bond(tetloop, checktet, outneighborvector.data());
					found = true;
					break;
				}
				enextself(checktet);
			}
			myErrchk(found);
			found = false;

			//for (checktet.ver = 0; checktet.ver < 4; checktet.ver++)
			//{
			//	q[3] = oppo(checktet, outtetvector.data());
			//	if (q[3] == p[2])
			//	{
			//		found = true;
			//		break;
			//	}
			//}
			//assert(found);
			//found = false;

			//// let checktet share the edge org->dest with tetloop
			//for (int j = 0; j < 3; j++)
			//{
			//	q[0] = org(checktet, outtetvector.data());
			//	q[1] = dest(checktet, outtetvector.data());
			//	if (q[0] == p[0] && q[1] == p[1]) // found the edge
			//	{
			//		esymself(checktet);
			//		bond(tetloop, checktet, outneighborvector.data());
			//		found = true;
			//		break;
			//	}
			//	enextself(checktet);
			//}
			//assert(found);
			//found = false;
		}
	}

	if (verbose)
	{
		tv[1] = clock();
		printf("time: %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Create hull tets
	if(verbose)
		printf("2. Create hull tets\n");
	int hullsize, outnumoftet = numoftet;
	int count = 0;
	while (count < outnumoftet)
	{
		tethandle tetloop(count, 11);
		tethandle tptr = tetloop;
		for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++)
		{
			if (outneighborvector[4 * tetloop.id + tetloop.ver].id == -1) // empty neighbor
			{
				// Create a hull tet.
				outneighborvector.push_back(tethandle(-1, 11));
				outneighborvector.push_back(tethandle(-1, 11));
				outneighborvector.push_back(tethandle(-1, 11));
				outneighborvector.push_back(tethandle(-1, 11));
				tethandle hulltet(outnumoftet, 11);
				int p[3];
				p[0] = org(tetloop, outtetvector.data());
				p[1] = dest(tetloop, outtetvector.data());
				p[2] = apex(tetloop, outtetvector.data());
				outtetvector.push_back(p[1]);
				outtetvector.push_back(p[0]);
				outtetvector.push_back(p[2]);
				outtetvector.push_back(-1);
				outnumoftet++;
				bond(tetloop, hulltet, outneighborvector.data());
				// Try connecting this to others that share common hull edges.
				for (int j = 0; j < 3; j++)
				{
					tethandle face1, face2;
					fsym(hulltet, face2, outneighborvector.data());
					while (1)
					{
						if (face2.id == -1)
							break;
						esymself(face2);
						if (apex(face2, outtetvector.data()) == -1)
							break;
						fsymself(face2, outneighborvector.data());
					}
					if (face2.id != -1)
					{
						// Found an adjacent hull tet.
						esym(hulltet, face1);
						bond(face1, face2, outneighborvector.data());
					}
					enextself(hulltet);
				}
			}
			// Update the point-to-tet map.
			int idx = outtetvector[4 * tetloop.id + tetloop.ver];
		}
		count++;
	}
	hullsize = outnumoftet - numoftet;
	if (verbose)
	{
		printf("Hull size = %d\n", hullsize);
		tv[0] = clock();
		printf("time: %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Subfaces will be inserted into the mesh
	if (verbose)
		printf("3. Insert subfaces\n");

	int outnumofface = numoftriface;
	trihandle* tet2trivector = new trihandle[4 * outnumoftet];
	tethandle* tri2tetvector = new tethandle[2 * outnumofface];
	for (int i = 0; i < 4 * outnumoftet; i++)
		tet2trivector[i] = trihandle(-1, 0);

	for (int i = 0; i < outnumofface; i++)
	{
		// Variables
		int j;
		int p[3], q[2];
		int neigh, ver;
		tethandle checktet, tetloop;
		trihandle neighsh, subloop;
		// Get endpoints
		for (j = 0; j < 3; j++)
		{
			p[j] = trifacelist[3 * i + j];
		}
		// Search the subface.
		tetloop.id = tri2tetlist[i];
		tetloop.ver = tri2tetverlist[i];
		bool found = false;
		for (j = 0; j < 3; j++)
		{
			if (apex(tetloop, outtetvector.data()) == p[2]) {
				q[0] = org(tetloop, outtetvector.data());
				q[1] = dest(tetloop, outtetvector.data());
				if ((q[0] == p[0] && q[1] == p[1]) || (q[0] == p[1] && q[1] == p[0]))
				{
					found = true;
					// Found the face.
					// Check if there exist a subface already?
					tspivot(tetloop, neighsh, tet2trivector);
					myErrchk(neighsh.id == -1);
					if (q[0] == p[1])
						fsymself(tetloop, outneighborvector.data());
					break;
				}
			}
			enextself(tetloop);
		}
		myErrchk(found);

		// bind suface and tetrahedra
		// Create a new subface.
		subloop.id = i;
		subloop.shver = 0;
		// Create the point-to-subface map.
		for (j = 0; j < 3; j++) {
			pointtype[p[j]] = FACETVERTEX; // initial type.
		}
		// Insert the subface into the mesh.
		tsbond(tetloop, subloop, tet2trivector, tri2tetvector);
		fsymself(tetloop, outneighborvector.data());
		sesymself(subloop);
		tsbond(tetloop, subloop, tet2trivector, tri2tetvector);
	}

	// Output all the needed information
	outpointtypelist = pointtype;
	outtri2tetlist = tri2tetvector;
	outnumoftetrahedron = outnumoftet;
	outtetlist = new int[4 * outnumoftetrahedron];
	std::copy(outtetvector.begin(), outtetvector.end(), outtetlist);
	outneighborlist = new tethandle[4 * outnumoftetrahedron];
	std::copy(outneighborvector.begin(), outneighborvector.end(), outneighborlist);
	outtet2trilist = tet2trivector;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// makesegment2parentmap()    Create a map from a segment to its parent.     //
//                                                                           //
// The map is saved in the array 'segment2parentlist' and					 //
// 'segmentendpointslist'. 													 //
// The length of 'segmentendpointslist'	is twice the number of segments.     //
// Each segment is assigned a unique index (starting from 0).                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

void makesegment2parentmap(
	int numofsegment,
	int* segmentlist,
	trihandle* seg2trilist,
	int*& segment2parentlist,
	int*& segmentendpointslist,
	int& numofparent
)
{
	trihandle segloop, prevseg, nextseg;
	int eorg, edest;
	int segindex = 0, idx = 0;
	int i, count;

	segment2parentlist = new int[numofsegment];
	std::vector<int> segptvec;

	// A segment s may have been split into many subsegments. Operate the one
	//   which contains the origin of s. Then mark the rest of subsegments.
	segloop.id = count = 0;
	segloop.shver = 0;
	while (count < numofsegment) {
		senext2(segloop, prevseg);
		spivotself(prevseg, seg2trilist);
		if (prevseg.id == -1) {
			eorg = sorg(segloop, segmentlist);
			edest = sdest(segloop, segmentlist);
			segment2parentlist[segloop.id] = segindex;
			senext(segloop, nextseg);
			spivotself(nextseg, seg2trilist);
			while (nextseg.id != -1) {
				segment2parentlist[nextseg.id] = segindex;
				nextseg.shver = 0;
				if (sorg(nextseg, segmentlist) != edest) sesymself(nextseg);
				assert(sorg(nextseg) == edest);
				edest = sdest(nextseg, segmentlist);
				// Go the next connected subsegment at edest.
				senextself(nextseg);
				spivotself(nextseg, seg2trilist);
			}
			segptvec.push_back(eorg);
			segptvec.push_back(edest);
			segindex++;
		}
		segloop.id = ++count;
	}

	segmentendpointslist = new int[2 * segindex];
	std::copy(segptvec.begin(), segptvec.end(), segmentendpointslist);
	numofparent = segindex;

	// debug
	//for (i = 0; i < numofsegment; i++)
	//{
	//	printf("seg #%d - %d\n", i, segment2parentlist[i]);
	//}
	//for (i = 0; i < segindex; i++)
	//{
	//	printf("seg parent #%d - %d, %d\n", i, segmentendpointslist[2 * i], segmentendpointslist[2 * i + 1]);
	//}
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// makesubfacepointsmap()    Create a map from facet to its vertices.        //
//                                                                           //
// All facets will be indexed (starting from 0).  The map is saved in three  //
// arrays: 'subface2parentlist', 'id2subfacelist', and 'subfacepointslist'.  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

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
)
{
	trihandle subloop, neighsh, parysh, parysh1;
	int pa, *ppt, parypt;
	verttype vt;
	int facetindex, totalvertices;
	int i, j, k, count;

	std::vector<std::vector<int>> facetvertexlist;
	facetindex = totalvertices = 0;
	std::vector<int> pmarker(numofpoint, 0);
	std::vector<int> smarker(numofsubface, 0);

	subface2parentlist = new int[numofsubface];

	subloop.id = count = 0;
	std::vector<trihandle> shlist;
	std::vector<int> vertlist;
	while (count < numofsubface) {
		if (smarker[subloop.id] == 0) {
			// A new facet. Create its vertices list.
			vertlist.clear();
			ppt = subfacelist + 3 * subloop.id;
			for (k = 0; k < 3; k++) {
				vt = pointtypelist[ppt[k]];
				if ((vt != FREESEGVERTEX) && (vt != FREEFACETVERTEX)) {
					pmarker[ppt[k]] = 1;
					vertlist.push_back(ppt[k]);
				}
			}
			smarker[subloop.id] = 1;
			shlist.push_back(subloop);
			for (i = 0; i < shlist.size(); i++) {
				parysh = shlist[i];
				subface2parentlist[parysh.id] = facetindex;
				for (j = 0; j < 3; j++) {
					if (!isshsubseg(parysh, subface2seglist)) {
						spivot(parysh, neighsh, subface2subfacelist);
						assert(neighsh.sh != NULL);
						//if (subloop.id == 27)
						//{
						//	int tmp[4];
						//	REAL* ctmp[4];
						//	REAL ang;
						//	tmp[0] = sorg(parysh, subfacelist);
						//	tmp[1] = sdest(parysh, subfacelist);
						//	tmp[2] = sapex(parysh, subfacelist);
						//	tmp[3] = sapex(neighsh, subfacelist);
						//	for (k = 0; k < 4; k++)
						//	{
						//		ctmp[k] = id2pointlist(tmp[k], pointlist);
						//	}
						//	ang = facedihedral(ctmp[0], ctmp[1], ctmp[2], ctmp[3]);
						//	if (ang > PI)
						//		ang = 2 * PI - ang;
						//	printf("Subface #%d - Edge %d, %d - ang = %g\n",
						//		parysh.id, tmp[0], tmp[1], ang);
						//}
						if (smarker[neighsh.id] == 0) {
							pa = sapex(neighsh, subfacelist);
							if (pmarker[pa] == 0) {
								vt = pointtypelist[pa];
								if ((vt != FREESEGVERTEX) && (vt != FREEFACETVERTEX)) {
									pmarker[pa] = 1;
									vertlist.push_back(pa);
								}
							}
							smarker[neighsh.id] = 1;
							shlist.push_back(neighsh);
						}
					}
					senextself(parysh);
				}
				//if(subloop.id == 27)
				//	printf("Subface #%d - %d, %d, %d\n", parysh.id, subfacelist[3*parysh.id], subfacelist[3 * parysh.id+1], subfacelist[3 * parysh.id+2]);
			} // i
			totalvertices += vertlist.size();
			// Uninfect facet vertices.
			for (k = 0; k < vertlist.size(); k++) {
				parypt = vertlist[k];
				pmarker[parypt] = 0;
			}
			//if (vertlist.size() != 3)
			//	printf("triface #%d - %d - %d, %d, %d\n", count, vertlist.size(), subfacelist[3 * count], subfacelist[3 * count + 1], subfacelist[3 * count + 2]);
			shlist.clear();
			// Save this vertex list.
			facetvertexlist.push_back(vertlist);
			facetindex++;
		}
		subloop.id = ++count;
	}

	id2subfacelist = new int[facetindex + 1];
	subfacepointslist = new int[totalvertices];

	id2subfacelist[0] = 0;
	for (i = 0, k = 0; i < facetindex; i++) {
		vertlist = facetvertexlist[i];
		id2subfacelist[i + 1] = (id2subfacelist[i] + vertlist.size());
		for (j = 0; j < vertlist.size(); j++) {
			parypt = vertlist[j];
			subfacepointslist[k] = parypt;
			k++;
		}
		//if (vertlist.size() == 4)
		//	printf("Ret = %g\n",
		//		orient3d(
		//			id2pointlist(vertlist[0], pointlist), 
		//			id2pointlist(vertlist[1], pointlist),
		//			id2pointlist(vertlist[2], pointlist),
		//			id2pointlist(vertlist[3], pointlist))
		//	);
	}
	assert(k == totalvertices);

	numofparent = facetindex;
	numofendpoints = totalvertices;
}