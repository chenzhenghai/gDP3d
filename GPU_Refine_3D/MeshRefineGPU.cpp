#include "MeshRefine.h"
#include "CudaRefine.h"

// AABB tree
typedef Mesh_domain::AABB_tree Tree;
typedef Tree::Bounding_box Box;
typedef CGAL::AABB_node<Tree::AABB_traits> Node;
typedef Tree::AABB_traits::Primitive Primitive;
typedef std::vector<Primitive> Primitives;
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;

void analyzeCGALTr(C3t3& c3t3)
{
	int numberofvertices = c3t3.triangulation().number_of_vertices();
	int numberoffaces = c3t3.triangulation().number_of_finite_facets();
	int numberoftets = c3t3.triangulation().number_of_finite_cells();
	printf("Number of vertices = %d\n", numberofvertices);
	printf("Number of triangular faces = %d\n", numberoffaces);
	printf("Number of tetrahedra = %d\n", numberoftets);

	int numberoffacets = c3t3.number_of_facets_in_complex();
	printf("Number of facets in complex = %d\n", numberoffacets);
	int created_by_ray = 0;
	int counter = 0;
	for (C3t3::Facets_in_complex_iterator it = c3t3.facets_in_complex_begin(),
		end = c3t3.facets_in_complex_end(); it != end; ++it)
	{
		Tr::Facet facet = *it;
		Tr::Cell_handle c = facet.first;
		int i = facet.second;
		Tr::Cell_handle n = c->neighbor(i);
		//std::cerr << counter << " "
		//	<< "(" << *facet.first->vertex((facet.second + 1) % 4) << ") "
		//	<< "(" << *facet.first->vertex((facet.second + 2) % 4) << ") "
		//	<< "(" << *facet.first->vertex((facet.second + 3) % 4) << ") ";
		counter++;
		if (c3t3.triangulation().is_infinite(c) || c3t3.triangulation().is_infinite(n))
		{
			//std::cerr << "R";
			created_by_ray++;
		}
		else
		{
			//std::cerr << "S";
		}

		//std::cerr << std::endl;
	}
	printf("  Added by segment:  %d\n", numberoffacets - created_by_ray);
	printf("  Added by ray:      %d\n", created_by_ray);

	int numberofcells = c3t3.number_of_cells_in_complex();
	printf("Number of cells in complex = %d\n", numberofcells);
}

void verifyTetNeighbor(MESHIO* meshio)
{
	for (int j = 0; j < meshio->numoftets; j++)
	{
		bool found = false;
		for (int i = 0; i <= 3; i++)
		{
			int neighid = meshio->tet2tetlist[4 * j + i];
			if (neighid != -1)
			{
				for (int k = 0; k <= 3; k++)
				{
					int nneighid = meshio->tet2tetlist[4 * neighid + k];
					if (nneighid == j)
					{
						found = true;
						break;
					}
				}
				if (!found)
				{
					printf("Tet #%d - %d, %d, %d, %d\n", j,
						meshio->tet2tetlist[4 * j + 0], meshio->tet2tetlist[4 * j + 1],
						meshio->tet2tetlist[4 * j + 2], meshio->tet2tetlist[4 * j + 3]);
					printf("Tet #%d - %d, %d, %d, %d\n", neighid,
						meshio->tet2tetlist[4 * neighid + 0], meshio->tet2tetlist[4 * neighid + 1],
						meshio->tet2tetlist[4 * neighid + 2], meshio->tet2tetlist[4 * neighid + 3]);
					printf("\n");
				}
			}
		}
	}
}

void convertCGALTr2Tr(
	C3t3& c3t3,
	MESHIO *meshio
)
{
	int numberofpoints = c3t3.triangulation().number_of_vertices();
	int numberoftets = c3t3.triangulation().number_of_finite_cells();
	int numberoffacets = c3t3.number_of_facets_in_complex();
	int numberoftets_indomain = c3t3.number_of_cells_in_complex();

	int counter = 0;

	// Set up point index hash table, coordinate and weight list
	std::unordered_map<Tr::Vertex_handle, int> hash_ptidx;
	meshio->numofpoints = numberofpoints;
	meshio->pointlist = new double[3 * numberofpoints];
	meshio->weightlist = new double[numberofpoints];
	for (Tr::Finite_vertices_iterator it = c3t3.triangulation().finite_vertices_begin(),
		end = c3t3.triangulation().finite_vertices_end(); it != end; ++it)
	{
		Tr::Vertex_handle vh = it;
		hash_ptidx.insert({vh, counter});
		Tr::Weighted_point wp = vh->point();
		meshio->pointlist[3 * counter + 0] = wp.x();
		meshio->pointlist[3 * counter + 1] = wp.y();
		meshio->pointlist[3 * counter + 2] = wp.z();
		meshio->weightlist[counter] = wp.weight();
		counter++;
	}
	cpuErrchk(numberofpoints == counter);

	// Set up tet index hash table and vertex list
	counter = 0;
	std::unordered_map<Tr::Cell_handle, int> hash_tetidx;
	meshio->numoftets = numberoftets;
	meshio->tetlist = new int[4 * numberoftets];
	meshio->tetstatuslist = new tetstatus[numberoftets];
	for (Tr::Finite_cells_iterator it = c3t3.triangulation().finite_cells_begin(),
		end = c3t3.triangulation().finite_cells_end(); it != end; ++it)
	{
		Tr::Cell_handle cell = it;
		hash_tetidx.insert({ cell, counter });

		meshio->tetstatuslist[counter].setEmpty(false);

		// The four vertices of a cell are indexed with 0, 1, 2 and 3 in positive orientation
		// https://doc.cgal.org/latest/Triangulation_3/index.html#fig__Triangulation3figorient
		// which means they are already oriented correctly.
		// Point 3 is above plane 0, 1, 2, which are in counterclockwise order.
		// So orient3d(0, 1, 2, 3) is always non-positive and fits our need.
		for (int j = 0; j <= 3; j++)
		{
			Tr::Vertex_handle vh = cell->vertex(j);
			auto search = hash_ptidx.find(vh);
			cpuErrchk(search != hash_ptidx.end());
			meshio->tetlist[4 * counter + j] = search->second;
		}

		counter++;
	}
	cpuErrchk(numberoftets == counter);

	// Set up tet neighbor list
	counter = 0;
	meshio->tet2tetlist = new int[4 * numberoftets];
	meshio->tet2tetverlist = new int[4 * numberoftets];
	for (Tr::Finite_cells_iterator it = c3t3.triangulation().finite_cells_begin(),
		end = c3t3.triangulation().finite_cells_end(); it != end; ++it)
	{
		Tr::Cell_handle cell = it;
		for (int j = 0; j <= 3; j++)
		{
			Tr::Facet mirror_f =
				c3t3.triangulation().mirror_facet(std::make_pair(cell, j));
			Tr::Cell_handle nh = mirror_f.first;
			//Tr::Cell_handle nh = cell->neighbor(j);
			if (c3t3.triangulation().is_infinite(nh))
			{
				meshio->tet2tetlist[4 * counter + j] = -1;
			}
			else
			{
				auto search = hash_tetidx.find(nh);
				cpuErrchk(search != hash_tetidx.end());
				meshio->tet2tetlist[4 * counter + j] = search->second;
				meshio->tet2tetverlist[4 * counter + j] = mirror_f.second;
			}
		}
		counter++;
	}
	cpuErrchk(numberoftets == counter);
	//verifyTetNeighbor(meshio);

	// Set up tetstatus
	counter = 0;
	for (C3t3::Cells_in_complex_iterator it = c3t3.cells_in_complex_begin(),
		end = c3t3.cells_in_complex_end(); it != end; ++it)
	{
		Tr::Cell_handle cell = it;
		auto search = hash_tetidx.find(cell);
		cpuErrchk(search != hash_tetidx.end());
		meshio->tetstatuslist[search->second].setInDomain(true);
	}

	// Set up triface list
	counter = 0;
	meshio->numoftrifaces = numberoffacets;
	meshio->trifacelist = new int[3 * numberoffacets];
	meshio->trifacecentlist = new double[3 * numberoffacets];
	meshio->tri2tetlist = new int[numberoffacets];
	meshio->tri2tetverlist = new int[numberoffacets];
	for (C3t3::Facets_in_complex_iterator it = c3t3.facets_in_complex_begin(),
		end = c3t3.facets_in_complex_end(); it != end; ++it)
	{
		Tr::Facet facet = *it;
		Tr::Cell_handle cell = facet.first;
		int i = facet.second;
		int k = i;

		for (int j = 0; j <= 2; j++)
		{
			Tr::Vertex_handle vh = cell->vertex((i + j + 1) % 4);
			auto search = hash_ptidx.find(vh);
			cpuErrchk(search != hash_ptidx.end());
			meshio->trifacelist[3 * counter + j] = search->second;
		}

		auto search = hash_tetidx.find(cell);
		if (search == hash_tetidx.end()) // need to check the other side
		{
			Tr::Facet mirror_f =
				c3t3.triangulation().mirror_facet(std::make_pair(cell, i));
			Tr::Cell_handle ncell = mirror_f.first;
			k = mirror_f.second;
			search = hash_tetidx.find(ncell);
		}
		cpuErrchk(search != hash_tetidx.end());
		meshio->tri2tetlist[counter] = search->second;
		meshio->tri2tetverlist[counter] = k;
		
		meshio->trifacecentlist[3 * counter + 0] = cell->get_facet_surface_center(i).x();
		meshio->trifacecentlist[3 * counter + 1] = cell->get_facet_surface_center(i).y();
		meshio->trifacecentlist[3 * counter + 2] = cell->get_facet_surface_center(i).z();

		counter++;
	}
}

void flattenAABBTree(const Node* node, int nb_primitives, int& ncounter, int& pcounter,
	int parentId, bool leftchild, std::vector<const Node*> & nodelist, std::vector<int>& lchildlist, 
	std::vector<int>& rchildlist, std::vector<Primitive>& pmlist)
{
	int nid = ++ncounter;
	nodelist.push_back(node);
	lchildlist.push_back(0); // set to 0 first
	rchildlist.push_back(0);
	if (parentId != -1) // not the root node
	{
		// the parent node should be in lists already
		// set its left/right child to the current node
		if (leftchild)
			lchildlist[parentId] = nid;
		else
			rchildlist[parentId] = nid;
	}

	int pid;
	switch (nb_primitives)
	{
	case 2:
	{
		Primitive left = node->left_data();
		pid = ++pcounter;
		lchildlist[nid] = -(pid + 1); // use negative value to indicate leaf
		pmlist.push_back(left);

		Primitive right = node->right_data();
		pid = ++pcounter;
		rchildlist[nid] = -(pid + 1);
		pmlist.push_back(right);
		break;
	}
	case 3:
	{
		Primitive left = node->left_data();
		pid = ++pcounter;
		lchildlist[nid] = -(pid + 1);
		pmlist.push_back(left);

		flattenAABBTree(&(node->right_child()), 2, ncounter, pcounter, nid, false,
			nodelist, lchildlist, rchildlist, pmlist);
		break;
	}
	default:
		flattenAABBTree(&(node->left_child()), nb_primitives / 2, ncounter, pcounter, nid, true,
			nodelist, lchildlist, rchildlist, pmlist);
		flattenAABBTree(&(node->right_child()), nb_primitives - nb_primitives / 2, ncounter, 
			pcounter, nid, false, nodelist, lchildlist, rchildlist, pmlist);
		break;
	}
}

void verifyAABBTree(Mesh_domain& domain, std::vector<const Node*> nodelist, 
	std::vector<int> lchildlist, std::vector<int> rchildlist, std::vector<Primitive> pmlist)
{
	int nb_primitives = domain.aabb_tree().size();
	cpuErrchk(nb_primitives == pmlist.size());
	
	int nb_nodes = nodelist.size();
	const Node *node, *lnode, *rnode;
	int lchild, rchild;
	bool ret;
	for (int i = 0; i < nb_nodes; i++)
	{
		node = nodelist[i];
		lchild = lchildlist[i];
		rchild = rchildlist[i];
		if (lchild >= 0)
		{
			lnode = nodelist[lchild];
			cpuErrchk(lnode == &(node->left_child()));
		}
		else
		{
			Primitive left = pmlist[-lchild - 1];
			cpuErrchk(left.id() == node->left_data().id());
		}
		if (rchild >= 0)
		{
			rnode = nodelist[rchild];
			cpuErrchk(rnode == &(node->right_child()));
		}
		else
		{
			Primitive right = pmlist[-rchild - 1];
			cpuErrchk(right.id() == node->right_data().id());
		}
	}
}

void convertDomain(
	Mesh_domain& domain,
	MESHIO *meshio
)
{
	// flatten aabb tree
	const Node* root_node = domain.aabb_tree().my_get_root_node();
	int nb_primitives = domain.aabb_tree().size();
	int ncounter = -1, pcounter = -1;
	std::vector<const Node*> vc_nodes;
	std::vector<int> vc_lchilds;
	std::vector<int> vc_rchilds;
	std::vector<Primitive> vc_pms;
	vc_nodes.reserve(nb_primitives - 1);
	vc_lchilds.reserve(nb_primitives - 1);
	vc_rchilds.reserve(nb_primitives - 1);
	vc_pms.reserve(nb_primitives);

	flattenAABBTree(root_node, nb_primitives, ncounter, pcounter, -1, false,
		vc_nodes, vc_lchilds, vc_rchilds, vc_pms);
	
	//verifyAABBTree(domain, vc_nodes, vc_lchilds, vc_rchilds, vc_pms);

	// set up nodes' bounding boxes
	int i, j;
	std::vector<double> vc_nodebbs;
	int nb_nodes = vc_nodes.size();
	for (i = 0; i < nb_nodes; i++)
	{
		const Node* n = vc_nodes[i];
		vc_nodebbs.push_back(n->bbox().xmin());
		vc_nodebbs.push_back(n->bbox().xmax());
		vc_nodebbs.push_back(n->bbox().ymin());
		vc_nodebbs.push_back(n->bbox().ymax());
		vc_nodebbs.push_back(n->bbox().zmin());
		vc_nodebbs.push_back(n->bbox().zmax());
		if (i == 0) // compute diagonal length
		{
			double bb[3], bt[3], dvec[3], dlen;
			bb[0] = n->bbox().xmin(); bb[1] = n->bbox().ymin(); bb[2] = n->bbox().zmin();
			bt[0] = n->bbox().xmax(); bt[1] = n->bbox().ymax(); bt[2] = n->bbox().zmax();
			dvec[0] = bt[0] - bb[0]; dvec[1] = bt[1] - bb[1]; dvec[2] = bt[2] - bb[2];
			dlen = sqrt(dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]);
			meshio->aabb_diglen = dlen;
			meshio->aabb_xmin = bb[0];
			meshio->aabb_ymin = bb[1];
			meshio->aabb_zmin = bb[2];
			meshio->aabb_xmax = bt[0];
			meshio->aabb_ymax = bt[1];
			meshio->aabb_zmax = bt[2];
		}
	}

	// set up primitives and their bounding boxes
	std::vector<double> vc_pmcs;
	std::vector<double> vc_pmbbs;
	for (i = 0; i < nb_primitives; i++)
	{
		Primitive p = vc_pms[i];
		CGAL::Triangle_3<CGAL::Epick> tg =
			p.datum(domain.aabb_tree().traits().shared_data());
		for (j = 0; j < 3; j++)
		{
			CGAL::Point_3<CGAL::Epick> pt = tg.vertex(j);
			vc_pmcs.push_back(pt.x());
			vc_pmcs.push_back(pt.y());
			vc_pmcs.push_back(pt.z());
		}
		vc_pmbbs.push_back(tg.bbox().xmin());
		vc_pmbbs.push_back(tg.bbox().xmax());
		vc_pmbbs.push_back(tg.bbox().ymin());
		vc_pmbbs.push_back(tg.bbox().ymax());
		vc_pmbbs.push_back(tg.bbox().zmin());
		vc_pmbbs.push_back(tg.bbox().zmax());
	}
	double log2 = log(nb_primitives) / log(2);
	int tree_level = ceil(log2);
	meshio->aabb_level = tree_level;

	// May use index-based structure to compress the domain by using unordered_map 
	// if it is too big. Currently, we record three points for one primitive directly.

	// Copy to meshio
	meshio->numofaabbnodes = nb_nodes;
	meshio->aabb_nodeleftchild = new int[nb_nodes];
	std::copy(vc_lchilds.begin(), vc_lchilds.end(), meshio->aabb_nodeleftchild);
	meshio->aabb_noderightchild = new int[nb_nodes];
	std::copy(vc_rchilds.begin(), vc_rchilds.end(), meshio->aabb_noderightchild);
	meshio->aabb_nodebbs = new double[6 * nb_nodes];
	std::copy(vc_nodebbs.begin(), vc_nodebbs.end(), meshio->aabb_nodebbs);
	meshio->numofaabbpms = nb_primitives;
	meshio->aabb_pmcoord = new double[9 * nb_primitives];
	std::copy(vc_pmcs.begin(), vc_pmcs.end(), meshio->aabb_pmcoord);
	meshio->aabb_pmbbs = new double[6 * nb_primitives];
	std::copy(vc_pmbbs.begin(), vc_pmbbs.end(), meshio->aabb_pmbbs);
}

void outputFacets2OFF(
	int numofpoint,
	double* pointlist,
	int numoftriface,
	int* trifacelist,
	char* filename
)
{
	// Save information into the OFF format
	FILE * fp;
	fp = fopen(filename, "w");
	fprintf(fp, "OFF\n");
	fprintf(fp, "%d %d %d\n", numofpoint, numoftriface, 0);
	for (int i = 0; i < numofpoint; i++)
	{
		fprintf(fp, "%lf %lf %lf\n", pointlist[3 * i], pointlist[3 * i + 1], pointlist[3 * i + 2]);
	}
	for (int i = 0; i < numoftriface; i++)
	{
		fprintf(fp, "%d  %d %d %d\n", 3,
			trifacelist[3 * i + 0], trifacelist[3 * i + 1], trifacelist[3 * i + 2]);
	}
	fclose(fp);
}

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
)
{
	// Save information into the mesh format
	FILE * fp;
	fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("Cannot write to file!");
		exit(0);
	}

	fprintf(fp, "MeshVersionFormatted 1\n");
	fprintf(fp, "Dimension 3\n");

	fprintf(fp, "Vertices\n");
	fprintf(fp, "%d\n", numofpoint);
	for (int i = 0; i < numofpoint; i++)
		fprintf(fp, "%lf %lf %lf 1\n", pointlist[3 * i + 0], pointlist[3 * i + 1], pointlist[3 * i + 2]);

	fprintf(fp, "Triangles\n");
	fprintf(fp, "%d\n", numoftriface);
	for (int i = 0; i < numoftriface; i++)
		fprintf(fp, "%d %d %d 1\n", trifacelist[3 * i + 0] + 1, trifacelist[3 * i + 1] + 1, trifacelist[3 * i + 2] + 1);

	fprintf(fp, "Tetrahedra\n");
	fprintf(fp, "%d\n", numoftet_indomain);
	for (int i = 0; i < numoftet; i++)
	{
		if(tetstatus[i].isInDomain())
			fprintf(fp, "%d %d %d %d 3\n", tetlist[4 * i + 0] + 1, tetlist[4 * i + 1] + 1, tetlist[4 * i + 2] + 1, tetlist[4 * i + 3] + 1);
	}
	fprintf(fp, "End\n");

	fclose(fp);
}

void elementStatistics(MESHIO* meshio, MESHCR* criteria, int& numberofbadfacets,
	int& numberofbadtets, double step, int& numberofslots, int*& fangledist, int*& tangledist)
{
	// angle distribution 
	numberofslots = 180.0 / step + 1;
	int i, j, slotIdx;
	int ip[4];
	double* p[4];
	double allangles[6];
	fangledist = new int[numberofslots];
	tangledist = new int[numberofslots];
	for (i = 0; i < numberofslots; i++)
	{
		fangledist[i] = 0;
		tangledist[i] = 0;
	}

	// count bad facets and angle distribution
	int badfacetcounter = 0;
	double center[3], w[4];
	for (i = 0; i < meshio->out_numoftrifaces; i++)
	{
		for (j = 0; j < 3; j++)
		{
			ip[j] = meshio->out_trifacelist[3 * i + j];
			p[j] = meshio->out_pointlist + 3 * ip[j];
			w[j] = meshio->out_weightlist[ip[j]];
			center[j] = meshio->out_trifacecent[3 * i + j];
		}
		if (isBadFacet(p[0], p[1], p[2], w[0], w[1], w[2], center,
			criteria->facet_angle, criteria->facet_size, criteria->facet_distance))
			badfacetcounter++;
		calAngles(p, allangles);
		for (j = 0; j < 3; j++)
		{
			slotIdx = allangles[j] / step;
			fangledist[slotIdx]++;
		}
	}
	numberofbadfacets = badfacetcounter;

	// count bad tets and dihedral angle distribution
	int badtetcounter = 0;
	for (i = 0; i < meshio->out_numoftets; i++)
	{
		if (!(meshio->out_tetstatus[i].isInDomain()))
			continue;
		for (j = 0; j < 4; j++)
		{
			ip[j] = meshio->out_tetlist[4 * i + j];
			p[j] = meshio->out_pointlist + 3 * ip[j];
		}
		if (isBadTet(p[0], p[1], p[2], p[3], criteria->cell_radius_edge_ratio, criteria->cell_size))
			badtetcounter++;
		calDihedral(p, allangles);
		for (j = 0; j < 6; j++)
		{
			slotIdx = allangles[j] / step;
			tangledist[slotIdx]++;
		}
	}
	numberofbadtets = badtetcounter;
}

void resultStatistics(MESHIO* meshio, MESHCR* criteria, MESHBH* behavior, double* times, char* filename)
{
	int numberofvertices = meshio->out_numofpoints;
	int numberoftets = meshio->out_numoftets;
	printf("Number of vertices = %d\n", numberofvertices);
	printf("Number of triangular faces = 0\n");
	printf("Number of tetrahedra = %d\n", numberoftets);

	int numberofbadfacets, numberofbadtets;
	int numberofslots, *fangledist, *tangledist;
	double step = 1.0;
	elementStatistics(meshio, criteria, numberofbadfacets, numberofbadtets, step,
		numberofslots, fangledist, tangledist);

	int numberoffacets = meshio->out_numoftrifaces;
	printf("Number of facets in complex = %d\n", numberoffacets);
	printf("Number of bad facets in complex = %d\n", numberofbadfacets);
	int numberofcells = meshio->out_numoftets_indomain;
	//int numberofcells = 0;
	//for (int i = 0; i < meshio->out_numoftets; i++)
	//{
	//	if (meshio->out_tetstatus[i].isInDomain())
	//		numberofcells++;
	//}
	printf("Number of cells in complex = %d\n", numberofcells);
	printf("Number of bad cells in complex = %d\n", numberofbadtets);

	printf("Total time = %lf\n", times[6]);
	printf("  CPU time = %lf\n", times[4]);
	printf("    Initialization time = %lf\n", times[0]);
#ifndef GQM3D_WITHOUT_1D_FEATURE
	double edge_protect_time = intertimer["Facets_scan_time"] +
		intertimer["Facets_refin_time"] + intertimer["Cells_scan_time"] +
		intertimer["Cells_refin_time"];
	edge_protect_time = times[1] - edge_protect_time * 1000;
	printf("    Edge protection time = %lf\n", edge_protect_time);
#endif
	printf("    Facet scan time = %lf\n", intertimer["Facets_scan_time"] * 1000);
	printf("    Facet refinement time = %lf\n", intertimer["Facets_refin_time"] * 1000);
	printf("    Cell scan time = %lf\n", intertimer["Cells_scan_time"] * 1000);
	printf("    Cell refinement time = %lf\n", intertimer["Cells_refin_time"] * 1000);
	printf("    Triangulation conversion time = %lf\n", times[2]);
	printf("    Domain conversion time = %lf\n", times[3]);
	printf("  GPU time = %lf\n", times[5]);
	printf("    Reconstruction time (CPU) = %lf\n", behavior->times[0]);
	printf("    Initialization time = %lf\n", behavior->times[1]);
	printf("    Elements refinement time = %lf\n", behavior->times[2]);
	printf("      Initialize quality time = %lf\n", behavior->times[4]);
	printf("      Splitting subfaces time = %lf\n", behavior->times[5]);
	printf("      Set tet status time = %lf\n", behavior->times[6]);
	printf("      Splitting tets time = %lf\n", behavior->times[7]);
	printf("    Mesh compaction time = %lf\n", behavior->times[3]);

	if (filename == NULL)
		return;

	FILE * fp;
	fp = fopen(filename, "w");
	fprintf(fp, "Number of vertices = %d\n", numberofvertices);
	fprintf(fp, "Number of triangular faces = %d\n", 0);
	fprintf(fp, "Number of tetrahedra = %d\n", numberoftets);
	fprintf(fp, "Number of facets in complex = %d\n", numberoffacets);
	fprintf(fp, "Number of bad facets in complex = %d\n", numberofbadfacets);
	fprintf(fp, "Number of cells in complex = %d\n", numberofcells);
	fprintf(fp, "Number of bad cells in complex = %d\n", numberofbadtets);
	fprintf(fp, "Total time = %lf\n", times[6]);
	fprintf(fp, "  CPU time = %lf\n", times[4]);
	fprintf(fp, "    Initialization time = %lf\n", times[0]);
#ifndef GQM3D_WITHOUT_1D_FEATURE
	fprintf(fp, "    Edge protection time = %lf\n", edge_protect_time);
#endif
	fprintf(fp, "    Facet scan time = %lf\n", intertimer["Facets_scan_time"] * 1000);
	fprintf(fp, "    Facet refinement time = %lf\n", intertimer["Facets_refin_time"] * 1000);
	fprintf(fp, "    Cell scan time = %lf\n", intertimer["Cells_scan_time"] * 1000);
	fprintf(fp, "    Cell refinement time = %lf\n", intertimer["Cells_refin_time"] * 1000);
	fprintf(fp, "    Triangulation conversion time = %lf\n", times[2]);
	fprintf(fp, "    Domain conversion time = %lf\n", times[3]);
	fprintf(fp, "  GPU time = %lf\n", times[5]);
	fprintf(fp, "    Reconstruction time (CPU) = %lf\n", behavior->times[0]);
	fprintf(fp, "    Initialization time = %lf\n", behavior->times[1]);
	fprintf(fp, "    Elements refinement time = %lf\n", behavior->times[2]);
	fprintf(fp, "      Initialize quality time = %lf\n", behavior->times[4]);
	fprintf(fp, "      Splitting subfaces time = %lf\n", behavior->times[5]);
	fprintf(fp, "      Set tet status time = %lf\n", behavior->times[6]);
	fprintf(fp, "      Splitting tets time = %lf\n", behavior->times[7]);
	fprintf(fp, "    Mesh compaction time = %lf\n", behavior->times[3]);

	fprintf(fp, "\nFacet angle distribution:\n");
	for (int i = 0; i < numberofslots; i++)
	{
		fprintf(fp, "%lf %d\n", i*step, fangledist[i]);
	}

	fprintf(fp, "\nDihedral angle distribution:\n");
	for (int i = 0; i < numberofslots; i++)
	{
		fprintf(fp, "%lf %d\n", i*step, tangledist[i]);
	}

	fclose(fp);
}

int refineInputOnGPU(
	char* infile,
	bool force_features,
	MESHCR* criteria,
	MESHBH* behavior,
	char* outmesh,
	char* outdata
)
{
	//CPU
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

	std::cout.precision(17);
	std::cerr.precision(17);
	double times[7];
	CGAL::Timer t;
	t.start();
	double last = 0, current;

	std::ifstream input(infile);
	Polyhedron polyhedron;
	input >> polyhedron;
	if (input.fail()) {
		std::cerr << "Error: Cannot read file " << infile << std::endl;
		return EXIT_FAILURE;
	}
	if (!CGAL::is_triangle_mesh(polyhedron)) {
		std::cerr << "Input geometry is not triangulated." << std::endl;
		return EXIT_FAILURE;
	}
	current = t.time() * 1000;
	printf("Input done. %lf\n", current - last);
	last = current;

	// Create domain
	Mesh_domain domain(polyhedron);
	current = t.time() * 1000;
	printf("Domain done. %lf\n", current - last);
	last = current;

#ifndef GQM3D_WITHOUT_1D_FEATURE
	// Get sharp features
	domain.detect_features();
	current = t.time() * 1000;
	printf("Detect features done. %lf\n", current - last);
	last = current;
#endif

	times[0] = current; // initialization time

	// Mesh criteria
	double inter_facet_angle = 0, inter_facet_size = 0, inter_facet_distance = 0;
	double inter_cell_radius_edge_ratio = 0, inter_cell_size = 0;
	if (behavior->mode == 1)
	{
		// do nothing
	}
	else if (behavior->mode == 2)
	{
		inter_facet_angle = criteria->facet_angle;
		inter_cell_radius_edge_ratio = criteria->cell_radius_edge_ratio;
	}
	else
	{
		printf("Error: Unknown mode %d\n", behavior->mode);
		exit(0);
	}

	Mesh_criteria mcriteria(
#ifndef GQM3D_WITHOUT_1D_FEATURE
		edge_size = criteria->edge_size,
#endif
		facet_angle = inter_facet_angle,
		facet_size = inter_facet_size,
		facet_distance = inter_facet_distance,
		cell_radius_edge_ratio = inter_cell_radius_edge_ratio,
		cell_size = inter_cell_size);

	// CGAL refinement
	intertimer["Facets_scan_time"] = 0;
	intertimer["Facets_refin_time"] = 0;
	intertimer["Cells_scan_time"] = 0;
	intertimer["Cells_refin_time"] = 0;
	C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, mcriteria, no_perturb(), no_exude());
	current = t.time() * 1000;
	printf("CGAL mesh refinement done. %lf\n", current - last);
	times[1] = current - last; // edge protection time
	last = current;
	//analyzeCGALTr(c3t3);

	MESHIO meshio;
	// Convert triangulation
	cpuErrchk(c3t3.triangulation().dimension() == 3);
	//   This may fail when there is only a planar curve as feature
	//   Need to handle it later by inserting extra points 
	convertCGALTr2Tr(c3t3, &meshio);
	current = t.time() * 1000;
	printf("Convert triangulation done. %lf\n", current - last);
	times[2] = current - last; // triangulation conversion time
	last = current;

	// Convert domain
	convertDomain(domain, &meshio);
	meshio.aabb_closed = polyhedron.is_closed();
	current = t.time() * 1000;
	printf("Convert domain done. %lf\n", current - last);
	times[3] = current - last; // domain conversion time
	last = current;
	times[4] = current; // CPU time

	// Launch refinement routine
	printf("Launch refinement routine...\n");
	GPU_Refine_3D(
		criteria,
		&meshio,
		behavior
	);
	current = t.time() * 1000;
	printf("refinement done. %lf\n\n", current - last);
	times[5] = current - last; // GPU time
	printf("Total time = %lf\n", current);
	times[6] = current; // total time

	// Output mesh
	char timestr[80];
	strftime(timestr, sizeof(timestr), "%Y%m%d%H%M", timeinfo);
	if (outmesh != NULL)
	{
#ifdef GQM3D_WITHOUT_1D_FEATURE
		criteria->edge_size = 0;
#endif

		char* filename;

		// save off
		std::ostringstream strs;
		strs << outmesh << "_"
			<< "es" << criteria->edge_size << "_";
		strs << "fa" << criteria->facet_angle << "_fs" << criteria->facet_size << "_fd" << criteria->facet_distance << "_";
		strs << "cr" << criteria->cell_radius_edge_ratio << "_cs" << criteria->cell_size << "_";
		if (behavior->mode == 1)
		{
			strs << "gpu" << "_" << timestr << ".off";
		}
		else
		{
#ifdef CGAL_CONCURRENT_MESH_3
			strs << "parallel_";
#endif
			strs << "cpu_gpu" << "_" << timestr <<  ".off";
		}
		filename = new char[strs.str().length() + 1];
		strcpy(filename, strs.str().c_str());
		outputFacets2OFF(
			meshio.out_numofpoints,
			meshio.out_pointlist,
			meshio.out_numoftrifaces,
			meshio.out_trifacelist,
			filename
		);
		delete[] filename;

		// save mesh
		std::ostringstream strs1;
		strs1 << outmesh << "_"
			<< "es" << criteria->edge_size << "_";
		strs1 << "fa" << criteria->facet_angle << "_fs" << criteria->facet_size << "_fd" << criteria->facet_distance << "_";
		strs1 << "cr" << criteria->cell_radius_edge_ratio << "_cs" << criteria->cell_size << "_";
		if (behavior->mode == 1)
		{
			strs1 << "gpu" << "_" << timestr << ".mesh";
		}
		else
		{
#ifdef CGAL_CONCURRENT_MESH_3
			strs1 << "parallel_";
#endif
			strs1 << "cpu_gpu" << "_" << timestr << ".mesh";
		}
		filename = new char[strs1.str().length() + 1];
		strcpy(filename, strs1.str().c_str());
		outputTr2Medit(
			meshio.out_numofpoints,
			meshio.out_pointlist,
			meshio.out_numoftrifaces,
			meshio.out_trifacelist,
			meshio.out_numoftets,
			meshio.out_numoftets_indomain,
			meshio.out_tetlist,
			meshio.out_tetstatus,
			filename
		);
		delete[] filename;
	}

	// Output statistic
	char* filename = NULL;
	if (outdata != NULL)
	{
		std::ostringstream strs;
		strs << outdata << "_"
			<< "es" << criteria->edge_size << "_";
		strs << "fa" << criteria->facet_angle << "_fs" << criteria->facet_size << "_fd" << criteria->facet_distance << "_";
		strs << "cr" << criteria->cell_radius_edge_ratio << "_cs" << criteria->cell_size << "_";
		if (behavior->mode == 1)
		{
			strs << "gpu" << "_" << timestr << ".txt";
		}
		else
		{
#ifdef CGAL_CONCURRENT_MESH_3
			strs << "parallel_";
#endif
			strs << "cpu_gpu" << "_" << timestr << ".txt";
		}
		filename = new char[strs.str().length() + 1];
		strcpy(filename, strs.str().c_str());
	}
	resultStatistics(&meshio, criteria, behavior, times, filename);
	if(filename != NULL)
		delete[] filename;

	return 1;
}