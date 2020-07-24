#include "MeshRefine.h"

std::map<std::string, double> intertimer;

// To avoid verbose function and named parameters call
using namespace CGAL::parameters;

void convertCGALTr2TrForOutput(
	C3t3& c3t3,
	MESHIO *meshio
)
{
	int numberofpoints = c3t3.triangulation().number_of_vertices();
	int numberoftets = c3t3.triangulation().number_of_finite_cells();
	int numberoftets_indomain = c3t3.number_of_cells_in_complex();
	int numberoffacets = c3t3.number_of_facets_in_complex();

	int counter = 0;
	
	// Set up point index hash table, coordinate and weight list
	std::unordered_map<Tr::Vertex_handle, int> hash_ptidx;
	meshio->numofpoints = numberofpoints;
	meshio->pointlist = new double[3 * numberofpoints];
	for (Tr::Finite_vertices_iterator it = c3t3.triangulation().finite_vertices_begin(),
		end = c3t3.triangulation().finite_vertices_end(); it != end; ++it)
	{
		Tr::Vertex_handle vh = it;
		hash_ptidx.insert({ vh, counter });
		Tr::Weighted_point wp = vh->point();
		meshio->pointlist[3 * counter + 0] = wp.x();
		meshio->pointlist[3 * counter + 1] = wp.y();
		meshio->pointlist[3 * counter + 2] = wp.z();
		counter++;
	}
	cpuErrchk(numberofpoints == counter);

	// Set up tet index hash table and vertex list
	counter = 0;
	std::unordered_map<Tr::Cell_handle, int> hash_tetidx;
	meshio->numoftets = numberoftets;
	meshio->numoftets_indomain = numberoftets_indomain;
	meshio->tetlist = new int[4 * numberoftets];
	meshio->tetstatuslist = new tetstatus[numberoftets];
	for (Tr::Finite_cells_iterator it = c3t3.triangulation().finite_cells_begin(),
		end = c3t3.triangulation().finite_cells_end(); it != end; ++it)
	{
		Tr::Cell_handle cell = it;
		hash_tetidx.insert({ cell, counter });

		meshio->tetstatuslist[counter].setEmpty(false);
		if (c3t3.is_in_complex(cell))
			meshio->tetstatuslist[counter].setInDomain(true);

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

	// Set up triface list
	counter = 0;
	meshio->numoftrifaces = numberoffacets;
	meshio->trifacelist = new int[3 * numberoffacets];
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

		if ((c3t3.subdomain_index(cell) > c3t3.subdomain_index(cell->neighbor(i))) == (i % 2 == 1))
		{
			int tmp = meshio->trifacelist[3 * counter + 0];
			meshio->trifacelist[3 * counter + 0] = meshio->trifacelist[3 * counter + 1];
			meshio->trifacelist[3 * counter + 1] = tmp;
		}

		counter++;
	}
}

void resultStatistics(C3t3& c3t3)
{
	int numberofvertices = c3t3.triangulation().number_of_vertices();
	int numberoffaces = c3t3.triangulation().number_of_finite_facets();
	int numberoftets = c3t3.triangulation().number_of_finite_cells();
	printf("Number of vertices = %d\n", numberofvertices);
	printf("Number of triangular faces = %d\n", numberoffaces);
	printf("Number of tetrahedra = %d\n", numberoftets);

	int numberoffacets = c3t3.number_of_facets_in_complex();
	printf("Number of facets in complex = %d\n", numberoffacets);
	int counter = 0;
	for (C3t3::Facets_in_complex_iterator it = c3t3.facets_in_complex_begin(),
		end = c3t3.facets_in_complex_end(); it != end; ++it)
	{
		Tr::Facet facet = *it;
		Tr::Cell_handle c = facet.first;
		int i = facet.second;
		Tr::Cell_handle n = c->neighbor(i);
		if (c3t3.triangulation().is_infinite(c) || c3t3.triangulation().is_infinite(n))
		{
			counter++;
		}
	}
	printf("  Added by segment:  %d\n", numberoffacets - counter);
	printf("  Added by ray:      %d\n", counter);

	int numberofcells = c3t3.number_of_cells_in_complex();
	printf("Number of cells in complex = %d\n", numberofcells);
}

void elementStatistics(C3t3& c3t3, MESHCR* criteria, int& numberofbadfacets,
	int& numberofbadtets, double step, int& numberofslots, int*& fangledist, int*& tangledist)
{
	// angle distribution 
	numberofslots = 180.0 / step + 1;
	int j, slotIdx;
	double* p[4];
	double allangles[6];
	fangledist = new int[numberofslots];
	tangledist = new int[numberofslots];
	for (j = 0; j < numberofslots; j++)
	{
		fangledist[j] = 0;
		tangledist[j] = 0;
	}

	// count bad facets
	int badfacetcounter = 0;
	double pa[3], pb[3], pc[3], pd[3], center[3], aw, bw, cw, dw;
	for (C3t3::Facets_in_complex_iterator it = c3t3.facets_in_complex_begin(),
		end = c3t3.facets_in_complex_end(); it != end; ++it)
	{
		Tr::Facet facet = *it;
		Tr::Cell_handle c = facet.first;
		int i = facet.second;
		center[0] = c->get_facet_surface_center(i).x();
		center[1] = c->get_facet_surface_center(i).y();
		center[2] = c->get_facet_surface_center(i).z();
		pa[0] = c->vertex((i + 1) & 3)->point().x();
		pa[1] = c->vertex((i + 1) & 3)->point().y();
		pa[2] = c->vertex((i + 1) & 3)->point().z();
		pb[0] = c->vertex((i + 2) & 3)->point().x();
		pb[1] = c->vertex((i + 2) & 3)->point().y();
		pb[2] = c->vertex((i + 2) & 3)->point().z();
		pc[0] = c->vertex((i + 3) & 3)->point().x();
		pc[1] = c->vertex((i + 3) & 3)->point().y();
		pc[2] = c->vertex((i + 3) & 3)->point().z();
		aw = c->vertex((i + 1) & 3)->point().weight();
		bw = c->vertex((i + 2) & 3)->point().weight();
		cw = c->vertex((i + 3) & 3)->point().weight();
		if (isBadFacet(pa, pb, pc, aw, bw, cw, center,
			criteria->facet_angle, criteria->facet_size, criteria->facet_distance))
			badfacetcounter++;
		p[0] = pa; p[1] = pb; p[2] = pc;
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
	for (C3t3::Cells_in_complex_iterator it = c3t3.cells_in_complex_begin(),
		end = c3t3.cells_in_complex_end(); it != end; ++it)
	{
		Tr::Cell_handle c = it;
		pa[0] = c->vertex(0)->point().x();
		pa[1] = c->vertex(0)->point().y();
		pa[2] = c->vertex(0)->point().z();
		pb[0] = c->vertex(1)->point().x();
		pb[1] = c->vertex(1)->point().y();
		pb[2] = c->vertex(1)->point().z();
		pc[0] = c->vertex(2)->point().x();
		pc[1] = c->vertex(2)->point().y();
		pc[2] = c->vertex(2)->point().z();
		pd[0] = c->vertex(3)->point().x();
		pd[1] = c->vertex(3)->point().y();
		pd[2] = c->vertex(3)->point().z();
		//aw = c->vertex(0)->point().weight();
		//bw = c->vertex(1)->point().weight();
		//cw = c->vertex(2)->point().weight();
		//dw = c->vertex(3)->point().weight();
		if (isBadTet(pa, pb, pc, pd, criteria->cell_radius_edge_ratio, criteria->cell_size))
			badtetcounter++;
		p[0] = pa; p[1] = pb; p[2] = pc; p[3] = pd;
		calDihedral(p, allangles);
		for (j = 0; j < 6; j++)
		{
			slotIdx = allangles[j] / step;
			tangledist[slotIdx]++;
		}
	}
	numberofbadtets = badtetcounter;
}

void resultStatistics(C3t3& c3t3, MESHCR* criteria, double* times, char* filename)
{
	int numberofvertices = c3t3.triangulation().number_of_vertices();
	int numberoffaces = c3t3.triangulation().number_of_finite_facets();
	int numberoftets = c3t3.triangulation().number_of_finite_cells();
	printf("Number of vertices = %d\n", numberofvertices);
	printf("Number of triangular faces = %d\n", numberoffaces);
	printf("Number of tetrahedra = %d\n", numberoftets);

	int numberofbadfacets, numberofbadtets;
	int numberofslots, *fangledist, *tangledist;
	double step = 1.0;
	elementStatistics(c3t3, criteria, numberofbadfacets, numberofbadtets, step,
		numberofslots, fangledist, tangledist);

	int numberoffacets = c3t3.number_of_facets_in_complex();
	printf("Number of facets in complex = %d\n", numberoffacets);
	printf("Number of bad facets in complex = %d\n", numberofbadfacets);
	int numberofcells = c3t3.number_of_cells_in_complex();
	printf("Number of cells in complex = %d\n", numberofcells);
	printf("Number of bad cells in complex = %d\n", numberofbadtets);
	printf("Total time = %lf\n", times[2]);
	printf("  Initialization time = %lf\n", times[0]);
	printf("  Mesh refinement time = %lf\n", times[1]);
#ifndef GQM3D_WITHOUT_1D_FEATURE
	double edge_protect_time = intertimer["Facets_scan_time"] +
		intertimer["Facets_refin_time"] + intertimer["Cells_scan_time"] +
		intertimer["Cells_refin_time"];
	edge_protect_time = times[1] - edge_protect_time * 1000;
	printf("    Edge protection time = %lf\n", edge_protect_time);
#endif
	printf("    Facet scan time = %lf\n", intertimer["Facets_scan_time"] * 1000);
	printf("    Facet refinement time = %lf\n", intertimer["Facets_refin_time"]*1000);
	printf("    Cell scan time = %lf\n", intertimer["Cells_scan_time"] * 1000);
	printf("    Cell refinement time = %lf\n", intertimer["Cells_refin_time"]*1000);

	FILE * fp;
	fp = fopen(filename, "w");
	fprintf(fp, "Number of vertices = %d\n", numberofvertices);
	fprintf(fp, "Number of triangular faces = %d\n", numberoffaces);
	fprintf(fp, "Number of tetrahedra = %d\n", numberoftets);
	fprintf(fp, "Number of facets in complex = %d\n", numberoffacets);
	fprintf(fp, "Number of bad facets in complex = %d\n", numberofbadfacets);
	fprintf(fp, "Number of cells in complex = %d\n", numberofcells);
	fprintf(fp, "Number of bad cells in complex = %d\n", numberofbadtets);
	fprintf(fp, "Total time = %lf\n", times[2]);
	fprintf(fp, "  Initialization time = %lf\n", times[0]);
	fprintf(fp, "  Mesh refinement time = %lf\n", times[1]);
#ifndef GQM3D_WITHOUT_1D_FEATURE
	fprintf(fp, "    Edge protection time = %lf\n", edge_protect_time);
#endif
	fprintf(fp, "    Facet scan time = %lf\n", intertimer["Facets_scan_time"] * 1000);
	fprintf(fp, "    Facet refinement time = %lf\n", intertimer["Facets_refin_time"] * 1000);
	fprintf(fp, "    Cell scan time = %lf\n", intertimer["Cells_scan_time"] * 1000);
	fprintf(fp, "    Cell refinement time = %lf\n", intertimer["Cells_refin_time"] * 1000);

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

int refineInputOnCGAL(
	char* infile,
	bool force_features,
	MESHCR* criteria,
	char* outmesh,
	char* outdata
)
{
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

	std::cout.precision(17);
	std::cerr.precision(17);
	double times[3];
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
	Mesh_criteria mcriteria(
#ifndef GQM3D_WITHOUT_1D_FEATURE
		edge_size = criteria->edge_size,
#endif
		facet_angle = criteria->facet_angle, facet_size = criteria->facet_size, facet_distance = criteria->facet_distance,
		cell_radius_edge_ratio = criteria->cell_radius_edge_ratio, cell_size = criteria->cell_size);

	// Mesh generation
	intertimer["Facets_scan_time"] = 0;
	intertimer["Facets_refin_time"] = 0;
	intertimer["Cells_scan_time"] = 0;
	intertimer["Cells_refin_time"] = 0;
	printf("\n=================== Mesh Generation Profiling ===================\n");
	C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, mcriteria, no_perturb(), no_exude());
	printf("=================== Mesh Generation Profiling ===================\n\n");
	current = t.time() * 1000;
	printf("Mesh generation done. %lf\n", current - last);
	printf("Total time = %lf\n\n", current);
	times[1] = current - last; // mesh refinement time
	times[2] = current; // total time

	// Output mesh
	char timestr[80];
	strftime(timestr, sizeof(timestr), "%Y%m%d%H%M", timeinfo);
	if (outmesh != NULL)
	{
#ifdef CGAL_CONCURRENT_MESH_3
		MESHIO out_mesh;
		convertCGALTr2TrForOutput(c3t3, &out_mesh);
#endif

#ifdef GQM3D_WITHOUT_1D_FEATURE
		criteria->edge_size = 0;
#endif

		char* filename;
		// save off
		std::ostringstream strs;
		strs << outmesh << "_" 
			 << "es" << criteria->edge_size << "_";
		strs << "fa" << criteria->facet_angle << "_fs" << criteria->facet_size << "_fd" << criteria->facet_distance << "_";
		strs << "cr" << criteria->cell_radius_edge_ratio << "_cs" << criteria->cell_size;
#ifdef CGAL_CONCURRENT_MESH_3
		strs << "_parallel";
#endif
		strs << "_" << timestr << ".off";
		filename = new char[strs.str().length() + 1];
		strcpy(filename, strs.str().c_str());
		std::ofstream off_file(filename);
#ifdef CGAL_CONCURRENT_MESH_3
		// In parallel version, the output function is very slow
		// so use our own function
		outputFacets2OFF(
			out_mesh.numofpoints,
			out_mesh.pointlist,
			out_mesh.numoftrifaces,
			out_mesh.trifacelist,
			filename
		);
#else
		c3t3.output_facets_in_complex_to_off(off_file);
#endif
		delete[] filename;

		// save mesh
		std::ostringstream strs1;
		strs1 << outmesh << "_"
			<< "es" << criteria->edge_size << "_";
		strs1 << "fa" << criteria->facet_angle << "_fs" << criteria->facet_size << "_fd" << criteria->facet_distance << "_";
		strs1 << "cr" << criteria->cell_radius_edge_ratio << "_cs" << criteria->cell_size;
#ifdef CGAL_CONCURRENT_MESH_3
		strs1 << "_parallel";
#endif
		strs1 << "_" << timestr << ".mesh";
		filename = new char[strs1.str().length() + 1];
		strcpy(filename, strs1.str().c_str());
		std::ofstream medit_file(filename);
#ifdef CGAL_CONCURRENT_MESH_3
		// In parallel version, the output function is very slow
		// so use our own function
		outputTr2Medit(
			out_mesh.numofpoints,
			out_mesh.pointlist,
			out_mesh.numoftrifaces,
			out_mesh.trifacelist,
			out_mesh.numoftets,
			out_mesh.numoftets_indomain,
			out_mesh.tetlist,
			out_mesh.tetstatuslist,
			filename
		);
#else
		c3t3.output_to_medit(medit_file);
#endif
		delete[] filename;
	}

	// Output statistic
	if (outdata != NULL)
	{
		char* filename;
		std::ostringstream strs;
		strs << outdata << "_"
			<< "es" << criteria->edge_size << "_";
		strs << "fa" << criteria->facet_angle << "_fs" << criteria->facet_size << "_fd" << criteria->facet_distance << "_";
		strs << "cr" << criteria->cell_radius_edge_ratio << "_cs" << criteria->cell_size;
#ifdef CGAL_CONCURRENT_MESH_3
		strs << "_parallel";
#endif
		strs << "_" << timestr << ".txt";
		filename = new char[strs.str().length() + 1];
		strcpy(filename, strs.str().c_str());
		resultStatistics(c3t3, criteria, times, filename);
		delete[] filename;
	}

	return 1;
}