#include <stdio.h>
#include <sstream>
#include <cuda_runtime.h>

#include "CudaAnimation.h"
#include "Mesh.h"

void outputStartingFrame(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	IntD& t_threadlist,
	IntD& t_insertidxlist,
	RealD& t_insertptlist,
	TetHandleD& t_locatedtet,
	int iter_seg,
	int iter_subface,
	int iter_tet
)
{
	// Prepare lists
	int numberofpoints = t_pointlist.size() / 3;
	drawmesh->pointlist = new REAL[3 * numberofpoints];
	cudaMemcpy(drawmesh->pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * numberofpoints * sizeof(double), cudaMemcpyDeviceToHost);

	int numberoftet = t_tetstatus.size();
	drawmesh->tetlist = new int[4 * numberoftet];
	cudaMemcpy(drawmesh->tetlist, thrust::raw_pointer_cast(&t_tetlist[0]), 4 * numberoftet * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->tetstatus = new tetstatus[numberoftet];
	cudaMemcpy(drawmesh->tetstatus, thrust::raw_pointer_cast(&t_tetstatus[0]), numberoftet * sizeof(tetstatus), cudaMemcpyDeviceToHost);

	int numberofthreads = t_threadlist.size();
	drawmesh->threadlist = new int[numberofthreads];
	cudaMemcpy(drawmesh->threadlist, thrust::raw_pointer_cast(&t_threadlist[0]), numberofthreads * sizeof(int), cudaMemcpyDeviceToHost);

	int numofinsertpt = t_insertidxlist.size();
	drawmesh->insertidxlist = new int[numofinsertpt];
	cudaMemcpy(drawmesh->insertidxlist, thrust::raw_pointer_cast(&t_insertidxlist[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->insertptlist = new REAL[3 * numofinsertpt];
	cudaMemcpy(drawmesh->insertptlist, thrust::raw_pointer_cast(&t_insertptlist[0]), 3 * numofinsertpt * sizeof(double), cudaMemcpyDeviceToHost);
	drawmesh->locatedtet = new tethandle[numofinsertpt];
	cudaMemcpy(drawmesh->locatedtet, thrust::raw_pointer_cast(&t_locatedtet[0]), numofinsertpt * sizeof(tethandle), cudaMemcpyDeviceToHost);

	// Write to mesh file
	// First frame: whole triangulation with bad tetrahedra colored
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		strs << "animation/F000_" << iter_seg << "_" << iter_subface << "_" << iter_tet << ".mesh";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "MeshVersionFormatted 1\n");
		fprintf(fp, "Dimension 3\n");

		fprintf(fp, "Vertices\n");
		fprintf(fp, "%d\n", numberofpoints);
		for (int i = 0; i < numberofpoints; i++)
			fprintf(fp, "%lf %lf %lf 1\n", drawmesh->pointlist[3 * i + 0], drawmesh->pointlist[3 * i + 1], drawmesh->pointlist[3 * i + 2]);

		// Color the faces of bad tetrahedra
		int numberoftriface = 0;
		for (int i = 0; i < numberoftet; i++)
		{
			if (!drawmesh->tetstatus[i].isEmpty())
				numberoftriface += 4;
		}
		numberoftriface += 4 * numberofthreads;
		fprintf(fp, "Triangles\n");
		fprintf(fp, "%d\n", numberoftriface);
		for (int i = 0; i < numberoftet; i++)
		{
			if (!drawmesh->tetstatus[i].isEmpty())
			{
				for (int j = 0; j < 4; j++)
					fprintf(fp, "%d %d %d 1\n", drawmesh->tetlist[4 * i + (j + 1) % 4] + 1,
						drawmesh->tetlist[4 * i + (j + 2) % 4] + 1, drawmesh->tetlist[4 * i + (j + 3) % 4] + 1);
			}
		}
		int threadIdx, tetIdx, colorIdx;
		for (int i = 0; i < numberofthreads; i++)
		{
			threadIdx = drawmesh->threadlist[i];
			colorIdx = (threadIdx % 32) + 2;
			tetIdx = drawmesh->insertidxlist[threadIdx];
			for (int j = 0; j < 4; j++)
				fprintf(fp, "%d %d %d %d\n", drawmesh->tetlist[4 * tetIdx + (j + 1) % 4] + 1,
					drawmesh->tetlist[4 * tetIdx + (j + 2) % 4] + 1, drawmesh->tetlist[4 * tetIdx + (j + 3) % 4] + 1, colorIdx);
		}

		fprintf(fp, "Tetrahedra\n");
		fprintf(fp, "%d\n", numberoftet);
		for (int i = 0; i < numberoftet; i++)
		{
			if (!drawmesh->tetstatus[i].isEmpty())
				fprintf(fp, "%d %d %d %d 1\n", drawmesh->tetlist[4 * i + 0] + 1, drawmesh->tetlist[4 * i + 1] + 1,
					drawmesh->tetlist[4 * i + 2] + 1, drawmesh->tetlist[4 * i + 3] + 1);
		}
		fprintf(fp, "End\n");
		fclose(fp);
		delete[] file;
	}
	// First frame info
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		strs << "animation/F000_" << iter_seg << "_" << iter_subface << "_" << iter_tet << ".txt";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "Number of points = %d\n", numberofpoints);
		fprintf(fp, "Number of tets = %d\n", numberoftet);
		fprintf(fp, "Number of bad tets = %d\n", numberofthreads);
		fclose(fp);
	}

	// Second frame: bad tetrahedra colored
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		strs << "animation/F001_" << iter_seg << "_" << iter_subface << "_" << iter_tet << ".mesh";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "MeshVersionFormatted 1\n");
		fprintf(fp, "Dimension 3\n");

		fprintf(fp, "Vertices\n");
		fprintf(fp, "%d\n", numberofpoints + numberofthreads);
		for (int i = 0; i < numberofpoints; i++)
			fprintf(fp, "%lf %lf %lf 1\n", drawmesh->pointlist[3 * i + 0], drawmesh->pointlist[3 * i + 1], drawmesh->pointlist[3 * i + 2]);
		int threadIdx, colorIdx;
		for (int i = 0; i < numberofthreads; i++)
		{
			threadIdx = drawmesh->threadlist[i];
			colorIdx = (threadIdx % 32) + 2;
			fprintf(fp, "%lf %lf %lf %d\n", drawmesh->insertptlist[3 * threadIdx + 0], drawmesh->insertptlist[3 * threadIdx + 1],
				drawmesh->insertptlist[3 * threadIdx + 2], colorIdx);
		}

		// Color the faces of bad tetrahedra
		int numberoftriface = 4 * numberofthreads;
		fprintf(fp, "Triangles\n");
		fprintf(fp, "%d\n", numberoftriface);
		int tetIdx;
		for (int i = 0; i < numberofthreads; i++)
		{
			threadIdx = drawmesh->threadlist[i];
			colorIdx = (threadIdx % 32) + 2;
			tetIdx = drawmesh->insertidxlist[threadIdx];
			for (int j = 0; j < 4; j++)
				fprintf(fp, "%d %d %d %d\n", drawmesh->tetlist[4 * tetIdx + (j + 1) % 4] + 1,
					drawmesh->tetlist[4 * tetIdx + (j + 2) % 4] + 1, drawmesh->tetlist[4 * tetIdx + (j + 3) % 4] + 1, colorIdx);
		}

		fprintf(fp, "Tetrahedra\n");
		fprintf(fp, "0\n");
		fprintf(fp, "End\n");
		fclose(fp);
		delete[] file;
	}

	// Third frame: located tetrahedra colored
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		strs << "animation/F002_" << iter_seg << "_" << iter_subface << "_" << iter_tet << ".mesh";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "MeshVersionFormatted 1\n");
		fprintf(fp, "Dimension 3\n");

		fprintf(fp, "Vertices\n");
		fprintf(fp, "%d\n", numberofpoints);
		for (int i = 0; i < numberofpoints; i++)
			fprintf(fp, "%lf %lf %lf 1\n", drawmesh->pointlist[3 * i + 0], drawmesh->pointlist[3 * i + 1], drawmesh->pointlist[3 * i + 2]);

		int numberoftriface = 4 * numberofthreads;
		fprintf(fp, "Triangles\n");
		fprintf(fp, "%d\n", numberoftriface);
		int threadIdx, colorIdx, tetIdx;
		for (int i = 0; i < numberofthreads; i++)
		{
			threadIdx = drawmesh->threadlist[i];
			colorIdx = (threadIdx % 32) + 2;
			tetIdx = drawmesh->locatedtet[threadIdx].id;
			for (int j = 0; j < 4; j++)
				fprintf(fp, "%d %d %d %d\n", drawmesh->tetlist[4 * tetIdx + (j + 1) % 4] + 1,
					drawmesh->tetlist[4 * tetIdx + (j + 2) % 4] + 1, drawmesh->tetlist[4 * tetIdx + (j + 3) % 4] + 1, colorIdx);
		}

		fprintf(fp, "Tetrahedra\n");
		fprintf(fp, "0\n");
		fprintf(fp, "End\n");
		fclose(fp);
		delete[] file;
	}
	// Third frame info
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		strs << "animation/F002_" << iter_seg << "_" << iter_subface << "_" << iter_tet << ".txt";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		bool* flag = new bool[numofinsertpt];
		for (int i = 0; i < numofinsertpt; i++)
			flag[i] = false;
		for (int i = 0; i < numberofthreads; i++)
		{
			int threadIdx = drawmesh->threadlist[i];;
			flag[threadIdx] = true;
		}
		for (int i = 0; i < numofinsertpt; i++)
		{
			if (flag[i])
			{
				int id1 = drawmesh->locatedtet[i].id;
				for (int j = i + 1; j < numofinsertpt; j++)
				{
					if (flag[j])
					{
						int id2 = drawmesh->locatedtet[j].id;
						if (id1 == id2)
							flag[j] = false;
					}
				}
			}
		}
		numberoftet = 0;
		for (int i = 0; i < numofinsertpt; i++)
		{
			if (flag[i])
				numberoftet++;
		}
		delete[] flag;

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "Number of located tets = %d\n", numberoftet);
		fclose(fp);
	}

	// Update frame number
	drawmesh->framenum = 3;

	// Clear memory
	delete[] drawmesh->pointlist;
	delete[] drawmesh->tetlist;
	delete[] drawmesh->tetstatus;
	delete[] drawmesh->threadlist;
	delete[] drawmesh->insertidxlist;
}

void outputCavityFrame(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	IntD& t_tetlist,
	UInt64D& t_tetmarker,
	IntD& t_threadmarker,
	TetHandleD& t_caveoldtetlist,
	IntD& t_caveoldtetnext,
	IntD& t_caveoldtethead,
	int iter_seg,
	int iter_subface,
	int iter_tet,
	int iter_expanding,
	int expandingsize
)
{
	// Prepare lists
	int numberofpoints = t_pointlist.size() / 3;
	drawmesh->pointlist = new REAL[3 * numberofpoints];
	cudaMemcpy(drawmesh->pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * numberofpoints * sizeof(double), cudaMemcpyDeviceToHost);

	drawmesh->tetlist = new int[t_tetlist.size()];
	cudaMemcpy(drawmesh->tetlist, thrust::raw_pointer_cast(&t_tetlist[0]), t_tetlist.size() * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->tetmarker = new unsigned long long[t_tetmarker.size()];
	cudaMemcpy(drawmesh->tetmarker, thrust::raw_pointer_cast(&t_tetmarker[0]), t_tetmarker.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	int numofinsertpt = t_threadmarker.size();
	drawmesh->threadmarker = new int[numofinsertpt];
	cudaMemcpy(drawmesh->threadmarker, thrust::raw_pointer_cast(&t_threadmarker[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->caveoldtethead = new int[numofinsertpt];
	cudaMemcpy(drawmesh->caveoldtethead, thrust::raw_pointer_cast(&t_caveoldtethead[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);

	drawmesh->caveoldtetlist = new tethandle[t_caveoldtetlist.size()];
	cudaMemcpy(drawmesh->caveoldtetlist, thrust::raw_pointer_cast(&t_caveoldtetlist[0]), t_caveoldtetlist.size() * sizeof(tethandle), cudaMemcpyDeviceToHost);
	drawmesh->caveoldtetnext = new int[t_caveoldtetnext.size()];
	cudaMemcpy(drawmesh->caveoldtetnext, thrust::raw_pointer_cast(&t_caveoldtetnext[0]), t_caveoldtetnext.size() * sizeof(int), cudaMemcpyDeviceToHost);

	// frame
	int numoftet = 0, numofhulltet = 0, numofsp = 0;
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		if(drawmesh->framenum < 10)
			strs << "animation/F00" << drawmesh->framenum << "_";
		else if(drawmesh->framenum < 100)
			strs << "animation/F0" << drawmesh->framenum << "_";
		else
			strs << "animation/F" << drawmesh->framenum << "_";

		strs << iter_seg << "_" << iter_subface << "_" << iter_tet << "_" << iter_expanding << ".mesh";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "MeshVersionFormatted 1\n");
		fprintf(fp, "Dimension 3\n");

		fprintf(fp, "Vertices\n");
		fprintf(fp, "%d\n", numberofpoints);
		for (int i = 0; i < numberofpoints; i++)
			fprintf(fp, "%lf %lf %lf 1\n", drawmesh->pointlist[3 * i + 0], drawmesh->pointlist[3 * i + 1], drawmesh->pointlist[3 * i + 2]);
		int threadIdx, tetIdx;
		for (int i = 0; i < numofinsertpt; i++)
		{
			if (drawmesh->threadmarker[i] != -1)
			{
				if(i != 42619)
					continue;
				threadIdx = i;
				numofsp++;
				int j = drawmesh->caveoldtethead[i];
				while (j != -1)
				{
					tetIdx = drawmesh->caveoldtetlist[j].id;
					if( (drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx + 1/* && drawmesh->tetlist[4*tetIdx + 3] != -1*/)
						numoftet++;
					if ((drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx + 1 && drawmesh->tetlist[4 * tetIdx + 3] == -1)
						numofhulltet++;
					j = drawmesh->caveoldtetnext[j];
				}
			}
		}
		int numberoftriface = 4 * numoftet;
		fprintf(fp, "Triangles\n");
		fprintf(fp, "%d\n", numberoftriface);
		int colorIdx;
		for (int i = 0; i < numofinsertpt; i++)
		{
			if (drawmesh->threadmarker[i] != -1)
			{
				if(i != 42619)
					continue;
				threadIdx = i;
				colorIdx = (threadIdx % 32) + 2;

				int k = drawmesh->caveoldtethead[i];
				while (k != -1)
				{
					tetIdx = drawmesh->caveoldtetlist[k].id;
					if ((drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx + 1/*&& drawmesh->tetlist[4 * tetIdx + 3] != -1*/)
					{
						for (int j = 0; j < 4; j++)
							fprintf(fp, "%d %d %d %d\n", drawmesh->tetlist[4 * tetIdx + (j + 1) % 4] + 1,
								drawmesh->tetlist[4 * tetIdx + (j + 2) % 4] + 1, drawmesh->tetlist[4 * tetIdx + (j + 3) % 4] + 1, colorIdx);
					}

					k = drawmesh->caveoldtetnext[k];
				}
			}
		}

		fprintf(fp, "Tetrahedra\n");
		fprintf(fp, "0\n");
		fprintf(fp, "End\n");
		fclose(fp);
		delete[] file;
	}

	// frame info
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		if (drawmesh->framenum < 10)
			strs << "animation/F00" << drawmesh->framenum << "_";
		else if (drawmesh->framenum < 100)
			strs << "animation/F0" << drawmesh->framenum << "_";
		else
			strs << "animation/F" << drawmesh->framenum << "_";

		strs << iter_seg << "_" << iter_subface << "_" << iter_tet << "_" << iter_expanding << ".txt";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "Number of splitting points = %d\n", numofsp);
		fprintf(fp, "Number of cavity tets = %d\n", numoftet);
		fprintf(fp, "Number of hull tets = %d\n", numofhulltet);
		fprintf(fp, "Number of expanded tets = %d\n", expandingsize);
		fclose(fp);
		delete[] file;
	}

	drawmesh->framenum++;

	// Clear memory
	delete[] drawmesh->pointlist;
	delete[] drawmesh->tetlist;
	delete[] drawmesh->tetmarker;
	delete[] drawmesh->threadmarker;
	delete[] drawmesh->caveoldtetlist;
	delete[] drawmesh->caveoldtetnext;
	delete[] drawmesh->caveoldtethead;
}

void outputLargeCavityFrame(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	IntD& t_tetlist,
	UInt64D& t_tetmarker,
	IntD& t_threadmarker,
	IntD& t_cavethreadidx,
	TetHandleD& t_caveoldtetlist,
	IntD& t_caveoldtetnext,
	IntD& t_caveoldtethead,
	int iter_seg,
	int iter_subface,
	int iter_tet,
	int iter_expanding,
	int expandingsize
)
{
	// Prepare lists
	int numberofpoints = t_pointlist.size() / 3;
	drawmesh->pointlist = new REAL[3 * numberofpoints];
	cudaMemcpy(drawmesh->pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * numberofpoints * sizeof(double), cudaMemcpyDeviceToHost);

	drawmesh->tetlist = new int[t_tetlist.size()];
	cudaMemcpy(drawmesh->tetlist, thrust::raw_pointer_cast(&t_tetlist[0]), t_tetlist.size() * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->tetmarker = new unsigned long long[t_tetmarker.size()];
	cudaMemcpy(drawmesh->tetmarker, thrust::raw_pointer_cast(&t_tetmarker[0]), t_tetmarker.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	int numofinsertpt = t_threadmarker.size();
	drawmesh->threadmarker = new int[numofinsertpt];
	cudaMemcpy(drawmesh->threadmarker, thrust::raw_pointer_cast(&t_threadmarker[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->caveoldtethead = new int[numofinsertpt];
	cudaMemcpy(drawmesh->caveoldtethead, thrust::raw_pointer_cast(&t_caveoldtethead[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);

	drawmesh->caveoldtetlist = new tethandle[t_caveoldtetlist.size()];
	cudaMemcpy(drawmesh->caveoldtetlist, thrust::raw_pointer_cast(&t_caveoldtetlist[0]), t_caveoldtetlist.size() * sizeof(tethandle), cudaMemcpyDeviceToHost);
	drawmesh->caveoldtetnext = new int[t_caveoldtetnext.size()];
	cudaMemcpy(drawmesh->caveoldtetnext, thrust::raw_pointer_cast(&t_caveoldtetnext[0]), t_caveoldtetnext.size() * sizeof(int), cudaMemcpyDeviceToHost);

	// Mark active threads
	int* expandthreadidx = new int[t_cavethreadidx.size()];
	cudaMemcpy(expandthreadidx, thrust::raw_pointer_cast(&t_cavethreadidx[0]), t_cavethreadidx.size() * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < t_cavethreadidx.size(); i++)
	{
		int threadId = expandthreadidx[i];
		if (threadId != -1)
			drawmesh->threadmarker[threadId] = 100;
	}

	// frame
	int numoftet = 0, numofhulltet = 0, numofsp = 0;
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		if (drawmesh->framenum < 10)
			strs << "animation/LF00" << drawmesh->framenum << "_";
		else if (drawmesh->framenum < 100)
			strs << "animation/LF0" << drawmesh->framenum << "_";
		else
			strs << "animation/LF" << drawmesh->framenum << "_";

		strs << iter_seg << "_" << iter_subface << "_" << iter_tet << "_" << iter_expanding << ".mesh";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "MeshVersionFormatted 1\n");
		fprintf(fp, "Dimension 3\n");

		fprintf(fp, "Vertices\n");
		fprintf(fp, "%d\n", numberofpoints);
		for (int i = 0; i < numberofpoints; i++)
			fprintf(fp, "%lf %lf %lf 1\n", drawmesh->pointlist[3 * i + 0], drawmesh->pointlist[3 * i + 1], drawmesh->pointlist[3 * i + 2]);
		int threadIdx, tetIdx;
		for (int i = 0; i < numofinsertpt; i++)
		{
			if (drawmesh->threadmarker[i] == 100)
			{
				threadIdx = i;
				numofsp++;
				int j = drawmesh->caveoldtethead[i];
				while (j != -1)
				{
					tetIdx = drawmesh->caveoldtetlist[j].id;
					if ((drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx + 1/* && drawmesh->tetlist[4*tetIdx + 3] != -1*/)
						numoftet++;
					if ((drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx + 1 && drawmesh->tetlist[4 * tetIdx + 3] == -1)
						numofhulltet++;
					j = drawmesh->caveoldtetnext[j];
				}
			}
		}
		int numberoftriface = 4 * numoftet;
		fprintf(fp, "Triangles\n");
		fprintf(fp, "%d\n", numberoftriface);
		int colorIdx;
		for (int i = 0; i < numofinsertpt; i++)
		{
			if (drawmesh->threadmarker[i] == 100)
			{
				threadIdx = i;
				colorIdx = (threadIdx % 32) + 2;

				int k = drawmesh->caveoldtethead[i];
				while (k != -1)
				{
					tetIdx = drawmesh->caveoldtetlist[k].id;
					if ((drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx + 1/*&& drawmesh->tetlist[4 * tetIdx + 3] != -1*/)
					{
						for (int j = 0; j < 4; j++)
							fprintf(fp, "%d %d %d %d\n", drawmesh->tetlist[4 * tetIdx + (j + 1) % 4] + 1,
								drawmesh->tetlist[4 * tetIdx + (j + 2) % 4] + 1, drawmesh->tetlist[4 * tetIdx + (j + 3) % 4] + 1, colorIdx);
					}

					k = drawmesh->caveoldtetnext[k];
				}
			}
		}

		fprintf(fp, "Tetrahedra\n");
		fprintf(fp, "0\n");
		fprintf(fp, "End\n");
		fclose(fp);
		delete[] file;
	}

	// frame info
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		if (drawmesh->framenum < 10)
			strs << "animation/LF00" << drawmesh->framenum << "_";
		else if (drawmesh->framenum < 100)
			strs << "animation/LF0" << drawmesh->framenum << "_";
		else
			strs << "animation/LF" << drawmesh->framenum << "_";

		strs << iter_seg << "_" << iter_subface << "_" << iter_tet << "_" << iter_expanding << ".txt";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "Number of splitting points = %d\n", numofsp);
		fprintf(fp, "Number of cavity tets = %d\n", numoftet);
		fprintf(fp, "Number of hull tets = %d\n", numofhulltet);
		fprintf(fp, "Number of expanded tets = %d\n", expandingsize);
		fclose(fp);
		delete[] file;
	}

	drawmesh->framenum++;

	// Clear memory
	delete[] drawmesh->pointlist;
	delete[] drawmesh->tetlist;
	delete[] drawmesh->tetmarker;
	delete[] drawmesh->threadmarker;
	delete[] drawmesh->caveoldtetlist;
	delete[] drawmesh->caveoldtetnext;
	delete[] drawmesh->caveoldtethead;
	delete[] expandthreadidx;
}

void outputCavityFrame(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	TetHandleD& t_tri2tetlist,
	IntD& t_tetlist,
	TetHandleD& t_neighborlist,
	UInt64D& t_tetmarker,
	IntD& t_threadmarker,
	TetHandleD& t_caveoldtetlist,
	IntD& t_caveoldtetnext,
	IntD& t_caveoldtethead,
	int iter_seg,
	int iter_subface,
	int iter_tet,
	int iter_expanding,
	int expandingsize
)
{
	// Prepare lists
	int numberofpoints = t_pointlist.size() / 3;
	drawmesh->pointlist = new REAL[3 * numberofpoints];
	cudaMemcpy(drawmesh->pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * numberofpoints * sizeof(double), cudaMemcpyDeviceToHost);

	int numofsubface = t_tristatus.size();
	drawmesh->numofsubface = numofsubface;
	drawmesh->trifacelist = new int[3 * numofsubface];
	cudaMemcpy(drawmesh->trifacelist, thrust::raw_pointer_cast(&t_trifacelist[0]), 3 * numofsubface * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->tristatus = new tristatus[numofsubface];
	cudaMemcpy(drawmesh->tristatus, thrust::raw_pointer_cast(&t_tristatus[0]), numofsubface * sizeof(tristatus), cudaMemcpyDeviceToHost);
	drawmesh->tri2tetlist = new tethandle[2 * numofsubface];
	cudaMemcpy(drawmesh->tri2tetlist, thrust::raw_pointer_cast(&t_tri2tetlist[0]), 2 * numofsubface * sizeof(tethandle), cudaMemcpyDeviceToHost);

	drawmesh->tetlist = new int[t_tetlist.size()];
	cudaMemcpy(drawmesh->tetlist, thrust::raw_pointer_cast(&t_tetlist[0]), t_tetlist.size() * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->tetmarker = new unsigned long long[t_tetmarker.size()];
	cudaMemcpy(drawmesh->tetmarker, thrust::raw_pointer_cast(&t_tetmarker[0]), t_tetmarker.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	drawmesh->neighborlist = new tethandle[t_neighborlist.size()];
	cudaMemcpy(drawmesh->neighborlist, thrust::raw_pointer_cast(&t_neighborlist[0]), t_neighborlist.size() * sizeof(tethandle), cudaMemcpyDeviceToHost);

	int numofinsertpt = t_threadmarker.size();
	drawmesh->threadmarker = new int[numofinsertpt];
	cudaMemcpy(drawmesh->threadmarker, thrust::raw_pointer_cast(&t_threadmarker[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->caveoldtethead = new int[numofinsertpt];
	cudaMemcpy(drawmesh->caveoldtethead, thrust::raw_pointer_cast(&t_caveoldtethead[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);

	drawmesh->caveoldtetlist = new tethandle[t_caveoldtetlist.size()];
	cudaMemcpy(drawmesh->caveoldtetlist, thrust::raw_pointer_cast(&t_caveoldtetlist[0]), t_caveoldtetlist.size() * sizeof(tethandle), cudaMemcpyDeviceToHost);
	drawmesh->caveoldtetnext = new int[t_caveoldtetnext.size()];
	cudaMemcpy(drawmesh->caveoldtetnext, thrust::raw_pointer_cast(&t_caveoldtetnext[0]), t_caveoldtetnext.size() * sizeof(int), cudaMemcpyDeviceToHost);

	// frame
	int numoftet = 0, numofhulltet = 0, numofsp = 0, numofcs = 0, numofttcs = 0;;
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		if (drawmesh->framenum < 10)
			strs << "animation/F00" << drawmesh->framenum << "_";
		else if (drawmesh->framenum < 100)
			strs << "animation/F0" << drawmesh->framenum << "_";
		else
			strs << "animation/F" << drawmesh->framenum << "_";

		strs << iter_seg << "_" << iter_subface << "_" << iter_tet << "_" << iter_expanding << ".mesh";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "MeshVersionFormatted 1\n");
		fprintf(fp, "Dimension 3\n");

		fprintf(fp, "Vertices\n");
		fprintf(fp, "%d\n", numberofpoints);
		for (int i = 0; i < numberofpoints; i++)
			fprintf(fp, "%lf %lf %lf 1\n", drawmesh->pointlist[3 * i + 0], drawmesh->pointlist[3 * i + 1], drawmesh->pointlist[3 * i + 2]);
		int threadIdx, tetIdx;
		for (int i = 0; i < numofinsertpt; i++)
		{
			if (drawmesh->threadmarker[i] != -1)
			{
				threadIdx = i;
				numofsp++;
				int j = drawmesh->caveoldtethead[i];
				while (j != -1)
				{
					tetIdx = drawmesh->caveoldtetlist[j].id;
					if ((drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx && drawmesh->tetlist[4 * tetIdx + 3] != -1)
						numoftet++;
					if ((drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx && drawmesh->tetlist[4 * tetIdx + 3] == -1)
						numofhulltet++;
					j = drawmesh->caveoldtetnext[j];
				}
			}
		}

		int numberoftriface = 4 * numoftet;
		for (int i = 0; i < numofsubface; i++)
		{
			if (!drawmesh->tristatus[i].isEmpty())
			{
				numofttcs++;
				trihandle checksh(i, 0);
				tethandle checktet;
				stpivot(checksh, checktet, drawmesh->tri2tetlist);
				if (ishulltet(checktet, drawmesh->tetlist))
					continue;
				fsymself(checktet, drawmesh->neighborlist);
				if (ishulltet(checktet, drawmesh->tetlist))
					continue;
				numberoftriface++;
			}
		}

		fprintf(fp, "Triangles\n");
		fprintf(fp, "%d\n", numberoftriface);
		int colorIdx;
		for (int i = 0; i < numofinsertpt; i++)
		{
			if (drawmesh->threadmarker[i] != -1)
			{
				threadIdx = i;
				colorIdx = (threadIdx % 32) + 2;

				int k = drawmesh->caveoldtethead[i];
				while (k != -1)
				{
					tetIdx = drawmesh->caveoldtetlist[k].id;
					if ((drawmesh->tetmarker[tetIdx] & 0xFFFFFFFF) == threadIdx && drawmesh->tetlist[4 * tetIdx + 3] != -1)
					{
						for (int j = 0; j < 4; j++)
							fprintf(fp, "%d %d %d %d\n", drawmesh->tetlist[4 * tetIdx + (j + 1) % 4] + 1,
								drawmesh->tetlist[4 * tetIdx + (j + 2) % 4] + 1, drawmesh->tetlist[4 * tetIdx + (j + 3) % 4] + 1, colorIdx);
					}

					k = drawmesh->caveoldtetnext[k];
				}
			}
		}
		for (int i = 0; i < numofsubface; i++)
		{
			if (!drawmesh->tristatus[i].isEmpty())
			{
				trihandle checksh(i, 0);
				tethandle checktet;
				stpivot(checksh, checktet, drawmesh->tri2tetlist);
				if (ishulltet(checktet, drawmesh->tetlist))
					continue;
				fsymself(checktet, drawmesh->neighborlist);
				if (ishulltet(checktet, drawmesh->tetlist))
					continue;
				fprintf(fp, "%d %d %d 1\n", drawmesh->trifacelist[3 * i] + 1, drawmesh->trifacelist[3 * i + 1] + 1,
					drawmesh->trifacelist[3 * i + 2] + 1);
			}
		}

		fprintf(fp, "Tetrahedra\n");
		fprintf(fp, "0\n");
		fprintf(fp, "End\n");
		fclose(fp);
		delete[] file;
	}

	// frame info
	{
		// filename
		FILE * fp;
		char *file;
		std::ostringstream strs;
		std::string fn;
		if (drawmesh->framenum < 10)
			strs << "animation/F00" << drawmesh->framenum << "_";
		else if (drawmesh->framenum < 100)
			strs << "animation/F0" << drawmesh->framenum << "_";
		else
			strs << "animation/F" << drawmesh->framenum << "_";

		strs << iter_seg << "_" << iter_subface << "_" << iter_tet << "_" << iter_expanding << ".txt";
		fn = strs.str();
		file = new char[fn.length() + 1];
		strcpy(file, fn.c_str());

		// writing
		fp = fopen(file, "w");
		fprintf(fp, "Number of splitting points = %d\n", numofsp);
		fprintf(fp, "Number of cavity tets = %d\n", numoftet);
		fprintf(fp, "Number of hull tets = %d\n", numofhulltet);
		fprintf(fp, "Number of expanded tets = %d\n", expandingsize);
		fprintf(fp, "Number of subfaces = %d\n", numofttcs);
		fclose(fp);
		delete[] file;
	}

	drawmesh->framenum++;

	// Clear memory
	delete[] drawmesh->pointlist;
	delete[] drawmesh->tetlist;
	delete[] drawmesh->tetmarker;
	delete[] drawmesh->threadmarker;
	delete[] drawmesh->caveoldtetlist;
	delete[] drawmesh->caveoldtetnext;
	delete[] drawmesh->caveoldtethead;
}

void outputTmpMesh(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	PointTypeD& t_pointtypelist,
	IntD& t_seglist,
	TriStatusD& t_segstatus,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	IntD& t_insertidxlist,
	RealD& t_insertptlist,
	IntD& t_threadlist,
	IntD& t_threadmarker,
	TetHandleD& t_cavebdrylist,
	IntD& t_cavebdrynext,
	IntD& t_cavebdryhead,
	int insertiontype
)
{
	int numofpoints = t_pointtypelist.size();
	drawmesh->numofpoints = numofpoints;
	drawmesh->pointlist = new REAL[3 * numofpoints];
	cudaMemcpy(drawmesh->pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * numofpoints * sizeof(double), cudaMemcpyDeviceToHost);
	drawmesh->pointtype = new verttype[numofpoints];
	cudaMemcpy(drawmesh->pointtype, thrust::raw_pointer_cast(&t_pointtypelist[0]), numofpoints * sizeof(verttype), cudaMemcpyDeviceToHost);

	int numofsubseg = t_segstatus.size();
	drawmesh->numofsubseg = numofsubseg;
	drawmesh->seglist = new int[3 * numofsubseg];
	cudaMemcpy(drawmesh->seglist, thrust::raw_pointer_cast(&t_seglist[0]), 3 * numofsubseg * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->segstatus = new tristatus[numofsubseg];
	cudaMemcpy(drawmesh->segstatus, thrust::raw_pointer_cast(&t_segstatus[0]), numofsubseg * sizeof(tristatus), cudaMemcpyDeviceToHost);

	int numofsubface = t_tristatus.size();
	drawmesh->numofsubface = numofsubface;
	drawmesh->trifacelist = new int[3 * numofsubface];
	cudaMemcpy(drawmesh->trifacelist, thrust::raw_pointer_cast(&t_trifacelist[0]), 3 * numofsubface * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->tristatus = new tristatus[numofsubface];
	cudaMemcpy(drawmesh->tristatus, thrust::raw_pointer_cast(&t_tristatus[0]), numofsubface * sizeof(tristatus), cudaMemcpyDeviceToHost);

	int numoftet = t_tetstatus.size();
	drawmesh->numoftet = numoftet;
	drawmesh->tetlist = new int[4 * numoftet];
	cudaMemcpy(drawmesh->tetlist, thrust::raw_pointer_cast(&t_tetlist[0]), 4 * numoftet * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->tetstatus = new tetstatus[numoftet];
	cudaMemcpy(drawmesh->tetstatus, thrust::raw_pointer_cast(&t_tetstatus[0]), numoftet * sizeof(tetstatus), cudaMemcpyDeviceToHost);

	// cavity
	int numberofthreads = t_threadlist.size();
	drawmesh->numofthread = numberofthreads;
	drawmesh->threadlist = new int[numberofthreads];
	cudaMemcpy(drawmesh->threadlist, thrust::raw_pointer_cast(&t_threadlist[0]), numberofthreads * sizeof(int), cudaMemcpyDeviceToHost);
	int numofinsertpt = t_insertidxlist.size();
	drawmesh->numofinsertpt = numofinsertpt;
	drawmesh->insertiontype = insertiontype;
	drawmesh->insertidxlist = new int[numofinsertpt];
	cudaMemcpy(drawmesh->insertidxlist, thrust::raw_pointer_cast(&t_insertidxlist[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->insertptlist = new REAL[3 * numofinsertpt];
	cudaMemcpy(drawmesh->insertptlist, thrust::raw_pointer_cast(&t_insertptlist[0]), 3 * numofinsertpt * sizeof(double), cudaMemcpyDeviceToHost);
	drawmesh->threadmarker = new int[numofinsertpt];
	cudaMemcpy(drawmesh->threadmarker, thrust::raw_pointer_cast(&t_threadmarker[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->cavebdrylist = new tethandle[t_cavebdrylist.size()];
	cudaMemcpy(drawmesh->cavebdrylist, thrust::raw_pointer_cast(&t_cavebdrylist[0]), t_cavebdrylist.size() * sizeof(tethandle), cudaMemcpyDeviceToHost);
	drawmesh->cavebdrynext = new int[t_cavebdrynext.size()];
	cudaMemcpy(drawmesh->cavebdrynext, thrust::raw_pointer_cast(&t_cavebdrynext[0]), t_cavebdrynext.size() * sizeof(int), cudaMemcpyDeviceToHost);
	drawmesh->cavebdryhead = new int[numofinsertpt];
	cudaMemcpy(drawmesh->cavebdryhead, thrust::raw_pointer_cast(&t_cavebdryhead[0]), numofinsertpt * sizeof(int), cudaMemcpyDeviceToHost);
}

void outputAllFacets(
	RealD& t_pointlist,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	TetHandleD& t_tri2tetlist,
	TetStatusD& t_tetstatus,
	char* filename,
	int iter
)
{
	int numofpoints = t_pointlist.size() / 3;
	double* pointlist = new double[3 * numofpoints];
	cudaMemcpy(pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * numofpoints * sizeof(double), cudaMemcpyDeviceToHost);

	int numoftris = t_tristatus.size();
	int* trifacelist = new int[3 * numoftris];
	cudaMemcpy(trifacelist, thrust::raw_pointer_cast(&t_trifacelist[0]), 3 * numoftris * sizeof(int), cudaMemcpyDeviceToHost);
	tristatus* tristatuslist = new tristatus[numoftris];
	cudaMemcpy(tristatuslist, thrust::raw_pointer_cast(&t_tristatus[0]), numoftris * sizeof(tristatus), cudaMemcpyDeviceToHost);
	tethandle* tri2tetlist = new tethandle[2 * numoftris];
	cudaMemcpy(tri2tetlist, thrust::raw_pointer_cast(&t_tri2tetlist[0]), 2 * numoftris * sizeof(tethandle), cudaMemcpyDeviceToHost);

	int numoftet = t_tetstatus.size();
	tetstatus* tetstatuslist = new tetstatus[numoftet];
	cudaMemcpy(tetstatuslist, thrust::raw_pointer_cast(&t_tetstatus[0]), numoftet * sizeof(tetstatus), cudaMemcpyDeviceToHost);

	int numofvalidtris = 0;
	int i;
	for (i = 0; i < numoftris; i++)
	{
		if (!tristatuslist[i].isEmpty())
			numofvalidtris++;
	}

	// Prepare for fileName
	std::ostringstream strs;
	strs << filename << "_" << iter << ".off";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	// output to off
	FILE * fp;
	fp = fopen(comtmp, "w");
	fprintf(fp, "OFF\n");
	fprintf(fp, "%d %d %d\n", numofpoints, numofvalidtris, 0);
	for (i = 0; i < numofpoints; i++)
	{
		fprintf(fp, "%lf %lf %lf\n", pointlist[3 * i], pointlist[3 * i + 1], pointlist[3 * i + 2]);
	}
	for (i = 0; i < numoftris; i++)
	{
		if (!tristatuslist[i].isEmpty())
		{
			tethandle neitet = tri2tetlist[2 * i + 0];
			if(!tetstatuslist[neitet.id].isInDomain())
				fprintf(fp, "%d  %d %d %d\n", 3,
					trifacelist[3 * i + 0], trifacelist[3 * i + 1], trifacelist[3 * i + 2]);
			else
				fprintf(fp, "%d  %d %d %d\n", 3,
					trifacelist[3 * i + 1], trifacelist[3 * i + 0], trifacelist[3 * i + 2]);
		}
	}
	fclose(fp);

	delete[] pointlist;
	delete[] trifacelist;
	delete[] tristatuslist;
}

void outputAllBadFacets(
	RealD& t_pointlist,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	TetHandleD& t_tri2tetlist,
	TetStatusD& t_tetstatus,
	char* filename,
	int iter
)
{
	int numofpoints = t_pointlist.size() / 3;
	double* pointlist = new double[3 * numofpoints];
	cudaMemcpy(pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * numofpoints * sizeof(double), cudaMemcpyDeviceToHost);

	int numoftris = t_tristatus.size();
	int* trifacelist = new int[3 * numoftris];
	cudaMemcpy(trifacelist, thrust::raw_pointer_cast(&t_trifacelist[0]), 3 * numoftris * sizeof(int), cudaMemcpyDeviceToHost);
	tristatus* tristatuslist = new tristatus[numoftris];
	cudaMemcpy(tristatuslist, thrust::raw_pointer_cast(&t_tristatus[0]), numoftris * sizeof(tristatus), cudaMemcpyDeviceToHost);
	tethandle* tri2tetlist = new tethandle[2 * numoftris];
	cudaMemcpy(tri2tetlist, thrust::raw_pointer_cast(&t_tri2tetlist[0]), 2 * numoftris * sizeof(tethandle), cudaMemcpyDeviceToHost);

	int numoftet = t_tetstatus.size();
	tetstatus* tetstatuslist = new tetstatus[numoftet];
	cudaMemcpy(tetstatuslist, thrust::raw_pointer_cast(&t_tetstatus[0]), numoftet * sizeof(tetstatus), cudaMemcpyDeviceToHost);

	int numofbadtris = 0;
	int i;
	for (i = 0; i < numoftris; i++)
	{
		if (!tristatuslist[i].isEmpty() && tristatuslist[i].isBad())
			numofbadtris++;
	}

	// Prepare for fileName
	std::ostringstream strs;
	strs << filename << "_" << iter << ".off";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	// output to off
	FILE * fp;
	fp = fopen(comtmp, "w");
	fprintf(fp, "OFF\n");
	fprintf(fp, "%d %d %d\n", numofpoints, numofbadtris, 0);
	for (i = 0; i < numofpoints; i++)
	{
		fprintf(fp, "%lf %lf %lf\n", pointlist[3 * i], pointlist[3 * i + 1], pointlist[3 * i + 2]);
	}
	for (i = 0; i < numoftris; i++)
	{
		if (!tristatuslist[i].isEmpty() && tristatuslist[i].isBad())
		{
			tethandle neitet = tri2tetlist[2 * i + 0];
			if (!tetstatuslist[neitet.id].isInDomain())
				fprintf(fp, "%d  %d %d %d\n", 3,
					trifacelist[3 * i + 0], trifacelist[3 * i + 1], trifacelist[3 * i + 2]);
			else
				fprintf(fp, "%d  %d %d %d\n", 3,
					trifacelist[3 * i + 1], trifacelist[3 * i + 0], trifacelist[3 * i + 2]);
		}
	}
	fclose(fp);

	delete[] pointlist;
	delete[] trifacelist;
	delete[] tristatuslist;
}

void outputInternalMesh(
	RealD& t_pointlist,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	char* filename,
	int iter
)
{
	int numofpoints = t_pointlist.size() / 3;
	double* pointlist = new double[3 * numofpoints];
	cudaMemcpy(pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * numofpoints * sizeof(double), cudaMemcpyDeviceToHost);

	int numoftris = t_tristatus.size();
	int* trifacelist = new int[3 * numoftris];
	cudaMemcpy(trifacelist, thrust::raw_pointer_cast(&t_trifacelist[0]), 3 * numoftris * sizeof(int), cudaMemcpyDeviceToHost);
	tristatus* tristatuslist = new tristatus[numoftris];
	cudaMemcpy(tristatuslist, thrust::raw_pointer_cast(&t_tristatus[0]), numoftris * sizeof(tristatus), cudaMemcpyDeviceToHost);

	int numofvalidtris = 0;
	int i;
	for (i = 0; i < numoftris; i++)
	{
		if (!tristatuslist[i].isEmpty())
			numofvalidtris++;
	}

	int numoftet = t_tetstatus.size();
	int* tetlist = new int[4 * numoftet];
	cudaMemcpy(tetlist, thrust::raw_pointer_cast(&t_tetlist[0]), 4 * numoftet * sizeof(int), cudaMemcpyDeviceToHost);
	tetstatus* tetstatuslist = new tetstatus[numoftet];
	cudaMemcpy(tetstatuslist, thrust::raw_pointer_cast(&t_tetstatus[0]), numoftet * sizeof(tetstatus), cudaMemcpyDeviceToHost);
	int numofvalidtets = 0;
	for (i = 0; i < numoftet; i++)
	{
		if (!tetstatuslist[i].isEmpty() && tetlist[4 * i + 3] != -1)
			numofvalidtets++;
	}

	// Prepare for fileName
	std::ostringstream strs;
	strs << filename << "_" << iter << ".mesh";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	// output to medit
	// writing
	FILE * fp = fopen(comtmp, "w");
	fprintf(fp, "MeshVersionFormatted 1\n");
	fprintf(fp, "Dimension 3\n");

	fprintf(fp, "Vertices\n");
	fprintf(fp, "%d\n", numofpoints);
	for (i = 0; i < numofpoints; i++)
		fprintf(fp, "%lf %lf %lf 1\n", pointlist[3 * i + 0], pointlist[3 * i + 1], pointlist[3 * i + 2]);

	fprintf(fp, "Triangles\n");
	fprintf(fp, "%d\n", numofvalidtris);

	for (i = 0; i < numoftris; i++)
	{
		if (!tristatuslist[i].isEmpty())
		{
			fprintf(fp, "%d %d %d 1\n", trifacelist[3 * i] + 1, trifacelist[3 * i + 1] + 1,
				trifacelist[3 * i + 2] + 1);
		}
	}

	fprintf(fp, "Tetrahedra\n");
	fprintf(fp, "%d\n", numofvalidtets);
	for (i = 0; i < numoftet; i++)
	{
		if (!tetstatuslist[i].isEmpty() && tetlist[4 * i + 3] != -1)
		{
			fprintf(fp, "%d %d %d %d 2\n", tetlist[4 * i] + 1, tetlist[4 * i + 1] + 1,
				tetlist[4 * i + 2] + 1, tetlist[4 * i + 3] + 1);
		}
	}
	fprintf(fp, "End\n");
	fclose(fp);

	delete[] pointlist;
	delete[] trifacelist;
	delete[] tristatuslist;
	delete[] tetlist;
	delete[] tetstatuslist;
}