# gDP3d
3D Restricted Delaunay Refinement on the GPU

Project Website: https://www.comp.nus.edu.sg/~tants/meshRefinement.html

Paper: On Designing GPU Algorithms with Applications to Mesh Refinement. Z. Chen, T.S. Tan, and H.Y. Ong. arXiv, 2020. (<a href="https://arxiv.org/abs/2007.00324">PDF</a>)

* A NVIDIA GPU is required since this project is implemented using CUDA.
* The development environment: Visual Studio 2017 and CUDA 9.0. (Please use x64 and Release mode.)
* CGAL 4.13 is used for comparision. Check https://www.cgal.org/index.html for more information.

--------------------------------------------------------------------------
Refinement Routine (located in GPU_Refine_3D/MeshRefine.h and GPU_Refine_3D/MeshRefineGPU.cpp):

void refineInputOnGPU(  
&nbsp;&nbsp;&nbsp;&nbsp; char* infile,  
&nbsp;&nbsp;&nbsp;&nbsp; bool force_features,  
&nbsp;&nbsp;&nbsp;&nbsp; MESHCR* criteria,  
&nbsp;&nbsp;&nbsp;&nbsp; MESHBH* behavior,  
&nbsp;&nbsp;&nbsp;&nbsp; char* outmesh,  
&nbsp;&nbsp;&nbsp;&nbsp; char* outdata)  

char* infile:  
The path for input file.

bool force_features:  
Has not been used yet.

MESHCR* criteria:  
The input criteria for refinement process; see GPU_Refine_3D/MeshIO.h for more details.

MESHBH* behavior:  
The input behaviors for refinement process; see GPU_Refine_3D/MeshIO.h for more details.

char* outmesh:  
The path for output mesh file.

char* outdata:  
The path for output statistic file.

--------------------------------------------------------------
Experiment

All experiments were conducted on a PC with an Intel i7-7700k 4.2GHz CPU, 32GB of DDR4 RAM and a GTX1080 Ti graphics card with 11GB of video memory.

* Real-world dataset:  
3D printing models from the <a href="https://ten-thousand-models.appspot.com/">Thingi10K</a> dataset were used. Some samples and result statistics by TetGen and this software are provided in GPU_Refine_3D/input_real/.
--------------------------------------------------------------

Proceed to GPU_Refine_3D/main.cpp to check how to call gpu and cpu refinement routines properly.
