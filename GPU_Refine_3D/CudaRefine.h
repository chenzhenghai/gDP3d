#pragma once

#include <cuda_runtime.h>
#include <helper_timer.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "MeshIO.h"
#include "MeshStructure.h"

void GPU_Refine_3D(
	MESHCR* input_criteria,
	MESHIO* input_mesh,
	MESHBH* input_behavior
);