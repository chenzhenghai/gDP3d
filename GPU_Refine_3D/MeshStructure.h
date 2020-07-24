#pragma once

#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for tetrahedron													 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct tethandle
{
	int id; // index for tetrahedron in tetrahedron list
	char ver; // Range from 0 to 11.
			  //           |   edge 0     edge 1     edge 2    //
			  //   --------|--------------------------------   //
			  //    face 0 |   0 (0/1)    4 (0/3)    8 (0/2)   //
			  //    face 1 |   1 (1/2)    5 (1/3)    9 (1/0)   //
			  //    face 2 |   2 (2/3)    6 (2/1)   10 (2/0)   //
			  //    face 3 |   3 (3/0)    7 (3/1)   11 (3/2)   //
	__forceinline__ __host__ __device__
	tethandle(void)
	{
		this->id = -1;
		ver = 11;
	}

	__forceinline__ __host__ __device__
	tethandle(int id, char ver)
	{
		this->id = id;
		this->ver = ver;
	}

} tethandle;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for triangle														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct trihandle
{
	int id; // index for triangle in triangle list
	char shver; // Range from 0 to 5.
				//                 | edge 0   edge 1   edge 2     //
				//  ---------------|--------------------------    //
				//   ccw orieation |   0        2        4        //
				//    cw orieation |   1        3        5		  //
	
	__forceinline__ __host__ __device__
	trihandle(void)
	{
		this->id = -1;
		shver = 0;
	}

	__forceinline__ __host__ __device__
	trihandle(int id, char ver)
	{
		this->id = id;
		this->shver = ver;
	}

} trihandle;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for tetrahedron status											 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct tetstatus
{
	unsigned char _status;
	// 76543210
	// ||||||||----- tetrahedron status
	// |||||||------ unsplittable
	// ||||||------- bad quality
	// |||||-------- in domain
	// ||||--------- new
	// |||---------- cavity reuse

	__forceinline__ __host__ __device__
	tetstatus(void)
	{
		_status = 0;
	}

	__forceinline__ __host__ __device__
	tetstatus(unsigned char s)
	{
		_status = s;
	}

	__forceinline__ __host__ __device__
	void clear()
	{
		_status = 0; // clear all information
	}

	//status of tetrahedron. 0 = empty; 1 = non-empty
	__forceinline__ __host__ __device__
	void setEmpty(bool n)
	{
		if (n)
			_status = _status & (~1);
		else
			_status = _status | 1;
	}

	__forceinline__ __host__ __device__
	bool isEmpty() const
	{
		return (_status & 1) == 0;
	}

	//0 = splittable; 1 = unsplittable
	__forceinline__ __host__ __device__
		void setUnsplittable(bool a)
	{
		_status = (_status & ~(1 << 1)) | (a ? 1 : 0) << 1;
	}

	__forceinline__ __host__ __device__
		bool isUnsplittable() const
	{
		return (_status & (1 << 1)) > 0;
	}

	__forceinline__ __host__ __device__
	void setBad(bool b)
	{
		_status = (_status & ~(1 << 2)) | (b ? 1 : 0) << 2;
	}

	__forceinline__ __host__ __device__
	bool isBad() const
	{
		return (_status & (1 << 2)) > 0;
	}

	__forceinline__ __host__ __device__
	void setInDomain(bool b)
	{
		_status = (_status & ~(1 << 3)) | (b ? 1 : 0) << 3;
	}

	__forceinline__ __host__ __device__
	bool isInDomain() const
	{
		return (_status & (1 << 3)) > 0;
	}

	__forceinline__ __host__ __device__
	void setNew(bool b)
	{
		_status = (_status & ~(1 << 4)) | (b ? 1 : 0) << 4;
	}

	__forceinline__ __host__ __device__
	bool isNew() const
	{
		return (_status & (1 << 4)) > 0;
	}

	__forceinline__ __host__ __device__
	void setCavityReuse(bool b)
	{
		_status = (_status & ~(1 << 5)) | (b ? 1 : 0) << 5;
	}

	__forceinline__ __host__ __device__
	bool isCavityReuse() const
	{
		return (_status & (1 << 5)) > 0;
	}

} tetstatus;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for subface status											     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct tristatus
{
	unsigned char _status;
	// 76543210
	// ||||||||----- triface status
	// |||||||------ unsplittable
	// ||||||------- bad quality
	// |||||-------- new
	// ||||--------- to check
	// |||---------- cavity reuse

	__forceinline__ __host__ __device__
	tristatus(void)
	{
		_status = 0;
	}

	__forceinline__ __host__ __device__
	tristatus(unsigned char s)
	{
		_status = s;
	}

	__forceinline__ __host__ __device__
	void clear()
	{
		_status = 0; // clear all information
	}

	//status of triface. 0 = empty; 1 = non-empty
	__forceinline__ __host__ __device__
	void setEmpty(bool n)
	{
		if (n)
			_status = _status & (~1);
		else
			_status = _status | 1;
	}

	__forceinline__ __host__ __device__
	bool isEmpty() const
	{
		return (_status & 1) == 0;
	}

	//0 = splittable; 1 = unsplittable
	__forceinline__ __host__ __device__
	void setUnsplittable(bool a)
	{
		_status = (_status & ~(1 << 1)) | (a ? 1 : 0) << 1;
	}

	__forceinline__ __host__ __device__
	bool isUnsplittable() const
	{
		return (_status & (1 << 1)) > 0;
	}

	__forceinline__ __host__ __device__
	void setBad(bool b)
	{
		_status = (_status & ~(1 << 2)) | (b ? 1 : 0) << 2;
	}

	__forceinline__ __host__ __device__
	bool isBad() const
	{
		return (_status & (1 << 2)) > 0;
	}

	__forceinline__ __host__ __device__
	void setNew(bool b)
	{
		_status = (_status & ~(1 << 3)) | (b ? 1 : 0) << 3;
	}

	__forceinline__ __host__ __device__
	bool isNew() const
	{
		return (_status & (1 << 3)) > 0;
	}

	__forceinline__ __host__ __device__
	void setToCheck(bool b)
	{
		_status = (_status & ~(1 << 4)) | (b ? 1 : 0) << 4;
	}

	__forceinline__ __host__ __device__
	bool isToCheck() const
	{
		return (_status & (1 << 4)) > 0;
	}

	__forceinline__ __host__ __device__
	void setCavityReuse(bool b)
	{
		_status = (_status & ~(1 << 5)) | (b ? 1 : 0) << 5;
	}

	__forceinline__ __host__ __device__
	bool isCavityReuse() const
	{
		return (_status & (1 << 5)) > 0;
	}

} tristatus;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Labels (enumeration declarations) used by TetGen.                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

// Labels that signify the type of a vertex. 
enum verttype {
	UNUSEDVERTEX, DUPLICATEDVERTEX, RIDGEVERTEX, ACUTEVERTEX,
	FACETVERTEX, VOLVERTEX, FREESEGVERTEX, FREEFACETVERTEX,
	FREEVOLVERTEX, NREGULARVERTEX, DEADVERTEX
};

// Labels that signify the result of triangle-triangle intersection test.
enum interresult {
	UNKNOWNINTER, DISJOINT, INTERSECT, COPLANAR, SHAREVERT, SHAREEDGE, SHAREFACE,
	TOUCHEDGE, TOUCHFACE, ACROSSVERT, ACROSSEDGE, ACROSSFACE,
	COLLISIONFACE, ACROSSSEG, ACROSSSUB
};

// Labels that signify the result of point location.
enum locateresult {
	UNKNOWN, OUTSIDE, INTETRAHEDRON, ONFACE, ONEDGE, ONVERTEX,
	ENCVERTEX, ENCSEGMENT, ENCSUBFACE, NEARVERTEX, NONREGULAR,
	INSTAR, BADELEMENT
};
