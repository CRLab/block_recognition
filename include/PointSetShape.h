#ifndef _POINT_SET_SHAPE_H_
#define _POINT_SET_SHAPE_H_

#include <UserData.h>
#include <vtkPolyData.h>
#include <list>

inline void mat_copy_rigid_transform_to4x4(const double* src, double** hommat4x4)
{
	// The rotation
	hommat4x4[0][0] = src[0]; hommat4x4[0][1] = src[1]; hommat4x4[0][2] = src[2];
	hommat4x4[1][0] = src[3]; hommat4x4[1][1] = src[4]; hommat4x4[1][2] = src[5];
	hommat4x4[2][0] = src[6]; hommat4x4[2][1] = src[7]; hommat4x4[2][2] = src[8];
	// The translation
	hommat4x4[0][3] = src[9];
	hommat4x4[1][3] = src[10];
	hommat4x4[2][3] = src[11];
	// The last row
	hommat4x4[3][0] = hommat4x4[3][1] = hommat4x4[3][2] = 0.0; hommat4x4[3][3] = 1.0;
}

template <class T>
inline void vec_copy12(const T* src, T* dst)
{
	dst[0]  = src[0];
	dst[1]  = src[1];
	dst[2]  = src[2];
	dst[3]  = src[3];
	dst[4]  = src[4];
	dst[5]  = src[5];
	dst[6]  = src[6];
	dst[7]  = src[7];
	dst[8]  = src[8];
	dst[9]  = src[9];
	dst[10] = src[10];
	dst[11] = src[11];
}

class PointSetShape
{
public:
	inline PointSetShape(const PointSetShape* src);
	inline PointSetShape(UserData* userdata, vtkPolyData* polydata, const double* rigid_transform, vtkPolyData* highResModel);
	virtual ~PointSetShape(){}

	UserData* getUserData()const{ return mUserData;}
	vtkPolyData* getPolyData()const{ return mPolyData;}

	vtkPolyData* getHighResModel()const{ return mHighResModel;}

	/** Copies the rigid transform of this shape to %mat4x4. %mat4x4 has to be a 4x4 matrix. */
	inline void getHomogeneousRigidTransform(double** mat4x4){ mat_copy_rigid_transform_to4x4(mRigidTransform, mat4x4);}
	/** Sets the rigid transform to 'mat'. */
	inline void setHomogeneousRigidTransform(const double mat[4][4]);

	/** Returns the rigid transform for this shape as a 12-vector: the first 9 elements are the elements of a rotation matrix
	  * (first the first row then the second and finally the third row) and the last three elements are the x, y, z coordinates
	  * of the translation vector. */
	const double* getRigidTransform()const{ return mRigidTransform;}
	/** This list contains the scene points which belong to this shape. */
	std::list<int>& getScenePointIds(){ return mScenePointIds;}

	double getConfidence()const{ return mConfidence;}
	void setConfidence(double value){ mConfidence = value;}

protected:
	double mRigidTransform[12];
	UserData* mUserData;
	vtkPolyData *mPolyData, *mHighResModel;
	std::list<int> mScenePointIds;
	double mConfidence;
};

//=== inline methods ===========================================================================================================

inline PointSetShape::PointSetShape(const PointSetShape* src)
{
	mUserData = src->getUserData();
	mPolyData = src->getPolyData();
	vec_copy12<double>(src->getRigidTransform(), mRigidTransform);
	mConfidence = src->getConfidence();
	mHighResModel = src->getHighResModel();
}

//==============================================================================================================================

inline PointSetShape::PointSetShape(UserData* userdata, vtkPolyData* polydata, const double* rigid_transform, vtkPolyData* highResModel)
{
	mUserData = userdata;
	mPolyData = polydata;
	vec_copy12<double>(rigid_transform, mRigidTransform);
	mHighResModel = highResModel;
	// Default values
	mConfidence = -1.0;
}

//==============================================================================================================================

inline void PointSetShape::setHomogeneousRigidTransform(const double mat[4][4])
{
	// Rotational part
	mRigidTransform[0] = mat[0][0]; mRigidTransform[1] = mat[0][1]; mRigidTransform[2] = mat[0][2];
	mRigidTransform[3] = mat[1][0]; mRigidTransform[4] = mat[1][1]; mRigidTransform[5] = mat[1][2];
	mRigidTransform[6] = mat[2][0]; mRigidTransform[7] = mat[2][1]; mRigidTransform[8] = mat[2][2];
	// The translation
	mRigidTransform[9]  = mat[0][3];
	mRigidTransform[10] = mat[1][3];
	mRigidTransform[11] = mat[2][3];
}

//==============================================================================================================================

#endif /*_POINT_SET_SHAPE_H_*/
