#include "ros/ros.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

#include <opencv/cv.h>
#include <image_geometry/pinhole_camera_model.h>
#include <boost/foreach.hpp>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <vtkPointData.h>
#include <vtkPolyDataReader.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkTransform.h>

#include <UserData.h>
#include <PointSetShape.h>
#include <block_recognition/FindObjects.h>
#include <block_recognition/RecognizedObjects.h>

#include <tf_conversions/tf_kdl.h>
#include <resource_retriever/retriever.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

using namespace std;
using namespace cv;

Mat rgbImg;
sensor_msgs::CameraInfoConstPtr info_msg;
bool rgbInit;
bool cam_info_init;
ros::NodeHandle* n;
ros::Publisher blocks_pc_pub;

float z_clip_min;
float z_clip_max;
float x_clip_min;
float x_clip_max;
float y_clip_min;
float y_clip_max;

int img_x_min;
int img_y_min;
int img_x_max;
int img_y_max;

tf::TransformListener *tf_listener;
std::string block_model_vtk;
std::string block_model_stl;
vtkPolyData *block_model_data;
UserData *block_user_data_small;
UserData *block_user_data_medium;
UserData *block_user_data_large;
void load_block_model();
ros::Publisher objects_pub_;
ros::Publisher markers_pub_;

boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud;

boost::mutex buffer_mutex_;
std::list<vtkSmartPointer<vtkPolyDataReader> > readers_;

static void array_to_pose(const double* array, geometry_msgs::Pose &pose_msg)
{
    tf::Matrix3x3 rot_m =  tf::Matrix3x3(
            array[0],array[1],array[2],
            array[3],array[4],array[5],
            array[6],array[7],array[8]);
    tf::Quaternion rot_q;
    rot_m.getRotation(rot_q);
    rot_q = rot_q * tf::Quaternion(tf::Vector3(1.0,0,0), M_PI/2.0);
    rot_q = rot_q * tf::Quaternion(tf::Vector3(0.0,0,1.0), M_PI);
    tf::quaternionTFToMsg(rot_q, pose_msg.orientation);

    pose_msg.position.x = array[9];
    pose_msg.position.y = array[10];
    pose_msg.position.z = array[11];
}

void saveRGBImg(const sensor_msgs::ImageConstPtr& msg)
{
    try {
        rgbImg = cv_bridge::toCvCopy(msg, "bgr8")->image;
        rgbInit = true;
    } catch(cv_bridge::Exception& e) {
        ROS_ERROR("could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void saveCamInfo(const sensor_msgs::CameraInfoConstPtr& msg)
{
    try {
        info_msg = msg;
        cam_info_init = true;
    } catch(cv_bridge::Exception& e) {
        ROS_ERROR("Could not load cam_info.");
    }
}

void getCloud(const sensor_msgs::PointCloud2ConstPtr &points_msg)
{
    // Lock the buffer mutex while we're capturing a new point cloud
    boost::mutex::scoped_lock buffer_lock(buffer_mutex_);

    // Convert to PCL cloud
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*points_msg, *cloud_tmp);

    cloud = cloud_tmp;
}

cv::Rect getImageRectTransform(int &_pxMin, int &_pyMin, tf::Transform &tf_to_cam)
{

    image_geometry::PinholeCameraModel cam_model_;
    cam_model_.fromCameraInfo(info_msg);

    //Minimum point transform
    tf::Vector3 world_min_pt_3d;
    world_min_pt_3d.setX(x_clip_min);
    world_min_pt_3d.setY(y_clip_min);
    world_min_pt_3d.setZ(z_clip_min);
    tf::Vector3 camera_min_pt_3d = tf_to_cam * world_min_pt_3d;

    cout << "---> World cropping bounds: " << x_clip_min << ", " << x_clip_max << ", " << y_clip_min << ", " << y_clip_max << ", " << z_clip_min << ", " << z_clip_max << std::endl;

    cv::Point3d min_pt_3d(camera_min_pt_3d.x(), camera_min_pt_3d.y(), camera_min_pt_3d.z());
    std::cout << "min_pt_3d: " << min_pt_3d.x << " " << min_pt_3d.y << " " << min_pt_3d.z << std::endl;
    cv::Point2d min_pt;
    min_pt = cam_model_.project3dToPixel(min_pt_3d);

    //Maximum point transform
    tf::Vector3 world_max_pt_3d;
    world_max_pt_3d.setX(x_clip_max);
    world_max_pt_3d.setY(y_clip_max);
    world_max_pt_3d.setZ(z_clip_min);
    tf::Vector3 camera_max_pt_3d = tf_to_cam * world_max_pt_3d;

    cv::Point3d max_pt_3d(camera_max_pt_3d.x(), camera_max_pt_3d.y(), camera_max_pt_3d.z());
    std::cout << "max_pt_3d: " << max_pt_3d.x << " " << max_pt_3d.y << " " << max_pt_3d.z << std::endl;
    cv::Point2d max_pt;
    max_pt = cam_model_.project3dToPixel(max_pt_3d);

    cout << "---> pre-cropping image bounds: " << min_pt.x << ", " << max_pt.x << ", " << min_pt.y << ", " << max_pt.y << endl;

    //Generate image crop
    img_x_min = max(0, (int) min(min_pt.x, max_pt.x));
    img_y_min = max(0, (int) min(min_pt.y, max_pt.y));
    img_x_max = min(1920, (int) max(min_pt.x, max_pt.x));
    img_y_max = min(1080, (int) max(min_pt.y, max_pt.y));

    cout << "---> cropping image bounds: " << img_x_min << ", " << img_x_max << ", " << img_y_min << ", " << img_y_max << endl;

    _pxMin = img_x_min;
    _pyMin = img_y_min;
    cv::Rect rect(img_x_min, img_y_min, img_x_max - img_x_min, img_y_max - img_y_min);
    return rect;
}

int findBlocks(list<boost::shared_ptr<PointSetShape> >& out)
{
    if (!rgbInit || !cam_info_init) return -1;

    tf::StampedTransform transform;
    tf::StampedTransform transform_world_to_camera;
    ros::Time now = ros::Time::now();
    tf_listener->waitForTransform ("/world", "/kinect2_rgb_optical_frame", now, ros::Duration(4.0));
    tf_listener->lookupTransform ("/world", "/kinect2_rgb_optical_frame", now, transform);
    tf_listener->lookupTransform ("/kinect2_rgb_optical_frame", "/world", now, transform_world_to_camera);
    tf::Transform tf_to_cam(transform_world_to_camera.getRotation(), transform_world_to_camera.getOrigin());

    // Lock the buffer mutex
    boost::mutex::scoped_lock buffer_lock(buffer_mutex_);

    // transform from camera frame to world frame
    pcl::PointCloud<pcl::PointXYZRGB> world_pc;
    cloud->header.frame_id = "/kinect2_rgb_optical_frame";
    pcl_ros::transformPointCloud("/world", *cloud, world_pc, *tf_listener);

    // get the clipped point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_xy(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_xyz(new pcl::PointCloud<pcl::PointXYZRGB>());

    pcl_ros::transformPointCloud("/world", *cloud, *cloud_transformed, *tf_listener);

    // TODO: try removing plane via SACSegmentation
    // crop point cloud based on 2d indices corresponding to rgb image
    pcl::ExtractIndices<pcl::PointXYZRGB > eifilter;
    eifilter.setInputCloud(cloud_transformed);
    pcl::IndicesPtr indices(new std::vector<int>);

    size_t row_width = 1920;
    size_t row_start = img_y_min * row_width;
    size_t row_end = img_y_max * row_width;
    size_t col_start = img_x_min;
    size_t col_end = img_x_max;
    for (int i = row_start; i < row_end; i += row_width) { //2073600
        for (int j = col_start; j < col_end; j++) {
            indices->push_back(i + j);
        }
    }
    eifilter.setIndices(indices);
    eifilter.filter(*cloud_filtered_xy);

    pcl::PassThrough<pcl::PointXYZRGB > pass;
    pass.setInputCloud(cloud_filtered_xy);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (z_clip_min, z_clip_max);
    pass.filter(*cloud_filtered_xyz);

    /* publish for debugging */
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed_back(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_filtered_xyz->header.frame_id = "/world";
    pcl_ros::transformPointCloud("/kinect2_rgb_optical_frame",
                *cloud_filtered_xyz,
                *cloud_transformed_back,
                *tf_listener);

    cloud_transformed_back->header.frame_id = "/kinect2_rgb_optical_frame";

    pcl::PCLPointCloud2 cloud_transformed_back_pc2;
    pcl::toPCLPointCloud2(*cloud_transformed_back, cloud_transformed_back_pc2);

    sensor_msgs::PointCloud2 cloud_transformed_back_msg;
    pcl_conversions::fromPCL(cloud_transformed_back_pc2, cloud_transformed_back_msg);

    blocks_pc_pub.publish(cloud_transformed_back_msg);

    // euclidean cluster extraction
    // create KdTree object for search method of extraction
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud_filtered_xyz);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.02); // 2cm
    ec.setMinClusterSize(100);
//    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered_xyz);
    ec.extract(cluster_indices);

    // find center of mass of each cluster
    std::cout << "found " << cluster_indices.size() << " clusters..." << std::endl;
    std::vector<pcl::PointXYZ> object_centers;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        float centerX = 0.0, centerY = 0.0, centerZ = 0.0;
        float numP = 0.0;

        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {

            centerX += cloud_filtered_xyz->points[*pit].x;
            centerY += cloud_filtered_xyz->points[*pit].y;
            centerZ += cloud_filtered_xyz->points[*pit].z;

            numP++;
        }
        centerX = centerX / numP;
        centerY = centerY / numP;
        centerZ = centerZ / numP;

        pcl::PointXYZ pt_xyz(centerX, centerY, centerZ);

        object_centers.push_back(pt_xyz);
    }

    /**************************************/
    /* find orientation from 2d RGB image */
    // crop the 2d image based on x and y clipping values
    int pxMin = 0, pyMin = 0;
    cv::Rect rect = getImageRectTransform(pxMin, pyMin, tf_to_cam);

    cv::Mat croppedImg;
    croppedImg = rgbImg(rect);

    Mat src_gray;

    cvtColor(croppedImg, src_gray, CV_BGR2GRAY);

    Mat src_binary;
    double threshold_value = 125;
    int threshold_type = 1; //binary inverted
    double max_BINARY_value = 255;
    threshold( src_gray, src_binary, threshold_value, max_BINARY_value, threshold_type);

//    blur(src_gray, src_gray, Size(3,3));

    RNG rng(12345);
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    std::cout << "---------- detect edges ----------" << std::endl;
    /// Detect edges using canny
    int thresh = 90;
    Canny( src_binary, canny_output, thresh, thresh*3, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<RotatedRect> minRect( contours.size() );
    vector<Moments> mu( contours.size() );
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ ) {
        /// Get the moments
        mu[i] = moments( contours[i], true );

        /// Get the rotated rectangles
        minRect[i] = minAreaRect( Mat(contours[i]) );

        ///  Get the mass centers
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    }

    std::cout << "found " << contours.size() << " contours..." << std::endl;

    /// Find orientation and draw contours
    for( int i = 0; i < contours.size(); i++ ) {

        if (object_centers.size() == 0) break; // TODO: possibly recognizes incorrect blocks this way

        float rectArea = minRect[i].size.width * minRect[i].size.height; // Ignore tiny contours found from noise in image (like a dent in foam)
        //if (mu[i].m00 < 500) continue; // use rectArea, moments sometimes returns too small for some reason
        if (rectArea < 600) continue;
        std::cout << "clusters left: " << object_centers.size() << std::endl;
        std::cout << "area " << i << ": " << mu[i].m00 << " vs rectArea: " << rectArea << std::endl;

        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

        drawContours( croppedImg, contours, i, color, 2, 8, hierarchy, 0, Point() );
        circle( croppedImg, mc[i], 4, color, -1, 8, 0 );

        // rotated rectangle
        Point2f rect_points[4]; minRect[i].points( rect_points );
        for( int j = 0; j < 4; j++ )
        line( croppedImg, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );

        float rotation = minRect[i].angle;
        ostringstream ss;
        if (rectArea > 7000) {
            ss << "large: ";
        } else if (rectArea < 4500) {
            ss << "small: ";
        } else {
            ss << "medium: ";
        }
        ss << rotation; string r(ss.str());
        putText(croppedImg, r.c_str(), cvPoint(mc[i].x,mc[i].y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

        // index into point cloud to get position of center of mass
        // TODO: figure out why sometimes mc is Nan?
        if (isnan(mc[i].x) || isnan(mc[i].y)) {
            std::cout << "contour " << i << " is nan!" << std::endl;
            continue;
        }
        int px = pxMin + mc[i].x;
        int py = pyMin + mc[i].y;

        // get corresponding point cloud point
        pcl::PointXYZRGB pt = world_pc[py*1920 + px];
        if (isnan(pt.x) || isnan(pt.y)) {
            pt = world_pc[py*1920 + px + 5]; // TODO: fix hacky way to deal with nan pc values
        }

        // transform point into camera frame to publish
        double min_dist = std::numeric_limits<double>::max();
        int min_idx = 0;
        // hacky? way to match point cloud cluster center to rgb img center by comparing distances
        for (int j = 0; j < object_centers.size(); j++) {
            double dist = (object_centers[j].x - pt.x) * (object_centers[j].x - pt.x);
            dist += (object_centers[j].y - pt.y) * (object_centers[j].y - pt.y);
            dist += (object_centers[j].z - pt.z) * (object_centers[j].z - pt.z);

            if (dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
        tf::Vector3 cam_pt;
        cam_pt.setX(object_centers[min_idx].x);
        cam_pt.setY(object_centers[min_idx].y);
        cam_pt.setZ(object_centers[min_idx].z);
        cam_pt = tf_to_cam * cam_pt; // transform into camera frame
        std::cout << "x: " << cam_pt.x() << ", y: " << cam_pt.y() << ", z: " << cam_pt.z() << std::endl;

        float sinTheta = sin(rotation * KDL::deg2rad);
        float cosTheta = cos(rotation * KDL::deg2rad);
        double rigid_transform[12];
        // Rotational part
        rigid_transform[0] = cosTheta;  rigid_transform[1] = -sinTheta; rigid_transform[2] = 0;
        rigid_transform[3] = sinTheta;  rigid_transform[4] = cosTheta;  rigid_transform[5] = 0;
        rigid_transform[6] = 0;         rigid_transform[7] = 0;         rigid_transform[8] = 1;
        // The translation
        rigid_transform[9]  = cam_pt.x();
        rigid_transform[10] = cam_pt.y();
        rigid_transform[11] = cam_pt.z();

        UserData *block_user_data;
        if (rectArea > 7000) {
            block_user_data = block_user_data_large;
        } else if (rectArea < 4500) {
            block_user_data = block_user_data_small;
        } else {
            block_user_data = block_user_data_medium;
        }
        //block_user_data = block_user_data_small;
        boost::shared_ptr<PointSetShape> shape = boost::make_shared<PointSetShape>(block_user_data, block_model_data, rigid_transform, block_model_data);
        out.push_back(shape);

        object_centers.erase(object_centers.begin() + min_idx); // make sure not to reuse the same cluster again
    }

    /// Show in a window for debugging
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", croppedImg );

    //namedWindow( "Gray", CV_WINDOW_AUTOSIZE );
    //imshow( "Gray", src_gray );

    namedWindow( "binary", CV_WINDOW_AUTOSIZE );
    imshow( "binary", src_binary );

    namedWindow( "canny", CV_WINDOW_AUTOSIZE );
    imshow( "canny", canny_output );

    return 0;
}

void publish_markers(const block_recognition::RecognizedObjects &objects_msg)
{
    visualization_msgs::MarkerArray marker_array;
    int id = 0;

    for(std::vector<block_recognition::PointSetShape>::const_iterator it = objects_msg.objects.begin();
            it != objects_msg.objects.end();
            ++it)
    {
        visualization_msgs::Marker marker;

        marker.header = objects_msg.header;
        marker.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(20.0);
        marker.ns = "objrec";
        marker.id = 0;

        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;

        marker.color.a = 0.75;
        marker.color.r = 1.0;
        marker.color.g = 0.1;
        marker.color.b = 0.3;

        marker.id = id++;
        marker.pose = it->pose;

        marker.mesh_resource = block_model_stl;
        marker_array.markers.push_back(marker);
    }

    // Publish the markers
    markers_pub_.publish(marker_array);
}


bool recognizeBlocks(block_recognition::FindObjects::Request &req, block_recognition::FindObjects::Response &res)
{
    std::list<boost::shared_ptr<PointSetShape> > detected_models;
    int failure = findBlocks(detected_models);
    // No objects recognized
    if (failure) {
        return false;
    }

    // Construct recognized objects message
    block_recognition::RecognizedObjects objects_msg;
    objects_msg.header.stamp = pcl_conversions::fromPCL(cloud->header).stamp;
    objects_msg.header.frame_id = cloud->header.frame_id;

    std::cout << "publishing " << detected_models.size() << "models..." << objects_msg.header.frame_id << std::endl;
    for(std::list<boost::shared_ptr<PointSetShape> >::iterator it = detected_models.begin();
            it != detected_models.end();
            ++it)
    {
        boost::shared_ptr<PointSetShape> detected_model = *it;

        // Construct and populate a message
        block_recognition::PointSetShape pss_msg;
        pss_msg.label = detected_model->getUserData()->getLabel();
        pss_msg.confidence = detected_model->getConfidence();
        array_to_pose(detected_model->getRigidTransform(), pss_msg.pose);

        // Transform into the world frame TODO: make this frame a parameter // keeping in camera frame?
        geometry_msgs::PoseStamped pose_stamped_in, pose_stamped_out;
        pose_stamped_in.header = pcl_conversions::fromPCL(cloud->header);
        pose_stamped_in.pose = pss_msg.pose;

        KDL::Frame pose;
        tf::poseMsgToKDL(pss_msg.pose, pose);
        KDL::Frame rot = KDL::Frame(KDL::Rotation::RotX(-(KDL::PI)/2.0), KDL::Vector(0,0,0));
        KDL::Frame newPose = pose * rot;
        tf::poseKDLToMsg(newPose, pss_msg.pose);

        objects_msg.objects.push_back(pss_msg);
    }

    // Publish the visualization markers
    publish_markers(objects_msg);

    // Publish the recognized objects
    objects_pub_.publish(objects_msg);

    for(std::vector<block_recognition::PointSetShape>::const_iterator it = objects_msg.objects.begin();
            it != objects_msg.objects.end();
            ++it)
    {
        block_recognition::PointSetShape object = *it;
        res.object_name.push_back(object.label);
        res.object_pose.push_back(object.pose);
    }

    ROS_INFO("sending back response ");
    return true;
}

vtkSmartPointer<vtkPolyData> scale_vtk_model(vtkSmartPointer<vtkPolyData> & m, double scale = 1.0/1000.0)
{
  vtkSmartPointer<vtkTransform> transp = vtkSmartPointer<vtkTransform>::New();
  transp->Scale(scale, scale, scale);
  vtkSmartPointer<vtkTransformPolyDataFilter> tpd = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
#if VTK_MAJOR_VERSION <= 5
  tpd->SetInput(m);
#else
  tpd->SetInputData(m);
#endif
  tpd->SetTransform(transp);
  tpd->Update();
  return tpd->GetOutput();
}

void load_block_model()
{
    ROS_INFO_STREAM("Loading block only...");

    std::string model_label = "block";//Change this based on size

    // Get the mesh uri & store it
    std::string param_name = "model_uris/"+model_label;
    if(!n->getParam(param_name, block_model_vtk)) {
        ROS_FATAL_STREAM("Required parameter not found! Namespace: "<<n->getNamespace()<<" Parameter: "<<param_name);
        throw ros::InvalidParameterException("Parameter not found!");
    }
    param_name = "stl_uris/"+model_label;
    if(!n->getParam(param_name, block_model_stl)) {
        ROS_FATAL_STREAM("Required parameter not found! Namespace: "<<n->getNamespace()<<" Parameter: "<<param_name);
        throw ros::InvalidParameterException("Parameter not found!");
    }

    ROS_INFO_STREAM("Adding model \""<<model_label<<"\" from "<<block_model_vtk);
    // Fetch the model data with a ros resource retriever
    resource_retriever::Retriever retriever;
    resource_retriever::MemoryResource resource;

    try {
        resource = retriever.get(block_model_vtk);
    } catch (resource_retriever::Exception& e) {
        ROS_ERROR_STREAM("Failed to retrieve \""<<model_label<<"\" model file from \""<<block_model_vtk<<"\" error: "<<e.what());
        return;
    }

    // Load the model into objrec
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();
    // This copies the data from the resource structure into the polydata reader
    reader->SetBinaryInputString(
            (const char*)resource.data.get(),
            resource.size);
    reader->ReadFromInputStringOn();
    reader->Update();
    readers_.push_back(reader);

    // Get the VTK normals
    vtkSmartPointer<vtkPolyData> polydata(reader->GetOutput());
    vtkSmartPointer<vtkFloatArray> point_normals(
            vtkFloatArray::SafeDownCast(polydata->GetPointData()->GetNormals()));

    if(!point_normals) {
        ROS_ERROR_STREAM("No vertex normals for mesh: "<<block_model_vtk);
        return;
    }

    // Get the VTK points
    size_t n_points = polydata->GetNumberOfPoints();
    size_t n_normals = point_normals->GetNumberOfTuples();

    if(n_points != n_normals) {
        ROS_ERROR_STREAM("Different numbers of vertices and vertex normals for mesh: "<<block_model_vtk);
        return;
    }

    // This is just here for reference
    for(vtkIdType i = 0; i < n_points; i++)
    {
        double pV[3];
        double pN[3];

        polydata->GetPoint(i, pV);
        point_normals->GetTuple(i, pN);
    }

    // Create new model user data
    block_user_data_small = new UserData();
    block_user_data_small->setLabel("block_50");
    block_user_data_medium = new UserData();
    block_user_data_medium->setLabel("block_64");
    block_user_data_large = new UserData();
    block_user_data_large->setLabel("block_76");

    vtkSmartPointer<vtkPolyData> model_data = reader->GetOutput();
    block_model_data = scale_vtk_model(model_data);

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "block_recognition_node");
    n = new ros::NodeHandle();
    tf::TransformListener tfl;
    tf_listener = &tfl;

    rgbInit = false;
    cam_info_init = false;

    n->getParam("x_clip_min", x_clip_min);
    n->getParam("x_clip_max", x_clip_max);
    n->getParam("y_clip_min", y_clip_min);
    n->getParam("y_clip_max", y_clip_max);
    n->getParam("z_clip_min", z_clip_min);
    n->getParam("z_clip_max", z_clip_max);

    ros::Subscriber original_pc_sub = n->subscribe("/kinect2/hd/points", 1, getCloud);

    cv::startWindowThread();
    image_transport::ImageTransport it(*n);
    image_transport::Subscriber sub = it.subscribe("/kinect2/hd/image_color_rect", 1, saveRGBImg);

    ros::Subscriber cam_info_sub = n->subscribe("/kinect2/hd/camera_info", 1, saveCamInfo);

    load_block_model();

    objects_pub_ = n->advertise<block_recognition::RecognizedObjects>("recognized_objects",20);
    markers_pub_ = n->advertise<visualization_msgs::MarkerArray>("recognized_objects_markers",20);

    ros::ServiceServer find_blocks_server_ = n->advertiseService("find_blocks", recognizeBlocks);

    blocks_pc_pub = n->advertise<sensor_msgs::PointCloud2>("filtered_blocks", 1);

    ros::spin();
    return 0;
}
