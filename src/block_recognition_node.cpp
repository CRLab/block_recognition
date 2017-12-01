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

#include <block_recognition/FindObjects.h>

#include <tf_conversions/tf_kdl.h>
#include <resource_retriever/retriever.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

#include <block_recognition/DetectedBlock.h>

#include <ros/package.h>

#include <string>

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
std::string block_user_data_small = "block_50";
std::string block_user_data_medium = "block_64";
std::string block_user_data_large = "block_76";
ros::Publisher markers_pub_;

boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud;

boost::mutex buffer_mutex_;

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

    ROS_INFO_STREAM("---> World cropping bounds: " << x_clip_min << ", " << x_clip_max << ", " << y_clip_min << ", " << y_clip_max << ", " << z_clip_min << ", " << z_clip_max << std::endl);

    cv::Point3d min_pt_3d(camera_min_pt_3d.x(), camera_min_pt_3d.y(), camera_min_pt_3d.z());
    ROS_INFO_STREAM("min_pt_3d: " << min_pt_3d.x << " " << min_pt_3d.y << " " << min_pt_3d.z << std::endl);
    cv::Point2d min_pt;
    min_pt = cam_model_.project3dToPixel(min_pt_3d);

    //Maximum point transform
    tf::Vector3 world_max_pt_3d;
    world_max_pt_3d.setX(x_clip_max);
    world_max_pt_3d.setY(y_clip_max);
    world_max_pt_3d.setZ(z_clip_min);
    tf::Vector3 camera_max_pt_3d = tf_to_cam * world_max_pt_3d;

    cv::Point3d max_pt_3d(camera_max_pt_3d.x(), camera_max_pt_3d.y(), camera_max_pt_3d.z());
    ROS_INFO_STREAM("max_pt_3d: " << max_pt_3d.x << " " << max_pt_3d.y << " " << max_pt_3d.z << std::endl);
    cv::Point2d max_pt;
    max_pt = cam_model_.project3dToPixel(max_pt_3d);

    ROS_INFO_STREAM("---> pre-cropping image bounds: " << min_pt.x << ", " << max_pt.x << ", " << min_pt.y << ", " << max_pt.y << endl);

    //Generate image crop
    img_x_min = max(0, (int) min(min_pt.x, max_pt.x));
    img_y_min = max(0, (int) min(min_pt.y, max_pt.y));
    img_x_max = min(1920, (int) max(min_pt.x, max_pt.x));
    img_y_max = min(1080, (int) max(min_pt.y, max_pt.y));

    ROS_INFO_STREAM("---> cropping image bounds: " << img_x_min << ", " << img_x_max << ", " << img_y_min << ", " << img_y_max << endl);

    _pxMin = img_x_min;
    _pyMin = img_y_min;
    cv::Rect rect(img_x_min, img_y_min, img_x_max - img_x_min, img_y_max - img_y_min);
    return rect;
}

bool findBlocks(std::vector<block_recognition::DetectedBlock>& detected_blocks)
{
    if (!rgbInit || !cam_info_init)
        return false;

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

    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered_xyz);
    ec.extract(cluster_indices);

    // find center of mass of each cluster
    ROS_INFO_STREAM("found " << cluster_indices.size() << " clusters..." << std::endl);
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

    ROS_INFO_STREAM("---------- detect edges ----------" << std::endl);
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

    ROS_INFO_STREAM("found " << contours.size() << " contours..." << std::endl);
    int unique_block_id = 0;

    /// Find orientation and draw contours
    for( int i = 0; i < contours.size(); i++ ) {

        if (object_centers.size() == 0) break; // TODO: possibly recognizes incorrect blocks this way

        float rectArea = minRect[i].size.width * minRect[i].size.height; // Ignore tiny contours found from noise in image (like a dent in foam)
        //if (mu[i].m00 < 500) continue; // use rectArea, moments sometimes returns too small for some reason
        if (rectArea < 600) continue;
        ROS_INFO_STREAM("clusters left: " << object_centers.size() << std::endl);
        ROS_INFO_STREAM("area " << i << ": " << mu[i].m00 << " vs rectArea: " << rectArea << std::endl);

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
            ROS_INFO_STREAM("contour " << i << " is nan!" << std::endl);
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
        ROS_INFO_STREAM("x: " << cam_pt.x() << ", y: " << cam_pt.y() << ", z: " << cam_pt.z() << std::endl);

        float sinTheta = sin(rotation * KDL::deg2rad);
        float cosTheta = cos(rotation * KDL::deg2rad);

        block_recognition::DetectedBlock detected_block;

        // Get header from cloud
        detected_block.pose_stamped.header = pcl_conversions::fromPCL(cloud->header);

        // Get rotation matrix
        tf::Matrix3x3 rot_m =  tf::Matrix3x3(
        cosTheta,-sinTheta,0,
        sinTheta,cosTheta,0,
        0,0,1);
        tf::Quaternion rot_q;
        rot_m.getRotation(rot_q);
        tf::quaternionTFToMsg(rot_q, detected_block.pose_stamped.pose.orientation);

        //Apply position information
        detected_block.pose_stamped.pose.position.x = cam_pt.x();
        detected_block.pose_stamped.pose.position.y = cam_pt.y();
        detected_block.pose_stamped.pose.position.z = cam_pt.z();

        if (rectArea > 7000) {
            detected_block.mesh_filename = block_user_data_large;
            detected_block.edge_length = 0.076;
        } else if (rectArea < 4500) {
            detected_block.mesh_filename = block_user_data_small;
            detected_block.edge_length = 0.050;
        } else {
            detected_block.mesh_filename = block_user_data_medium;
            detected_block.edge_length = 0.064;
        }

        // Camera to center of box. Because Moveit does not add boxes with the TF at the base of the box
        KDL::Frame pose;
        tf::poseMsgToKDL(detected_block.pose_stamped.pose, pose);
        KDL::Frame rot = KDL::Frame(KDL::Rotation::RotX(KDL::PI), KDL::Vector(0,0,detected_block.edge_length/2.0));
        KDL::Frame newPose = pose * rot;
        tf::poseKDLToMsg(newPose, detected_block.pose_stamped.pose);
//        tf::poseKDLToMsg(pose, detected_block.pose_stamped.pose);

        detected_block.unique_block_name = std::string("block") + std::to_string(unique_block_id);
        detected_block.unique_id = unique_block_id++;

        detected_blocks.push_back(detected_block);

        object_centers.erase(object_centers.begin() + min_idx); // make sure not to reuse the same cluster again
    }

    return true;
}


void publish_detected_blocks_as_marker(const std::vector<block_recognition::DetectedBlock> detected_blocks)
{
    visualization_msgs::MarkerArray marker_array;

    for(auto detected_block : detected_blocks) {
        visualization_msgs::Marker marker;

        marker.header = detected_block.pose_stamped.header;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(20.0);
        marker.ns = "objrec";

        // Add scaling factor based on the size of the cube
        marker.scale.x = detected_block.edge_length;
        marker.scale.y = detected_block.edge_length;
        marker.scale.z = detected_block.edge_length;

        marker.color.a = 0.75;
        marker.color.r = 1.0;
        marker.color.g = 0.1;
        marker.color.b = 0.3;

        marker.id = detected_block.unique_id;
        marker.pose = detected_block.pose_stamped.pose;

        marker_array.markers.push_back(marker);
    }

    // Publish the markers
    markers_pub_.publish(marker_array);
}


bool recognizeBlocks(block_recognition::FindObjects::Request &req, block_recognition::FindObjects::Response &res)
{
    // No objects recognized
    if (!findBlocks(res.detected_blocks)) {
        return false;
    }

    ROS_INFO_STREAM("publishing " << res.detected_blocks.size() << "models..." << pcl_conversions::fromPCL(cloud->header).stamp << std::endl);
    publish_detected_blocks_as_marker(res.detected_blocks);

    ROS_INFO("sending back response ");
    return true;
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

    markers_pub_ = n->advertise<visualization_msgs::MarkerArray>("recognized_objects_markers",20);

    ros::ServiceServer find_blocks_server_ = n->advertiseService("find_blocks", recognizeBlocks);

    blocks_pc_pub = n->advertise<sensor_msgs::PointCloud2>("filtered_blocks", 1);

    ros::spin();
    return 0;
}
