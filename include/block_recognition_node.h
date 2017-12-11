//
// Created by david on 12/7/17.
//

#ifndef PROJECT_BLOCK_RECOGNITION_NODE_H
#define PROJECT_BLOCK_RECOGNITION_NODE_H

#include "ros/ros.h"
#include <ros/package.h>

#include <tf_conversions/tf_kdl.h>
#include <tf/transform_listener.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>


#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <image_geometry/pinhole_camera_model.h>

#include <boost/foreach.hpp>

#include <block_recognition/DetectedBlock.h>
#include <block_recognition/FindObjects.h>

#include <string>

static std::string BLOCK_SMALL_FILENAME = "block_50";
static std::string BLOCK_MEDIUM_FILENAME = "block_64";
static std::string BLOCK_LARGE_FILENAME = "block_76";

/*!
 * finds a cosine of angle between vectors
 * from pt0->pt1 and from pt0->pt2
 * @param pt1
 * @param pt2
 * @param pt0
 * @return Angle between vectors
 */
static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 );

/*!
 * returns sequence of squares detected on the image.
 * the sequence is stored in the specified memory storage
 * @param threshold
 * @param image
 * @param squares
 */
static void findSquares( int threshold, cv::Mat& image, std::vector<std::vector<cv::Point> >& squares );

class BlockRecognitionNode {
 private:
  tf::TransformListener tfListener_;
  ros::Publisher markersPublisher_;
  ros::Publisher blockPointcloudPublisher_;

  ros::Subscriber cameraPointCloudSubscriber_;
  image_transport::Subscriber rgbImageSubscriber_;
  ros::Subscriber cameraInfoSubscriber_;
  ros::ServiceServer findBlocksService_;

  std::shared_ptr<ros::NodeHandle> n_;

  sensor_msgs::CameraInfoConstPtr cameraInfoMsg_;
  bool cameraInfoInit_;
  cv::Mat rgbImage_;
  bool rgbInit_;

  boost::mutex bufferMutex_;
  pcl::PointCloud<pcl::PointXYZRGB> cloudInCameraFrame_;

  float xClipMin_, yClipMin_, zClipMin_;
  float xClipMax_, yClipMax_, zClipMax_;

  std::string worldFrame_, cameraFrame_;
  std::string cameraInfoTopic_, imageColorTopic_, pointCloudTopic_;

 public:
  BlockRecognitionNode();

  void saveRGBImage(const sensor_msgs::ImageConstPtr& msg);

  void saveCamInfo(const sensor_msgs::CameraInfoConstPtr& msg);

  void savePointCloud(const sensor_msgs::PointCloud2ConstPtr &points_msg);

  cv::Rect getImageCroppingRect(tf::StampedTransform &transformWorldToCamera);

  bool findBlocks(std::vector<block_recognition::DetectedBlock>& detected_blocks);

  void publishBlockPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr blockPointCloud);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, cv::Rect imageCroppingRectangle);

  void captureEuclidianClusters(std::vector<pcl::PointXYZ> &objectCenters, pcl::PointCloud<pcl::PointXYZRGB> &filteredPointCloud);

  bool recognizeBlocks(block_recognition::FindObjects::Request &req, block_recognition::FindObjects::Response &res);

  void publishDetectedBlocksAsMarkers(const std::vector<block_recognition::DetectedBlock> detected_blocks);
};


#endif //PROJECT_BLOCK_RECOGNITION_NODE_H
