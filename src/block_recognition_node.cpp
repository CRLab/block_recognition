#include <block_recognition_node.h>


static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 )
{
  double dx1 = pt1.x - pt0.x;
  double dy1 = pt1.y - pt0.y;
  double dx2 = pt2.x - pt0.x;
  double dy2 = pt2.y - pt0.y;
  return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


static void findSquares( int threshold, int N, const cv::Mat& image, std::vector<std::vector<cv::Point> >& squares )
{
  squares.clear();

//s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

  // down-scale and upscale the image to filter out the noise
  //pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
  //pyrUp(pyr, timg, image.size());


  // blur will enhance edge detection
  cv::Mat timg(image);
  medianBlur(image, timg, 9);
  cv::Mat gray0(timg.size(), CV_8U), gray;

  std::vector<std::vector<cv::Point> > contours;

  // find squares in every color plane of the image
  for( int c = 0; c < 3; c++ )
  {
    int ch[] = {c, 0};
    cv::mixChannels(&timg, 1, &gray0, 1, ch, 1);

    // try several threshold levels
    for( int l = 0; l < N; l++ )
    {
      // hack: use Canny instead of zero threshold level.
      // Canny helps to catch squares with gradient shading
      if( l == 0 )
      {
        // apply Canny. Take the upper threshold from slider
        // and set the lower to 0 (which forces edges merging)
        Canny(gray0, gray, 5, threshold, 5);
        // dilate canny output to remove potential
        // holes between edge segments
        dilate(gray, gray, cv::Mat(), cv::Point(-1,-1));
      }
      else
      {
        // apply threshold if l!=0:
        //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
        gray = gray0 >= (l+1)*255/N;
      }

      // find contours and store them all as a list
      findContours(gray, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

      std::vector<cv::Point> approx;

      // test each contour
      for( size_t i = 0; i < contours.size(); i++ )
      {
        // approximate contour with accuracy proportional
        // to the contour perimeter
        cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);

        // square contours should have 4 vertices after approximation
        // relatively large area (to filter out noisy contours)
        // and be convex.
        // Note: absolute value of an area is used because
        // area may be positive or negative - in accordance with the
        // contour orientation
        if( approx.size() == 4 &&
            fabs(cv::contourArea(cv::Mat(approx))) > 1000 &&
            cv::isContourConvex(cv::Mat(approx)) )
        {
          double maxCosine = 0;

          for( int j = 2; j < 5; j++ )
          {
            // find the maximum cosine of the angle between joint edges
            double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
            maxCosine = MAX(maxCosine, cosine);
          }

          // if cosines of all angles are small
          // (all angles are ~90 degree) then write quandrange
          // vertices to resultant sequence
          if( maxCosine < 0.3 )
            squares.push_back(approx);
        }
      }
    }
  }
}

void BlockRecognitionNode::saveRGBImage(const sensor_msgs::ImageConstPtr& msg) {
  try {
    rgbImage_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    rgbInit_ = true;
  } catch(cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("Could not convert from '" << msg->encoding.c_str() << "' to 'bgr8'.");
  }
}

void BlockRecognitionNode::saveCamInfo(const sensor_msgs::CameraInfoConstPtr& msg)
{
  try {
    cameraInfoMsg_ = msg;
    cameraInfoInit_ = true;
  } catch(cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("Could not load cam_info.");
  }
}

void BlockRecognitionNode::savePointCloud(const sensor_msgs::PointCloud2ConstPtr &points_msg)
{
  // Lock the buffer mutex while we're capturing a new point cloud
  boost::mutex::scoped_lock buffer_lock(bufferMutex_);

  // Convert to PCL cloud
  pcl::fromROSMsg(*points_msg, cloudInCameraFrame_);
}

cv::Rect BlockRecognitionNode::getImageCroppingRect(tf::StampedTransform &transformWorldToCamera)
{
  ROS_INFO_STREAM("World cropping bounds: (" << xClipMin_ << ", " << yClipMax_ << ", " << zClipMin_ << ") to ("
                                             << xClipMax_ << ", " << yClipMax_ << ", " << zClipMax_ << ")");

  image_geometry::PinholeCameraModel cameraModel;
  cameraModel.fromCameraInfo(cameraInfoMsg_);

  // Minimum point transform
  tf::Vector3 worldMinimumVector3(xClipMin_, yClipMin_, zClipMin_);
  tf::Vector3 cameraMinimumVector3 = transformWorldToCamera * worldMinimumVector3;
  cv::Point3d cameraMinimumPoint3d(cameraMinimumVector3.x(), cameraMinimumVector3.y(), cameraMinimumVector3.z());
  cv::Point2d pixelMinimumPoint = cameraModel.project3dToPixel(cameraMinimumPoint3d);

  // Maximum point transform
  tf::Vector3 worldMaximumVector3(xClipMin_, yClipMin_, zClipMin_);
  tf::Vector3 cameraMaximumVector3 = transformWorldToCamera * worldMaximumVector3;
  cv::Point3d cameraMaximumPoint3d(cameraMaximumVector3.x(), cameraMaximumVector3.y(), cameraMaximumVector3.z());
  cv::Point2d pixelMaximumPoint = cameraModel.project3dToPixel(cameraMaximumPoint3d);

  // 2D Image cropping bounds
  int imgXMin = std::max(0, (int) std::min(pixelMinimumPoint.x, pixelMaximumPoint.x));
  int imgYMin = std::max(0, (int) std::min(pixelMinimumPoint.y, pixelMaximumPoint.y));
  int imgXMax = std::min(cameraInfoMsg_->width, (unsigned int) std::max(pixelMinimumPoint.x, pixelMaximumPoint.x));
  int imgYMax = std::min(cameraInfoMsg_->height, (unsigned int) std::max(pixelMinimumPoint.y, pixelMaximumPoint.y));

  ROS_INFO_STREAM("Cropping image bounds: (" << imgXMin << ", " << imgYMin << ") to ("
                                             << imgXMax << ", " << imgYMax << ")");

  return cv::Rect(imgXMin, imgYMin, imgXMax - imgXMin, imgYMax - imgYMin);
}

//static void filterPointCloud(PointCloud::Ptr inputPointCloud, float xMin, float yMin, float zMin, float xMax, float yMax, float zMax) {
//  pcl::PassThrough<pcl::PointXYZ> passThrough;
//  passThrough.setInputCloud(inputPointCloud);
//  passThrough.setFilterFieldName("x");
//  passThrough.setFilterLimits(xMin, xMax);
//  passThrough.filter(*inputPointCloud);
//  passThrough.setFilterFieldName("y");
//  passThrough.setFilterLimits(yMin, yMax);
//  passThrough.filter(*inputPointCloud);
//  passThrough.setFilterFieldName("z");
//  passThrough.setFilterLimits(zMin, zMax);
//  passThrough.filter(*inputPointCloud);
//}

void BlockRecognitionNode::captureEuclidianClusters(std::vector<pcl::PointXYZ> &objectCenters, pcl::PointCloud<pcl::PointXYZRGB> &filteredPointCloud) {
  // euclidean cluster extraction
  // create KdTree object for search method of extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(&filteredPointCloud);
  tree->setInputCloud(tmp);

  std::vector<pcl::PointIndices> clusterIndices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(0.02); // 2cm
  ec.setMinClusterSize(100);

  ec.setSearchMethod(tree);
  ec.setInputCloud(tmp);
  ec.extract(clusterIndices);

  // find center of mass of each cluster
  ROS_INFO_STREAM("Found " << clusterIndices.size() << " clusters" << std::endl);
  ;
  for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin(); it != clusterIndices.end(); ++it)
  {
    float centerX = 0.0, centerY = 0.0, centerZ = 0.0;
    float numP = 0.0;

    for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {

      centerX += filteredPointCloud.points[*pit].x;
      centerY += filteredPointCloud.points[*pit].y;
      centerZ += filteredPointCloud.points[*pit].z;

      numP++;
    }
    centerX = centerX / numP;
    centerY = centerY / numP;
    centerZ = centerZ / numP;

    pcl::PointXYZ pt_xyz(centerX, centerY, centerZ);

    objectCenters.push_back(pt_xyz);
  }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr BlockRecognitionNode::filterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, cv::Rect imageCroppingRectangle) {
  // Crop the point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr worldPointCloudFilteredXY(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr worldPointCloudFilteredXYZ(new pcl::PointCloud<pcl::PointXYZRGB>());

  // crop point cloud based on 2d indices corresponding to rgb image
  ROS_INFO("Initialize index filter");
  pcl::ExtractIndices<pcl::PointXYZRGB > xyFilter;
  xyFilter.setInputCloud(pointCloud);
  pcl::IndicesPtr indices(new std::vector<int>);

  ROS_INFO("Crop the point cloud based on 2d indices");

  int row_start = imageCroppingRectangle.tl().y * cameraInfoMsg_->width;
  int row_end = imageCroppingRectangle.br().y * cameraInfoMsg_->width;
  int col_start = imageCroppingRectangle.tl().x;
  int col_end = imageCroppingRectangle.br().x;
  for (int i = row_start; i < row_end; i += cameraInfoMsg_->width) { //2073600
    for (int j = col_start; j < col_end; j++) {
      indices->push_back(i + j);
    }
  }
  xyFilter.setIndices(indices);
  xyFilter.filter(*worldPointCloudFilteredXY);

  // Filter on z bounds
  ROS_INFO("Filter on Z");
  pcl::PassThrough<pcl::PointXYZRGB> zFilter;
  zFilter.setInputCloud(worldPointCloudFilteredXY);
  zFilter.setFilterFieldName ("z");
  zFilter.setFilterLimits (zClipMin_, zClipMax_);
  zFilter.filter(*worldPointCloudFilteredXYZ);

  ROS_INFO("Finished filtering");

  return worldPointCloudFilteredXYZ;
}

bool BlockRecognitionNode::findBlocks(std::vector<block_recognition::DetectedBlock>& detected_blocks)
{
  if (!rgbInit_ || !cameraInfoInit_)
    return false;

  ROS_INFO("Collecting transforms");
  tf::StampedTransform transformCameraToWorld;
  tf::StampedTransform transformWorldToCamera;
  ros::Time now = ros::Time::now();
  tfListener_.waitForTransform (worldFrame_, cameraFrame_, now, ros::Duration(4.0));
  tfListener_.lookupTransform (worldFrame_, cameraFrame_, now, transformCameraToWorld);
  tfListener_.lookupTransform (cameraFrame_, worldFrame_, now, transformWorldToCamera);
  ROS_INFO("Collected transforms");

  // Get image cropping rectangle
  cv::Rect imageCroppingRectangle = this->getImageCroppingRect(transformWorldToCamera);

  // Lock the buffer mutex
  ROS_INFO("Locking buffer");
  boost::mutex::scoped_lock buffer_lock(bufferMutex_);

  // Transform camera point cloud to world frame
  ROS_INFO("Converting camera point cloud to world frame");
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr worldPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  cloudInCameraFrame_.header.frame_id = cameraFrame_;
  pcl_ros::transformPointCloud(worldFrame_, cloudInCameraFrame_, *worldPointCloud, tfListener_);

  ROS_INFO("Filtering point cloud using imageCroppingRectangle");
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr worldPointCloudFilteredXYZ = this->filterPointCloud(worldPointCloud, imageCroppingRectangle);

  // Publish world point cloud for debugging
  ROS_INFO("Publishing filtered point cloud");
  worldPointCloudFilteredXYZ->header.frame_id = worldFrame_;
  publishBlockPointCloud(worldPointCloudFilteredXYZ);

  // Find all the centers of blocks in the point cloud
  std::vector<pcl::PointXYZ> objectCenters;
  captureEuclidianClusters(objectCenters, *worldPointCloudFilteredXYZ);

  /**************************************/
  /* find orientation from 2d RGB image */
  // crop the 2d image based on x and y clipping values

  cv::Mat croppedImg = rgbImage_(imageCroppingRectangle);

  cv::Mat src_gray, src_binary;
  cvtColor(croppedImg, src_gray, CV_BGR2GRAY);

  double threshold_value = 125;
  int threshold_type = 1; //binary inverted
  double max_BINARY_value = 255;

  threshold( src_gray, src_binary, threshold_value, max_BINARY_value, threshold_type);

//    blur(src_gray, src_gray, Size(3,3));

  cv::RNG rng(12345);
  cv::Mat canny_output;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  ROS_INFO("Detecting edges");
  // Detect edges using canny
  int thresh = 90;
  Canny( src_binary, canny_output, thresh, thresh*3, 3 );

  // Find contours
  findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

  std::vector<cv::RotatedRect> minRect( contours.size() );
  std::vector<cv::Moments> mu( contours.size() );
  std::vector<cv::Point2f> mc( contours.size() );
  for( int i = 0; i < contours.size(); i++ ) {
    /// Get the moments
    mu[i] = cv::moments( contours[i], true );

    /// Get the rotated rectangles
    minRect[i] = cv::minAreaRect( cv::Mat(contours[i]) );

    ///  Get the mass centers
    mc[i] = cv::Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
  }

  ROS_INFO_STREAM("found " << contours.size() << " contours..." << std::endl);
  int unique_block_id = 0;

  /// Find orientation and draw contours
  for( int i = 0; i < contours.size(); i++ ) {

    if (objectCenters.size() == 0) break; // TODO: possibly recognizes incorrect blocks this way

    float rectArea = minRect[i].size.width * minRect[i].size.height; // Ignore tiny contours found from noise in image (like a dent in foam)
    //if (mu[i].m00 < 500) continue; // use rectArea, moments sometimes returns too small for some reason
    if (rectArea < 600) continue;
    ROS_INFO_STREAM("clusters left: " << objectCenters.size() << std::endl);
    ROS_INFO_STREAM("area " << i << ": " << mu[i].m00 << " vs rectArea: " << rectArea << std::endl);

    cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

    cv::drawContours( croppedImg, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
    cv::circle( croppedImg, mc[i], 4, color, -1, 8, 0 );

    // rotated rectangle
    cv::Point2f rect_points[4]; minRect[i].points( rect_points );
    for( int j = 0; j < 4; j++ )
      line( croppedImg, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );

    float rotation = minRect[i].angle;

    // index into point cloud to get position of center of mass
    // TODO: figure out why sometimes mc is Nan?
    if (std::isnan(mc[i].x) || std::isnan(mc[i].y)) {
      ROS_INFO_STREAM("contour " << i << " is nan!" << std::endl);
      continue;
    }
    int px = imageCroppingRectangle.tl().x + (int) mc[i].x;
    int py = imageCroppingRectangle.tl().y + (int) mc[i].y;

    // get corresponding point cloud point
    pcl::PointXYZRGB pt = (*worldPointCloud)[py * cameraInfoMsg_->width + px];
    if (std::isnan(pt.x) || std::isnan(pt.y)) {
      pt = (*worldPointCloud)[py * cameraInfoMsg_->width + px + 5]; // TODO: fix hacky way to deal with nan pc values
    }

    // transform point into camera frame to publish
    double min_dist = std::numeric_limits<double>::max();
    int min_idx = 0;
    // hacky? way to match point cloud cluster center to rgb img center by comparing distances
    for (int j = 0; j < objectCenters.size(); j++) {
      double dist = (objectCenters[j].x - pt.x) * (objectCenters[j].x - pt.x);
      dist += (objectCenters[j].y - pt.y) * (objectCenters[j].y - pt.y);
      dist += (objectCenters[j].z - pt.z) * (objectCenters[j].z - pt.z);

      if (dist < min_dist) {
        min_dist = dist;
        min_idx = j;
      }
    }
    tf::Vector3 blockCenterInCameraFrame;
    blockCenterInCameraFrame.setX(objectCenters[min_idx].x);
    blockCenterInCameraFrame.setY(objectCenters[min_idx].y);
    blockCenterInCameraFrame.setZ(objectCenters[min_idx].z);
    blockCenterInCameraFrame = transformWorldToCamera * blockCenterInCameraFrame; // transform into camera frame

    double sinTheta = sin(rotation * KDL::deg2rad);
    double cosTheta = cos(rotation * KDL::deg2rad);

    block_recognition::DetectedBlock detected_block;

    // Get header from cloud
    detected_block.pose_stamped.header = pcl_conversions::fromPCL(cloudInCameraFrame_.header);

    // Get rotation matrix
    tf::Matrix3x3 rot_m =  tf::Matrix3x3(
        cosTheta,-sinTheta,0,
        sinTheta,cosTheta,0,
        0,0,1);
    tf::Quaternion rot_q;
    rot_m.getRotation(rot_q);
    tf::quaternionTFToMsg(rot_q, detected_block.pose_stamped.pose.orientation);

    //Apply position information
    detected_block.pose_stamped.pose.position.x = blockCenterInCameraFrame.x();
    detected_block.pose_stamped.pose.position.y = blockCenterInCameraFrame.y();
    detected_block.pose_stamped.pose.position.z = blockCenterInCameraFrame.z();

    if (rectArea > 7000) {
      detected_block.mesh_filename = BLOCK_LARGE_FILENAME;
      detected_block.edge_length = 0.076;
    } else if (rectArea < 4500) {
      detected_block.mesh_filename = BLOCK_SMALL_FILENAME;
      detected_block.edge_length = 0.050;
    } else {
      detected_block.mesh_filename = BLOCK_MEDIUM_FILENAME;
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

    objectCenters.erase(objectCenters.begin() + min_idx); // make sure not to reuse the same cluster again
  }

  return true;
}

void BlockRecognitionNode::publishBlockPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr blockPointCloud) {
  pcl::PCLPointCloud2 blockPointCloud2;
  pcl::toPCLPointCloud2(*blockPointCloud, blockPointCloud2);

  sensor_msgs::PointCloud2 blockPointCloudMsg;
  pcl_conversions::fromPCL(blockPointCloud2, blockPointCloudMsg);

  blockPointcloudPublisher_.publish(blockPointCloudMsg);
}

void BlockRecognitionNode::publishDetectedBlocksAsMarkers(const std::vector<block_recognition::DetectedBlock> detected_blocks)
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
  markersPublisher_.publish(marker_array);
}


bool BlockRecognitionNode::recognizeBlocks(block_recognition::FindObjects::Request &req, block_recognition::FindObjects::Response &res)
{
  // No objects recognized
  if (!findBlocks(res.detected_blocks)) {
    return false;
  }

  ROS_INFO_STREAM("Publishing " << res.detected_blocks.size() << "models..." << pcl_conversions::fromPCL(cloudInCameraFrame_.header).stamp);
  publishDetectedBlocksAsMarkers(res.detected_blocks);

  ROS_INFO("Sending back response ");
  return true;
}

BlockRecognitionNode::BlockRecognitionNode() {
  n_ = std::shared_ptr<ros::NodeHandle>(new ros::NodeHandle());

  rgbInit_ = false;
  cameraInfoInit_ = false;

  n_->getParam("x_clip_min", xClipMin_);
  n_->getParam("x_clip_max", xClipMax_);
  n_->getParam("y_clip_min", yClipMin_);
  n_->getParam("y_clip_max", yClipMax_);
  n_->getParam("z_clip_min", zClipMin_);
  n_->getParam("z_clip_max", zClipMax_);

  n_->getParam("world_frame", worldFrame_);
  n_->getParam("camera_frame", cameraFrame_);
  n_->getParam("camera_info_topic", cameraInfoTopic_);
  n_->getParam("image_color_topic", imageColorTopic_);
  n_->getParam("point_cloud_topic", pointCloudTopic_);

  cameraPointCloudSubscriber_ = n_->subscribe(pointCloudTopic_, 1, &BlockRecognitionNode::savePointCloud, this);

  image_transport::ImageTransport imageTransport(*n_);
  rgbImageSubscriber_ = imageTransport.subscribe(imageColorTopic_, 1, &BlockRecognitionNode::saveRGBImage, this);

  cameraInfoSubscriber_ = n_->subscribe(cameraInfoTopic_, 1, &BlockRecognitionNode::saveCamInfo, this);

  markersPublisher_ = n_->advertise<visualization_msgs::MarkerArray>("recognized_objects_markers", 20);

  findBlocksService_ = n_->advertiseService("find_blocks", &BlockRecognitionNode::recognizeBlocks, this);

  blockPointcloudPublisher_ = n_->advertise<sensor_msgs::PointCloud2>("filtered_blocks", 1);

  ROS_INFO("READY!");
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "block_recognition_node");

  std::shared_ptr<BlockRecognitionNode> node(new BlockRecognitionNode());

  ros::spin();
  return 0;
}
