
#ifndef CONVEX_OPTIMIZE_FITTING_H
#define CONVEX_OPTIMIZE_FITTING_H

#include <ros/ros.h>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#define EIGEN_MPL2_ONLY

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "autoware_msgs/DetectedObject.h"
#include "autoware_msgs/DetectedObjectArray.h"

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/PolygonArray.h>

#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <chrono>
#include <std_msgs/Float32.h>
#include<algorithm>


#define __APP_NAME__ "RANSAC L-shape Fitting"

static ros::Publisher time_ransacLshape_pub;

static std::chrono::time_point<std::chrono::system_clock> ransacLshape_start, ransacLshape_end;
static std_msgs::Float32 time_ransacLshape;
static double exe_time = 0.0;

class ConvexOptimizeFitting
{
private:

  tf2_ros::TransformListener tf2_listener;
  tf2_ros::Buffer tf2_buffer;

  std::string input_cluster_topic_;
  std::string output_bbox_topic_;
  std::string corner_point_topic_;

  std::string bbox_source_frame_;
  std::string bbox_target_frame_;

  ros::NodeHandle node_handle_;
  ros::Subscriber sub_object_array_;
  ros::Publisher pub_object_array_;
  ros::Publisher pub_autoware_bboxs_array_;
  ros::Publisher pub_jsk_bboxs_array_;
  ros::Publisher pub_corner_point_;
  ros::Publisher pub_convex_corner_points_;

  ros::Publisher pub_left_side_point_;
  ros::Publisher pub_right_side_point_;
  ros::Publisher pub_ransac_line_left_;
  ros::Publisher pub_ransac_line_right_;
  ros::Publisher pub_rec_corner_points_;
  ros::Publisher pub_local_obstacle_info_;
  ros::Publisher pub_cluster_polygon_;

  void MainLoop(const autoware_msgs::DetectedObjectArray& in_cluster_array);

  int index_decrease(const int &index, const int &size);

  jsk_recognition_msgs::BoundingBox jsk_bbox_transform(const autoware_msgs::DetectedObject &autoware_bbox, 
          const std_msgs::Header& header);
  void calculateDimPos(double &theta_star, autoware_msgs::DetectedObject &output, std::vector<cv::Point2f>& rec_corner_points, const pcl::PointCloud<pcl::PointXYZ> &cluster);
  
  void calcuBoxbyPolygon(double &theta_star, autoware_msgs::DetectedObject &output, std::vector<cv::Point2f>& rec_corner_points, 
                          const std::vector<cv::Point2f> &polygon_cluster, const pcl::PointCloud<pcl::PointXYZ> &cluster);
  void optim_convex_fitting(const pcl::PointCloud<pcl::PointXYZ> &current_cluster, 
              std::vector<cv::Point2f> &hull_cluster, pcl::PointCloud<pcl::PointXYZI>::Ptr &convex_corner_visual, double &theta_optim);

  void calcuProjLinebyConvex(double &theta_star, const std::vector<cv::Point2f> &hull_cluster, pcl::PointXYZI &convex_corner_point,
                             cv::Point2f &rec_corner_near, double &rec_minl_a, double &rec_minl_b, double &rec_minl_c);
  
public:
  ConvexOptimizeFitting();
};

#endif  // CONVEX_OPTIMIZE_FITTING_H