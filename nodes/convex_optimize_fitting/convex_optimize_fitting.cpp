#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_datatypes.h>
#include <chrono>
#include "convex_optimize_fitting.h"

ConvexOptimizeFitting::ConvexOptimizeFitting() : tf2_listener(tf2_buffer)
{
  ros::NodeHandle private_nh_("~");

  private_nh_.param<std::string>("input_cluster_topic", input_cluster_topic_, "/travel/autoware_objects");
  ROS_INFO("Input_cluster_topic: %s", input_cluster_topic_.c_str());

  node_handle_.param<std::string>("output_bbox_topic_", output_bbox_topic_, "/convex_optim_fitting/bbox_visual_jsk");
  ROS_INFO("output_bbox_topic_: %s", output_bbox_topic_.c_str());

  private_nh_.param<std::string>("bbox_target_frame", bbox_target_frame_, "base_link");
  ROS_INFO("[%s] bounding box's target frame is: %s", __APP_NAME__, bbox_target_frame_);

  sub_object_array_ = node_handle_.subscribe(input_cluster_topic_, 1, &ConvexOptimizeFitting::MainLoop, this);

  pub_autoware_bboxs_array_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>("/convex_optim_fitting/autoware_bboxs_array", 1);
  pub_jsk_bboxs_array_ = node_handle_.advertise<jsk_recognition_msgs::BoundingBoxArray>("/convex_optim_fitting/jsk_bboxs_array",1);

  pub_rec_corner_points_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/convex_optim_fitting/rec_corner_points", 1);

  pub_convex_corner_points_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/convex_optim_fitting/convex_corner_points", 1);

  pub_cluster_polygon_ = node_handle_.advertise<jsk_recognition_msgs::PolygonArray>("/convex_optim_fitting/cluster_polygon", 2);

  time_ransacLshape_pub = node_handle_.advertise<std_msgs::Float32>("/time_convex_fitting", 1);
}


void ConvexOptimizeFitting::calcuProjLinebyConvex(double &theta_star, const std::vector<cv::Point2f> &hull_cluster, pcl::PointXYZI &convex_corner_point,
                                                    cv::Point2f &rec_corner_near, double &rec_minl_a, double &rec_minl_b, double &rec_minl_c)
{
  Eigen::Vector2d e_1_star;  
  Eigen::Vector2d e_2_star;
  e_1_star << std::cos(theta_star), std::sin(theta_star);
  e_2_star << -std::sin(theta_star), std::cos(theta_star);
  std::vector<double> C_1_star;
  std::vector<double> C_2_star;

  double rec_a[4] = {0};
  double rec_b[4] = {0};
  double rec_c[4] = {0};

  for (int i = 0; i < hull_cluster.size(); i++)
  {
    C_1_star.push_back(hull_cluster[i].x * e_1_star.x() + hull_cluster[i].y * e_1_star.y());
    C_2_star.push_back(hull_cluster[i].x * e_2_star.x() + hull_cluster[i].y * e_2_star.y());
  }

  rec_a[0] = std::cos(theta_star);
  rec_b[0] = std::sin(theta_star);
  rec_c[0] = *std::min_element(C_1_star.begin(), C_1_star.end());

  rec_a[1] = -1.0 * std::sin(theta_star);
  rec_b[1] = std::cos(theta_star);
  rec_c[1] = *std::min_element(C_2_star.begin(), C_2_star.end());

  rec_a[2] = std::cos(theta_star);
  rec_b[2] = std::sin(theta_star);
  rec_c[2] = *std::max_element(C_1_star.begin(), C_1_star.end());

  rec_a[3] = -1.0 * std::sin(theta_star);
  rec_b[3] = std::cos(theta_star);
  rec_c[3] = *std::max_element(C_2_star.begin(), C_2_star.end());

  cv::Point2f rec_corner_p1, rec_corner_p2, rec_corner_p3, rec_corner_p4;
  std::vector<cv::Point2f> rec_corner_points;

  double a_1 = rec_a[0];
  double b_1 = rec_b[0];
  double c_1 = rec_c[0];

  double a_2 = rec_a[1];
  double b_2 = rec_b[1];
  double c_2 = rec_c[1];

  double a_3 = rec_a[2];
  double b_3 = rec_b[2];
  double c_3 = rec_c[2];

  double a_4 = rec_a[3];
  double b_4 = rec_b[3];
  double c_4 = rec_c[3];

  rec_corner_p1.x = (b_1 * c_2 - b_2 * c_1) / (a_2 * b_1 - a_1 * b_2);
  rec_corner_p1.y = (a_1 * c_2 - a_2 * c_1) / (a_1 * b_2 - a_2 * b_1);

  rec_corner_p2.x = (b_2 * c_3 - b_3 * c_2) / (a_3 * b_2 - a_2 * b_3);
  rec_corner_p2.y = (a_2 * c_3 - a_3 * c_2) / (a_2 * b_3 - a_3 * b_2);

  rec_corner_p3.x = (b_3 * c_4 - b_4 * c_3) / (a_4 * b_3 - a_3 * b_4);
  rec_corner_p3.y = (a_3 * c_4 - a_4 * c_3) / (a_3 * b_4 - a_4 * b_3);

  rec_corner_p4.x = (b_1 * c_4 - b_4 * c_1) / (a_4 * b_1 - a_1 * b_4);
  rec_corner_p4.y = (a_1 * c_4 - a_4 * c_1) / (a_1 * b_4 - a_4 * b_1);

  rec_corner_points.push_back(rec_corner_p1);
  rec_corner_points.push_back(rec_corner_p2);
  rec_corner_points.push_back(rec_corner_p3);
  rec_corner_points.push_back(rec_corner_p4);

  double min_dist_reccp = std::numeric_limits<double>::max();
  int min_reccp_index;

  for (int i = 0; i < rec_corner_points.size(); ++i)
  {
    if ((rec_corner_points[i].x * rec_corner_points[i].x + rec_corner_points[i].y * rec_corner_points[i].y) < min_dist_reccp || i == 0)
    {
      min_dist_reccp = rec_corner_points[i].x * rec_corner_points[i].x + rec_corner_points[i].y * rec_corner_points[i].y;
      min_reccp_index = i;
    }
    float min_dist_reccp_debug = rec_corner_points[i].x * rec_corner_points[i].x + rec_corner_points[i].y * rec_corner_points[i].y;
  }

  int min_reccp_index_m1;
  double dist_pro_1 = pow((rec_corner_points[min_reccp_index].x - rec_corner_points[(min_reccp_index + 1) % 4].x), 2) + 
                      pow((rec_corner_points[min_reccp_index].y - rec_corner_points[(min_reccp_index + 1) % 4].y), 2);

  if (min_reccp_index - 1 < 0)
    min_reccp_index_m1 = 3;
  else
    min_reccp_index_m1 = min_reccp_index - 1;

  double dist_pro_2 = pow((rec_corner_points[min_reccp_index].x - rec_corner_points[min_reccp_index_m1].x), 2) 
                    + pow((rec_corner_points[min_reccp_index].y - rec_corner_points[min_reccp_index_m1].y), 2);

  int project_lin_index;

  if (dist_pro_1 > dist_pro_2)
  {
    project_lin_index = (min_reccp_index + 1) % 4;
  }
  else
  {
    project_lin_index = min_reccp_index;
  }

  rec_corner_near.x = rec_corner_points[min_reccp_index].x;
  rec_corner_near.y = rec_corner_points[min_reccp_index].y;

  rec_minl_a = rec_a[project_lin_index];
  rec_minl_b = rec_b[project_lin_index];
  rec_minl_c = rec_c[project_lin_index];
}

int ConvexOptimizeFitting::index_decrease(const int &index, const int &size)
{
  if (index - 1 < 0)
    return size - 1;
  else
    return index - 1;
}


jsk_recognition_msgs::BoundingBox ConvexOptimizeFitting::jsk_bbox_transform(const autoware_msgs::DetectedObject &autoware_bbox, 
          const std_msgs::Header& header)
{
  jsk_recognition_msgs::BoundingBox jsk_bbox;
  jsk_bbox.header = header;
  jsk_bbox.pose = autoware_bbox.pose;
  jsk_bbox.dimensions = autoware_bbox.dimensions;
  jsk_bbox.label = autoware_bbox.id;
  jsk_bbox.value = 1.0f;

  return std::move(jsk_bbox);
}

void ConvexOptimizeFitting::optim_convex_fitting(const pcl::PointCloud<pcl::PointXYZ> &current_cluster, 
              std::vector<cv::Point2f> &hull_cluster, pcl::PointCloud<pcl::PointXYZI>::Ptr &convex_corner_visual, double &theta_optim)
{
  std::vector<cv::Point2f> current_points;
  for (unsigned int i = 0; i < current_cluster.points.size(); i++)
  {
    cv::Point2f pt;
    pt.x = current_cluster.points[i].x;
    pt.y = current_cluster.points[i].y;
    current_points.push_back(pt);
  }

  cv::convexHull(current_points, hull_cluster);

  double distance_hull;
  double min_dist = std::numeric_limits<double>::max();
  int corner_index;

  // reorder the hull point
  // search the closest point of the cluster
  for (int i = 0; i < hull_cluster.size(); i++)
  {
    distance_hull = hull_cluster[i].x * hull_cluster[i].x + hull_cluster[i].y * hull_cluster[i].y;

    if (distance_hull < min_dist)
    {
      min_dist = distance_hull;
      corner_index = i;
    }

    if (min_dist == std::numeric_limits<double>::max())
    {
      continue;
    }
  }

  pcl::PointXYZI convex_corne_point;
  convex_corne_point.x = hull_cluster[corner_index].x;
  convex_corne_point.y = hull_cluster[corner_index].y;
  convex_corne_point.z = 0;
  convex_corne_point.intensity = min_dist;
  convex_corner_visual->push_back(convex_corne_point);

  pcl::PointXYZI nearest_line;
  double rec_minl_a, rec_minl_b, rec_minl_c;

  const double max_angle = M_PI / 2.0;

  const double angle_reso = 0.5 * M_PI / 180.0;
  cv::Point2f evaluate_pt;

  double trapezoid_h, trapezoid_upper, trapezoid_lower, trapezoid_area;
  double convex_area_l, convex_area_r, convex_area_s, break_condition;

  std::vector<std::pair<double /*theta*/, double /*q*/>> Q;
  std::vector<std::pair<double /*convex_area_l*/, double /*convex_area_r*/>> debug;

  for (double theta_search = 0; theta_search < max_angle; theta_search += angle_reso)
  {
    cv::Point2f rec_corner_near;
    calcuProjLinebyConvex(theta_search, hull_cluster, convex_corne_point, rec_corner_near, rec_minl_a, rec_minl_b, rec_minl_c);
  
    // solve error 1
    Eigen::Vector2f line_vector;
    line_vector << rec_minl_b, -rec_minl_a;

    Eigen::Vector2f corner_index_p1;
    Eigen::Vector2f corner_index_m1;
    corner_index_p1 << hull_cluster[(corner_index + 1) % hull_cluster.size()].x - hull_cluster[corner_index].x, 
                       hull_cluster[(corner_index + 1) % hull_cluster.size()].y - hull_cluster[corner_index].y;
    corner_index_m1 << hull_cluster[index_decrease(corner_index, hull_cluster.size())].x - hull_cluster[corner_index].x, 
                       hull_cluster[index_decrease(corner_index, hull_cluster.size())].y - hull_cluster[corner_index].y;

    bool invalid_area_l = false;
    bool invalid_area_r = false;

    if (corner_index_p1.dot(line_vector) * corner_index_m1.dot(line_vector) > 0)
    {
      Eigen::Vector2f vector_c2r;
      Eigen::Vector2f vector_c2l;
      Eigen::Vector2f vector_c2rc;
      vector_c2r << hull_cluster[(corner_index + 1) % hull_cluster.size()].x - hull_cluster[corner_index].x,
                     hull_cluster[(corner_index + 1) % hull_cluster.size()].y - hull_cluster[corner_index].y;
      vector_c2l << hull_cluster[index_decrease(corner_index, hull_cluster.size())].x - hull_cluster[corner_index].x,
                     hull_cluster[index_decrease(corner_index, hull_cluster.size())].y - hull_cluster[corner_index].y;
      vector_c2rc << rec_corner_near.x - hull_cluster[corner_index].x,
                     rec_corner_near.y - hull_cluster[corner_index].y;

      // ROS_INFO("vector_c2rc=%f",vector_c2rc.norm());

      float angle_r, angle_l;

      if (vector_c2rc.norm() > 0.001)
      {
        angle_r = acos(vector_c2r.dot(vector_c2rc) / (vector_c2r.norm() * vector_c2rc.norm())) * 180 / M_PI;
        angle_l = acos(vector_c2l.dot(vector_c2rc) / (vector_c2l.norm() * vector_c2rc.norm())) * 180 / M_PI;
        if (angle_r > 90){invalid_area_r = true;}
        if (angle_l > 90){invalid_area_l = true;}
      }
      else
      {
        angle_r = acos(vector_c2r.dot(line_vector) / (vector_c2r.norm() * line_vector.norm())) * 180 / M_PI;
        angle_l = acos(vector_c2l.dot(line_vector) / (vector_c2l.norm() * line_vector.norm())) * 180 / M_PI;
        if (angle_r < angle_l){invalid_area_r = true;}
        if (angle_l < angle_r){invalid_area_l = true;}
      }
    }

    convex_area_r = 0;
    int iter_num = 0;
    for (int i = corner_index; iter_num < hull_cluster.size(); i++)
    {
      if (invalid_area_r)
        break;

      Eigen::Vector2f evaluate_pt_c;
      Eigen::Vector2f evaluate_pt_n;
      Eigen::Vector2f evaluate_lineseg;


      evaluate_pt_c << hull_cluster[i % hull_cluster.size()].x, hull_cluster[i % hull_cluster.size()].y;
      evaluate_pt_n << hull_cluster[(i + 1) % hull_cluster.size()].x, hull_cluster[(i + 1) % hull_cluster.size()].y;
      evaluate_lineseg = evaluate_pt_n - evaluate_pt_c;

      trapezoid_h = evaluate_lineseg.dot(line_vector);

      if (i != corner_index && trapezoid_h * break_condition < 0)
      {
        break;
      }

      // calculate the area of trapezoid
      trapezoid_upper = abs(rec_minl_a * evaluate_pt_c(0) + rec_minl_b * evaluate_pt_c(1) - rec_minl_c) / sqrt(rec_minl_a * rec_minl_a + rec_minl_b * rec_minl_b);
      trapezoid_lower = abs(rec_minl_a * evaluate_pt_n(0) + rec_minl_b * evaluate_pt_n(1) - rec_minl_c) / sqrt(rec_minl_a * rec_minl_a + rec_minl_b * rec_minl_b);
      trapezoid_area = (trapezoid_upper + trapezoid_lower) * abs(trapezoid_h) / 2;

      convex_area_r += trapezoid_area;
      break_condition = trapezoid_h;
      iter_num++;

      // in case the index over the size of vector
      if (i > hull_cluster.size() - 1)
        i = 0;
    }

    convex_area_l = 0;
    iter_num = 0;

    for (int i = corner_index; iter_num < hull_cluster.size(); i--)
    {
      if (i < 0)
        i = hull_cluster.size() - 1;

      if (invalid_area_l)
        break;

      Eigen::Vector2f evaluate_pt_c;
      Eigen::Vector2f evaluate_pt_n;
      Eigen::Vector2f evaluate_lineseg;
      // Eigen::Vector2f line_vector;

      evaluate_pt_c << hull_cluster[i].x, hull_cluster[i].y;
      evaluate_pt_n << hull_cluster[index_decrease(i, hull_cluster.size())].x, hull_cluster[index_decrease(i, hull_cluster.size())].y;
      evaluate_lineseg = evaluate_pt_n - evaluate_pt_c;

      trapezoid_h = evaluate_lineseg.dot(line_vector);

      if (i != corner_index && trapezoid_h * break_condition < 0)
      {
        break;
      }

      // calculate the area of trapezoid
      trapezoid_upper = abs(rec_minl_a * evaluate_pt_c(0) + rec_minl_b * evaluate_pt_c(1) - rec_minl_c) / sqrt(rec_minl_a * rec_minl_a + rec_minl_b * rec_minl_b);
      trapezoid_lower = abs(rec_minl_a * evaluate_pt_n(0) + rec_minl_b * evaluate_pt_n(1) - rec_minl_c) / sqrt(rec_minl_a * rec_minl_a + rec_minl_b * rec_minl_b);
      trapezoid_area = (trapezoid_upper + trapezoid_lower) * abs(trapezoid_h) / 2;
      
      convex_area_l += trapezoid_area;
      break_condition = trapezoid_h;
      iter_num++;
    }

    convex_area_s = convex_area_l + convex_area_r;

    Q.push_back(std::make_pair(theta_search, convex_area_s));

    debug.push_back(std::make_pair(convex_area_l, convex_area_r));
  }

  double min_q;
  double convex_area_l_debug, convex_area_r_debug;

  for (size_t i = 0; i < Q.size(); ++i)
  {
    if (Q.at(i).second < min_q  || i == 0)
    {
      min_q = Q.at(i).second;
      theta_optim = Q.at(i).first;

      convex_area_l_debug = debug.at(i).first;
      convex_area_r_debug = debug.at(i).second;
    }
  }

  // ROS_INFO("DEBUG_trapezoid_area=%f, %f,%f", theta_optim, convex_area_l_debug, convex_area_r_debug);
  // std::cout << "--------------------------------" << std::endl;
}


void ConvexOptimizeFitting::MainLoop(const autoware_msgs::DetectedObjectArray& in_cluster_array)
{
  ransacLshape_start = std::chrono::system_clock::now();  

  autoware_msgs::DetectedObjectArray out_object_array;
  
  pcl::PointCloud<pcl::PointXYZI>::Ptr corner_points_visual(new pcl::PointCloud<pcl::PointXYZI>());

  pcl::PointCloud<pcl::PointXYZI>::Ptr convex_corner_visual(new pcl::PointCloud<pcl::PointXYZI>());

  out_object_array.header = in_cluster_array.header;

  int intensity_mark = 1;

  jsk_recognition_msgs::BoundingBox jsk_bbox;
  jsk_recognition_msgs::BoundingBoxArray jsk_bbox_array;

/*----------------------------------transform the bounding box to target frame.-------------------------------------------*/
  geometry_msgs::TransformStamped transform_stamped;
  geometry_msgs::Pose pose, pose_transformed;
  auto bbox_header = in_cluster_array.header;
  bbox_source_frame_ = bbox_header.frame_id;
  bbox_header.frame_id = bbox_target_frame_;
  jsk_bbox_array.header = bbox_header;

  try
  {
    transform_stamped = tf2_buffer.lookupTransform(bbox_target_frame_, bbox_source_frame_, ros::Time());
    // ROS_INFO("target_frame is %s",bbox_target_frame_.c_str());
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    ROS_WARN("Frame Transform Given Up! Outputing obstacles in the original LiDAR frame %s instead...", bbox_source_frame_.c_str());
    bbox_header.frame_id = bbox_source_frame_;
    try
    {
      transform_stamped = tf2_buffer.lookupTransform(bbox_source_frame_, bbox_source_frame_, ros::Time(0));
    }
    catch (tf2::TransformException& ex2)
    {
      ROS_ERROR("%s", ex2.what());
      return;
    }
  }

/*-----------------------------------------------------------------------------------------------------------------------*/
  jsk_recognition_msgs::PolygonArray clusters_polygon_;

  for (const auto& in_object : in_cluster_array.objects)
  {
    pcl::PointCloud<pcl::PointXYZ> current_cluster;
    pcl::fromROSMsg(in_object.pointcloud, current_cluster);
    double theta_optim;
    std::vector<cv::Point2f> hull_cluster;

    optim_convex_fitting(current_cluster, hull_cluster, convex_corner_visual, theta_optim);

    autoware_msgs::DetectedObject output_object;
    std::vector<cv::Point2f> rec_corner_points(4);
    calcuBoxbyPolygon(theta_optim, output_object, rec_corner_points, hull_cluster, current_cluster);

    // transform the bounding box
    pose.position = output_object.pose.position;
    pose.orientation = output_object.pose.orientation;

    tf2::doTransform(pose, pose_transformed, transform_stamped);

    output_object.header = bbox_header;
    output_object.pose = pose_transformed;

    //copy the autoware box to jsk box
    jsk_bbox = jsk_bbox_transform(output_object, bbox_header);
    jsk_bbox_array.boxes.push_back(jsk_bbox);

    //push the autoware bounding box in the array
    out_object_array.objects.push_back(output_object);

    //visulization the rectangle four corner points
    pcl::PointXYZI rec_corner_pt;

    //visulization the convex hull
    geometry_msgs::PolygonStamped cluster_polygon_;

    for (size_t i = 0; i < hull_cluster.size() + 1; i++)
    {
      geometry_msgs::Point32 point;
      point.x = hull_cluster[i % hull_cluster.size()].x;
      point.y = hull_cluster[i % hull_cluster.size()].y;
      point.z = 0;
      cluster_polygon_.polygon.points.push_back(point);
      cluster_polygon_.header.stamp = ros::Time::now();
      cluster_polygon_.header.frame_id = in_object.header.frame_id;
    }
    clusters_polygon_.polygons.push_back(cluster_polygon_);

    for (int i = 0; i < 4; i++)
    {
      rec_corner_pt.x = rec_corner_points[i].x;
      rec_corner_pt.y = rec_corner_points[i].y;
      rec_corner_pt.z = output_object.pose.position.z + 0.5 * output_object.dimensions.z;
      rec_corner_pt.intensity = intensity_mark;
      corner_points_visual->push_back(rec_corner_pt);
    }
    intensity_mark++;
  }

  out_object_array.header = bbox_header;
  pub_autoware_bboxs_array_.publish(out_object_array);

  jsk_bbox_array.header = bbox_header;
  pub_jsk_bboxs_array_.publish(jsk_bbox_array);

  clusters_polygon_.header.frame_id = in_cluster_array.header.frame_id;
  clusters_polygon_.header.stamp = ros::Time::now();
  pub_cluster_polygon_.publish(clusters_polygon_);

  // rectangle corner points
  sensor_msgs::PointCloud2 corner_points_visual_ros;
  pcl::toROSMsg(*corner_points_visual, corner_points_visual_ros);
  corner_points_visual_ros.header = in_cluster_array.header;
  pub_rec_corner_points_.publish(corner_points_visual_ros);

  // rectangle corner points
  sensor_msgs::PointCloud2 convex_corner_visual_ros;
  pcl::toROSMsg(*convex_corner_visual, convex_corner_visual_ros);
  convex_corner_visual_ros.header = in_cluster_array.header;
  pub_convex_corner_points_.publish(convex_corner_visual_ros);

  ransacLshape_end = std::chrono::system_clock::now();
  exe_time = std::chrono::duration_cast<std::chrono::microseconds>(ransacLshape_end - ransacLshape_start).count() / 1000.0;
  time_ransacLshape.data = exe_time;
  time_ransacLshape_pub.publish(time_ransacLshape);
}

void ConvexOptimizeFitting::calcuBoxbyPolygon(double &theta_star, autoware_msgs::DetectedObject &output, std::vector<cv::Point2f>& rec_corner_points, 
                   const std::vector<cv::Point2f> &polygon_cluster, const pcl::PointCloud<pcl::PointXYZ> &cluster)
{
  // calc min and max z for cylinder length
  double min_z = 0;
  double max_z = 0;
  for (size_t i = 0; i < cluster.size(); ++i)
  {
    if (cluster.at(i).z < min_z || i == 0)
      min_z = cluster.at(i).z;
    if (max_z < cluster.at(i).z || i == 0)
      max_z = cluster.at(i).z;
  }

  Eigen::Vector2d e_1_star;  
  Eigen::Vector2d e_2_star;
  e_1_star << std::cos(theta_star), std::sin(theta_star);
  e_2_star << -std::sin(theta_star), std::cos(theta_star);
  std::vector<double> C_1_star;
  std::vector<double> C_2_star;

  for (size_t i = 0; i < polygon_cluster.size(); i++)
  {
    C_1_star.push_back(polygon_cluster[i].x * e_1_star.x() + polygon_cluster[i].y * e_1_star.y());
    C_2_star.push_back(polygon_cluster[i].x * e_2_star.x() + polygon_cluster[i].y * e_2_star.y());
  }

  const double min_C_1_star = *std::min_element(C_1_star.begin(), C_1_star.end());
  const double max_C_1_star = *std::max_element(C_1_star.begin(), C_1_star.end());
  const double min_C_2_star = *std::min_element(C_2_star.begin(), C_2_star.end());
  const double max_C_2_star = *std::max_element(C_2_star.begin(), C_2_star.end());

  const double a_1 = std::cos(theta_star);
  const double b_1 = std::sin(theta_star);
  const double c_1 = min_C_1_star;

  const double a_2 = -1.0 * std::sin(theta_star);
  const double b_2 = std::cos(theta_star);
  const double c_2 = min_C_2_star;

  const double a_3 = std::cos(theta_star);
  const double b_3 = std::sin(theta_star);
  const double c_3 = max_C_1_star;

  const double a_4 = -1.0 * std::sin(theta_star);
  const double b_4 = std::cos(theta_star);
  const double c_4 = max_C_2_star;

  // calc center of bounding box
  double intersection_x_1 = (b_1 * c_2 - b_2 * c_1) / (a_2 * b_1 - a_1 * b_2);
  double intersection_y_1 = (a_1 * c_2 - a_2 * c_1) / (a_1 * b_2 - a_2 * b_1);
  double intersection_x_2 = (b_3 * c_4 - b_4 * c_3) / (a_4 * b_3 - a_3 * b_4);
  double intersection_y_2 = (a_3 * c_4 - a_4 * c_3) / (a_3 * b_4 - a_4 * b_3);

  cv::Point2f rec_corner_p1, rec_corner_p2, rec_corner_p3, rec_corner_p4;

  rec_corner_p1.x = intersection_x_1;
  rec_corner_p1.y = intersection_y_1;

  rec_corner_p2.x = (b_2 * c_3 - b_3 * c_2) / (a_3 * b_2 - a_2 * b_3);
  rec_corner_p2.y = (a_2 * c_3 - a_3 * c_2) / (a_2 * b_3 - a_3 * b_2);

  rec_corner_p3.x = intersection_x_2;
  rec_corner_p3.y = intersection_y_2;

  rec_corner_p4.x = (b_1 * c_4 - b_4 * c_1) / (a_4 * b_1 - a_1 * b_4);
  rec_corner_p4.y = (a_1 * c_4 - a_4 * c_1) / (a_1 * b_4 - a_4 * b_1);

  rec_corner_points[0] = rec_corner_p1;
  rec_corner_points[1] = rec_corner_p2;
  rec_corner_points[2] = rec_corner_p3;
  rec_corner_points[3] = rec_corner_p4;

  // calc dimention of bounding box
  Eigen::Vector2d e_x;
  Eigen::Vector2d e_y;
  e_x << a_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1)), b_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1));
  e_y << a_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2)), b_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2));
  Eigen::Vector2d diagonal_vec;
  diagonal_vec << intersection_x_1 - intersection_x_2, intersection_y_1 - intersection_y_2;

  // calc yaw
  tf2::Quaternion quat;
  quat.setEuler(/* roll */ 0, /* pitch */ 0, /* yaw */ std::atan2(e_1_star.y(), e_1_star.x()));

  output.pose.position.x = (intersection_x_1 + intersection_x_2) / 2.0;
  output.pose.position.y = (intersection_y_1 + intersection_y_2) / 2.0;
  // output.pose.position.z = centroid.z;
  output.pose.position.z = (max_z + min_z) / 2.0;
  output.pose.orientation = tf2::toMsg(quat);

  double ep = 0.01;
  output.dimensions.x = std::fabs(e_x.dot(diagonal_vec));
  output.dimensions.y = std::fabs(e_y.dot(diagonal_vec));
  output.dimensions.z = std::max((max_z - min_z), ep);
  output.pose_reliable = true;

  output.dimensions.x = std::max(output.dimensions.x, ep);
  output.dimensions.y = std::max(output.dimensions.y, ep);
  output.dimensions.z = std::max(output.dimensions.z, ep);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ransac_lshape_fitting_node");
  ConvexOptimizeFitting app;
  ros::spin();

  return 0;
}