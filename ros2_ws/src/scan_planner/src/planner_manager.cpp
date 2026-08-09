// #include <fstream>
#include <plan_manage/planner_manager.h>
#include <plan_manage/reference_execution.h>
#include <plan_manage/reference_spline_boundary.h>
#include <plan_manage/reference_trajectory_initialization.h>
#include <plan_manage/trajectory_progress.h>
#include <plan_manage/trajectory_timing.h>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace scan_planner
{
  namespace
  {
    constexpr double kMaximumReferenceInitialProgressTolerance = 0.005;
    constexpr double kMaximumReferenceProgressMeasurementTolerance = 0.002;
    constexpr double kMaximumReferenceCorridorProgressLead = 0.035;
    constexpr double kMaximumBlockedCorridorProgressLead = 0.050;

    void applyLinearZReference(std::vector<Eigen::Vector3d> &points, const double start_z, const double target_z)
    {
      if (points.empty())
        return;

      if (points.size() == 1)
      {
        points.front()(2) = start_z;
        return;
      }

      std::vector<double> accumulated_xy_length(points.size(), 0.0);
      for (size_t i = 1; i < points.size(); ++i)
      {
        accumulated_xy_length[i] = accumulated_xy_length[i - 1] +
                                   (points[i].head<2>() - points[i - 1].head<2>()).norm();
      }

      const double total_xy_length = accumulated_xy_length.back();
      for (size_t i = 0; i < points.size(); ++i)
      {
        const double ratio = total_xy_length > 1e-6
                                 ? accumulated_xy_length[i] / total_xy_length
                                 : static_cast<double>(i) / static_cast<double>(points.size() - 1);
        points[i](2) = start_z + ratio * (target_z - start_z);
      }

      points.front()(2) = start_z;
      points.back()(2) = target_z;
    }

  } // namespace

  // SECTION interfaces for setup and query

  SCANPlannerManager::SCANPlannerManager() {}

  SCANPlannerManager::~SCANPlannerManager() { std::cout << "des manager" << std::endl; }

  void SCANPlannerManager::initPlanModules(rclcpp::Node *node, PlanningVisualization::Ptr vis)
  {
    node_ = node;
    /* read algorithm parameters */
    const auto get_double = [node](const std::string &name, double default_value) {
      if (!node->has_parameter(name)) node->declare_parameter<double>(name, default_value);
      return node->get_parameter(name).as_double();
    };
    const auto get_bool = [node](const std::string &name, bool default_value) {
      if (!node->has_parameter(name)) node->declare_parameter<bool>(name, default_value);
      return node->get_parameter(name).as_bool();
    };
    pp_.max_vel_ = get_double("manager.max_vel", -1.0);
    pp_.max_acc_ = get_double("manager.max_acc", -1.0);
    pp_.max_jerk_ = get_double("manager.max_jerk", -1.0);
    pp_.vel_tolerance_ = get_double("optimization.vel_tolerance", 0.005);
    pp_.acc_tolerance_ = get_double("optimization.acc_tolerance", 0.01);
    pp_.feasibility_tolerance_ = get_double("manager.feasibility_tolerance", 0.01);
    pp_.reference_profile_acceleration_scale_ = get_double(
        "manager.reference_profile_acceleration_scale", 1.0);
    pp_.reference_free_guide_refine_enabled_ = get_bool(
        "manager.reference_free_guide_refine_enabled", false);
    pp_.reference_free_guide_refine_minimum_duration_gain_ = get_double(
        "manager.reference_free_guide_refine_minimum_duration_gain", 0.05);
    pp_.ctrl_pt_dist = get_double("manager.control_points_distance", -1.0);
    pp_.planning_horizon_ = get_double("manager.planning_horizon", 5.0);
    pp_.max_reference_reverse_distance_ =
        get_double("manager.max_reference_reverse_distance", 0.02);
    pp_.max_reference_reverse_speed_ =
        get_double("manager.max_reference_reverse_speed", 0.02);
    pp_.reference_corridor_max_deviation_ =
        get_double("manager.reference_corridor_max_deviation", 0.10);
    pp_.reference_obstacle_corridor_max_deviation_ =
        get_double(
            "manager.reference_obstacle_corridor_max_deviation", 0.35);
    pp_.reference_corridor_max_progress_lead_ =
        get_double("manager.reference_corridor_max_progress_lead", 0.035);
    pp_.reference_obstacle_corridor_max_progress_lead_ =
        get_double(
            "manager.reference_obstacle_corridor_max_progress_lead", 0.05);
    pp_.reference_corridor_initial_progress_tolerance_ =
        get_double(
            "manager.reference_corridor_initial_progress_tolerance", 0.001);
    pp_.reference_corridor_progress_measurement_tolerance_ =
        get_double(
            "manager.reference_corridor_progress_measurement_tolerance",
            0.001);
    if (!std::isfinite(pp_.max_vel_) || pp_.max_vel_ <= 0.0 ||
        !std::isfinite(pp_.max_acc_) || pp_.max_acc_ <= 0.0 ||
        !std::isfinite(pp_.vel_tolerance_) || pp_.vel_tolerance_ < 0.0 ||
        !std::isfinite(pp_.acc_tolerance_) || pp_.acc_tolerance_ < 0.0 ||
        !std::isfinite(pp_.feasibility_tolerance_) ||
        pp_.feasibility_tolerance_ < 0.0 ||
        !std::isfinite(pp_.reference_profile_acceleration_scale_) ||
        pp_.reference_profile_acceleration_scale_ <= 0.0 ||
        pp_.reference_profile_acceleration_scale_ > 1.0 ||
        !std::isfinite(
            pp_.reference_free_guide_refine_minimum_duration_gain_) ||
        pp_.reference_free_guide_refine_minimum_duration_gain_ < 0.0)
      throw std::runtime_error(
          "SCAN 速度、加速度、reference 速度剖面/平滑收益或可行性容差参数非法");
    if (!dynamicFeasibilityTolerancesCompatible(
            pp_.max_vel_, pp_.max_acc_, pp_.feasibility_tolerance_,
            pp_.vel_tolerance_, pp_.acc_tolerance_))
      throw std::runtime_error(
          "SCAN 连续可行性容差不得宽于最终动态采样门");
    if (!std::isfinite(pp_.reference_corridor_max_deviation_) ||
        pp_.reference_corridor_max_deviation_ <= 0.0 ||
        !std::isfinite(
            pp_.reference_obstacle_corridor_max_deviation_) ||
        pp_.reference_obstacle_corridor_max_deviation_ <
            pp_.reference_corridor_max_deviation_ ||
        !std::isfinite(pp_.reference_corridor_max_progress_lead_) ||
        pp_.reference_corridor_max_progress_lead_ < 0.0 ||
        pp_.reference_corridor_max_progress_lead_ >
            kMaximumReferenceCorridorProgressLead ||
        !std::isfinite(
            pp_.reference_obstacle_corridor_max_progress_lead_) ||
        pp_.reference_obstacle_corridor_max_progress_lead_ <
            pp_.reference_corridor_max_progress_lead_ ||
        pp_.reference_obstacle_corridor_max_progress_lead_ >
            kMaximumBlockedCorridorProgressLead ||
        !std::isfinite(
            pp_.reference_corridor_initial_progress_tolerance_) ||
        pp_.reference_corridor_initial_progress_tolerance_ < 0.0 ||
        pp_.reference_corridor_initial_progress_tolerance_ >
            kMaximumReferenceInitialProgressTolerance ||
        !std::isfinite(
            pp_.reference_corridor_progress_measurement_tolerance_) ||
        pp_.reference_corridor_progress_measurement_tolerance_ < 0.0 ||
        pp_.reference_corridor_progress_measurement_tolerance_ >
            kMaximumReferenceProgressMeasurementTolerance)
      throw std::runtime_error(
          "reference corridor 门限必须为有限安全值，空旷/受阻进度上限不得超过 35/50mm，首点拟合容差不得超过 5mm，进度测量容差不得超过 2mm");

    local_data_.traj_id_ = 0;
    grid_map_.reset(new GridMap);
    grid_map_->initMap(node_);

    bspline_optimizer_rebound_.reset(new BsplineOptimizer);
    bspline_optimizer_rebound_->setParam(node_);
    bspline_optimizer_rebound_->setEnvironment(grid_map_);
    bspline_optimizer_rebound_->a_star_.reset(new AStar);
    bspline_optimizer_rebound_->a_star_->initGridMap(grid_map_, Eigen::Vector3i(100, 100, 100));

    visualization_ = vis;
  }

  // !SECTION

  // SECTION rebond replanning

  bool SCANPlannerManager::reboundReplan(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel,
                                        Eigen::Vector3d start_acc, Eigen::Vector3d local_target_pt,
                                        Eigen::Vector3d local_target_vel, bool flag_polyInit,
                                        bool flag_randomPolyTraj, bool require_forward_progress,
                                        const std::vector<Eigen::Vector3d> &reference_guide,
                                        const std::vector<Eigen::Vector3d> &reference_corridor_guide)
  {

    local_plan_failure_state_.beginAttempt();

    static int count = 0;
    std::cout << endl
              << "[rebo replan]: -------------------------------------" << count++ << std::endl;
    cout.precision(3);
    cout << "start: " << start_pt.transpose() << ", " << start_vel.transpose() << "\ngoal:" << local_target_pt.transpose() << ", " << local_target_vel.transpose()
         << endl;

    const double start_to_target_distance =
        (start_pt - local_target_pt).norm();
    if (reboundPlanRejectedBeforeInitialization(
            start_to_target_distance, require_forward_progress,
            reference_guide.size()))
    {
      if (require_forward_progress && reference_guide.size() < 2U)
        local_plan_failure_state_.set(
            LocalPlanFailureReason::ReferenceCorridorRejected);
      cout << "Close to goal" << endl;
      continuous_failures_count_++;
      return false;
    }

    auto t_start = std::chrono::steady_clock::now();
    double t_init = 0.0, t_opt = 0.0, t_refine = 0.0;

    /*** STEP 1: INIT ***/
    double ts = (start_pt - local_target_pt).norm() > 0.1 ? pp_.ctrl_pt_dist / pp_.max_vel_ * 1.2 : pp_.ctrl_pt_dist / pp_.max_vel_ * 5; // pp_.ctrl_pt_dist / pp_.max_vel_ is too tense, and will surely exceed the acc/vel limits
    vector<Eigen::Vector3d> point_set, start_end_derivatives;
    vector<Eigen::Vector3d> reference_corridor_point_set;
    vector<Eigen::Vector3d> reference_spatial_guide;
    if (require_forward_progress)
    {
      ReferenceTrajectoryInitialization corridor_initialization;
      if (reference_corridor_guide.size() < 2 ||
          !initializeReferenceTrajectory(
              reference_corridor_guide,
              reference_corridor_guide.front(), local_target_pt,
              pp_.max_vel_,
              pp_.max_acc_ * pp_.reference_profile_acceleration_scale_,
              std::max(
                  grid_map_->getResolution(),
                  0.25 * pp_.ctrl_pt_dist),
              corridor_initialization))
      {
        local_plan_failure_state_.set(
            LocalPlanFailureReason::ReferenceCorridorRejected);
        RCLCPP_ERROR(
            node_->get_logger(),
            "无法从 PCT semantic guide 构造有序走廊");
        continuous_failures_count_++;
        return false;
      }
      reference_corridor_point_set =
          std::move(corridor_initialization.spatial_guide);
    }
    static bool flag_first_call = true, flag_force_polynomial = false;
    bool flag_regenerate = false;
    bool used_reference_guide = false;
    do
    {
      point_set.clear();
      start_end_derivatives.clear();
      flag_regenerate = false;

      if (flag_first_call || flag_polyInit || flag_force_polynomial /*|| ( start_pt - local_target_pt ).norm() < 1.0*/) // Initial path generated from a min-snap traj by order.
      {
        flag_first_call = false;
        flag_force_polynomial = false;

        if (require_forward_progress && !reference_guide.empty())
        {
          ReferenceTrajectoryInitialization reference_initialization;
          used_reference_guide = initializeReferenceTrajectory(
              reference_guide, start_pt, local_target_pt,
              pp_.max_vel_,
              pp_.max_acc_ * pp_.reference_profile_acceleration_scale_,
              std::max(
                  grid_map_->getResolution(),
                  0.25 * pp_.ctrl_pt_dist),
              start_vel, local_target_vel,
              reference_initialization);
          if (!used_reference_guide)
          {
            local_plan_failure_state_.set(
                LocalPlanFailureReason::ReferenceCorridorRejected);
            RCLCPP_ERROR(
                node_->get_logger(),
                "无法从 reference 折线构造局部 B-spline 初值");
            continuous_failures_count_++;
            return false;
          }
          point_set =
              std::move(reference_initialization.parameterization_points);
          reference_spatial_guide =
              std::move(reference_initialization.spatial_guide);
          ts = reference_initialization.time_step;
          start_end_derivatives.push_back(start_vel);
          start_end_derivatives.push_back(local_target_vel);
          start_end_derivatives.push_back(start_acc);
          start_end_derivatives.push_back(Eigen::Vector3d::Zero());
        }
        else
        {

        PolynomialTraj gl_traj;

        double dist = (start_pt - local_target_pt).norm();
        double time = pow(pp_.max_vel_, 2) / pp_.max_acc_ > dist ? sqrt(dist / pp_.max_acc_) : (dist - pow(pp_.max_vel_, 2) / pp_.max_acc_) / pp_.max_vel_ + 2 * pp_.max_vel_ / pp_.max_acc_;

        if (!flag_randomPolyTraj)
        {
          gl_traj = PolynomialTraj::one_segment_traj_gen(start_pt, start_vel, start_acc, local_target_pt, local_target_vel, Eigen::Vector3d::Zero(), time);
        }
        else
        {
          Eigen::Vector3d horizon_dir = ((start_pt - local_target_pt).cross(Eigen::Vector3d(0, 0, 1))).normalized();
          Eigen::Vector3d vertical_dir = ((start_pt - local_target_pt).cross(horizon_dir)).normalized();
          Eigen::Vector3d random_inserted_pt = (start_pt + local_target_pt) / 2 +
                                               (((double)rand()) / RAND_MAX - 0.5) * (start_pt - local_target_pt).norm() * horizon_dir * 0.8 * (-0.978 / (continuous_failures_count_ + 0.989) + 0.989) +
                                               (((double)rand()) / RAND_MAX - 0.5) * (start_pt - local_target_pt).norm() * vertical_dir * 0.4 * (-0.978 / (continuous_failures_count_ + 0.989) + 0.989);
          Eigen::MatrixXd pos(3, 3);
          pos.col(0) = start_pt;
          pos.col(1) = random_inserted_pt;
          pos.col(2) = local_target_pt;
          Eigen::VectorXd t(2);
          t(0) = t(1) = time / 2;
          gl_traj = PolynomialTraj::minSnapTraj(pos, start_vel, local_target_vel, start_acc, Eigen::Vector3d::Zero(), t);
        }

        double t;
        bool flag_too_far;
        ts *= 1.5; // ts will be divided by 1.5 in the next
        do
        {
          ts /= 1.5;
          point_set.clear();
          flag_too_far = false;
          Eigen::Vector3d last_pt = gl_traj.evaluate(0);
          for (t = 0; t < time; t += ts)
          {
            Eigen::Vector3d pt = gl_traj.evaluate(t);
            if ((last_pt - pt).norm() > pp_.ctrl_pt_dist * 1.5)
            {
              flag_too_far = true;
              break;
            }
            last_pt = pt;
            point_set.push_back(pt);
          }
        } while (flag_too_far || point_set.size() < 7); // To make sure the initial path has enough points.
        t -= ts;
        start_end_derivatives.push_back(gl_traj.evaluateVel(0));
        start_end_derivatives.push_back(local_target_vel);
        start_end_derivatives.push_back(gl_traj.evaluateAcc(0));
        start_end_derivatives.push_back(gl_traj.evaluateAcc(t));
        }
      }
      else // Initial path generated from previous trajectory.
      {

        double t;
        double t_cur = (node_->now() - local_data_.start_time_).seconds();

        vector<double> pseudo_arc_length;
        vector<Eigen::Vector3d> segment_point;
        pseudo_arc_length.push_back(0.0);
        for (t = t_cur; t < local_data_.duration_ + 1e-3; t += ts)
        {
          segment_point.push_back(local_data_.position_traj_.evaluateDeBoorT(t));
          if (t > t_cur)
          {
            pseudo_arc_length.push_back((segment_point.back() - segment_point[segment_point.size() - 2]).norm() + pseudo_arc_length.back());
          }
        }
        t -= ts;

        double poly_time = (local_data_.position_traj_.evaluateDeBoorT(t) - local_target_pt).norm() / pp_.max_vel_ * 2;
        if (poly_time > ts)
        {
          PolynomialTraj gl_traj = PolynomialTraj::one_segment_traj_gen(local_data_.position_traj_.evaluateDeBoorT(t),
                                                                        local_data_.velocity_traj_.evaluateDeBoorT(t),
                                                                        local_data_.acceleration_traj_.evaluateDeBoorT(t),
                                                                        local_target_pt, local_target_vel, Eigen::Vector3d::Zero(), poly_time);

          for (t = ts; t < poly_time; t += ts)
          {
            if (!pseudo_arc_length.empty())
            {
              segment_point.push_back(gl_traj.evaluate(t));
              pseudo_arc_length.push_back((segment_point.back() - segment_point[segment_point.size() - 2]).norm() + pseudo_arc_length.back());
            }
            else
            {
              RCLCPP_ERROR(node_->get_logger(), "pseudo_arc_length is empty; aborting replan");
              continuous_failures_count_++;
              return false;
            }
          }
        }

        double sample_length = 0;
        double cps_dist = pp_.ctrl_pt_dist * 1.5; // cps_dist will be divided by 1.5 in the next
        size_t id = 0;
        do
        {
          cps_dist /= 1.5;
          point_set.clear();
          sample_length = 0;
          id = 0;
          while ((id <= pseudo_arc_length.size() - 2) && sample_length <= pseudo_arc_length.back())
          {
            if (sample_length >= pseudo_arc_length[id] && sample_length < pseudo_arc_length[id + 1])
            {
              point_set.push_back((sample_length - pseudo_arc_length[id]) / (pseudo_arc_length[id + 1] - pseudo_arc_length[id]) * segment_point[id + 1] +
                                  (pseudo_arc_length[id + 1] - sample_length) / (pseudo_arc_length[id + 1] - pseudo_arc_length[id]) * segment_point[id]);
              sample_length += cps_dist;
            }
            else
              id++;
          }
          point_set.push_back(local_target_pt);
        } while (point_set.size() < 7); // If the start point is very close to end point, this will help

        start_end_derivatives.push_back(local_data_.velocity_traj_.evaluateDeBoorT(t_cur));
        start_end_derivatives.push_back(local_target_vel);
        start_end_derivatives.push_back(local_data_.acceleration_traj_.evaluateDeBoorT(t_cur));
        start_end_derivatives.push_back(Eigen::Vector3d::Zero());

        if (point_set.size() > pp_.planning_horizon_ / pp_.ctrl_pt_dist * 3) // The initial path is abnormally too long!
        {
          flag_force_polynomial = true;
          flag_regenerate = true;
        }
      }
    } while (flag_regenerate);

    if (!used_reference_guide)
      applyLinearZReference(point_set, start_pt(2), local_target_pt(2));

    Eigen::MatrixXd ctrl_pts;
    const bool parameterized = used_reference_guide
        ? parameterizeCubicBsplineWithExactStartBoundary(
            ts, point_set, start_end_derivatives, ctrl_pts)
        : ([&]() {
            UniformBspline::parameterizeToBspline(
                ts, point_set, start_end_derivatives, ctrl_pts);
            return ctrl_pts.rows() == 3 && ctrl_pts.cols() >= 3 &&
                   ctrl_pts.allFinite();
          })();
    if (!parameterized)
    {
      local_plan_failure_state_.set(
          LocalPlanFailureReason::ReferenceCorridorRejected);
      RCLCPP_ERROR(
          node_->get_logger(),
          "无法参数化具有精确起点边界的 reference B-spline");
      continuous_failures_count_++;
      return false;
    }

    // reference 初始化会把折线首尾严格锚定到本轮真实 Odometry 与局部目标，
    // 并只在原折线段内补点。它继续负责初值和碰撞检测；受阻时的有序
    // 走廊则在下方改用不含人工回归连接的 PCT semantic guide。
    const std::vector<Eigen::Vector3d> &execution_reference_guide =
        used_reference_guide ? reference_spatial_guide : reference_guide;

    vector<vector<Eigen::Vector3d>> a_star_paths;
    const bool execution_reference_guide_free =
        used_reference_guide &&
        checkReferenceGuideCollisionFree(execution_reference_guide);
    const bool initial_reference_spline_free =
        used_reference_guide &&
        checkTrajectoryCollisionFree(UniformBspline(ctrl_pts, 3, ts));
    const bool preserve_collision_free_reference_guide =
        execution_reference_guide_free && initial_reference_spline_free;
    // 空旷时保留包含真实起点连接的严格初始化 guide。只有 reference
    // 真正受阻、需要 rebound 绕障时，走廊才改用不含人工回归连接的
    // PCT semantic guide；这不会给后续 U 形或自交路径增加进度额度。
    const bool use_semantic_reference_corridor =
        require_forward_progress && !execution_reference_guide_free;
    const std::vector<Eigen::Vector3d> &ordered_progress_guide =
        use_semantic_reference_corridor
            ? reference_corridor_point_set
            : execution_reference_guide;
    if (!preserve_collision_free_reference_guide)
      a_star_paths = bspline_optimizer_rebound_->initControlPoints(ctrl_pts, true);

    t_init = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    static int vis_id = 0;
    visualization_->displayInitPathList(point_set, 0.2, 0);
    visualization_->displayAStarList(a_star_paths, vis_id);

    t_start = std::chrono::steady_clock::now();

    /*** STEP 2: OPTIMIZE ***/
    bool flag_step_1_success = true;
    if (preserve_collision_free_reference_guide)
    {
      bool accepted_refined_candidate = false;
      if (pp_.reference_free_guide_refine_enabled_)
      {
        UniformBspline original_candidate(ctrl_pts, 3, ts);
        original_candidate.setPhysicalLimits(
            pp_.max_vel_, pp_.max_acc_, pp_.feasibility_tolerance_);
        double original_ratio = 1.0;
        if (!original_candidate.checkFeasibility(original_ratio, false))
        {
          UniformBspline original_timed = original_candidate;
          const bool original_timing_valid =
              rescaleTrajectoryToPhysicalLimits(
                  original_timed, pp_.max_vel_, pp_.max_acc_,
                  pp_.feasibility_tolerance_);

          bspline_optimizer_rebound_->ref_pts_.clear();
          const double reference_sample_step =
              original_candidate.getTimeSum() /
              static_cast<double>(ctrl_pts.cols() - 3);
          for (double sample_time = 0.0;
               sample_time < original_candidate.getTimeSum() + 1.0e-4;
               sample_time += reference_sample_step)
          {
            bspline_optimizer_rebound_->ref_pts_.push_back(
                original_candidate.evaluateDeBoorT(sample_time));
          }

          Eigen::MatrixXd refined_control_points;
          const bool refined =
              bspline_optimizer_rebound_->BsplineOptimizeTrajRefine(
                  ctrl_pts, ts, refined_control_points);
          if (refined && original_timing_valid)
          {
            UniformBspline refined_candidate(
                refined_control_points, 3, ts);
            UniformBspline refined_timed = refined_candidate;
            const bool refined_timing_valid =
                rescaleTrajectoryToPhysicalLimits(
                    refined_timed, pp_.max_vel_, pp_.max_acc_,
                    pp_.feasibility_tolerance_);
            const ReferenceCorridorCheck refined_corridor =
                refined_timing_valid
                    ? checkTrajectoryReferenceCorridor(
                        refined_timed, execution_reference_guide,
                        ordered_progress_guide,
                        pp_.reference_corridor_max_deviation_,
                        pp_.reference_corridor_max_progress_lead_,
                        0.01,
                        pp_.reference_corridor_initial_progress_tolerance_,
                        pp_.reference_corridor_progress_measurement_tolerance_)
                    : ReferenceCorridorCheck{};
            const double duration_gain =
                original_timed.getTimeSum() - refined_timed.getTimeSum();
            if (refined_timing_valid &&
                refined_corridor.safe &&
                checkTrajectoryCollisionFree(refined_timed) &&
                checkDynamicFeasibility(refined_timed) &&
                duration_gain + 1.0e-9 >=
                    pp_.reference_free_guide_refine_minimum_duration_gain_)
            {
              ctrl_pts = std::move(refined_control_points);
              accepted_refined_candidate = true;
              RCLCPP_INFO(
                  node_->get_logger(),
                  "空旷 reference 折线平滑候选通过安全门，局部轨迹预计缩短 %.3f 秒",
                  duration_gain);
            }
          }
        }
      }
      if (!accepted_refined_candidate)
      {
        // 平滑候选没有同时满足碰撞、ordered corridor、动力学和收益门时，
        // 继续使用原始折线初值；不能为了提速把 90° 平台切成对角捷径。
        RCLCPP_INFO(
            node_->get_logger(),
            "reference guide 全段无碰撞，保留安全折线参数化初值");
      }
    }
    else
    {
      flag_step_1_success =
          bspline_optimizer_rebound_->BsplineOptimizeTrajRebound(ctrl_pts, ts);
    }
    cout << "first_optimize_step_success=" << flag_step_1_success << endl;
    if (!flag_step_1_success)
    {
      // visualization_->displayOptimalList( ctrl_pts, vis_id );
      local_plan_failure_state_.setFromReboundFailure(
          bspline_optimizer_rebound_->lastReboundFailureReason());
      continuous_failures_count_++;
      return false;
    }
    //visualization_->displayOptimalList( ctrl_pts, vis_id );

    t_opt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    t_start = std::chrono::steady_clock::now();

    /*** STEP 3: REFINE(RE-ALLOCATE TIME) IF NECESSARY ***/
    UniformBspline pos = UniformBspline(ctrl_pts, 3, ts);
    pos.setPhysicalLimits(pp_.max_vel_, pp_.max_acc_, pp_.feasibility_tolerance_);

    double ratio;
    bool flag_step_2_success = true;
    if (!pos.checkFeasibility(ratio, false))
    {
      cout << "Need to reallocate time." << endl;
      if (!preserve_collision_free_reference_guide)
      {
        Eigen::MatrixXd optimal_control_points;
        flag_step_2_success = refineTrajAlgo(
            pos, start_end_derivatives, ratio, ts,
            optimal_control_points);
        if (flag_step_2_success)
          pos = UniformBspline(optimal_control_points, 3, ts);
      }
    }

    if (flag_step_2_success)
    {
      // refine optimizer 会在重参数化后再次移动控制点，不能把“优化成功”
      // 等同于动态可行。无论走空旷 guide 还是 rebound 分支，发布前都对
      // 最终空间曲线做统一时间缩放；这样不会改变已验证的碰撞与走廊几何。
      flag_step_2_success = rescaleTrajectoryToPhysicalLimits(
          pos, pp_.max_vel_, pp_.max_acc_, pp_.feasibility_tolerance_);
    }

    if (!flag_step_2_success || !checkDynamicFeasibility(pos))
    {
      local_plan_failure_state_.set(
          LocalPlanFailureReason::DynamicFeasibilityFailed);
      printf("\033[34mThis trajectory is dynamically infeasible. Skip publishing it.\n\033[0m");
      continuous_failures_count_++;
      return false;
    }

    if (!checkTrajectoryCollisionFree(pos))
    {
      printf("\033[34mThis trajectory is colliding. Skip publishing it.\n\033[0m");
      continuous_failures_count_++;
      return false;
    }

    ReferenceCorridorCheck corridor;
    bool reference_corridor_checked = false;
    const double reference_corridor_deviation_limit =
        execution_reference_guide_free
            ? pp_.reference_corridor_max_deviation_
            : pp_.reference_obstacle_corridor_max_deviation_;
    const double reference_corridor_progress_lead_limit =
        execution_reference_guide_free
            ? pp_.reference_corridor_max_progress_lead_
            : pp_.reference_obstacle_corridor_max_progress_lead_;
    if (require_forward_progress)
    {
      reference_corridor_checked = true;
      corridor = checkTrajectoryReferenceCorridor(
          pos, execution_reference_guide, ordered_progress_guide,
          reference_corridor_deviation_limit,
          reference_corridor_progress_lead_limit,
          0.01,
          pp_.reference_corridor_initial_progress_tolerance_,
          pp_.reference_corridor_progress_measurement_tolerance_);
      if (!corridor.safe)
      {
        local_plan_failure_state_.set(
            LocalPlanFailureReason::ReferenceCorridorRejected);
        RCLCPP_WARN(
            node_->get_logger(),
            "Reference trajectory left ordered guide corridor: "
            "guide_free=%s, trajectory=%.4f m, anchor=%.4f m, "
            "initial_progress=%.6f m, relative_progress_lead=%.6f m, "
            "progress_lead=%.6f m, limits=%.4f/%.4f m, "
            "initial_tolerance=%.4f m, measurement_tolerance=%.4f m, "
            "semantic=%s",
            execution_reference_guide_free ? "true" : "false",
            corridor.maximum_trajectory_deviation,
            corridor.maximum_guide_anchor_deviation,
            corridor.initial_guide_progress,
            corridor.maximum_relative_guide_progress_lead,
            corridor.maximum_guide_progress_lead,
            reference_corridor_deviation_limit,
            reference_corridor_progress_lead_limit,
            pp_.reference_corridor_initial_progress_tolerance_,
            pp_.reference_corridor_progress_measurement_tolerance_,
            use_semantic_reference_corridor ? "true" : "false");
        continuous_failures_count_++;
        return false;
      }
    }

    if (require_forward_progress)
    {
      const TrajectoryProgressCheck progress = checkTrajectoryForwardProgress(
          pos, start_pt, local_target_pt,
          pp_.max_reference_reverse_distance_,
          pp_.max_reference_reverse_speed_);
      if (!progress.safe)
      {
        local_plan_failure_state_.set(
            LocalPlanFailureReason::ReferenceCorridorRejected);
        RCLCPP_WARN(
            node_->get_logger(),
            "Reference trajectory reverses progress: distance=%.4f m, speed=%.4f m/s",
            progress.maximum_reverse_distance,
            progress.minimum_projected_velocity);
        continuous_failures_count_++;
        return false;
      }
    }

    t_refine = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    // save planned results
    updateTrajInfo(pos, node_->now());
    local_data_.reference_corridor_checked_ =
        reference_corridor_checked;
    local_data_.reference_corridor_safe_ = corridor.safe;
    if (reference_corridor_checked)
    {
      local_data_.maximum_trajectory_deviation_ =
          corridor.maximum_trajectory_deviation;
      local_data_.maximum_guide_anchor_deviation_ =
          corridor.maximum_guide_anchor_deviation;
      local_data_.maximum_guide_progress_lead_ =
          corridor.maximum_guide_progress_lead;
      local_data_.reference_corridor_deviation_limit_ =
          reference_corridor_deviation_limit;
      // diagnostics 发布的是本次判定实际使用的有效上限，保证独立 bridge
      // 复核与 planner 的 fail-closed 结论一致；业务门限仍由上面的原值记录。
      local_data_.reference_corridor_progress_lead_limit_ =
          reference_corridor_progress_lead_limit +
          pp_.reference_corridor_progress_measurement_tolerance_;
      local_data_.ordered_reference_guide_ = execution_reference_guide;
    }

    cout << "total time:\033[42m" << (t_init + t_opt + t_refine)
         << "\033[0m,optimize:" << (t_init + t_opt) << ",refine:" << t_refine << endl;

    // success. YoY
    continuous_failures_count_ = 0;
    local_plan_failure_state_.completeSuccess();
    return true;
  }

  bool SCANPlannerManager::EmergencyStop(Eigen::Vector3d stop_pos)
  {
    Eigen::MatrixXd control_points(3, 6);
    for (int i = 0; i < 6; i++)
    {
      control_points.col(i) = stop_pos;
    }

    updateTrajInfo(UniformBspline(control_points, 3, 1.0), node_->now());

    return true;
  }

  bool SCANPlannerManager::planGlobalTrajWaypoints(const Eigen::Vector3d &start_pos, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
                                                  const std::vector<Eigen::Vector3d> &waypoints, const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc)
  {

    // generate global reference trajectory

    if (waypoints.empty())
      return false;

    vector<Eigen::Vector3d> points;
    points.push_back(start_pos);

    for (size_t wp_i = 0; wp_i < waypoints.size(); wp_i++)
    {
      points.push_back(waypoints[wp_i]);
    }

    double total_len = 0;
    for (size_t i = 0; i < points.size() - 1; i++)
    {
      total_len += (points[i + 1] - points[i]).norm();
    }

    // insert intermediate points if too far
    vector<Eigen::Vector3d> inter_points;
    double dist_thresh = max(total_len / 8, 4.0);

    for (size_t i = 0; i < points.size() - 1; ++i)
    {
      inter_points.push_back(points.at(i));
      double dist = (points.at(i + 1) - points.at(i)).norm();

      if (dist > dist_thresh)
      {
        int id_num = floor(dist / dist_thresh) + 1;

        for (int j = 1; j < id_num; ++j)
        {
          Eigen::Vector3d inter_pt =
              points.at(i) * (1.0 - double(j) / id_num) + points.at(i + 1) * double(j) / id_num;
          inter_points.push_back(inter_pt);
        }
      }
    }

    inter_points.push_back(points.back());

    // for ( int i=0; i<inter_points.size(); i++ )
    // {
    //   cout << inter_points[i].transpose() << endl;
    // }

    // write position matrix
    int pt_num = inter_points.size();
    Eigen::MatrixXd pos(3, pt_num);
    for (int i = 0; i < pt_num; ++i)
      pos.col(i) = inter_points[i];

    Eigen::Vector3d zero(0, 0, 0);
    Eigen::VectorXd time(pt_num - 1);
    for (int i = 0; i < pt_num - 1; ++i)
    {
      time(i) = (pos.col(i + 1) - pos.col(i)).norm() / (pp_.max_vel_);
    }

    time(0) *= 2.0;
    time(time.rows() - 1) *= 2.0;

    PolynomialTraj gl_traj;
    if (pos.cols() >= 3)
      gl_traj = PolynomialTraj::minSnapTraj(pos, start_vel, end_vel, start_acc, end_acc, time);
    else if (pos.cols() == 2)
      gl_traj = PolynomialTraj::one_segment_traj_gen(start_pos, start_vel, start_acc, pos.col(1), end_vel, end_acc, time(0));
    else
      return false;

    auto time_now = node_->now();
    global_data_.setGlobalTraj(gl_traj, time_now);

    return true;
  }

  bool SCANPlannerManager::planGlobalTraj(const Eigen::Vector3d &start_pos, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
                                         const Eigen::Vector3d &end_pos, const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc)
  {

    // generate global reference trajectory

    vector<Eigen::Vector3d> points;
    points.push_back(start_pos);
    points.push_back(end_pos);

    // insert intermediate points if too far
    vector<Eigen::Vector3d> inter_points;
    const double dist_thresh = 4.0;

    for (size_t i = 0; i < points.size() - 1; ++i)
    {
      inter_points.push_back(points.at(i));
      double dist = (points.at(i + 1) - points.at(i)).norm();

      if (dist > dist_thresh)
      {
        int id_num = floor(dist / dist_thresh) + 1;

        for (int j = 1; j < id_num; ++j)
        {
          Eigen::Vector3d inter_pt =
              points.at(i) * (1.0 - double(j) / id_num) + points.at(i + 1) * double(j) / id_num;
          inter_points.push_back(inter_pt);
        }
      }
    }

    inter_points.push_back(points.back());

    // write position matrix
    int pt_num = inter_points.size();
    Eigen::MatrixXd pos(3, pt_num);
    for (int i = 0; i < pt_num; ++i)
      pos.col(i) = inter_points[i];

    Eigen::Vector3d zero(0, 0, 0);
    Eigen::VectorXd time(pt_num - 1);
    for (int i = 0; i < pt_num - 1; ++i)
    {
      time(i) = (pos.col(i + 1) - pos.col(i)).norm() / (pp_.max_vel_);
    }

    time(0) *= 2.0;
    time(time.rows() - 1) *= 2.0;

    PolynomialTraj gl_traj;
    if (pos.cols() >= 3)
      gl_traj = PolynomialTraj::minSnapTraj(pos, start_vel, end_vel, start_acc, end_acc, time);
    else if (pos.cols() == 2)
      gl_traj = PolynomialTraj::one_segment_traj_gen(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, time(0));
    else
      return false;

    auto time_now = node_->now();
    global_data_.setGlobalTraj(gl_traj, time_now);

    return true;
  }

  bool SCANPlannerManager::refineTrajAlgo(UniformBspline &traj, vector<Eigen::Vector3d> &start_end_derivative, double ratio, double &ts, Eigen::MatrixXd &optimal_control_points)
  {
    double t_inc;

    Eigen::MatrixXd ctrl_pts; // = traj.getControlPoint()

    // std::cout << "ratio: " << ratio << std::endl;
    reparamBspline(traj, start_end_derivative, ratio, ctrl_pts, ts, t_inc);

    traj = UniformBspline(ctrl_pts, 3, ts);

    double t_step = traj.getTimeSum() / (ctrl_pts.cols() - 3);
    bspline_optimizer_rebound_->ref_pts_.clear();
    for (double t = 0; t < traj.getTimeSum() + 1e-4; t += t_step)
      bspline_optimizer_rebound_->ref_pts_.push_back(traj.evaluateDeBoorT(t));

    bool success = bspline_optimizer_rebound_->BsplineOptimizeTrajRefine(ctrl_pts, ts, optimal_control_points);

    return success;
  }

  void SCANPlannerManager::updateTrajInfo(const UniformBspline &position_traj, const rclcpp::Time time_now)
  {
    local_data_.start_time_ = time_now;
    local_data_.position_traj_ = position_traj;
    local_data_.velocity_traj_ = local_data_.position_traj_.getDerivative();
    local_data_.acceleration_traj_ = local_data_.velocity_traj_.getDerivative();
    local_data_.start_pos_ = local_data_.position_traj_.evaluateDeBoorT(0.0);
    local_data_.duration_ = local_data_.position_traj_.getTimeSum();
    local_data_.traj_id_ += 1;
    local_data_.reference_corridor_checked_ = false;
    local_data_.reference_corridor_safe_ = false;
    local_data_.maximum_trajectory_deviation_ = 0.0;
    local_data_.maximum_guide_anchor_deviation_ = 0.0;
    local_data_.maximum_guide_progress_lead_ = 0.0;
    local_data_.reference_corridor_deviation_limit_ = 0.0;
    local_data_.reference_corridor_progress_lead_limit_ = 0.0;
    local_data_.ordered_reference_guide_.clear();
  }

  bool SCANPlannerManager::checkDynamicFeasibility(UniformBspline position_traj)
  {
    if (!trajectoryTimingStateIsValid(position_traj))
      return false;
    const double duration = position_traj.getTimeSum();
    const double vel_limit = pp_.max_vel_ + pp_.vel_tolerance_;
    const double acc_limit = pp_.max_acc_ + pp_.acc_tolerance_;
    if (!std::isfinite(duration) || duration <= 0.0 ||
        !std::isfinite(vel_limit) || vel_limit <= 0.0 ||
        !std::isfinite(acc_limit) || acc_limit <= 0.0)
      return false;

    UniformBspline vel_traj = position_traj.getDerivative();
    UniformBspline acc_traj = vel_traj.getDerivative();
    if (!vel_traj.getControlPoint().allFinite() ||
        !vel_traj.getKnot().allFinite() ||
        !acc_traj.getControlPoint().allFinite() ||
        !acc_traj.getKnot().allFinite())
      return false;
    const double sample_dt =
        std::max(0.01, std::min(0.05, duration / 50.0));

    for (double t = 0.0; t < duration + 1e-6; t += sample_dt)
    {
      const double tc = std::min(t, duration);
      Eigen::Vector3d vel = vel_traj.evaluateDeBoorT(tc);
      const double velocity_norm = vel.norm();
      if (!vel.allFinite() || !std::isfinite(velocity_norm) ||
          velocity_norm > vel_limit)
      {
        RCLCPP_WARN(node_->get_logger(),
                    "Dynamic feasibility failed: velocity at t=%.3f is %.3f > %.3f",
                    tc, velocity_norm, vel_limit);
        return false;
      }

      Eigen::Vector3d acc = acc_traj.evaluateDeBoorT(tc);
      const double acceleration_norm = acc.norm();
      if (!acc.allFinite() || !std::isfinite(acceleration_norm) ||
          acceleration_norm > acc_limit)
      {
        RCLCPP_WARN(node_->get_logger(),
                    "Dynamic feasibility failed: acceleration at t=%.3f is %.3f > %.3f",
                    tc, acceleration_norm, acc_limit);
        return false;
      }
    }

    return true;
  }

  bool SCANPlannerManager::checkTrajectoryCollisionFree(
      UniformBspline position_traj) const
  {
    if (!grid_map_)
      return false;
    const double duration = position_traj.getTimeSum();
    if (!std::isfinite(duration) || duration <= 0.0)
      return false;

    // 碰撞采样必须只由空间曲线决定，不能随统一时间缩放改变。旧实现用
    // 固定秒数差分；零速起步在缩放后会把极小但有效的切向误判为零，继而
    // 回退到 yaw=0，把沿 -Y 的二楼窄走廊轨迹横着检查并错误拒绝。
    UniformBspline velocity_traj = position_traj.getDerivative();
    const Eigen::MatrixXd velocity_control_points =
        velocity_traj.getControlPoint();
    if (velocity_control_points.rows() != 3 ||
        velocity_control_points.cols() < 1 ||
        !velocity_control_points.allFinite())
      return false;
    double maximum_velocity_bound = 0.0;
    for (Eigen::Index column = 0;
         column < velocity_control_points.cols(); ++column)
      maximum_velocity_bound = std::max(
          maximum_velocity_bound,
          velocity_control_points.col(column).norm());
    if (!std::isfinite(maximum_velocity_bound) ||
        maximum_velocity_bound <= 1.0e-9)
      return false;

    const double spatial_sample_distance = std::max(
        0.0025, std::min(0.01, 0.20 * grid_map_->getResolution()));
    const int sample_count = std::max(
        2,
        static_cast<int>(std::ceil(
            duration * maximum_velocity_bound /
            spatial_sample_distance)));
    std::vector<Eigen::Vector3d> positions;
    positions.reserve(static_cast<std::size_t>(sample_count + 1));
    for (int index = 0; index <= sample_count; ++index)
    {
      const double current_time =
          duration * static_cast<double>(index) /
          static_cast<double>(sample_count);
      positions.push_back(position_traj.evaluateDeBoorT(current_time));
      if (!positions.back().allFinite())
        return false;
    }

    const double minimum_tangent_chord =
        std::max(1.0e-4, 0.5 * spatial_sample_distance);
    for (int index = 0; index <= sample_count; ++index)
    {
      int previous_index = std::max(0, index - 1);
      int next_index = std::min(sample_count, index + 1);
      Eigen::Vector2d tangent =
          (positions[static_cast<std::size_t>(next_index)] -
           positions[static_cast<std::size_t>(previous_index)]).head<2>();
      while (tangent.norm() < minimum_tangent_chord &&
             (previous_index > 0 || next_index < sample_count))
      {
        previous_index = std::max(0, previous_index - 1);
        next_index = std::min(sample_count, next_index + 1);
        tangent =
            (positions[static_cast<std::size_t>(next_index)] -
             positions[static_cast<std::size_t>(previous_index)]).head<2>();
      }
      if (!tangent.allFinite() || tangent.squaredNorm() <= 1.0e-12)
      {
        RCLCPP_WARN(
            node_->get_logger(),
            "局部轨迹在采样 %d/%d 没有可判定的平面切向，拒绝碰撞审计",
            index, sample_count);
        return false;
      }
      const double yaw = std::atan2(tangent(1), tangent(0));
      const Eigen::Vector3d &position =
          positions[static_cast<std::size_t>(index)];
      const int occupancy =
          grid_map_->getInflateOccupancy(position, yaw);
      if (occupancy != 0)
      {
        RCLCPP_WARN(
            node_->get_logger(),
            "局部轨迹碰撞：采样=%d/%d，位置=(%.4f, %.4f, %.4f)，"
            "空间切向 yaw=%.4f，弦长=%.6f m，占据=%d",
            index, sample_count,
            position.x(), position.y(), position.z(), yaw,
            tangent.norm(), occupancy);
        return false;
      }
    }
    return true;
  }

  bool SCANPlannerManager::checkReferenceGuideCollisionFree(
      const std::vector<Eigen::Vector3d> &reference_guide) const
  {
    if (!grid_map_ || reference_guide.size() < 2)
      return false;
    const double sample_distance =
        std::max(0.01, 0.5 * grid_map_->getResolution());
    for (std::size_t index = 0; index + 1 < reference_guide.size(); ++index)
    {
      const Eigen::Vector3d &start = reference_guide[index];
      const Eigen::Vector3d &end = reference_guide[index + 1];
      if (!start.allFinite() || !end.allFinite())
        return false;
      const Eigen::Vector3d delta = end - start;
      const double length = delta.norm();
      if (!std::isfinite(length) || length <= 1.0e-9)
        return false;
      const double yaw = std::atan2(delta(1), delta(0));
      const int sample_count =
          std::max(1, static_cast<int>(std::ceil(length / sample_distance)));
      for (int sample = 0; sample <= sample_count; ++sample)
      {
        const double ratio =
            static_cast<double>(sample) / static_cast<double>(sample_count);
        const Eigen::Vector3d position = start + ratio * delta;
        if (grid_map_->getInflateOccupancy(position, yaw) != 0)
          return false;
      }
    }
    return true;
  }

  void SCANPlannerManager::reparamBspline(UniformBspline &bspline, vector<Eigen::Vector3d> &start_end_derivative, double ratio,
                                         Eigen::MatrixXd &ctrl_pts, double &dt, double &time_inc)
  {
    double time_origin = bspline.getTimeSum();
    int seg_num = bspline.getControlPoint().cols() - 3;
    // double length = bspline.getLength(0.1);
    // int seg_num = ceil(length / pp_.ctrl_pt_dist);

    bspline.lengthenTime(ratio);
    double duration = bspline.getTimeSum();
    dt = duration / double(seg_num);
    time_inc = duration - time_origin;

    vector<Eigen::Vector3d> point_set;
    for (double time = 0.0; time <= duration + 1e-4; time += dt)
    {
      point_set.push_back(bspline.evaluateDeBoorT(time));
    }
    UniformBspline::parameterizeToBspline(dt, point_set, start_end_derivative, ctrl_pts);
  }

} // namespace scan_planner
