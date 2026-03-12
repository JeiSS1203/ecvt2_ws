#include "ecvt2_controller/kdl_utils.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <Eigen/Geometry>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ecvt2_controller
{

  // =============================
  //  회전 / 평행이동 유틸 함수
  // =============================
  Eigen::Matrix4d RotX(double q)
  {
    double c = std::cos(q), s = std::sin(q);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(1, 1) = c;
    T(1, 2) = -s;
    T(2, 1) = s;
    T(2, 2) = c;
    return T;
  }

  Eigen::Matrix4d RotY(double q)
  {
    double c = std::cos(q), s = std::sin(q);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 0) = c;
    T(0, 2) = s;
    T(2, 0) = -s;
    T(2, 2) = c;
    return T;
  }

  Eigen::Matrix4d RotZ(double q)
  {
    double c = std::cos(q), s = std::sin(q);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 0) = c;
    T(0, 1) = -s;
    T(1, 0) = s;
    T(1, 1) = c;
    return T;
  }

  Eigen::Matrix4d Trans(double x, double y, double z)
  {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = x;
    T(1, 3) = y;
    T(2, 3) = z;
    return T;
  }

  void printMatrixPretty(const Eigen::MatrixXd &M, const std::string &name)
  {
    const int width = 12;    // 열 간격
    const int precision = 6; // 소수점 자리수

    std::cout << name << ":\n";
    for (int i = 0; i < M.rows(); ++i)
    {
      for (int j = 0; j < M.cols(); ++j)
      {
        std::cout << std::setw(width) << std::fixed << std::setprecision(precision)
                  << M(i, j);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  Eigen::Matrix3d eulerZYXToRot(double yaw, double pitch, double roll)
  {
    const double cz = std::cos(yaw), sz = std::sin(yaw);
    const double cy = std::cos(pitch), sy = std::sin(pitch);
    const double cx = std::cos(roll), sx = std::sin(roll);

    Eigen::Matrix3d Rz;
    Rz << cz, -sz, 0,
        sz, cz, 0,
        0, 0, 1;
    Eigen::Matrix3d Ry;
    Ry << cy, 0, sy,
        0, 1, 0,
        -sy, 0, cy;
    Eigen::Matrix3d Rx;
    Rx << 1, 0, 0,
        0, cx, -sx,
        0, sx, cx;
    return Rz * Ry * Rx;
  }

  // ---- small helper: apply Pc operator without forming Pc ----
  // Pc(v) = v - Jc^T * ( (Jc*Jc^T + lambda I)^-1 * (Jc*v) )
  static inline Eigen::VectorXd applyPc(
      const Eigen::MatrixXd &Jc,
      const Eigen::LDLT<Eigen::MatrixXd> &S_ldlt,
      const Eigen::Ref<const Eigen::VectorXd> &v)
  {
    // y = Jc * v   (12x1)
    Eigen::VectorXd y = Jc * v;
    // x = S^{-1} y (12x1)
    Eigen::VectorXd x = S_ldlt.solve(y);
    // Pc(v) = v - Jc^T x (22x1)
    return v - Jc.transpose() * x;
  }

  // ---- small helper: build diagonal of W^{-1} without forming W ----
  static inline Eigen::VectorXd buildWinvDiag(const Eigen::Ref<const Eigen::VectorXd> &q)
  {
    // W is 22x22 identity except indices 6..21 (16 leg joints)
    // We return winv_diag such that (W^{-1} * v) == winv_diag.array() * v.array()
    Eigen::VectorXd winv = Eigen::VectorXd::Ones(22);

    Eigen::VectorXd q_legs(16), q_min(16), q_max(16);
    q_legs << q[0], q[1], q[2], q[3],
        q[5], q[6], q[7], q[8],
        q[10], q[11], q[12], q[13],
        q[15], q[16], q[17], q[18];

    const double deg2rad = M_PI / 180.0;
    q_min << -49 * deg2rad, -58.4 * deg2rad, -55.9 * deg2rad, -180 * deg2rad, // FR
        -28.9 * deg2rad, -58.4 * deg2rad, -55.9 * deg2rad, -180 * deg2rad,    // FL
        -28.9 * deg2rad, -56.9 * deg2rad, -56.3 * deg2rad, -180 * deg2rad,    // RR
        -49 * deg2rad, -56.9 * deg2rad, -56.3 * deg2rad, -180 * deg2rad;      // RL

    q_max << 28.9 * deg2rad, 56.9 * deg2rad, 56.3 * deg2rad, 180 * deg2rad, // FR
        49 * deg2rad, 56.9 * deg2rad, 56.3 * deg2rad, 180 * deg2rad,        // FL
        49 * deg2rad, 58.4 * deg2rad, 55.9 * deg2rad, 180 * deg2rad,        // RR
        28.9 * deg2rad, 58.4 * deg2rad, 55.9 * deg2rad, 180 * deg2rad;      // RL

    const Eigen::VectorXd c = 0.5 * (q_max + q_min);
    const Eigen::VectorXd r = 0.5 * (q_max - q_min);
    const Eigen::VectorXd s = (q_legs - c).cwiseQuotient(r);

    // Joint-limit barrier shape parameters.
    // - k: penalty strength near normalized limit |s| -> 1
    // - eps: avoids singularity at denominator (1 - s^2)
    //
    // Effect:
    // - larger k => manipulability metric de-emphasizes near-limit leg motion more aggressively
    const double k = 20.0;
    const double eps = 1e-4;

    // original: W(i,i)=1 + k * (( (s^2)/(1-s^2+eps) )^2)
    // we store winv(i)=1/W(i,i)
    for (int i = 0; i < 16; ++i)
    {
      const double si = s(i);
      const double si2 = si * si;
      const double denom = (1.0 - si2 + eps);
      const double frac = (si2 / denom);
      const double Wii = 1.0 + k * (frac * frac);
      winv(6 + i) = 1.0 / Wii;
    }

    return winv;
  }

  // ============================================================
  //  Optimized computeManipulabilityMatrix (same signature)
  //
  //  목표 quantity:
  //    A = Jb * Pc * W^{-1} * Pc * Jb^T   (3x3)
  //
  //  where
  //    Jb: base translation Jacobian part (3x22)
  //    Pc: contact-consistent projection
  //        Pc(v) = v - Jc^T (Jc Jc^T + lambda I)^{-1} Jc v
  //    W : joint-limit aware metric (diagonal)
  //
  //  구현 최적화:
  //  - Pc matrix를 직접 만들지 않고 Pc(v) 연산으로만 처리
  //  - W^{-1}는 diagonal scaling으로 처리
  //  - (JcJc^T + lambdaI)^{-1}는 LDLT solve 사용
  // ============================================================
  Eigen::Matrix3d computeManipulabilityMatrix(
      const FKJacobianResult &result, const Eigen::Ref<const Eigen::VectorXd> &q)
  {
    // Assemble J_b (3x22) and J_c (12x22) exactly as original mapping
    Eigen::MatrixXd J_b = Eigen::MatrixXd::Zero(3, 22);
    Eigen::MatrixXd J_c = Eigen::MatrixXd::Zero(12, 22);

    // Base (3x6)
    J_b.block<3, 6>(0, 0) = result.J[0].block<3, 6>(0, 0);

    // Contacts: FR, FL, RR, RL
    // FR rows 0..2: base cols 0..5 + FR leg cols 6..9
    J_c.block<3, 6>(0, 0) = result.J[1].block<3, 6>(0, 0);
    J_c.block<3, 4>(0, 6) = result.J[1].block<3, 4>(0, 6);

    // FL rows 3..5: base cols 0..5 + FL leg cols 10..13
    J_c.block<3, 6>(3, 0) = result.J[2].block<3, 6>(0, 0);
    J_c.block<3, 4>(3, 10) = result.J[2].block<3, 4>(0, 6);

    // RR rows 6..8: base cols 0..5 + RR leg cols 14..17
    J_c.block<3, 6>(6, 0) = result.J[3].block<3, 6>(0, 0);
    J_c.block<3, 4>(6, 14) = result.J[3].block<3, 4>(0, 6);

    // RL rows 9..11: base cols 0..5 + RL leg cols 18..21
    J_c.block<3, 6>(9, 0) = result.J[4].block<3, 6>(0, 0);
    J_c.block<3, 4>(9, 18) = result.J[4].block<3, 4>(0, 6);

    // W^{-1} diagonal (22)
    const Eigen::VectorXd winv = buildWinvDiag(q);

    // S = Jc*Jc^T + lambda I (12x12), factorize once
    constexpr double lambda = 1e-6;
    Eigen::MatrixXd S = J_c * J_c.transpose();
    S.diagonal().array() += lambda;

    Eigen::LDLT<Eigen::MatrixXd> S_ldlt(S);

    // Build A column-wise by injecting each basis direction e_i.
    // This is equivalent to evaluating
    //   A[:,i] = Jb * Pc * W^{-1} * Pc * Jb^T * e_i
    // without constructing dense intermediate operators.
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();

    for (int i = 0; i < 3; ++i)
    {
      // u = Jb^T e_i  (22x1) -> i-th column of Jb^T
      Eigen::VectorXd u = J_b.transpose().col(i);

      // w = Pc(u)
      Eigen::VectorXd w = applyPc(J_c, S_ldlt, u);

      // z = W^{-1} w  (diag scaling)
      Eigen::VectorXd z = winv.array() * w.array();

      // t = Pc(z)
      Eigen::VectorXd t = applyPc(J_c, S_ldlt, z);

      // A_col = Jb * t
      A.col(i) = (J_b * t);
    }

    return A;
  }

  // ---- helper: stable damped pseudo-inverse for 3x4 J (same form you used) ----
  // J_pinv = J^T * (J J^T + mu I)^{-1}
  static inline Eigen::MatrixXd pinv3x4_damped(const Eigen::MatrixXd &J, double mu)
  {
    // J: 3x4
    Eigen::Matrix3d M = (J * J.transpose()).eval();
    M.diagonal().array() += mu;

    Eigen::LLT<Eigen::Matrix3d> llt(M);
    // Solve (J J^T + mu I)^{-1} for I3, then left-multiply by J^T
    Eigen::Matrix3d Minv = llt.solve(Eigen::Matrix3d::Identity());
    return J.transpose() * Minv; // 4x3
  }

  static inline void ensureSizeZero(Eigen::MatrixXd &M, int r, int c)
  {
    if (M.rows() != r || M.cols() != c)
      M.resize(r, c);
    M.setZero();
  }

  // revolute joint around local Z axis of frame Tj, expressed in world
  static inline void addRevoluteZCol(Eigen::MatrixXd &J, int col,
                                     const Eigen::Matrix4d &Tj_world,
                                     const Eigen::Vector3d &p_end_world)
  {
    const Eigen::Vector3d z = Tj_world.block<3, 1>(0, 2);
    const Eigen::Vector3d p = Tj_world.block<3, 1>(0, 3);
    J.block<3, 1>(0, col) = z.cross(p_end_world - p);
    J.block<3, 1>(3, col) = z;
  }

  // ======================================
  //  베이스 위치 + 자세(Quaternion) → 4x4
  // ======================================
  static Eigen::Matrix4d makeBaseT(const Eigen::Ref<const Eigen::Vector3d> &pos,
                                   const Eigen::Ref<const Eigen::Vector4d> &quat_xyzw)
  {
    // Eigen은 (w,x,y,z) 순서를 사용하므로 입력 xyzw에서 순서를 맞춰줌
    Eigen::Quaterniond q(quat_xyzw(3), quat_xyzw(0), quat_xyzw(1), quat_xyzw(2));

    if (q.norm() == 0.0)
    {
      throw std::invalid_argument("base_quat norm is zero");
    }
    q.normalize();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.toRotationMatrix(); // 회전
    T.block<3, 1>(0, 3) = pos;                  // 위치
    return T;
  }

  struct PinocchioTarget
  {
    pinocchio::FrameIndex frame_id{0};
    std::vector<Eigen::Index> joint_v_cols;
    bool use_leg_mimic_projection{false};
  };

  struct PinocchioLegCols
  {
    Eigen::Index j1{-1};
    Eigen::Index j2{-1};
    Eigen::Index j3{-1};
    Eigen::Index j4{-1};
    Eigen::Index j5{-1};
    Eigen::Index jw{-1};
  };

  struct PinocchioContext
  {
    pinocchio::Model model;
    std::unique_ptr<pinocchio::Data> data;
    Eigen::VectorXd q;
    Eigen::MatrixXd jac_frame_full;
    std::array<PinocchioTarget, 5> targets;
    std::array<PinocchioLegCols, 4> leg_cols;
    std::array<Eigen::Index, 4> arm_cols{{-1, -1, -1, -1}};
  };

  static std::string resolvePinocchioUrdfPath(const std::string &urdf_path)
  {
    if (!urdf_path.empty())
      return urdf_path;

    std::string resolved =
      ament_index_cpp::get_package_share_directory("forestry_robot_mjcf") +
      "/xml/ecvt_v2_simplify.urdf";
    if (std::filesystem::exists(resolved))
      return resolved;

    const std::string src_fallback =
      "/home/harco/ecvt2_ws/src/forestry_robot_mjcf/xml/ecvt_v2_simplify.urdf";
    if (std::filesystem::exists(src_fallback))
      return src_fallback;

    throw std::runtime_error("Unable to locate ecvt_v2_simplify.urdf");
  }

  static Eigen::Index requireJointVCol(const pinocchio::Model &model, const std::string &joint_name)
  {
    const pinocchio::JointIndex jid = model.getJointId(joint_name);
    if (jid == 0)
      throw std::runtime_error("Pinocchio joint not found: " + joint_name);
    const auto &jm = model.joints[jid];
    if (jm.nv() != 1)
      throw std::runtime_error("Unsupported joint nv for: " + joint_name);
    return static_cast<Eigen::Index>(jm.idx_v());
  }

  static pinocchio::FrameIndex requireFrameId(const pinocchio::Model &model, const std::string &frame_name)
  {
    const pinocchio::FrameIndex id = model.getFrameId(frame_name);
    if (id >= model.frames.size())
      throw std::runtime_error("Pinocchio frame not found: " + frame_name);
    return id;
  }

  static PinocchioContext makePinocchioContext(const std::string &urdf_path)
  {
    PinocchioContext ctx;
    pinocchio::urdf::buildModel(
      resolvePinocchioUrdfPath(urdf_path),
      pinocchio::JointModelFreeFlyer(),
      ctx.model, false);
    ctx.data = std::make_unique<pinocchio::Data>(ctx.model);
    ctx.q = pinocchio::neutral(ctx.model);
    ctx.jac_frame_full = Eigen::MatrixXd::Zero(6, ctx.model.nv);

    auto fill_target = [&](PinocchioTarget &target,
                           const std::string &frame_name,
                           const std::vector<std::string> &joint_names,
                           bool use_leg_mimic) {
      target.frame_id = requireFrameId(ctx.model, frame_name);
      target.use_leg_mimic_projection = use_leg_mimic;
      target.joint_v_cols.clear();
      target.joint_v_cols.reserve(joint_names.size());
      for (const auto &joint_name : joint_names)
        target.joint_v_cols.push_back(requireJointVCol(ctx.model, joint_name));
    };

    fill_target(ctx.targets[0], "FRW", {"FRJ1", "FRJ2", "FRJ3", "FRJ4", "FRJ5", "FRJW"}, true);
    fill_target(ctx.targets[1], "FLW", {"FLJ1", "FLJ2", "FLJ3", "FLJ4", "FLJ5", "FLJW"}, true);
    fill_target(ctx.targets[2], "RRW", {"RRJ1", "RRJ2", "RRJ3", "RRJ4", "RRJ5", "RRJW"}, true);
    fill_target(ctx.targets[3], "RLW", {"RLJ1", "RLJ2", "RLJ3", "RLJ4", "RLJ5", "RLJW"}, true);
    fill_target(ctx.targets[4], "UP5", {"UPJ1", "UPJ2", "UPJ3", "UPJ4"}, false);

    auto set_leg_cols = [&](size_t i, const std::string &prefix) {
      ctx.leg_cols[i] = {
        requireJointVCol(ctx.model, prefix + "J1"),
        requireJointVCol(ctx.model, prefix + "J2"),
        requireJointVCol(ctx.model, prefix + "J3"),
        requireJointVCol(ctx.model, prefix + "J4"),
        requireJointVCol(ctx.model, prefix + "J5"),
        requireJointVCol(ctx.model, prefix + "JW")
      };
    };

    set_leg_cols(0, "FR");
    set_leg_cols(1, "FL");
    set_leg_cols(2, "RR");
    set_leg_cols(3, "RL");
    ctx.arm_cols[0] = requireJointVCol(ctx.model, "UPJ1");
    ctx.arm_cols[1] = requireJointVCol(ctx.model, "UPJ2");
    ctx.arm_cols[2] = requireJointVCol(ctx.model, "UPJ3");
    ctx.arm_cols[3] = requireJointVCol(ctx.model, "UPJ4");

    return ctx;
  }

  static PinocchioContext &getPinocchioContext(const std::string &urdf_path)
  {
    static std::unique_ptr<PinocchioContext> ctx;
    static std::string ctx_urdf;

    const std::string resolved = resolvePinocchioUrdfPath(urdf_path);
    if (!ctx || ctx_urdf != resolved)
    {
      ctx = std::make_unique<PinocchioContext>(makePinocchioContext(resolved));
      ctx_urdf = resolved;
    }
    return *ctx;
  }

  static void setJointIfExists(pinocchio::Model &model, Eigen::VectorXd &q,
                               const std::string &joint_name, double value)
  {
    const pinocchio::JointIndex jid = model.getJointId(joint_name);
    if (jid == 0)
      return;

    const auto &jmodel = model.joints[jid];
    if (jmodel.nq() == 1)
    {
      q(jmodel.idx_q()) = value;
      return;
    }
    if (jmodel.nq() == 2)
    {
      q(jmodel.idx_q()) = std::cos(value);
      q(jmodel.idx_q() + 1) = std::sin(value);
    }
  }

  static void mapLegToPinQ(pinocchio::Model &model, Eigen::VectorXd &q_pin,
                           const Eigen::Ref<const Eigen::VectorXd> &q,
                           const std::string &prefix,
                           const int idx_j1, const int idx_j2, const int idx_j3,
                           const int idx_j5, const int idx_jw)
  {
    const double j1 = q(idx_j1);
    const double j2 = q(idx_j2);
    const double j3 = q(idx_j3);
    const double j4 = -(j2 + j3);
    const double j5 = q(idx_j5);
    const double jw = q(idx_jw);

    setJointIfExists(model, q_pin, prefix + "J1", j1);
    setJointIfExists(model, q_pin, prefix + "J2", j2);
    setJointIfExists(model, q_pin, prefix + "J3", j3);
    setJointIfExists(model, q_pin, prefix + "J4", j4);
    setJointIfExists(model, q_pin, prefix + "J5", j5);
    setJointIfExists(model, q_pin, prefix + "JW", jw);
  }

  static void buildPinocchioQ(
    PinocchioContext &ctx,
    const Eigen::Ref<const Eigen::Vector3d> &base_pos,
    const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
    const Eigen::Ref<const Eigen::VectorXd> &q)
  {
    if (q.size() < 24)
      throw std::invalid_argument("Pinocchio q mapping requires at least 24 articulation elements.");

    ctx.q = pinocchio::neutral(ctx.model);
    ctx.q.segment<3>(0) = base_pos;

    Eigen::Vector4d quat = base_quat_xyzw;
    const double quat_norm = quat.norm();
    if (quat_norm < 1e-12)
      throw std::invalid_argument("base_quat_xyzw norm is zero.");
    quat /= quat_norm;
    ctx.q.segment<4>(3) = quat;

    mapLegToPinQ(ctx.model, ctx.q, q, "FR", 0, 1, 2, 3, 4);
    mapLegToPinQ(ctx.model, ctx.q, q, "FL", 5, 6, 7, 8, 9);
    mapLegToPinQ(ctx.model, ctx.q, q, "RR", 10, 11, 12, 13, 14);
    mapLegToPinQ(ctx.model, ctx.q, q, "RL", 15, 16, 17, 18, 19);

    setJointIfExists(ctx.model, ctx.q, "UPJ1", q(20));
    setJointIfExists(ctx.model, ctx.q, "UPJ2", q(21));
    setJointIfExists(ctx.model, ctx.q, "UPJ3", q(22));
    setJointIfExists(ctx.model, ctx.q, "UPJ4", q(23));
    if (q.size() > 24)
      setJointIfExists(ctx.model, ctx.q, "UPJ5", q(24));
    if (q.size() > 25)
      setJointIfExists(ctx.model, ctx.q, "UPJ6", q(25));
    if (q.size() > 26)
      setJointIfExists(ctx.model, ctx.q, "TOOLJ1", q(26));
  }

  static inline Eigen::Matrix<double, 6, 1> toLinearAngularColumn(
    const Eigen::Ref<const Eigen::MatrixXd> &jac_pin, const Eigen::Index col)
  {
    Eigen::Matrix<double, 6, 1> out;
    out.segment<3>(0) = jac_pin.block<3, 1>(0, col);
    out.segment<3>(3) = jac_pin.block<3, 1>(3, col);
    return out;
  }

  static Eigen::Matrix<double, 6, 10> computePinFrameJacobianKdlLayout(
    PinocchioContext &ctx, const PinocchioTarget &target)
  {
    pinocchio::getFrameJacobian(
      ctx.model, *ctx.data, target.frame_id,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      ctx.jac_frame_full);

    Eigen::MatrixXd full_linear_angular = Eigen::MatrixXd::Zero(6, ctx.model.nv);
    for (int col = 0; col < ctx.model.nv; ++col)
      full_linear_angular.col(col) = toLinearAngularColumn(ctx.jac_frame_full, static_cast<Eigen::Index>(col));

    Eigen::MatrixXd reduced = Eigen::MatrixXd::Zero(
      6, 6 + static_cast<Eigen::Index>(target.joint_v_cols.size()));
    reduced.block(0, 0, 6, 6) = full_linear_angular.block(0, 0, 6, 6);
    for (size_t i = 0; i < target.joint_v_cols.size(); ++i)
      reduced.col(6 + static_cast<Eigen::Index>(i)) = full_linear_angular.col(target.joint_v_cols[i]);

    Eigen::Matrix<double, 6, 10> out = Eigen::Matrix<double, 6, 10>::Zero();
    out.block<6, 6>(0, 0) = reduced.block(0, 0, 6, 6);

    if (target.use_leg_mimic_projection)
    {
      const Eigen::Matrix<double, 6, 1> J1 = reduced.col(6);
      const Eigen::Matrix<double, 6, 1> J2 = reduced.col(7);
      const Eigen::Matrix<double, 6, 1> J3 = reduced.col(8);
      const Eigen::Matrix<double, 6, 1> J4 = reduced.col(9);
      const Eigen::Matrix<double, 6, 1> J5 = reduced.col(10);
      out.col(6) = J1;
      out.col(7) = J2 - J4;
      out.col(8) = J3 - J4;
      out.col(9) = J5;
      return out;
    }

    for (int i = 0; i < 4; ++i)
      out.col(6 + i) = reduced.col(6 + i);
    return out;
  }

  static Eigen::Matrix<double, 3, 30> computeReducedComJacobian(
    const PinocchioContext &ctx, const Eigen::Ref<const Eigen::MatrixXd> &jcom_full)
  {
    Eigen::Matrix<double, 3, 30> out = Eigen::Matrix<double, 3, 30>::Zero();
    out.block<3, 6>(0, 0) = jcom_full.block(0, 0, 3, 6);

    int out_col = 6;
    for (const auto &leg : ctx.leg_cols)
    {
      out.col(out_col++) = jcom_full.col(leg.j1);
      out.col(out_col++) = jcom_full.col(leg.j2) - jcom_full.col(leg.j4);
      out.col(out_col++) = jcom_full.col(leg.j3) - jcom_full.col(leg.j4);
      out.col(out_col++) = jcom_full.col(leg.j5);
      out.col(out_col++) = jcom_full.col(leg.jw);
    }

    for (size_t i = 0; i < ctx.arm_cols.size(); ++i)
      out.col(out_col++) = jcom_full.col(ctx.arm_cols[i]);
    return out;
  }

  static Eigen::Matrix4d se3ToMatrix4(const pinocchio::SE3 &M)
  {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = M.rotation();
    T.block<3, 1>(0, 3) = M.translation();
    return T;
  }

  // // ===================================================
  // //  FK + Jacobian을 동시에 계산하는 메인 함수
  // // ===================================================
  // FKJacobianResult computeFKAllAndJacobian(const Eigen::Ref<const Eigen::Vector3d> &base_pos,
  //                                          const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
  //                                          const Eigen::Ref<const Eigen::VectorXd> &q)
  // {
  //   FKJacobianResult out;
  //   for (int i = 0; i < 6; ++i)
  //   {
  //     out.T[i] = Eigen::Matrix4d::Identity();
  //   }
  //   for (int i = 0; i < 6; ++i)
  //   {
  //     if (i == 0)
  //     {
  //       out.J[i].resize(6, 6);
  //       out.J[i].setZero();
  //     }
  //     else if (i <= 4)
  //     {
  //       out.J[i].resize(3, 10);
  //       out.J[i].setZero();
  //     }
  //     else /* i==5 */
  //     {
  //       out.J[i].resize(6, 10);
  //       out.J[i].setZero();
  //     }
  //   }
  //   // 베이스 좌표계 변환 (월드 → 로봇 베이스)
  //   const Eigen::Matrix4d T_world_base = makeBaseT(base_pos, base_quat_xyzw);
  //   // 베이스 부위의 FK, Jacobian
  //   // ---------------------- Base (index 0) ----------------------
  //   out.T[0] = T_world_base; // 베이스 변환행렬 저장
  //   Eigen::MatrixXd J_base = Eigen::MatrixXd::Zero(6, 6);
  //   J_base.block<3, 3>(0, 0) = T_world_base.block<3, 3>(0, 0);
  //   J_base.block<3, 3>(3, 3) = T_world_base.block<3, 3>(0, 0);
  //   out.J[0] = J_base; // 베이스 Jacobian 저장

  //   // ==================================================================
  //   // 아래 5개 블록에서 개별 부위(FR, FL, RR, RL, ARM)의
  //   // - FK : T_part를 계산 후, 최종 월드 좌표계로 변환해 out.T[i] 저장
  //   // - Jacobian : out.J[i]의 해당 부위 조인트 인덱스 열만 채우기
  //   // ==================================================================

  //   // T Matrix 선언
  //   Eigen::Matrix4d T_base_FR1, T_base_FL1, T_base_RR1, T_base_RL1,
  //       T_FR1_FR2, T_FR2_FR3, T_FR3_FR4, T_FR4_FR5, T_FR5_FRW,
  //       T_FL1_FL2, T_FL2_FL3, T_FL3_FL4, T_FL4_FL5, T_FL5_FLW,
  //       T_RR1_RR2, T_RR2_RR3, T_RR3_RR4, T_RR4_RR5, T_RR5_RRW,
  //       T_RL1_RL2, T_RL2_RL3, T_RL3_RL4, T_RL4_RL5, T_RL5_RLW,
  //       T_base_UP1, T_UP1_UP2, T_UP2_UP3, T_UP3_UP4, T_UP4_UP5, T_UP5_UP6, T_UP6_TOOL;

  //   Eigen::Matrix4d T_world_FR1, T_world_FL1, T_world_RR1, T_world_RL1,
  //       T_world_FR2, T_world_FR3, T_world_FR4, T_world_FR5, T_world_FRW,
  //       T_world_FL2, T_world_FL3, T_world_FL4, T_world_FL5, T_world_FLW,
  //       T_world_RR2, T_world_RR3, T_world_RR4, T_world_RR5, T_world_RRW,
  //       T_world_RL2, T_world_RL3, T_world_RL4, T_world_RL5, T_world_RLW,
  //       T_world_UP1, T_world_UP2, T_world_UP3, T_world_UP4, T_world_UP5, T_world_UP6, T_world_TOOL;

  //   // ---------------------- FR (index 1) ----------------------
  //   {
  //     // FR FK
  //     T_base_FR1 = Trans(1.2832, -0.9125, -0.1807) * RotZ(q[0]);
  //     T_FR1_FR2 = Trans(0.337, 0.0, -0.085) * RotX(M_PI / 2) * RotZ(q[1]);
  //     T_FR2_FR3 = Trans(1.4, 0.0, 0.0) * RotZ(q[2]);
  //     T_FR3_FR4 = Trans(0.0, -1.0, 0.0) * RotZ(-q[1] - q[2]);
  //     T_FR4_FR5 = Trans(0.015, -0.288, 0.0) * RotX(-M_PI / 2) * RotZ(q[3]);
  //     T_FR5_FRW = Trans(0.0, -0.66506, -0.171) * RotZ(M_PI) * RotX(M_PI / 2) * RotZ(q[4]);

  //     T_world_FR1 = T_world_base * T_base_FR1;
  //     T_world_FR2 = T_world_FR1 * T_FR1_FR2;
  //     T_world_FR3 = T_world_FR2 * T_FR2_FR3;
  //     T_world_FR4 = T_world_FR3 * T_FR3_FR4;
  //     T_world_FR5 = T_world_FR4 * T_FR4_FR5;
  //     T_world_FRW = T_world_FR5 * T_FR5_FRW;

  //     out.T[1] = T_world_FRW;

  //     // FR Jacobian
  //     Eigen::MatrixXd J_FR = Eigen::MatrixXd::Zero(6, 10);
  //     J_FR.block<3, 3>(0, 0) = T_world_base.block<3, 3>(0, 0);
  //     J_FR.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();

  //     Eigen::Vector3d base_to_FRW_in_world = T_world_FRW.block<3, 1>(0, 3) - T_world_base.block<3, 1>(0, 3);
  //     J_FR.block<3, 1>(0, 3) = T_world_base.block<3, 1>(0, 0).cross(base_to_FRW_in_world);
  //     J_FR.block<3, 1>(0, 4) = T_world_base.block<3, 1>(0, 1).cross(base_to_FRW_in_world);
  //     J_FR.block<3, 1>(0, 5) = T_world_base.block<3, 1>(0, 2).cross(base_to_FRW_in_world);
  //     J_FR.block<3, 3>(3, 3) = T_world_base.block<3, 3>(0, 0);

  //     J_FR.block<3, 1>(0, 6) = T_world_FR1.block<3, 1>(0, 2).cross(T_world_FRW.block<3, 1>(0, 3) - T_world_FR1.block<3, 1>(0, 3));                                                                                                      // z축 회전
  //     J_FR.block<3, 1>(0, 7) = T_world_FR2.block<3, 1>(0, 2).cross(T_world_FRW.block<3, 1>(0, 3) - T_world_FR2.block<3, 1>(0, 3)) - T_world_FR4.block<3, 1>(0, 2).cross(T_world_FRW.block<3, 1>(0, 3) - T_world_FR4.block<3, 1>(0, 3)); // mimic joint constraint 반영
  //     J_FR.block<3, 1>(0, 8) = T_world_FR3.block<3, 1>(0, 2).cross(T_world_FRW.block<3, 1>(0, 3) - T_world_FR3.block<3, 1>(0, 3)) - T_world_FR4.block<3, 1>(0, 2).cross(T_world_FRW.block<3, 1>(0, 3) - T_world_FR4.block<3, 1>(0, 3)); // mimic joint constraint 반영
  //     J_FR.block<3, 1>(0, 9) = T_world_FR5.block<3, 1>(0, 2).cross(T_world_FRW.block<3, 1>(0, 3) - T_world_FR5.block<3, 1>(0, 3));                                                                                                      // z축 회전

  //     J_FR.block<3, 1>(3, 6) = T_world_FR1.block<3, 1>(0, 2);
  //     J_FR.block<3, 1>(3, 7) = T_world_FR2.block<3, 1>(0, 2) - T_world_FR4.block<3, 1>(0, 2); // mimic joint constraint 반영
  //     J_FR.block<3, 1>(3, 8) = T_world_FR3.block<3, 1>(0, 2) - T_world_FR4.block<3, 1>(0, 2); // mimic joint constraint 반영
  //     J_FR.block<3, 1>(3, 9) = T_world_FR5.block<3, 1>(0, 2);
  //     out.J[1] = J_FR;
  //   }

  //   // ---------------------- FL (index 2) ----------------------
  //   {
  //     T_base_FL1 = Trans(1.2832, 0.9125, -0.1807) * RotZ(q[5]);
  //     T_FL1_FL2 = Trans(0.337, 0.0, -0.085) * RotX(M_PI / 2) * RotZ(q[6]);
  //     T_FL2_FL3 = Trans(1.4, 0.0, 0.0) * RotZ(q[7]);
  //     T_FL3_FL4 = Trans(0.0, -1.0, 0.0) * RotZ(-q[6] - q[7]);
  //     T_FL4_FL5 = Trans(0.015, -0.288, 0.0) * RotX(-M_PI / 2) * RotZ(q[8]);
  //     T_FL5_FLW = Trans(0.0, 0.66506, -0.171) * RotZ(M_PI) * RotX(M_PI / 2) * RotZ(q[9]);

  //     T_world_FL1 = T_world_base * T_base_FL1;
  //     T_world_FL2 = T_world_FL1 * T_FL1_FL2;
  //     T_world_FL3 = T_world_FL2 * T_FL2_FL3;
  //     T_world_FL4 = T_world_FL3 * T_FL3_FL4;
  //     T_world_FL5 = T_world_FL4 * T_FL4_FL5;
  //     T_world_FLW = T_world_FL5 * T_FL5_FLW;

  //     out.T[2] = T_world_FLW;
  //     // FL Jacobian
  //     Eigen::MatrixXd J_FL = Eigen::MatrixXd::Zero(6, 10);
  //     J_FL.block<3, 3>(0, 0) = T_world_base.block<3, 3>(0, 0);
  //     J_FL.block<3, 3>(3, 3) = Eigen::Matrix3d::Zero();

  //     Eigen::Vector3d base_to_FLW_in_world = T_world_FLW.block<3, 1>(0, 3) - T_world_base.block<3, 1>(0, 3);
  //     J_FL.block<3, 1>(0, 3) = T_world_base.block<3, 1>(0, 0).cross(base_to_FLW_in_world);
  //     J_FL.block<3, 1>(0, 4) = T_world_base.block<3, 1>(0, 1).cross(base_to_FLW_in_world);
  //     J_FL.block<3, 1>(0, 5) = T_world_base.block<3, 1>(0, 2).cross(base_to_FLW_in_world);
  //     J_FL.block<3, 3>(3, 3) = T_world_base.block<3, 3>(0, 0);

  //     J_FL.block<3, 1>(0, 6) = T_world_FL1.block<3, 1>(0, 2).cross(T_world_FLW.block<3, 1>(0, 3) - T_world_FL1.block<3, 1>(0, 3));                                                                                                      // z축 회전
  //     J_FL.block<3, 1>(0, 7) = T_world_FL2.block<3, 1>(0, 2).cross(T_world_FLW.block<3, 1>(0, 3) - T_world_FL2.block<3, 1>(0, 3)) - T_world_FL4.block<3, 1>(0, 2).cross(T_world_FLW.block<3, 1>(0, 3) - T_world_FL4.block<3, 1>(0, 3)); // mimic joint constraint 반영
  //     J_FL.block<3, 1>(0, 8) = T_world_FL3.block<3, 1>(0, 2).cross(T_world_FLW.block<3, 1>(0, 3) - T_world_FL3.block<3, 1>(0, 3)) - T_world_FL4.block<3, 1>(0, 2).cross(T_world_FLW.block<3, 1>(0, 3) - T_world_FL4.block<3, 1>(0, 3)); // mimic joint constraint 반영
  //     J_FL.block<3, 1>(0, 9) = T_world_FL5.block<3, 1>(0, 2).cross(T_world_FLW.block<3, 1>(0, 3) - T_world_FL5.block<3, 1>(0, 3));                                                                                                      // z축 회전

  //     J_FL.block<3, 1>(3, 6) = T_world_FL1.block<3, 1>(0, 2);
  //     J_FL.block<3, 1>(3, 7) = T_world_FL2.block<3, 1>(0, 2) - T_world_FL4.block<3, 1>(0, 2); // mimic joint constraint 반영
  //     J_FL.block<3, 1>(3, 8) = T_world_FL3.block<3, 1>(0, 2) - T_world_FL4.block<3, 1>(0, 2); // mimic joint constraint 반영
  //     J_FL.block<3, 1>(3, 9) = T_world_FL5.block<3, 1>(0, 2);

  //     out.J[2] = J_FL;
  //   }

  //   // ---------------------- RR (index 3) ----------------------
  //   {
  //     T_base_RR1 = Trans(-1.3668, -0.9125, -0.1807) * RotZ(M_PI) * RotZ(q[10]);
  //     T_RR1_RR2 = Trans(0.337, 0.0, -0.085) * RotX(-M_PI / 2) * RotZ(q[11]);
  //     T_RR2_RR3 = Trans(1.4, 0.0, 0.0) * RotZ(M_PI) * RotZ(q[12]);
  //     T_RR3_RR4 = Trans(0.0, -1.0, 0.0) * RotZ(-q[11] - q[12]);
  //     T_RR4_RR5 = Trans(-0.015, -0.288, 0.0) * RotZ(M_PI) * RotX(M_PI / 2) * RotZ(q[13]);
  //     T_RR5_RRW = Trans(0.0, 0.66506, -0.171) * RotX(M_PI / 2) * RotZ(q[14]);

  //     T_world_RR1 = T_world_base * T_base_RR1;
  //     T_world_RR2 = T_world_RR1 * T_RR1_RR2;
  //     T_world_RR3 = T_world_RR2 * T_RR2_RR3;
  //     T_world_RR4 = T_world_RR3 * T_RR3_RR4;
  //     T_world_RR5 = T_world_RR4 * T_RR4_RR5;
  //     T_world_RRW = T_world_RR5 * T_RR5_RRW;

  //     out.T[3] = T_world_RRW;

  //     // RR Jacobian
  //     Eigen::MatrixXd J_RR = Eigen::MatrixXd::Zero(6, 10);
  //     J_RR.block<3, 3>(0, 0) = T_world_base.block<3, 3>(0, 0);
  //     J_RR.block<3, 3>(3, 3) = Eigen::Matrix3d::Zero();

  //     Eigen::Vector3d base_to_RRW_in_world = T_world_RRW.block<3, 1>(0, 3) - T_world_base.block<3, 1>(0, 3);
  //     J_RR.block<3, 1>(0, 3) = T_world_base.block<3, 1>(0, 0).cross(base_to_RRW_in_world);
  //     J_RR.block<3, 1>(0, 4) = T_world_base.block<3, 1>(0, 1).cross(base_to_RRW_in_world);
  //     J_RR.block<3, 1>(0, 5) = T_world_base.block<3, 1>(0, 2).cross(base_to_RRW_in_world);
  //     J_RR.block<3, 3>(3, 3) = T_world_base.block<3, 3>(0, 0);

  //     J_RR.block<3, 1>(0, 6) = T_world_RR1.block<3, 1>(0, 2).cross(T_world_RRW.block<3, 1>(0, 3) - T_world_RR1.block<3, 1>(0, 3));                                                                                                      // z축 회전
  //     J_RR.block<3, 1>(0, 7) = T_world_RR2.block<3, 1>(0, 2).cross(T_world_RRW.block<3, 1>(0, 3) - T_world_RR2.block<3, 1>(0, 3)) - T_world_RR4.block<3, 1>(0, 2).cross(T_world_RRW.block<3, 1>(0, 3) - T_world_RR4.block<3, 1>(0, 3)); // mimic joint constraint 반영
  //     J_RR.block<3, 1>(0, 8) = T_world_RR3.block<3, 1>(0, 2).cross(T_world_RRW.block<3, 1>(0, 3) - T_world_RR3.block<3, 1>(0, 3)) - T_world_RR4.block<3, 1>(0, 2).cross(T_world_RRW.block<3, 1>(0, 3) - T_world_RR4.block<3, 1>(0, 3)); // mimic joint constraint 반영
  //     J_RR.block<3, 1>(0, 9) = T_world_RR5.block<3, 1>(0, 2).cross(T_world_RRW.block<3, 1>(0, 3) - T_world_RR5.block<3, 1>(0, 3));                                                                                                      // z축 회전

  //     J_RR.block<3, 1>(3, 6) = T_world_RR1.block<3, 1>(0, 2);
  //     J_RR.block<3, 1>(3, 7) = T_world_RR2.block<3, 1>(0, 2) - T_world_RR4.block<3, 1>(0, 2); // mimic joint constraint 반영
  //     J_RR.block<3, 1>(3, 8) = T_world_RR3.block<3, 1>(0, 2) - T_world_RR4.block<3, 1>(0, 2); // mimic joint constraint 반영
  //     J_RR.block<3, 1>(3, 9) = T_world_RR5.block<3, 1>(0, 2);

  //     out.J[3] = J_RR;
  //   }

  //   // ---------------------- RL (index 4) ----------------------
  //   {
  //     T_base_RL1 = Trans(-1.3668, 0.9125, -0.1807) * RotZ(M_PI) * RotZ(q[15]);
  //     T_RL1_RL2 = Trans(0.337, 0.0, -0.085) * RotX(-M_PI / 2) * RotZ(q[16]);
  //     T_RL2_RL3 = Trans(1.4, 0.0, 0.0) * RotZ(M_PI) * RotZ(q[17]);
  //     T_RL3_RL4 = Trans(0.0, -1.0, 0.0) * RotZ(-q[16] - q[17]);
  //     T_RL4_RL5 = Trans(-0.015, -0.288, 0.0) * RotZ(M_PI) * RotX(M_PI / 2) * RotZ(q[18]);
  //     T_RL5_RLW = Trans(0.0, -0.66506, -0.171) * RotX(M_PI / 2) * RotZ(q[19]);

  //     T_world_RL1 = T_world_base * T_base_RL1;
  //     T_world_RL2 = T_world_RL1 * T_RL1_RL2;
  //     T_world_RL3 = T_world_RL2 * T_RL2_RL3;
  //     T_world_RL4 = T_world_RL3 * T_RL3_RL4;
  //     T_world_RL5 = T_world_RL4 * T_RL4_RL5;
  //     T_world_RLW = T_world_RL5 * T_RL5_RLW;

  //     out.T[4] = T_world_RLW;

  //     // RL Jacobian
  //     Eigen::MatrixXd J_RL = Eigen::MatrixXd::Zero(6, 10);
  //     J_RL.block<3, 3>(0, 0) = T_world_base.block<3, 3>(0, 0);
  //     J_RL.block<3, 3>(3, 3) = Eigen::Matrix3d::Zero();

  //     Eigen::Vector3d base_to_RLW_in_world = T_world_RLW.block<3, 1>(0, 3) - T_world_base.block<3, 1>(0, 3);
  //     J_RL.block<3, 1>(0, 3) = T_world_base.block<3, 1>(0, 0).cross(base_to_RLW_in_world);
  //     J_RL.block<3, 1>(0, 4) = T_world_base.block<3, 1>(0, 1).cross(base_to_RLW_in_world);
  //     J_RL.block<3, 1>(0, 5) = T_world_base.block<3, 1>(0, 2).cross(base_to_RLW_in_world);
  //     J_RL.block<3, 3>(3, 3) = T_world_base.block<3, 3>(0, 0);

  //     J_RL.block<3, 1>(0, 6) = T_world_RL1.block<3, 1>(0, 2).cross(T_world_RLW.block<3, 1>(0, 3) - T_world_RL1.block<3, 1>(0, 3));                                                                                                      // z축 회전
  //     J_RL.block<3, 1>(0, 7) = T_world_RL2.block<3, 1>(0, 2).cross(T_world_RLW.block<3, 1>(0, 3) - T_world_RL2.block<3, 1>(0, 3)) - T_world_RL4.block<3, 1>(0, 2).cross(T_world_RLW.block<3, 1>(0, 3) - T_world_RL4.block<3, 1>(0, 3)); // mimic joint constraint 반영
  //     J_RL.block<3, 1>(0, 8) = T_world_RL3.block<3, 1>(0, 2).cross(T_world_RLW.block<3, 1>(0, 3) - T_world_RL3.block<3, 1>(0, 3)) - T_world_RL4.block<3, 1>(0, 2).cross(T_world_RLW.block<3, 1>(0, 3) - T_world_RL4.block<3, 1>(0, 3)); // mimic joint constraint 반영
  //     J_RL.block<3, 1>(0, 9) = T_world_RL5.block<3, 1>(0, 2).cross(T_world_RLW.block<3, 1>(0, 3) - T_world_RL5.block<3, 1>(0, 3));                                                                                                      // z축 회전

  //     J_RL.block<3, 1>(3, 6) = T_world_RL1.block<3, 1>(0, 2);
  //     J_RL.block<3, 1>(3, 7) = T_world_RL2.block<3, 1>(0, 2) - T_world_RL4.block<3, 1>(0, 2); // mimic joint constraint 반영
  //     J_RL.block<3, 1>(3, 8) = T_world_RL3.block<3, 1>(0, 2) - T_world_RL4.block<3, 1>(0, 2); // mimic joint constraint 반영
  //     J_RL.block<3, 1>(3, 9) = T_world_RL5.block<3, 1>(0, 2);

  //     out.J[4] = J_RL;
  //   }

  //   // ---------------------- ARM (index 5) ----------------------
  //   {
  //     T_base_UP1 = Trans(0.370195, 0.0, 0.345303) * RotZ(q[20]);
  //     T_UP1_UP2 = Trans(-0.35, 0.006485, 1.402) * RotX(M_PI / 2) * RotZ(q[21]);
  //     T_UP2_UP3 = Trans(4.0, -0.065, 0.001927) * RotZ(q[22]);
  //     T_UP3_UP4 = Trans(0.2, -0.02, -0.00094) * RotY(M_PI / 2) * Trans(0.0, 0.0, q[23]); // prismatic joint
  //     T_UP4_UP5 = Trans(-0.00227, -0.065809, 3.158) * RotY(M_PI / 2) * RotX(M_PI) * RotZ(q[24]);
  //     T_UP5_UP6 = Trans(0.0, 0.16, -0.00227) * RotY(-M_PI / 2) * RotZ(q[25]);
  //     T_UP6_TOOL = Trans(0.0, 0.343, 0.0) * RotY(M_PI / 2) * RotX(M_PI / 2) * RotZ(q[26]);

  //     T_world_UP1 = T_world_base * T_base_UP1;
  //     T_world_UP2 = T_world_UP1 * T_UP1_UP2;
  //     T_world_UP3 = T_world_UP2 * T_UP2_UP3;
  //     T_world_UP4 = T_world_UP3 * T_UP3_UP4;
  //     T_world_UP5 = T_world_UP4 * T_UP4_UP5;
  //     T_world_UP6 = T_world_UP5 * T_UP5_UP6;
  //     T_world_TOOL = T_world_UP6 * T_UP6_TOOL;

  //     out.T[5] = T_world_UP5;

  //     Eigen::MatrixXd J_ARM = Eigen::MatrixXd::Zero(6, 10);
  //     J_ARM.block<3, 3>(0, 0) = T_world_base.block<3, 3>(0, 0);
  //     J_ARM.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();

  //     Eigen::Vector3d base_to_UP5_in_world = T_world_UP5.block<3, 1>(0, 3) - T_world_base.block<3, 1>(0, 3);
  //     J_ARM.block<3, 1>(0, 3) = T_world_base.block<3, 1>(0, 0).cross(base_to_UP5_in_world);
  //     J_ARM.block<3, 1>(0, 4) = T_world_base.block<3, 1>(0, 1).cross(base_to_UP5_in_world);
  //     J_ARM.block<3, 1>(0, 5) = T_world_base.block<3, 1>(0, 2).cross(base_to_UP5_in_world);
  //     J_ARM.block<3, 3>(3, 3) = T_world_base.block<3, 3>(0, 0);

  //     J_ARM.block<3, 1>(0, 6) = T_world_UP1.block<3, 1>(0, 2).cross(T_world_UP5.block<3, 1>(0, 3) - T_world_UP1.block<3, 1>(0, 3)); // z축 회전
  //     J_ARM.block<3, 1>(3, 6) = T_world_UP1.block<3, 1>(0, 2);
  //     J_ARM.block<3, 1>(0, 7) = T_world_UP2.block<3, 1>(0, 2).cross(T_world_UP5.block<3, 1>(0, 3) - T_world_UP2.block<3, 1>(0, 3)); // z축 회전
  //     J_ARM.block<3, 1>(3, 7) = T_world_UP2.block<3, 1>(0, 2);
  //     J_ARM.block<3, 1>(0, 8) = T_world_UP3.block<3, 1>(0, 2).cross(T_world_UP5.block<3, 1>(0, 3) - T_world_UP3.block<3, 1>(0, 3)); // z축 회전
  //     J_ARM.block<3, 1>(3, 8) = T_world_UP3.block<3, 1>(0, 2);
  //     J_ARM.block<3, 1>(0, 9) = T_world_UP4.block<3, 1>(0, 0);
  //     J_ARM.block<3, 1>(3, 9) = Eigen::Vector3d::Zero();

  //     out.J[5] = J_ARM;
  //   }

  //   return out;
  // }

  // Eigen::Matrix3d computeManipulabilityMatrix(
  //     const FKJacobianResult &result, const Eigen::Ref<const Eigen::VectorXd> &q)
  // {
  //   Eigen::MatrixXd J_b = Eigen::MatrixXd::Zero(3, 22);
  //   Eigen::MatrixXd J_c = Eigen::MatrixXd::Zero(12, 22);

  //   J_b.block<3, 6>(0, 0) = result.J[0].block<3, 6>(0, 0); // 베이스

  //   // FR
  //   J_c.block<3, 6>(0, 0) = result.J[1].block<3, 6>(0, 0);
  //   J_c.block<3, 4>(0, 6) = result.J[1].block<3, 4>(0, 6);

  //   // FL
  //   J_c.block<3, 6>(3, 0) = result.J[2].block<3, 6>(0, 0);
  //   J_c.block<3, 4>(3, 10) = result.J[2].block<3, 4>(0, 6);

  //   // RR
  //   J_c.block<3, 6>(6, 0) = result.J[3].block<3, 6>(0, 0);
  //   J_c.block<3, 4>(6, 14) = result.J[3].block<3, 4>(0, 6);

  //   // RL
  //   J_c.block<3, 6>(9, 0) = result.J[4].block<3, 6>(0, 0);
  //   J_c.block<3, 4>(9, 18) = result.J[4].block<3, 4>(0, 6);

  //   // Joint limit weight matrix
  //   Eigen::MatrixXd W = Eigen::MatrixXd::Identity(22, 22);
  //   Eigen::VectorXd q_legs(16);
  //   Eigen::VectorXd q_min(16);
  //   Eigen::VectorXd q_max(16);
  //   Eigen::VectorXd c(16);
  //   Eigen::VectorXd r(16);
  //   Eigen::VectorXd s(16);

  //   q_legs << q[0], q[1], q[2], q[3],
  //       q[5], q[6], q[7], q[8],
  //       q[10], q[11], q[12], q[13],
  //       q[15], q[16], q[17], q[18];

  //   double deg2rad = M_PI / 180.0;
  //   q_min << -49 * deg2rad, -58.4 * deg2rad, -55.9 * deg2rad, -180 * deg2rad, // FR
  //       -28.9 * deg2rad, -58.4 * deg2rad, -55.9 * deg2rad, -180 * deg2rad,      // FL
  //       -28.9 * deg2rad, -56.9 * deg2rad, -56.3 * deg2rad, -180 * deg2rad,      // RR
  //       -49 * deg2rad, -56.9 * deg2rad, -56.3 * deg2rad, -180 * deg2rad;      // RL

  //   q_max << 28.9 * deg2rad, 56.9 * deg2rad, 56.3 * deg2rad, 180 * deg2rad, // FR
  //       49 * deg2rad, 56.9 * deg2rad, 56.3 * deg2rad, 180 * deg2rad,      // FL
  //       49 * deg2rad, 58.4 * deg2rad, 55.9 * deg2rad, 180 * deg2rad,      // RR
  //       28.9 * deg2rad, 58.4 * deg2rad, 55.9 * deg2rad, 180 * deg2rad;      // RL

  //   c = (q_max + q_min) / 2.0;
  //   r = (q_max - q_min) / 2.0;
  //   // s = (q_legs - c) / r;
  //   s = (q_legs - c).cwiseQuotient(r);

  //   double k = 20.0;
  //   double eps = 0.0001;
  //   for (int i = 6; i < 22; i++)
  //   {
  //     W(i, i) = 1.0 + k * pow((pow(s(i - 6), 2) / (1.0 - pow(s(i - 6), 2) + eps)), 2);
  //   }

  //   Eigen::MatrixXd P_c = Eigen::MatrixXd::Identity(22, 22) - J_c.transpose() * (J_c * J_c.transpose() + Eigen::MatrixXd::Identity(12, 12) * 1e-6).inverse() * J_c;
  //   Eigen::MatrixXd A = J_b * P_c * W.inverse() * P_c.transpose() * J_b.transpose();
  //   return A;
  // }

  // Eigen::Vector3d computeVMaxDirection(const Eigen::Ref<const Eigen::Vector3d> &base_pos,
  //                                      const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
  //                                      const Eigen::Ref<const Eigen::VectorXd> &q,
  //                                      const Eigen::Ref<const Eigen::Matrix3d> &A)
  // {
  //   Eigen::Vector3d ex = Eigen::Vector3d::UnitX();
  //   Eigen::Vector3d ey = Eigen::Vector3d::UnitY();
  //   Eigen::Vector3d ez = Eigen::Vector3d::UnitZ();
  //   double eps = 1e-3;

  //   auto result = computeFKAllAndJacobian(base_pos, base_quat_xyzw, q);
  //   Eigen::MatrixXd J_b_FR = result.J[1].block<3, 4>(0, 6);
  //   Eigen::MatrixXd J_b_FL = result.J[2].block<3, 4>(0, 6);
  //   Eigen::MatrixXd J_b_RR = result.J[3].block<3, 4>(0, 6);
  //   Eigen::MatrixXd J_b_RL = result.J[4].block<3, 4>(0, 6);
  //   Eigen::Matrix3d R_wb = result.T[0].block<3, 3>(0, 0);

  //   Eigen::VectorXd dq_legs = Eigen::VectorXd::Zero(20);

  //   Eigen::MatrixXd J_b_FR_pinv = J_b_FR.transpose() * (J_b_FR * J_b_FR.transpose()).inverse();
  //   Eigen::MatrixXd J_b_FL_pinv = J_b_FL.transpose() * (J_b_FL * J_b_FL.transpose()).inverse();
  //   Eigen::MatrixXd J_b_RR_pinv = J_b_RR.transpose() * (J_b_RR * J_b_RR.transpose()).inverse();
  //   Eigen::MatrixXd J_b_RL_pinv = J_b_RL.transpose() * (J_b_RL * J_b_RL.transpose()).inverse();

  //   // calculate dx_A
  //   dq_legs.segment<4>(0) = -eps * J_b_FR_pinv * R_wb.transpose() * ex;
  //   dq_legs.segment<4>(5) = -eps * J_b_FL_pinv * R_wb.transpose() * ex;
  //   dq_legs.segment<4>(10) = -eps * J_b_RR_pinv * R_wb.transpose() * ex;
  //   dq_legs.segment<4>(15) = -eps * J_b_RL_pinv * R_wb.transpose() * ex;

  //   Eigen::VectorXd q_new = q;
  //   q_new.segment<4>(0) += dq_legs.segment<4>(0);
  //   q_new.segment<4>(5) += dq_legs.segment<4>(5);
  //   q_new.segment<4>(10) += dq_legs.segment<4>(10);
  //   q_new.segment<4>(15) += dq_legs.segment<4>(15);

  //   auto result_xp = computeFKAllAndJacobian(base_pos + eps * ex, base_quat_xyzw, q_new);
  //   Eigen::Matrix3d A_xp = computeManipulabilityMatrix(result_xp, q_new);

  //   dq_legs.segment<4>(0) = -eps * J_b_FR_pinv * R_wb.transpose() * (-ex);
  //   dq_legs.segment<4>(5) = -eps * J_b_FL_pinv * R_wb.transpose() * (-ex);
  //   dq_legs.segment<4>(10) = -eps * J_b_RR_pinv * R_wb.transpose() * (-ex);
  //   dq_legs.segment<4>(15) = -eps * J_b_RL_pinv * R_wb.transpose() * (-ex);

  //   q_new = q;
  //   q_new.segment<4>(0) += dq_legs.segment<4>(0);
  //   q_new.segment<4>(5) += dq_legs.segment<4>(5);
  //   q_new.segment<4>(10) += dq_legs.segment<4>(10);
  //   q_new.segment<4>(15) += dq_legs.segment<4>(15);

  //   auto result_xm = computeFKAllAndJacobian(base_pos - eps * ex, base_quat_xyzw, q_new);
  //   Eigen::Matrix3d A_xm = computeManipulabilityMatrix(result_xm, q_new);

  //   Eigen::Matrix3d dA_dx = (A_xp - A_xm) / (2.0 * eps);

  //   // calculate dy_A
  //   dq_legs.segment<4>(0) = -eps * J_b_FR_pinv * R_wb.transpose() * ey;
  //   dq_legs.segment<4>(5) = -eps * J_b_FL_pinv * R_wb.transpose() * ey;
  //   dq_legs.segment<4>(10) = -eps * J_b_RR_pinv * R_wb.transpose() * ey;
  //   dq_legs.segment<4>(15) = -eps * J_b_RL_pinv * R_wb.transpose() * ey;

  //   q_new = q;
  //   q_new.segment<4>(0) += dq_legs.segment<4>(0);
  //   q_new.segment<4>(5) += dq_legs.segment<4>(5);
  //   q_new.segment<4>(10) += dq_legs.segment<4>(10);
  //   q_new.segment<4>(15) += dq_legs.segment<4>(15);

  //   auto result_yp = computeFKAllAndJacobian(base_pos + eps * ey, base_quat_xyzw, q_new);
  //   Eigen::Matrix3d A_yp = computeManipulabilityMatrix(result_yp, q_new);

  //   dq_legs.segment<4>(0) = -eps * J_b_FR_pinv * R_wb.transpose() * (-ey);
  //   dq_legs.segment<4>(5) = -eps * J_b_FL_pinv * R_wb.transpose() * (-ey);
  //   dq_legs.segment<4>(10) = -eps * J_b_RR_pinv * R_wb.transpose() * (-ey);
  //   dq_legs.segment<4>(15) = -eps * J_b_RL_pinv * R_wb.transpose() * (-ey);

  //   q_new = q;
  //   q_new.segment<4>(0) += dq_legs.segment<4>(0);
  //   q_new.segment<4>(5) += dq_legs.segment<4>(5);
  //   q_new.segment<4>(10) += dq_legs.segment<4>(10);
  //   q_new.segment<4>(15) += dq_legs.segment<4>(15);

  //   auto result_ym = computeFKAllAndJacobian(base_pos - eps * ey, base_quat_xyzw, q_new);
  //   Eigen::Matrix3d A_ym = computeManipulabilityMatrix(result_ym, q_new);

  //   Eigen::Matrix3d dA_dy = (A_yp - A_ym) / (2.0 * eps);

  //   // calculate dz_A
  //   dq_legs.segment<4>(0) = -eps * J_b_FR_pinv * R_wb.transpose() * ez;
  //   dq_legs.segment<4>(5) = -eps * J_b_FL_pinv * R_wb.transpose() * ez;
  //   dq_legs.segment<4>(10) = -eps * J_b_RR_pinv * R_wb.transpose() * ez;
  //   dq_legs.segment<4>(15) = -eps * J_b_RL_pinv * R_wb.transpose() * ez;

  //   q_new = q;
  //   q_new.segment<4>(0) += dq_legs.segment<4>(0);
  //   q_new.segment<4>(5) += dq_legs.segment<4>(5);
  //   q_new.segment<4>(10) += dq_legs.segment<4>(10);
  //   q_new.segment<4>(15) += dq_legs.segment<4>(15);

  //   auto result_zp = computeFKAllAndJacobian(base_pos + eps * ez, base_quat_xyzw, q_new);
  //   Eigen::Matrix3d A_zp = computeManipulabilityMatrix(result_zp, q_new);

  //   dq_legs.segment<4>(0) = -eps * J_b_FR_pinv * R_wb.transpose() * (-ez);
  //   dq_legs.segment<4>(5) = -eps * J_b_FL_pinv * R_wb.transpose() * (-ez);
  //   dq_legs.segment<4>(10) = -eps * J_b_RR_pinv * R_wb.transpose() * (-ez);
  //   dq_legs.segment<4>(15) = -eps * J_b_RL_pinv * R_wb.transpose() * (-ez);

  //   q_new = q;
  //   q_new.segment<4>(0) += dq_legs.segment<4>(0);
  //   q_new.segment<4>(5) += dq_legs.segment<4>(5);
  //   q_new.segment<4>(10) += dq_legs.segment<4>(10);
  //   q_new.segment<4>(15) += dq_legs.segment<4>(15);

  //   auto result_zm = computeFKAllAndJacobian(base_pos - eps * ez, base_quat_xyzw, q_new);
  //   Eigen::Matrix3d A_zm = computeManipulabilityMatrix(result_zm, q_new);

  //   Eigen::Matrix3d dA_dz = (A_zp - A_zm) / (2.0 * eps);

  //   Eigen::Vector3d grad;
  //   grad(0) = (A.inverse()*dA_dx).trace();
  //   grad(1) = (A.inverse()*dA_dy).trace();
  //   grad(2) = (A.inverse()*dA_dz).trace();
  //   // return grad.normalized();
  //   return grad;
  // }

  FKJacobianResult computeFKAllAndJacobian(const Eigen::Ref<const Eigen::Vector3d> &base_pos,
                                           const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
                                           const Eigen::Ref<const Eigen::VectorXd> &q)
  {
    FKJacobianResult out;

    // ---- allocate once / avoid heap churn ----
    for (int i = 0; i < 6; ++i)
      out.T[i] = Eigen::Matrix4d::Identity();

    ensureSizeZero(out.J[0], 6, 6);
    for (int i = 1; i <= 4; ++i)
      ensureSizeZero(out.J[i], 6, 10);
    ensureSizeZero(out.J[5], 6, 10);

    // ---- Base ----
    const Eigen::Matrix4d T_world_base = makeBaseT(base_pos, base_quat_xyzw);
    out.T[0] = T_world_base;

    const Eigen::Matrix3d Rwb = T_world_base.block<3, 3>(0, 0);
    const Eigen::Vector3d pwb = T_world_base.block<3, 1>(0, 3);
    const Eigen::Vector3d xw = Rwb.col(0);
    const Eigen::Vector3d yw = Rwb.col(1);
    const Eigen::Vector3d zw = Rwb.col(2);

    // base Jacobian (same as your original)
    out.J[0].setZero();
    out.J[0].block<3, 3>(0, 0) = Rwb;
    out.J[0].block<3, 3>(3, 3) = Rwb;

    // ===========================================================
    // Precomputed CONSTANT transforms (do once, reuse every call)
    // ===========================================================
    // NOTE: anything that depends on q must stay dynamic.
    // We factor "constant * RotZ(qi)" patterns to reduce ops.

    // FR constants
    static const Eigen::Matrix4d C_base_FR1 = Trans(1.2832, -0.9125, -0.1807);
    static const Eigen::Matrix4d C_FR1_FR2 = Trans(0.337, 0.0, -0.085) * RotX(M_PI / 2);
    static const Eigen::Matrix4d C_FR2_FR3 = Trans(1.4, 0.0, 0.0);
    static const Eigen::Matrix4d C_FR3_FR4 = Trans(0.0, -1.0, 0.0);
    static const Eigen::Matrix4d C_FR4_FR5 = Trans(0.015, -0.288, 0.0) * RotX(-M_PI / 2);
    static const Eigen::Matrix4d C_FR5_FRW = Trans(0.0, -0.66506, -0.171) * RotZ(M_PI) * RotX(M_PI / 2);

    // FL constants
    static const Eigen::Matrix4d C_base_FL1 = Trans(1.2832, 0.9125, -0.1807);
    static const Eigen::Matrix4d C_FL1_FL2 = Trans(0.337, 0.0, -0.085) * RotX(M_PI / 2);
    static const Eigen::Matrix4d C_FL2_FL3 = Trans(1.4, 0.0, 0.0);
    static const Eigen::Matrix4d C_FL3_FL4 = Trans(0.0, -1.0, 0.0);
    static const Eigen::Matrix4d C_FL4_FL5 = Trans(0.015, -0.288, 0.0) * RotX(-M_PI / 2);
    static const Eigen::Matrix4d C_FL5_FLW = Trans(0.0, 0.66506, -0.171) * RotZ(M_PI) * RotX(M_PI / 2);

    // RR constants
    static const Eigen::Matrix4d C_base_RR1 = Trans(-1.3668, -0.9125, -0.1807) * RotZ(M_PI);
    static const Eigen::Matrix4d C_RR1_RR2 = Trans(0.337, 0.0, -0.085) * RotX(-M_PI / 2);
    static const Eigen::Matrix4d C_RR2_RR3 = Trans(1.4, 0.0, 0.0) * RotZ(M_PI);
    static const Eigen::Matrix4d C_RR3_RR4 = Trans(0.0, -1.0, 0.0);
    static const Eigen::Matrix4d C_RR4_RR5 = Trans(-0.015, -0.288, 0.0) * RotZ(M_PI) * RotX(M_PI / 2);
    static const Eigen::Matrix4d C_RR5_RRW = Trans(0.0, 0.66506, -0.171) * RotX(M_PI / 2);

    // RL constants
    static const Eigen::Matrix4d C_base_RL1 = Trans(-1.3668, 0.9125, -0.1807) * RotZ(M_PI);
    static const Eigen::Matrix4d C_RL1_RL2 = Trans(0.337, 0.0, -0.085) * RotX(-M_PI / 2);
    static const Eigen::Matrix4d C_RL2_RL3 = Trans(1.4, 0.0, 0.0) * RotZ(M_PI);
    static const Eigen::Matrix4d C_RL3_RL4 = Trans(0.0, -1.0, 0.0);
    static const Eigen::Matrix4d C_RL4_RL5 = Trans(-0.015, -0.288, 0.0) * RotZ(M_PI) * RotX(M_PI / 2);
    static const Eigen::Matrix4d C_RL5_RLW = Trans(0.0, -0.66506, -0.171) * RotX(M_PI / 2);

    // ARM constants (only factor out constant parts; prismatic stays dynamic)
    static const Eigen::Matrix4d C_base_UP1 = Trans(0.370195, 0.0, 0.345303);
    static const Eigen::Matrix4d C_UP1_UP2 = Trans(-0.35, 0.006485, 1.402) * RotX(M_PI / 2);
    static const Eigen::Matrix4d C_UP2_UP3 = Trans(4.0, -0.065, 0.001927);
    static const Eigen::Matrix4d C_UP3_UP4 = Trans(0.2, -0.02, -0.00094) * RotY(M_PI / 2); // * Trans(0,0,q[23]) later
    static const Eigen::Matrix4d C_UP4_UP5 = Trans(-0.00227, -0.065809, 3.158) * RotY(M_PI / 2) * RotX(M_PI);
    static const Eigen::Matrix4d C_UP5_UP6 = Trans(0.0, 0.16, -0.00227) * RotY(-M_PI / 2);
    static const Eigen::Matrix4d C_UP6_TOOL = Trans(0.0, 0.343, 0.0) * RotY(M_PI / 2) * RotX(M_PI / 2);

    // ===========================================================
    // FK + Jacobian for each part
    // ===========================================================

    // ---------------------- FR (index 1) ----------------------
    {
      // dynamic transforms (minimized)
      const Eigen::Matrix4d T_base_FR1 = C_base_FR1 * RotZ(q[0]);
      const Eigen::Matrix4d T_FR1_FR2 = C_FR1_FR2 * RotZ(q[1]);
      const Eigen::Matrix4d T_FR2_FR3 = C_FR2_FR3 * RotZ(q[2]);
      const Eigen::Matrix4d T_FR3_FR4 = C_FR3_FR4 * RotZ(-q[1] - q[2]); // mimic link rotation depends on q1,q2
      const Eigen::Matrix4d T_FR4_FR5 = C_FR4_FR5 * RotZ(q[3]);
      const Eigen::Matrix4d T_FR5_FRW = C_FR5_FRW * RotZ(q[4]);

      const Eigen::Matrix4d T_world_FR1 = T_world_base * T_base_FR1;
      const Eigen::Matrix4d T_world_FR2 = T_world_FR1 * T_FR1_FR2;
      const Eigen::Matrix4d T_world_FR3 = T_world_FR2 * T_FR2_FR3;
      const Eigen::Matrix4d T_world_FR4 = T_world_FR3 * T_FR3_FR4;
      const Eigen::Matrix4d T_world_FR5 = T_world_FR4 * T_FR4_FR5;
      const Eigen::Matrix4d T_world_FRW = T_world_FR5 * T_FR5_FRW;

      out.T[1] = T_world_FRW;

      Eigen::MatrixXd &J = out.J[1];
      J.setZero();

      const Eigen::Vector3d p_end = T_world_FRW.block<3, 1>(0, 3);
      const Eigen::Vector3d r = p_end - pwb;

      // base columns (same as your original)
      J.block<3, 3>(0, 0) = Rwb;
      J.block<3, 1>(0, 3) = xw.cross(r);
      J.block<3, 1>(0, 4) = yw.cross(r);
      J.block<3, 1>(0, 5) = zw.cross(r);
      J.block<3, 3>(3, 3) = Rwb;

      // leg joint columns (same mimic handling)
      addRevoluteZCol(J, 6, T_world_FR1, p_end);

      // mimic: col7 = (FR2) - (FR4)
      {
        const Eigen::Vector3d z2 = T_world_FR2.block<3, 1>(0, 2);
        const Eigen::Vector3d p2 = T_world_FR2.block<3, 1>(0, 3);
        const Eigen::Vector3d z4 = T_world_FR4.block<3, 1>(0, 2);
        const Eigen::Vector3d p4 = T_world_FR4.block<3, 1>(0, 3);
        J.block<3, 1>(0, 7) = z2.cross(p_end - p2) - z4.cross(p_end - p4);
        J.block<3, 1>(3, 7) = z2 - z4;
      }

      // mimic: col8 = (FR3) - (FR4)
      {
        const Eigen::Vector3d z3 = T_world_FR3.block<3, 1>(0, 2);
        const Eigen::Vector3d p3 = T_world_FR3.block<3, 1>(0, 3);
        const Eigen::Vector3d z4 = T_world_FR4.block<3, 1>(0, 2);
        const Eigen::Vector3d p4 = T_world_FR4.block<3, 1>(0, 3);
        J.block<3, 1>(0, 8) = z3.cross(p_end - p3) - z4.cross(p_end - p4);
        J.block<3, 1>(3, 8) = z3 - z4;
      }

      addRevoluteZCol(J, 9, T_world_FR5, p_end);
    }

    // ---------------------- FL (index 2) ----------------------
    {
      const Eigen::Matrix4d T_base_FL1 = C_base_FL1 * RotZ(q[5]);
      const Eigen::Matrix4d T_FL1_FL2 = C_FL1_FL2 * RotZ(q[6]);
      const Eigen::Matrix4d T_FL2_FL3 = C_FL2_FL3 * RotZ(q[7]);
      const Eigen::Matrix4d T_FL3_FL4 = C_FL3_FL4 * RotZ(-q[6] - q[7]);
      const Eigen::Matrix4d T_FL4_FL5 = C_FL4_FL5 * RotZ(q[8]);
      const Eigen::Matrix4d T_FL5_FLW = C_FL5_FLW * RotZ(q[9]);

      const Eigen::Matrix4d T_world_FL1 = T_world_base * T_base_FL1;
      const Eigen::Matrix4d T_world_FL2 = T_world_FL1 * T_FL1_FL2;
      const Eigen::Matrix4d T_world_FL3 = T_world_FL2 * T_FL2_FL3;
      const Eigen::Matrix4d T_world_FL4 = T_world_FL3 * T_FL3_FL4;
      const Eigen::Matrix4d T_world_FL5 = T_world_FL4 * T_FL4_FL5;
      const Eigen::Matrix4d T_world_FLW = T_world_FL5 * T_FL5_FLW;

      out.T[2] = T_world_FLW;

      Eigen::MatrixXd &J = out.J[2];
      J.setZero();

      const Eigen::Vector3d p_end = T_world_FLW.block<3, 1>(0, 3);
      const Eigen::Vector3d r = p_end - pwb;

      J.block<3, 3>(0, 0) = Rwb;
      J.block<3, 1>(0, 3) = xw.cross(r);
      J.block<3, 1>(0, 4) = yw.cross(r);
      J.block<3, 1>(0, 5) = zw.cross(r);
      J.block<3, 3>(3, 3) = Rwb;

      addRevoluteZCol(J, 6, T_world_FL1, p_end);

      {
        const Eigen::Vector3d z2 = T_world_FL2.block<3, 1>(0, 2);
        const Eigen::Vector3d p2 = T_world_FL2.block<3, 1>(0, 3);
        const Eigen::Vector3d z4 = T_world_FL4.block<3, 1>(0, 2);
        const Eigen::Vector3d p4 = T_world_FL4.block<3, 1>(0, 3);
        J.block<3, 1>(0, 7) = z2.cross(p_end - p2) - z4.cross(p_end - p4);
        J.block<3, 1>(3, 7) = z2 - z4;
      }

      {
        const Eigen::Vector3d z3 = T_world_FL3.block<3, 1>(0, 2);
        const Eigen::Vector3d p3 = T_world_FL3.block<3, 1>(0, 3);
        const Eigen::Vector3d z4 = T_world_FL4.block<3, 1>(0, 2);
        const Eigen::Vector3d p4 = T_world_FL4.block<3, 1>(0, 3);
        J.block<3, 1>(0, 8) = z3.cross(p_end - p3) - z4.cross(p_end - p4);
        J.block<3, 1>(3, 8) = z3 - z4;
      }

      addRevoluteZCol(J, 9, T_world_FL5, p_end);
    }

    // ---------------------- RR (index 3) ----------------------
    {
      const Eigen::Matrix4d T_base_RR1 = C_base_RR1 * RotZ(q[10]);
      const Eigen::Matrix4d T_RR1_RR2 = C_RR1_RR2 * RotZ(q[11]);
      const Eigen::Matrix4d T_RR2_RR3 = C_RR2_RR3 * RotZ(q[12]);
      const Eigen::Matrix4d T_RR3_RR4 = C_RR3_RR4 * RotZ(-q[11] - q[12]);
      const Eigen::Matrix4d T_RR4_RR5 = C_RR4_RR5 * RotZ(q[13]);
      const Eigen::Matrix4d T_RR5_RRW = C_RR5_RRW * RotZ(q[14]);

      const Eigen::Matrix4d T_world_RR1 = T_world_base * T_base_RR1;
      const Eigen::Matrix4d T_world_RR2 = T_world_RR1 * T_RR1_RR2;
      const Eigen::Matrix4d T_world_RR3 = T_world_RR2 * T_RR2_RR3;
      const Eigen::Matrix4d T_world_RR4 = T_world_RR3 * T_RR3_RR4;
      const Eigen::Matrix4d T_world_RR5 = T_world_RR4 * T_RR4_RR5;
      const Eigen::Matrix4d T_world_RRW = T_world_RR5 * T_RR5_RRW;

      out.T[3] = T_world_RRW;

      Eigen::MatrixXd &J = out.J[3];
      J.setZero();

      const Eigen::Vector3d p_end = T_world_RRW.block<3, 1>(0, 3);
      const Eigen::Vector3d r = p_end - pwb;

      J.block<3, 3>(0, 0) = Rwb;
      J.block<3, 1>(0, 3) = xw.cross(r);
      J.block<3, 1>(0, 4) = yw.cross(r);
      J.block<3, 1>(0, 5) = zw.cross(r);
      J.block<3, 3>(3, 3) = Rwb;

      addRevoluteZCol(J, 6, T_world_RR1, p_end);

      {
        const Eigen::Vector3d z2 = T_world_RR2.block<3, 1>(0, 2);
        const Eigen::Vector3d p2 = T_world_RR2.block<3, 1>(0, 3);
        const Eigen::Vector3d z4 = T_world_RR4.block<3, 1>(0, 2);
        const Eigen::Vector3d p4 = T_world_RR4.block<3, 1>(0, 3);
        J.block<3, 1>(0, 7) = z2.cross(p_end - p2) - z4.cross(p_end - p4);
        J.block<3, 1>(3, 7) = z2 - z4;
      }

      {
        const Eigen::Vector3d z3 = T_world_RR3.block<3, 1>(0, 2);
        const Eigen::Vector3d p3 = T_world_RR3.block<3, 1>(0, 3);
        const Eigen::Vector3d z4 = T_world_RR4.block<3, 1>(0, 2);
        const Eigen::Vector3d p4 = T_world_RR4.block<3, 1>(0, 3);
        J.block<3, 1>(0, 8) = z3.cross(p_end - p3) - z4.cross(p_end - p4);
        J.block<3, 1>(3, 8) = z3 - z4;
      }

      addRevoluteZCol(J, 9, T_world_RR5, p_end);
    }

    // ---------------------- RL (index 4) ----------------------
    {
      const Eigen::Matrix4d T_base_RL1 = C_base_RL1 * RotZ(q[15]);
      const Eigen::Matrix4d T_RL1_RL2 = C_RL1_RL2 * RotZ(q[16]);
      const Eigen::Matrix4d T_RL2_RL3 = C_RL2_RL3 * RotZ(q[17]);
      const Eigen::Matrix4d T_RL3_RL4 = C_RL3_RL4 * RotZ(-q[16] - q[17]);
      const Eigen::Matrix4d T_RL4_RL5 = C_RL4_RL5 * RotZ(q[18]);
      const Eigen::Matrix4d T_RL5_RLW = C_RL5_RLW * RotZ(q[19]);

      const Eigen::Matrix4d T_world_RL1 = T_world_base * T_base_RL1;
      const Eigen::Matrix4d T_world_RL2 = T_world_RL1 * T_RL1_RL2;
      const Eigen::Matrix4d T_world_RL3 = T_world_RL2 * T_RL2_RL3;
      const Eigen::Matrix4d T_world_RL4 = T_world_RL3 * T_RL3_RL4;
      const Eigen::Matrix4d T_world_RL5 = T_world_RL4 * T_RL4_RL5;
      const Eigen::Matrix4d T_world_RLW = T_world_RL5 * T_RL5_RLW;

      out.T[4] = T_world_RLW;

      Eigen::MatrixXd &J = out.J[4];
      J.setZero();

      const Eigen::Vector3d p_end = T_world_RLW.block<3, 1>(0, 3);
      const Eigen::Vector3d r = p_end - pwb;

      J.block<3, 3>(0, 0) = Rwb;
      J.block<3, 1>(0, 3) = xw.cross(r);
      J.block<3, 1>(0, 4) = yw.cross(r);
      J.block<3, 1>(0, 5) = zw.cross(r);
      J.block<3, 3>(3, 3) = Rwb;

      addRevoluteZCol(J, 6, T_world_RL1, p_end);

      {
        const Eigen::Vector3d z2 = T_world_RL2.block<3, 1>(0, 2);
        const Eigen::Vector3d p2 = T_world_RL2.block<3, 1>(0, 3);
        const Eigen::Vector3d z4 = T_world_RL4.block<3, 1>(0, 2);
        const Eigen::Vector3d p4 = T_world_RL4.block<3, 1>(0, 3);
        J.block<3, 1>(0, 7) = z2.cross(p_end - p2) - z4.cross(p_end - p4);
        J.block<3, 1>(3, 7) = z2 - z4;
      }

      {
        const Eigen::Vector3d z3 = T_world_RL3.block<3, 1>(0, 2);
        const Eigen::Vector3d p3 = T_world_RL3.block<3, 1>(0, 3);
        const Eigen::Vector3d z4 = T_world_RL4.block<3, 1>(0, 2);
        const Eigen::Vector3d p4 = T_world_RL4.block<3, 1>(0, 3);
        J.block<3, 1>(0, 8) = z3.cross(p_end - p3) - z4.cross(p_end - p4);
        J.block<3, 1>(3, 8) = z3 - z4;
      }

      addRevoluteZCol(J, 9, T_world_RL5, p_end);
    }

    // ---------------------- ARM (index 5) ----------------------
    {
      const Eigen::Matrix4d T_base_UP1 = C_base_UP1 * RotZ(q[20]);
      const Eigen::Matrix4d T_UP1_UP2 = C_UP1_UP2 * RotZ(q[21]);
      const Eigen::Matrix4d T_UP2_UP3 = C_UP2_UP3 * RotZ(q[22]);
      const Eigen::Matrix4d T_UP3_UP4 = C_UP3_UP4 * Trans(0.0, 0.0, q[23]); // prismatic
      const Eigen::Matrix4d T_UP4_UP5 = C_UP4_UP5 * RotZ(q[24]);
      const Eigen::Matrix4d T_UP5_UP6 = C_UP5_UP6 * RotZ(q[25]);
      const Eigen::Matrix4d T_UP6_TOOL = C_UP6_TOOL * RotZ(q[26]);

      const Eigen::Matrix4d T_world_UP1 = T_world_base * T_base_UP1;
      const Eigen::Matrix4d T_world_UP2 = T_world_UP1 * T_UP1_UP2;
      const Eigen::Matrix4d T_world_UP3 = T_world_UP2 * T_UP2_UP3;
      const Eigen::Matrix4d T_world_UP4 = T_world_UP3 * T_UP3_UP4;
      const Eigen::Matrix4d T_world_UP5 = T_world_UP4 * T_UP4_UP5;
      const Eigen::Matrix4d T_world_UP6 = T_world_UP5 * T_UP5_UP6;
      const Eigen::Matrix4d T_world_TOOL = T_world_UP6 * T_UP6_TOOL;

      // NOTE: keep same output as your original: out.T[5] = T_world_UP5
      out.T[5] = T_world_UP5;

      Eigen::MatrixXd &J = out.J[5];
      J.setZero();

      const Eigen::Vector3d p_end = T_world_UP5.block<3, 1>(0, 3);
      const Eigen::Vector3d r = p_end - pwb;

      J.block<3, 3>(0, 0) = Rwb;
      J.block<3, 1>(0, 3) = xw.cross(r);
      J.block<3, 1>(0, 4) = yw.cross(r);
      J.block<3, 1>(0, 5) = zw.cross(r);
      J.block<3, 3>(3, 3) = Rwb;

      // arm joint mapping kept identical to your original (cols 6..9 used here)
      addRevoluteZCol(J, 6, T_world_UP1, p_end);
      addRevoluteZCol(J, 7, T_world_UP2, p_end);
      addRevoluteZCol(J, 8, T_world_UP3, p_end);

      // prismatic at UP4 along its joint axis (URDF axis: local Z)
      J.block<3, 1>(0, 9) = T_world_UP4.block<3, 1>(0, 2);
      J.block<3, 1>(3, 9) = Eigen::Vector3d::Zero();
    }

    return out;
  }

  FKJacobianResult computePinocchioFKAllAndJacobian(
    const Eigen::Ref<const Eigen::Vector3d> &base_pos,
    const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
    const Eigen::Ref<const Eigen::VectorXd> &q,
    const std::string &urdf_path)
  {
    PinocchioContext &ctx = getPinocchioContext(urdf_path);
    buildPinocchioQ(ctx, base_pos, base_quat_xyzw, q);

    pinocchio::computeJointJacobians(ctx.model, *ctx.data, ctx.q);
    pinocchio::updateFramePlacements(ctx.model, *ctx.data);
    pinocchio::centerOfMass(ctx.model, *ctx.data, ctx.q);
    pinocchio::jacobianCenterOfMass(ctx.model, *ctx.data, ctx.q, false);

    FKJacobianResult out;
    for (int i = 0; i < 6; ++i)
      out.T[i] = Eigen::Matrix4d::Identity();
    ensureSizeZero(out.J[0], 6, 6);
    for (int i = 1; i <= 5; ++i)
      ensureSizeZero(out.J[i], 6, 10);

    // Base transform/jacobian in same layout as KDL result.
    out.T[0] = makeBaseT(base_pos, base_quat_xyzw);
    const Eigen::Matrix3d Rwb = out.T[0].block<3, 3>(0, 0);
    out.J[0].setZero();
    out.J[0].block<3, 3>(0, 0) = Rwb;
    out.J[0].block<3, 3>(3, 3) = Rwb;

    out.com = ctx.data->com[0];
    out.com_jacobian = computeReducedComJacobian(ctx, ctx.data->Jcom);

    for (size_t i = 0; i < ctx.targets.size(); ++i)
    {
      out.T[i + 1] = se3ToMatrix4(ctx.data->oMf[ctx.targets[i].frame_id]);
      out.J[i + 1] = computePinFrameJacobianKdlLayout(ctx, ctx.targets[i]);
    }
    return out;
  }

  PinocchioKinematicsResult computePinocchioComAndJacobians(
    const Eigen::Ref<const Eigen::Vector3d> &base_pos,
    const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
    const Eigen::Ref<const Eigen::VectorXd> &q,
    const std::string &urdf_path)
  {
    const FKJacobianResult full = computePinocchioFKAllAndJacobian(base_pos, base_quat_xyzw, q, urdf_path);
    PinocchioKinematicsResult out;
    out.com = full.com;
    out.com_jacobian = full.com_jacobian;
    for (size_t i = 0; i < out.jacobians.size(); ++i)
      out.jacobians[i] = full.J[i + 1];
    return out;
  }

  // ============================================================
  //  Optimized computeVMaxDirection (same signature)
  //
  //  Returns gradient of log-det manipulability wrt base translation:
  //    grad_i = trace(A^{-1} * dA/dx_i),  i in {x,y,z}
  //
  //  Steps:
  //  1) For each axis perturbation ±eps, move base and compensate leg joints
  //     so contact geometry remains consistent (approximate).
  //  2) Recompute A at perturbed states.
  //  3) Central-difference dA/dx_i.
  //  4) Compute trace(A^{-1} dA/dx_i) via linear solve.
  //
  //  Tuning:
  //  - eps: finite-difference step (smaller = less bias, more noise)
  //  - mu_pinv: damping for leg compensation pseudoinverse stability
  // ============================================================
  Eigen::Vector3d computeVMaxDirection(const Eigen::Ref<const Eigen::Vector3d> &base_pos,
                                       const Eigen::Ref<const Eigen::Vector4d> &base_quat_xyzw,
                                       const Eigen::Ref<const Eigen::VectorXd> &q,
                                       const Eigen::Ref<const Eigen::Matrix3d> &A)
  {
    const Eigen::Vector3d ex = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d ey = Eigen::Vector3d::UnitY();
    const Eigen::Vector3d ez = Eigen::Vector3d::UnitZ();

    const double eps = 1e-3;
    const double mu_pinv = 1e-8; // damped pinv for robust leg compensation near singular configs

    // Base FK/J
    auto result0 = computeFKAllAndJacobian(base_pos, base_quat_xyzw, q);

    Eigen::MatrixXd J_b_FR = result0.J[1].block<3, 4>(0, 6);
    Eigen::MatrixXd J_b_FL = result0.J[2].block<3, 4>(0, 6);
    Eigen::MatrixXd J_b_RR = result0.J[3].block<3, 4>(0, 6);
    Eigen::MatrixXd J_b_RL = result0.J[4].block<3, 4>(0, 6);

    const Eigen::Matrix3d R_wb = result0.T[0].block<3, 3>(0, 0);

    // Damped pinv (4x3)
    const Eigen::MatrixXd J_b_FR_pinv = pinv3x4_damped(J_b_FR, mu_pinv);
    const Eigen::MatrixXd J_b_FL_pinv = pinv3x4_damped(J_b_FL, mu_pinv);
    const Eigen::MatrixXd J_b_RR_pinv = pinv3x4_damped(J_b_RR, mu_pinv);
    const Eigen::MatrixXd J_b_RL_pinv = pinv3x4_damped(J_b_RL, mu_pinv);

    auto evalA_with_compensated_legs = [&](const Eigen::Vector3d &dpos_world) -> Eigen::Matrix3d
    {
      // dq_legs is just a scratch; we only touch q segments [0..3], [5..8], [10..13], [15..18]
      Eigen::VectorXd q_new = q;

      // Desired base delta expressed in base frame for leg correction.
      // We solve dq_legs so that foot-point disturbance from base motion is reduced.
      const Eigen::Vector3d dpos_base = R_wb.transpose() * dpos_world;

      // dq = - J^+ * dpos_base
      const Eigen::VectorXd dq_fr = -J_b_FR_pinv * dpos_base; // 4
      const Eigen::VectorXd dq_fl = -J_b_FL_pinv * dpos_base; // 4
      const Eigen::VectorXd dq_rr = -J_b_RR_pinv * dpos_base; // 4
      const Eigen::VectorXd dq_rl = -J_b_RL_pinv * dpos_base; // 4

      q_new.segment<4>(0) += dq_fr;
      q_new.segment<4>(5) += dq_fl;
      q_new.segment<4>(10) += dq_rr;
      q_new.segment<4>(15) += dq_rl;

      auto res = computeFKAllAndJacobian(base_pos + dpos_world, base_quat_xyzw, q_new);
      return computeManipulabilityMatrix(res, q_new);
    };

    // Central-difference derivatives of A.
    const Eigen::Matrix3d A_xp = evalA_with_compensated_legs(+eps * ex);
    const Eigen::Matrix3d A_xm = evalA_with_compensated_legs(-eps * ex);
    const Eigen::Matrix3d dA_dx = (A_xp - A_xm) / (2.0 * eps);

    const Eigen::Matrix3d A_yp = evalA_with_compensated_legs(+eps * ey);
    const Eigen::Matrix3d A_ym = evalA_with_compensated_legs(-eps * ey);
    const Eigen::Matrix3d dA_dy = (A_yp - A_ym) / (2.0 * eps);

    const Eigen::Matrix3d A_zp = evalA_with_compensated_legs(+eps * ez);
    const Eigen::Matrix3d A_zm = evalA_with_compensated_legs(-eps * ez);
    const Eigen::Matrix3d dA_dz = (A_zp - A_zm) / (2.0 * eps);

    // grad_i = trace( A^{-1} dA/di )
    // Use solve instead of inverse (faster & stable)
    Eigen::LDLT<Eigen::Matrix3d> A_ldlt(A);

    Eigen::Vector3d grad;
    grad(0) = (A_ldlt.solve(dA_dx)).trace();
    grad(1) = (A_ldlt.solve(dA_dy)).trace();
    grad(2) = (A_ldlt.solve(dA_dz)).trace();

    return grad;
  }
} // namespace ecvt_controller