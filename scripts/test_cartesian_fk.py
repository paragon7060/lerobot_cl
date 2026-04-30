#!/usr/bin/env python3
"""
train_groot_cartesian.py의 FK / 회전 변환 검증 스크립트.

3가지 체크:
  1. q=0 sanity check  — zero-pose에서 xyz가 알려진 값과 일치하는지
  2. RPY ↔ quat 일관성 — 같은 joint에서 두 표현이 동일한 R을 복원하는지
  3. roboticstoolbox 대조 (설치된 경우) — 독립 구현과 xyz/R 비교

실행:
    conda run -n lerobot050_groot python scripts/test_cartesian_fk.py
"""

import sys
from pathlib import Path

import numpy as np

# train_groot_cartesian.py의 함수를 직접 import
sys.path.insert(0, str(Path(__file__).parent))
from train_groot_cartesian import (
    _franka_fk_batch,
    _rotation_to_euler_zyx,
    _rotation_to_quat,
    joint_to_cartesian,
)

# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """quat (N, 4) [qx, qy, qz, qw] → R (N, 3, 3)."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw),
        2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw),
        2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2),
    ], axis=1).reshape(-1, 3, 3)
    return R


def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """rpy (N, 3) [roll, pitch, yaw] → R (N, 3, 3). ZYX 컨벤션."""
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.stack([
        cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr,
        sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr,
        -sp,    cp*sr,              cp*cr,
    ], axis=1).reshape(-1, 3, 3)
    return R


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """두 회전행렬 세트 간의 angular error (degree). (N,3,3) x2 → (N,)."""
    R_diff = np.einsum("nij,nkj->nik", R1, R2)  # R1 @ R2^T
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: q=0 sanity check
# ─────────────────────────────────────────────────────────────────────────────
def test_zero_pose():
    """
    q=[0,0,0,0,0,0,0]에서 Franka end-effector 위치.
    DH 파라미터로 계산한 이론값:
      x = 0.0, y = 0.0, z = d1+d3+d5+d7 = 0.333+0.316+0.384+0.107 = 1.140 m
    (a, alpha에 의한 오프셋은 q=0에서 x/y에 기여)
    """
    print("\n── Test 1: q=0 sanity check ──")
    q = np.zeros((1, 7))
    xyz, R = _franka_fk_batch(q)

    print(f"  xyz = {xyz[0].tolist()}")
    print(f"  R   =\n{R[0]}")

    # z 방향 높이 체크 (d들의 합이 z에 기여, 단 alpha에 의한 회전도 있음)
    # 정확한 값은 roboticstoolbox와 대조하는 게 맞고 여기서는 reasonable range 체크
    z_val = xyz[0, 2]
    ok = 0.9 < z_val < 1.3
    print(f"  z={z_val:.4f}m  {'(reasonable range 0.9~1.3m)' if ok else '(out of expected range!)'}")
    print(f"  {PASS if ok else FAIL} zero-pose z range")

    # R이 회전행렬인지 (det=1, R^T R = I)
    det = np.linalg.det(R[0])
    ortho_err = np.max(np.abs(R[0] @ R[0].T - np.eye(3)))
    ok_R = abs(det - 1.0) < 1e-6 and ortho_err < 1e-6
    print(f"  det(R)={det:.8f}, ortho_err={ortho_err:.2e}")
    print(f"  {PASS if ok_R else FAIL} R is valid rotation matrix")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: RPY ↔ quat 일관성
# ─────────────────────────────────────────────────────────────────────────────
def test_rpy_quat_consistency(n=200, seed=42):
    """
    랜덤 joint angle에서 rpy와 quat이 동일한 회전행렬을 복원하는지 확인.
    """
    print("\n── Test 2: RPY ↔ quat consistency ──")
    rng = np.random.default_rng(seed)
    # Franka joint limits 내 랜덤 샘플
    q_limits = np.array([
        [-2.8973,  2.8973],
        [-1.7628,  1.7628],
        [-2.8973,  2.8973],
        [-3.0718, -0.0698],
        [-2.8973,  2.8973],
        [-0.0175,  3.7525],
        [-2.8973,  2.8973],
    ])
    q = rng.uniform(q_limits[:, 0], q_limits[:, 1], size=(n, 7))

    _, R_ref = _franka_fk_batch(q)

    rpy  = _rotation_to_euler_zyx(R_ref)
    quat = _rotation_to_quat(R_ref)

    R_from_rpy  = _rpy_to_rotation_matrix(rpy)
    R_from_quat = _quat_to_rotation_matrix(quat)

    err_rpy  = rotation_error_deg(R_ref, R_from_rpy)
    err_quat = rotation_error_deg(R_ref, R_from_quat)

    print(f"  RPY  round-trip error  — max: {err_rpy.max():.4f}°  mean: {err_rpy.mean():.4f}°")
    print(f"  Quat round-trip error  — max: {err_quat.max():.4f}°  mean: {err_quat.mean():.4f}°")

    rpy_ok  = err_rpy.max()  < 0.01
    quat_ok = err_quat.max() < 0.001
    print(f"  {PASS if rpy_ok  else FAIL} RPY  round-trip (tol 0.01°)")
    print(f"  {PASS if quat_ok else FAIL} Quat round-trip (tol 0.001°)")

    # quat unit norm 체크
    norms = np.linalg.norm(quat, axis=1)
    norm_ok = np.allclose(norms, 1.0, atol=1e-6)
    print(f"  quat norm: min={norms.min():.8f} max={norms.max():.8f}")
    print(f"  {PASS if norm_ok else FAIL} Quat unit norm")

    # quat의 rpy 결과와도 일치하는지 (같은 회전을 두 표현으로 나타내는지)
    err_cross = rotation_error_deg(R_from_rpy, R_from_quat)
    cross_ok = err_cross.max() < 0.01
    print(f"  RPY vs Quat cross-check — max: {err_cross.max():.4f}°")
    print(f"  {PASS if cross_ok else FAIL} RPY == Quat (same rotation)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: roboticstoolbox 대조 (설치된 경우만)
# ─────────────────────────────────────────────────────────────────────────────
def test_against_roboticstoolbox(n=50, seed=0):
    print("\n── Test 3: roboticstoolbox 대조 ──")
    try:
        import roboticstoolbox as rtb
        from spatialmath import SE3
    except ImportError:
        print("  [SKIP] roboticstoolbox not installed")
        print("         pip install roboticstoolbox-python spatialmath")
        return

    robot = rtb.models.Panda()
    rng = np.random.default_rng(seed)
    q_limits = np.array([
        [-2.8973,  2.8973],
        [-1.7628,  1.7628],
        [-2.8973,  2.8973],
        [-3.0718, -0.0698],
        [-2.8973,  2.8973],
        [-0.0175,  3.7525],
        [-2.8973,  2.8973],
    ])
    qs = rng.uniform(q_limits[:, 0], q_limits[:, 1], size=(n, 7))

    xyz_ours, R_ours = _franka_fk_batch(qs)
    xyz_errs, R_errs = [], []

    for i, q in enumerate(qs):
        T = robot.fkine(q)
        xyz_ref = T.t
        R_ref   = T.R

        xyz_errs.append(np.linalg.norm(xyz_ours[i] - xyz_ref))
        R_diff = R_ours[i] @ R_ref.T
        trace  = R_diff[0,0] + R_diff[1,1] + R_diff[2,2]
        angle  = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
        R_errs.append(angle)

    xyz_errs = np.array(xyz_errs)
    R_errs   = np.array(R_errs)
    print(f"  xyz error  — max: {xyz_errs.max()*1000:.4f}mm  mean: {xyz_errs.mean()*1000:.4f}mm")
    print(f"  R error    — max: {R_errs.max():.4f}°  mean: {R_errs.mean():.4f}°")

    xyz_ok = xyz_errs.max() < 1e-4   # 0.1mm
    R_ok   = R_errs.max()  < 0.01    # 0.01°
    print(f"  {PASS if xyz_ok else FAIL} xyz vs roboticstoolbox (tol 0.1mm)")
    print(f"  {PASS if R_ok   else FAIL} R   vs roboticstoolbox (tol 0.01°)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: joint_to_cartesian output shape/dim 체크
# ─────────────────────────────────────────────────────────────────────────────
def test_output_shape():
    print("\n── Test 4: output shape ──")
    a = np.random.randn(16, 32).astype(np.float32)  # 32-dim zero-padded
    rpy_out  = joint_to_cartesian(a, "rpy")
    quat_out = joint_to_cartesian(a, "quat")

    print(f"  input shape:      {a.shape}")
    print(f"  rpy  output:      {rpy_out.shape}  (expected (16, 7))")
    print(f"  quat output:      {quat_out.shape}  (expected (16, 8))")
    print(f"  {PASS if rpy_out.shape  == (16, 7) else FAIL} rpy  shape")
    print(f"  {PASS if quat_out.shape == (16, 8) else FAIL} quat shape")

    # gripper 값이 그대로 전달되는지
    gripper_in   = a[:, 7]
    gripper_rpy  = rpy_out[:, 6]
    gripper_quat = quat_out[:, 7]
    g_ok = np.allclose(gripper_in, gripper_rpy, atol=1e-5) and \
           np.allclose(gripper_in, gripper_quat, atol=1e-5)
    print(f"  {PASS if g_ok else FAIL} gripper pass-through")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_zero_pose()
    test_rpy_quat_consistency()
    test_against_roboticstoolbox()
    test_output_shape()
    print("\n── Done ──")
