import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def create_kalman_filter(init_x, init_y, meas_noise=150.0, proc_noise=5.0, dt=1.0):
    """
    Standard Constant-Velocity (CV) Kalman filter.

    State: [x, y, vx, vy]  (dim_x=4)
    Measurement: [x, y]    (dim_z=2)

    After kf.update(z), the innovation (residual) is available as:
        kf.y  — innovation vector (z - H @ x_prior)
        kf.S  — innovation covariance

    Args:
        init_x, init_y: Initial position (pixels)
        meas_noise: Measurement noise variance (px²).
            YOLOv8 bbox centers jitter ~10-15px between frames on 1080p
            surveillance → σ² ≈ 100-225 px². Default 150 is empirically
            calibrated so authentic NIS ≈ χ²(2).
        proc_noise: Process noise variance (px²/frame²).
            Captures unmodeled accelerations/jitter in constant-velocity
            assumption. Default 5.0 reflects ~2-3 px/frame² typical
            pedestrian acceleration.
        dt: Time step (frames or seconds). Default 1.0.
    """

    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Constant velocity transition model
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    # Measurement model
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)

    # Initial state
    kf.x = np.array([[init_x], [init_y], [0.0], [0.0]])

    # Initial state covariance — large but not extreme
    # P_pos = meas_noise (we know position to ~measurement accuracy)
    # P_vel = large (velocity is unknown)
    kf.P = np.diag([meas_noise, meas_noise, 50.0, 50.0])

    # Measurement noise — calibrated for YOLOv8 on 1080p surveillance
    # R = σ² I₂, where σ ≈ 10-15 px for YOLO bbox center jitter
    kf.R = np.array([
        [meas_noise, 0.0],
        [0.0, meas_noise]
    ])

    # Process noise — discrete white noise model for constant-velocity
    q_block = Q_discrete_white_noise(dim=2, dt=dt, var=proc_noise)
    kf.Q = np.zeros((4, 4))
    # x and vx (indices 0 and 2)
    kf.Q[0, 0] = q_block[0, 0]
    kf.Q[0, 2] = q_block[0, 1]
    kf.Q[2, 0] = q_block[1, 0]
    kf.Q[2, 2] = q_block[1, 1]
    # y and vy (indices 1 and 3)
    kf.Q[1, 1] = q_block[0, 0]
    kf.Q[1, 3] = q_block[0, 1]
    kf.Q[3, 1] = q_block[1, 0]
    kf.Q[3, 3] = q_block[1, 1]

    return kf


def create_ca_kalman_filter(init_x, init_y, meas_noise=150.0, proc_noise=2.0, dt=1.0):
    """
    Constant-Acceleration (CA) Kalman filter.

    State: [x, y, vx, vy, ax, ay]  (dim_x=6)
    Measurement: [x, y]             (dim_z=2)

    This filter explains authentic accelerations/turns that the CV filter
    cannot, reducing false-positive NIS spikes on legitimate maneuvers.
    The combined min-NIS approach (Lever 2) takes min(nis_cv, nis_ca) to
    ensure high NIS only when NO physics model explains the motion.

    After kf.update(z), the innovation is available as:
        kf.y  — innovation vector
        kf.S  — innovation covariance

    Args:
        init_x, init_y: Initial position (pixels)
        meas_noise: Measurement noise variance (px²). Same as CV filter.
        proc_noise: Process noise variance (px²/frame⁴).
            Lower than CV because the CA model already captures acceleration.
            Default 2.0.
        dt: Time step (frames or seconds). Default 1.0.
    """

    kf = KalmanFilter(dim_x=6, dim_z=2)

    # Constant-acceleration transition model
    # x' = x + vx*dt + 0.5*ax*dt^2
    # vx' = vx + ax*dt
    # ax' = ax
    kf.F = np.array([
        [1, 0, dt, 0,  0.5*dt**2, 0],
        [0, 1, 0,  dt, 0,         0.5*dt**2],
        [0, 0, 1,  0,  dt,        0],
        [0, 0, 0,  1,  0,         dt],
        [0, 0, 0,  0,  1,         0],
        [0, 0, 0,  0,  0,         1]
    ], dtype=float)

    # Measurement model — observe position only
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ], dtype=float)

    # Initial state
    kf.x = np.array([[init_x], [init_y], [0.0], [0.0], [0.0], [0.0]])

    # Initial state covariance
    # Position: measurement accuracy; velocity: uncertain; accel: very uncertain
    kf.P = np.diag([meas_noise, meas_noise, 50.0, 50.0, 10.0, 10.0])

    # Measurement noise — same as CV (same detector)
    kf.R = np.array([
        [meas_noise, 0.0],
        [0.0, meas_noise]
    ])

    # Process noise — discrete white noise for (pos, vel, accel) triples
    q_block = Q_discrete_white_noise(dim=3, dt=dt, var=proc_noise)
    kf.Q = np.zeros((6, 6))
    
    # x, vx, ax (indices 0, 2, 4)
    kf.Q[0, 0] = q_block[0, 0]
    kf.Q[0, 2] = q_block[0, 1]
    kf.Q[0, 4] = q_block[0, 2]
    
    kf.Q[2, 0] = q_block[1, 0]
    kf.Q[2, 2] = q_block[1, 1]
    kf.Q[2, 4] = q_block[1, 2]
    
    kf.Q[4, 0] = q_block[2, 0]
    kf.Q[4, 2] = q_block[2, 1]
    kf.Q[4, 4] = q_block[2, 2]
    
    # y, vy, ay (indices 1, 3, 5)
    kf.Q[1, 1] = q_block[0, 0]
    kf.Q[1, 3] = q_block[0, 1]
    kf.Q[1, 5] = q_block[0, 2]
    
    kf.Q[3, 1] = q_block[1, 0]
    kf.Q[3, 3] = q_block[1, 1]
    kf.Q[3, 5] = q_block[1, 2]
    
    kf.Q[5, 1] = q_block[2, 0]
    kf.Q[5, 3] = q_block[2, 1]
    kf.Q[5, 5] = q_block[2, 2]

    return kf
