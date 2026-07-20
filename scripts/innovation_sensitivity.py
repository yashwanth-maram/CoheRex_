import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from ultralytics import YOLO
from coherex.tracking.kalman import create_kalman_filter, create_ca_kalman_filter

def main():
    print("="*80)
    print("Innovation Sensitivity Analysis")
    print("="*80)
    print("Simulating a well-behaved authentic track in a Kalman filter...")

    # Initialize filters at (100, 100)
    kf_cv = create_kalman_filter(100.0, 100.0)
    kf_ca = create_ca_kalman_filter(100.0, 100.0)

    # Burn-in period (30 frames of straight motion, vx=5)
    for i in range(30):
        kf_cv.predict()
        kf_ca.predict()
        
        # True motion
        meas_x = 100.0 + i * 5.0 + np.random.normal(0, 0.5)
        meas_y = 100.0 + np.random.normal(0, 0.5)
        
        measurement = np.array([[meas_x], [meas_y]])
        kf_cv.update(measurement)
        kf_ca.update(measurement)

    print("Filters burned in and converged.")
    print(f"Condition number of CV Covariance S: {np.linalg.cond(kf_cv.S):.4e}")
    print(f"Determinant of CV Covariance S:      {np.linalg.det(kf_cv.S):.4e}")
    
    # Store state to reset
    x_cv_base = kf_cv.x.copy()
    P_cv_base = kf_cv.P.copy()
    x_ca_base = kf_ca.x.copy()
    P_ca_base = kf_ca.P.copy()

    displacements = [10, 50, 100, 500, 1000, 3000]

    print("\n")
    print("| Artificial Displacement (px) | Innovation Norm ||y|| | Peak NIS (Combined) |")
    print("|-----------------------------:|----------------------:|--------------------:|")

    for d in displacements:
        # Reset state
        kf_cv.x = x_cv_base.copy()
        kf_cv.P = P_cv_base.copy()
        kf_ca.x = x_ca_base.copy()
        kf_ca.P = P_ca_base.copy()
        
        # Predict
        kf_cv.predict()
        kf_ca.predict()
        
        # Base true next position should be approx x=250, y=100
        # Displace it by +d in x
        meas_x = (100.0 + 30 * 5.0) + d
        meas_y = 100.0
        
        measurement = np.array([[meas_x], [meas_y]])
        kf_cv.update(measurement)
        kf_ca.update(measurement)
        
        # Compute NIS
        y_cv = kf_cv.y
        S_cv = kf_cv.S
        nis_cv = float(y_cv.T @ np.linalg.inv(S_cv) @ y_cv)
        
        y_ca = kf_ca.y
        S_ca = kf_ca.S
        nis_ca = float(y_ca.T @ np.linalg.inv(S_ca) @ y_ca)
        
        nis_combined = min(nis_cv, nis_ca)
        y_norm = np.linalg.norm(y_cv)
        
        print(f"| {d:>28} | {y_norm:>21.2f} | {nis_combined:>19.2f} |")

    print("\nSensitivity Analysis Complete.")

if __name__ == "__main__":
    main()
