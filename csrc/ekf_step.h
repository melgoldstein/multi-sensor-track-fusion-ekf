#ifndef TRACK_FUSION_EKF_STEP_H
#define TRACK_FUSION_EKF_STEP_H
/* Row-major buffers. Caller validates finite values and positive covariances.
 * state: 4 doubles; covariance: 16; jacobian: dimensions*4;
 * innovation: dimensions; variance: dimensions*dimensions.
 * Distinct, non-overlapping writable state/covariance buffers are required. */
int ekf_predict(double *state, double *covariance, double dt, double noise);
int ekf_update(double *state, double *covariance, const double *jacobian,
               const double *innovation, const double *variance, int dimensions);
#endif
