#include <math.h>
#include <string.h>

int ekf_predict(double *state, double *covariance, double dt, double noise) {
    if (!state || !covariance || !isfinite(dt) || dt<0 || !isfinite(noise) || noise<0) return 0;
    double transition[16]={1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1};
    double product[16]={0}, next[16]={0};
    for (int row=0;row<4;++row) for (int col=0;col<4;++col)
        for (int inner=0;inner<4;++inner)
            product[row*4+col]+=transition[row*4+inner]*covariance[inner*4+col];
    for (int row=0;row<4;++row) for (int col=0;col<4;++col)
        for (int inner=0;inner<4;++inner)
            next[row*4+col]+=product[row*4+inner]*transition[col*4+inner];
    for (int axis=0;axis<2;++axis) {
        next[axis*4+axis]+=noise*dt*dt*dt/3;
        next[axis*4+axis+2]+=noise*dt*dt/2;
        next[(axis+2)*4+axis]+=noise*dt*dt/2;
        next[(axis+2)*4+axis+2]+=noise*dt;
    }
    state[0]+=dt*state[2]; state[1]+=dt*state[3];
    memcpy(covariance,next,sizeof next);
    return 1;
}

int ekf_update(double *state, double *covariance, const double *jacobian,
               const double *innovation, const double *variance, int dimensions) {
    if (dimensions<1 || dimensions>2) return 0;
    double cross[8]={0}, residual_cov[4]={0}, inverse[4]={0};
    for (int row=0;row<4;++row) for (int col=0;col<dimensions;++col)
        for (int inner=0;inner<4;++inner)
            cross[row*dimensions+col]+=covariance[row*4+inner]*jacobian[col*4+inner];
    for (int row=0;row<dimensions;++row) for (int col=0;col<dimensions;++col) {
        residual_cov[row*dimensions+col]=variance[row*dimensions+col];
        for (int inner=0;inner<4;++inner)
            residual_cov[row*dimensions+col]+=jacobian[row*4+inner]*cross[inner*dimensions+col];
    }
    if (dimensions==1) {
        if (!(residual_cov[0]>0)) return 0;
        inverse[0]=1/residual_cov[0];
    } else {
        double determinant=residual_cov[0]*residual_cov[3]-residual_cov[1]*residual_cov[2];
        if (!(determinant>0) || !isfinite(determinant)) return 0;
        inverse[0]=residual_cov[3]/determinant; inverse[1]=-residual_cov[1]/determinant;
        inverse[2]=-residual_cov[2]/determinant; inverse[3]=residual_cov[0]/determinant;
    }
    double gain[8]={0}, correction[16]={0}, product[16]={0}, posterior[16]={0};
    for (int row=0;row<4;++row) for (int col=0;col<dimensions;++col)
        for (int inner=0;inner<dimensions;++inner)
            gain[row*dimensions+col]+=cross[row*dimensions+inner]*inverse[inner*dimensions+col];
    for (int row=0;row<4;++row) for (int col=0;col<4;++col) {
        correction[row*4+col]=(row==col);
        for (int inner=0;inner<dimensions;++inner)
            correction[row*4+col]-=gain[row*dimensions+inner]*jacobian[inner*4+col];
    }
    for (int row=0;row<4;++row) for (int col=0;col<4;++col)
        for (int inner=0;inner<4;++inner)
            product[row*4+col]+=correction[row*4+inner]*covariance[inner*4+col];
    for (int row=0;row<4;++row) for (int col=0;col<4;++col) {
        for (int inner=0;inner<4;++inner)
            posterior[row*4+col]+=product[row*4+inner]*correction[col*4+inner];
        for (int first=0;first<dimensions;++first) for (int second=0;second<dimensions;++second)
            posterior[row*4+col]+=gain[row*dimensions+first]*variance[first*dimensions+second]*gain[col*dimensions+second];
    }
    for (int row=0;row<4;++row) for (int col=0;col<dimensions;++col)
        state[row]+=gain[row*dimensions+col]*innovation[col];
    memcpy(covariance,posterior,sizeof posterior);
    return 1;
}
