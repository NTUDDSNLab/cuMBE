__global__ void CUDA_GLOBAL_EXTENSION(int *g_u2L, int *g_v2P, int *g_v2Q, int *g_L, int *g_R, int *g_P, int *g_Q, int *g_x,
                                      int *g_L_lp, int *g_R_lp, int *g_P_lp, int *g_Q_lp, int *g_L_buf, int *g_num_N_u, int *g_pre_min,
                                      int *u2L, int *v2P, int *v2Q, int *L, int *R, int *P, int *Q, int *x,
                                      int *L_lp, int *R_lp, int *P_lp, int *Q_lp, int *L_buf, int *num_N_u, int *pre_min,
                                      int *NUM_L, int *NUM_R) {
    
    __shared__ int offset_L, offset_R;

    if (!threadIdx.x) {
        offset_L = blockIdx.x * (*NUM_L);
        offset_R = blockIdx.x * (*NUM_R);
    }

    __syncthreads();
    
    for (int i = threadIdx.x; i < *NUM_L; i += blockDim.x) {
        g_u2L  [offset_L + i] = u2L  [i];
        g_L    [offset_L + i] = L    [i];
        g_L_buf[offset_L + i] = L_buf[i];
    }
    
    for (int i = threadIdx.x; i < *NUM_R; i += blockDim.x) {
        g_v2P    [offset_R + i] = v2P    [i];
        g_v2Q    [offset_R + i] = v2Q    [i];
        g_R      [offset_R + i] = R      [i];
        g_P      [offset_R + i] = P      [i];
        g_Q      [offset_R + i] = Q      [i];
        g_x      [offset_R + i] = x      [i];
        g_L_lp   [offset_R + i] = L_lp   [i];
        g_R_lp   [offset_R + i] = R_lp   [i];
        g_P_lp   [offset_R + i] = P_lp   [i];
        g_Q_lp   [offset_R + i] = Q_lp   [i];
        g_num_N_u[offset_R + i] = num_N_u[i];
        g_pre_min[offset_R + i] = pre_min[i];
    }
    
}