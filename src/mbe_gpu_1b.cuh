__global__ void CUDA_MBE(int *NUM_L, int *NUM_R, int *NUM_EDGES, Node *node, int *edge,
                         int *u2L, int *L, int *R, int *P, int *Q,
                         int *x, int *L_lp, int *R_lp, int *P_lp, int *Q_lp) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_total_thds = gridDim.x * blockDim.x;
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 0x1f;
    int num_warps = blockDim.x >> 5;
    int num_maximal_bicliques = 0;
    __shared__ int lvl;
    __shared__ int *x_cur;
    __shared__ int *L_lp_cur, *L_lp_nxt;
    __shared__ int *R_lp_cur, *R_lp_nxt;
    __shared__ int *P_lp_cur, *P_lp_nxt;
    __shared__ int *Q_lp_cur, *Q_lp_nxt;
    __shared__ bool is_recursive;
    __shared__ bool is_maximal;
    __shared__ int num_L_nxt, num_N_v;
    // __shared__ int num_N_v[WARP_SIZE];

    __shared__ long long clk[NUM_CLK], clk_;

    if (!threadIdx.x) {
        clk_ = clock();
        for (int i = 0; i < NUM_CLK; i++)
            clk[i] = 0;
    }

    if (!threadIdx.x)
        lvl = 0;

    __syncthreads();

    for (; lvl >= 0; ) {

        // if (!threadIdx.x)
        //     printf("\nlvl: %d", lvl);

        x_cur    = &(   x[lvl]);
        L_lp_cur = &(L_lp[lvl]); L_lp_nxt = &(L_lp[lvl+1]);
        R_lp_cur = &(R_lp[lvl]); R_lp_nxt = &(R_lp[lvl+1]);
        P_lp_cur = &(P_lp[lvl]); P_lp_nxt = &(P_lp[lvl+1]);
        Q_lp_cur = &(Q_lp[lvl]); Q_lp_nxt = &(Q_lp[lvl+1]);

        if (!threadIdx.x)
            is_recursive = false;
        
        __syncthreads();

        // while P ≠ ∅ do
        while (*P_lp_cur != 0) {

            CLK(0);

            if (!threadIdx.x) {

                // printf("\n");
                // for (int i = 0; i < lvl; i++) printf("        ");
                // printf("L:");
                // for (int i = 0; i < *NUM_L; i++)
                //     printf(" %d", L[i]);
                // printf("\n");
                // for (int i = 0; i < lvl; i++) printf("        ");
                // printf("R:");
                // for (int i = 0; i < *R_lp_cur; i++)
                //     printf(" %d", R[i]);
                // printf("\n");
                // for (int i = 0; i < lvl; i++) printf("        ");
                // printf("P:");
                // for (int i = 0; i < *P_lp_cur; i++)
                //     printf(" %d", P[i]);
                // printf("\n");
                // for (int i = 0; i < lvl; i++) printf("        ");
                // printf("Q:");
                // for (int i = 0; i < *Q_lp_cur; i++)
                //     printf(" %d", Q[i]);

                // Select x from P;
                // P <--- P \ {x};
                *x_cur = P[--(*P_lp_cur)];
                //// printf("x: %d\n", *x_cur);
                
                // R' <--- R ∪ {x};
                *R_lp_nxt = *R_lp_cur;
                R[(*R_lp_nxt)++] = *x_cur;

                //// *L_lp_nxt = 0;
                num_L_nxt = 0;
            }
            
            __syncthreads();

            CLK(1);

            // |L'|
            for (int l = tid; l < *NUM_L; l += num_total_thds)
                L[l] = L[l] > lvl ? lvl : L[l];

            __syncthreads();

            CLK(2);
            
            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node[*x_cur].start + threadIdx.x, eid_end = node[*x_cur].start + node[*x_cur].length; eid < eid_end; eid += blockDim.x) {
                int l = edge[eid];
                if (L[l] == lvl) {
                    L[l]++;
                    atomicAdd(&num_L_nxt, 1);
                }
            }

            __syncthreads();

            CLK(3);

            //// printf("L':");
            //// for (int i = 0; i < *L_lp_nxt; i++)
            ////     printf(" %d", L[i]);
            //// printf("\n");
            
            if (!threadIdx.x) {
                
                // P' ← ∅; Q' ← ∅;
                *P_lp_nxt = 0; *Q_lp_nxt = *Q_lp_cur;
                is_maximal = true;

            }

            // foreach v ∈ Q
            for (int i = 0; i < *Q_lp_cur; i++) {

                int v = Q[i];

                if (!threadIdx.x)
                    num_N_v = 0; // |N[v]|

                __syncthreads();

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                    int l = edge[eid];
                    if (L[l] > lvl)
                        atomicAdd(&num_N_v, 1);
                }

                __syncthreads();
                
                // if |N[v]| = |L'| then
                if (num_N_v == num_L_nxt) {
                    is_maximal = false;
                    break;
                }
                // // else if |N[v]| > 0 then
                // else if (num_N_v == 0)
                //     // Q' ← Q' ∪ {v};
                //     swap(Q[(*Q_ls_nxt)++], Q[i]);

                __syncthreads();
                
            }
            
            CLK(4);

            __syncthreads();

            //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int i = 0; i < *P_lp_cur; i++) {
                    int v = P[i];

                    if (!threadIdx.x)
                        num_N_v = 0; // |N[v]|

                    __syncthreads();

                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                        int l = edge[eid];
                        if (L[l] > lvl)
                            atomicAdd(&num_N_v, 1);
                    }

                    __syncthreads();
                    
                    if (!threadIdx.x) {

                        // if |N[v]| = |L'| then
                        if (num_N_v == num_L_nxt)
                            // R' ← R' ∪ {v};
                            R[(*R_lp_nxt)++] = v;
                        // else if |N[v]| > 0 then
                        else if (num_N_v > 0) {
                            // P' ← P' ∪ {v};
                            int P_tmp = P[*P_lp_nxt];
                            P[(*P_lp_nxt)++] = P[i];
                            P[i] = P_tmp;
                        }

                    }

                    __syncthreads();

                }
            
                CLK(5);

                if (!threadIdx.x) {

#ifdef DEBUG
                    //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

                    // PRINT(L', R');
                    // printf("\n> Find maximal biclique (No. %d)", num_maximal_bicliques++);
                    // printf("\nL':");
                    // for (int i = 0; i < *NUM_L; i++)
                    //     if (L[i] > lvl)
                    //         printf(" %d", i);
                    // printf("\nR':");
                    // for (int i = 0; i < *R_lp_nxt; i++)
                    //     printf(" %d", R[i]);
                    // printf("\n");

                    // save maximal bicliques
                    //// Biclique new_maximal_bicliques;
                    //// for (int i = 0; i < *L_lp_nxt; i++)
                    ////     new_maximal_bicliques.L.insert(L[i]);
                    //// for (int i = 0; i < *R_lp_nxt; i++)
                    ////     new_maximal_bicliques.R.insert(R[i]);
                    //// maximal_bicliques.push_back(new_maximal_bicliques);
                    if (++num_maximal_bicliques << 22 == 0)
                        printf("%d\n", num_maximal_bicliques);
#endif /* DEBUG */

                    // if P' ≠ ∅ then
                    if (*P_lp_nxt != 0) {
                        // biclique_find(G, L', R', P', Q');
                        //// printf("\n往 下 安安");
                        lvl++;
                        is_recursive = true;
                    }

                }

                __syncthreads();
            
                CLK(6);

                if (is_recursive)
                    break;

            }
            else {
                //// printf("\n不安安");
            }

            if (!threadIdx.x) {

                // Q ← Q ∪ {x};
                Q[(*Q_lp_cur)++] = *x_cur;
                //// printf("\n往 右 安安");

            }

        }
        
        __syncthreads();

        // printf("tid: %d, lvl: %d\n", tid, lvl);

        if (!threadIdx.x) {

            if (!is_recursive) {
                lvl--;
                Q[Q_lp[lvl]++] = x[lvl];
                //// printf("\n往 上 安安");
                //// printf("\n往 右 安安");
            }

        }

        __syncthreads();

    }

    if (!threadIdx.x) {
        printf("maximal bicliques: %d\n", num_maximal_bicliques);
        printf("time:");
        for (int i = 0; i < NUM_CLK; i++) {
            // clk[i] >>= 21;
            printf(" %lld", clk[i]);
        }
        printf("\n");
    }
}