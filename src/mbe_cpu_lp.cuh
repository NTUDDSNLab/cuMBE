void maximal_bic_enum(int *NUM_L, int *NUM_R, int *NUM_EDGES, Node *node, int *edge,
                      int *u2L, int *L, int *R, int *P, int *Q,
                      int *x, int *L_lp, int *R_lp, int *P_lp, int *Q_lp) {
    vector<Biclique> maximal_bicliques;
    int num_maximal_bicliques = 0;

    long long clk[NUM_CLK] = { 0 }, clk_ = clock();
    
    for (int lvl = 0; lvl >= 0; ) {

        // printf("lvl: %d\n", lvl);

        int *x_cur    = &(   x[lvl]);
        int *L_lp_cur = &(L_lp[lvl]), *L_lp_nxt = &(L_lp[lvl+1]);
        int *R_lp_cur = &(R_lp[lvl]), *R_lp_nxt = &(R_lp[lvl+1]);
        int *P_lp_cur = &(P_lp[lvl]), *P_lp_nxt = &(P_lp[lvl+1]);
        int *Q_lp_cur = &(Q_lp[lvl]), *Q_lp_nxt = &(Q_lp[lvl+1]);
        bool is_recursive = false;

        // while P ≠ ∅ do
        while (*P_lp_cur != 0) {

            CLK_CPU(0);

            //string tab_level(lvl << 3, ' ');
            //printf("\n%sL:", tab_level.c_str());
            //for (int i = 0; i < *L_lp_cur; i++)
            //    printf(" %d", L[i]);
            //printf("\n%sR:", tab_level.c_str());
            //for (int i = 0; i < *R_lp_cur; i++)
            //    printf(" %d", R[i]);
            //printf("\n%sP:", tab_level.c_str());
            //for (int i = 0; i < *P_lp_cur; i++)
            //    printf(" %d", P[i]);
            //printf("\n%sQ:", tab_level.c_str());
            //for (int i = 0; i < *Q_lp_cur; i++)
            //    printf(" %d", Q[i]);

            // Select x from P;
            // P <--- P \ {x};
            *x_cur = P[--(*P_lp_cur)];
            //// printf("x: %d\n", *x_cur);
            
            // R' <--- R ∪ {x};
            *R_lp_nxt = *R_lp_cur;
            R[(*R_lp_nxt)++] = *x_cur;

            CLK_CPU(1);

            *L_lp_nxt = 0; // |L'|

            CLK_CPU(2);

            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node[*x_cur].start, eid_end = eid + node[*x_cur].length; eid < eid_end; eid++) {
                int u = edge[eid];
                int l = u2L[u];
                if (l < *L_lp_cur) {
                    swap(L[(*L_lp_nxt)++], L[l]);
                    swap(u2L[L[l]], u2L[u]);
                }
            }

            CLK_CPU(3);

            //// printf("L':");
            //// for (int i = 0; i < *L_lp_nxt; i++)
            ////     printf(" %d", L[i]);
            //// printf("\n");
            
            // P' ← ∅; Q' ← ∅;
            *P_lp_nxt = 0; *Q_lp_nxt = *Q_lp_cur;

            bool is_maximal = true;

            // foreach v ∈ Q
            for (int i = 0; i < *Q_lp_cur; i++) {

                int v = Q[i];

                int num_N_v = 0; // |N[v]|
                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start, eid_end = eid + node[v].length; eid < eid_end; eid++) {
                    int u = edge[eid];
                    int l = u2L[u];
                    if (l < *L_lp_nxt)
                        num_N_v++;
                }
                
                // if |N[v]| = |L'| then
                if (num_N_v == *L_lp_nxt) {
                    is_maximal = false;
                    break;
                }
                // // else if |N[v]| > 0 then
                // else if (num_N_v == 0)
                //     // Q' ← Q' ∪ {v};
                //     swap(Q[(*Q_ls_nxt)++], Q[i]);
                
            }

            CLK_CPU(4);

            //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int i = 0; i < *P_lp_cur; i++) {
                    int v = P[i];

                    int num_N_v = 0; // |N[v]|
                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node[v].start, eid_end = eid + node[v].length; eid < eid_end; eid++) {
                        int u = edge[eid];
                        int l = u2L[u];
                        if (l < *L_lp_nxt)
                            num_N_v++;
                    }
                    
                    // if |N[v]| = |L'| then
                    if (num_N_v == *L_lp_nxt)
                        // R' ← R' ∪ {v};
                        R[(*R_lp_nxt)++] = v;
                    // else if |N[v]| > 0 then
                    else if (num_N_v > 0)
                        // P' ← P' ∪ {v};
                        swap(P[(*P_lp_nxt)++], P[i]);

                }

                CLK_CPU(5);

#ifdef DEBUG
                //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

                // PRINT(L', R');
                //                printf("\n> Find maximal biclique (No. %d)", num_maximal_bicliques++);
                //                printf("\nL':");
                //                for (int i = 0; i < *L_lp_nxt; i++)
                //                    printf(" %d", L[i]);
                //                printf("\nR':");
                //                for (int i = 0; i < *R_lp_nxt; i++)
                //                    printf(" %d", R[i]);
                //                printf("\n");
                //
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

                CLK_CPU(6);

                // if P' ≠ ∅ then
                if (*P_lp_nxt != 0) {
                    // biclique_find(G, L', R', P', Q');
                    //// printf("\n往 下 安安");
                    lvl++;
                    is_recursive = true;
                    break;
                }

            }
            else {
                //// printf("\n不安安");
            }

            // Q ← Q ∪ {x};
            Q[(*Q_lp_cur)++] = *x_cur;
            //// printf("\n往 右 安安");
        }

        if (!is_recursive) {
            lvl--;
            Q[Q_lp[lvl]++] = x[lvl];
            //// printf("\n往 上 安安");
            //// printf("\n往 右 安安");
        }

    }

    printf("maximal bicliques: %d\n", num_maximal_bicliques);
    printf("time:");
    for (int i = 0; i < NUM_CLK; i++) {
        // clk[i] >>= 21;
        printf(" %lld", clk[i]);
    }
    printf("\n");

    if (*NUM_R > 20 || *NUM_L > 20) return;

    string _ = "";
    printf("\33[2J\33[0;0H");

    printf("  ");
    for (int i = 0; i < *NUM_L; i++)
        printf(" %d", i / 10);
    printf("\n  ");
    for (int i = 0; i < *NUM_L; i++)
        printf(" %d", i % 10);
    printf("\n");
    for (int i = 0; i < *NUM_R; i++) {
        bool adj_vec[*NUM_L] = { false };
        for (int j = node[i].start, j_end = j + node[i].length; j < j_end; j++)
            adj_vec[edge[j]] = true;
        printf("%d%d", i / 10, i % 10);
        for (int j = 0; j < *NUM_L; j++)
            printf(" %c", adj_vec[j] ? '#' : '-');
        printf("\n");
    }

    for (int i = 0, i_end = maximal_bicliques.size(); i < i_end; i++) {
        printf("\33[7m");
        for (const auto &r: maximal_bicliques[i].R)
            for (const auto &l: maximal_bicliques[i].L)
                printf("\33[%d;%dH#\n", 3 + r, 4 + l * 2);
        printf("\33[0m\n\33[%d;0H\n", 3 + (*NUM_R));
        if      (_ == "auto") usleep(800000);
        else if (_ != "exit") cin >> _;
        for (const auto &r: maximal_bicliques[i].R)
            for (const auto &l: maximal_bicliques[i].L)
                printf("\33[%d;%dH#\n", 3 + r, 4 + l * 2);
    }
    printf("\33[%d;0H\n", 3 + (*NUM_R));
}

