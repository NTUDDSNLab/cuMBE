#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <sys/ioctl.h>
using namespace std;
using namespace std::chrono;
using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define NUM_THDS 512

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

struct EdgePair
{
	int row;
	int col;
    EdgePair(int row_, int col_): row(row_), col(col_) {}
};

typedef struct
{
    int num_rows;
    int num_cols;
    unordered_set<int> row;
    unordered_set<int> col;
} Biclique;

typedef struct
{
    int row_end, r;
    int col_end, c;
    int *row;
    int *col;
} Candidate;

random_device rd;
mt19937 en(rd());
uniform_int_distribution<unsigned long long> rand_64;

char t_ms_idx = 0; long long t_ms_start = 0, t_ms_end = 0; vector<long long> t_ms(13, 0);
void mark_time_ms(short new_init_idx) {
    // t_ms_end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    t_ms_end = clock();
    if (new_init_idx >= 0) t_ms_idx = new_init_idx;
    else t_ms[t_ms_idx++] += t_ms_end - t_ms_start;
    t_ms_start = t_ms_end;
    return;
}

char choose_op(int num_rows_cand, int num_cols_cand, int num_rows_bic, int num_cols_bic) {
    // cout << num_rows_cand << ' ' << num_cols_cand << ' ' << num_rows_bic << ' ' << num_cols_bic << ' ';
    if ((num_rows_cand == 0 && num_rows_bic == 1) || (num_cols_cand == 0 && num_cols_bic == 1) || (num_rows_cand | num_cols_cand) == 0) return 'x';
    long long prob_rows = 1, prob_cols = 1;
    if (num_rows_bic != 1 || num_cols_bic != 1) {
        prob_rows = (num_rows_cand != 0) * (num_cols_bic - 1);
        prob_cols = (num_cols_cand != 0) * (num_rows_bic - 1) * 2;
    }
    uniform_int_distribution<> rand_cand(0, prob_rows + prob_cols - 1);
    return (rand_cand(en) < prob_rows ? 'r' : 'c');
}

__global__ void CUDA_TRANS_CSPARSE(int *tmp, Node *node_i, int *edge_i, Node *node_o, int *edge_o, long long *NUM_NODES, int *NUM_EDGES)
{
    grid_group grid = this_grid();
    int num_thds  = blockDim.x * gridDim.x;
    // int num_warps = num_thds >> 5;
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id == 0)
        tmp[0] = node_o[0].start = node_o[0].length = 0;
    grid.sync();
    for (int nid = 1 + id; nid < *NUM_NODES; nid += num_thds)
        node_o[nid].length = 0;
    grid.sync();
    for (int eid = id; eid < *NUM_EDGES; eid += num_thds)
        atomicAdd(&(node_o[edge_i[eid]].length), 1);
    grid.sync();
    for (int nid = 1 + id; nid < *NUM_NODES; nid += num_thds)
        node_o[nid].start = node_o[nid - 1].length;
    grid.sync();
    for (int offset = 1; offset < *NUM_NODES; offset <<= 1) {
        for (int nid = *NUM_NODES - (num_thds - id); nid >= offset; nid -= num_thds)
            tmp[nid] = node_o[nid - offset].start + node_o[nid].start;
        grid.sync();
        for (int nid = *NUM_NODES - (num_thds - id); nid >= offset; nid -= num_thds)
            node_o[nid].start = tmp[nid];
        grid.sync();
    }
    // for (int nid = id >> 5; nid < *NUM_NODES; nid += num_warps)
    //     for (int eid = node_i[nid].start + threadIdx.x & 0x1f, eid_end = node_i[nid].start + node_i[nid].length; eid < eid_end; eid += 32)
    //         edge_o[atomicAdd(&(tmp[edge_i[eid]]), 1)] = nid;
    for (int nid = 0; nid < *NUM_NODES; nid++) {
        for (int eid = node_i[nid].start + id, eid_end = node_i[nid].start + node_i[nid].length; eid < eid_end; eid += num_thds)
            edge_o[tmp[edge_i[eid]]++] = nid;
        grid.sync();
    }
}

int main(int argc, char* argv[])
{
    string str_dataset = argv[1];
    // printf("\033[0;1;33m");
    cout << str_dataset.substr(str_dataset.rfind('/')+1) << "\n";
    // printf("\033[0;1m");
	ifstream fin;
    int _, *NUM_EDGES, SOURCE;
    long long *NUM_NODES;
    cudaMallocManaged(&NUM_EDGES, sizeof(int));
    cudaMallocManaged(&NUM_NODES, sizeof(long long));
    fin.open(argv[1]);
    fin >> *NUM_NODES >> *NUM_EDGES >> SOURCE;

	Node* node;
	int* edge;
    cudaMallocManaged(&node, sizeof(Node)*(*NUM_NODES));
    cudaMallocManaged(&edge, sizeof(int)*(*NUM_EDGES));
    for(int i=0;i<*NUM_NODES;i++) fin >> node[i].start >> node[i].length;
    for(int i=0;i<*NUM_EDGES;i++) fin >> edge[i] >> _;
    fin.close();

    Node *node_r = node, *node_c;
	int *edge_r = edge, *edge_c, *tmp;
    // Edge_Pair *edge_p;
    cudaMallocManaged(&tmp, sizeof(int)*(*NUM_NODES));
    cudaMallocManaged(&node_c, sizeof(Node)*(*NUM_NODES));
    cudaMallocManaged(&edge_c, sizeof(int)*(*NUM_EDGES));
    // cudaMallocManaged(&edge_p, sizeof(Edge_Pair)*(*NUM_EDGES));
    
    cout << "Nodes: " << *NUM_NODES << "\n";
    cout << "Edges: " << *NUM_EDGES << "\n";

    int numBlocksPerSM = 1;
    int numThreads = NUM_THDS;
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, CUDA_TRANS_CSPARSE, numThreads, 0);
    dim3 num_blocks_TRANS_CSPARSE(deviceProp.multiProcessorCount * numBlocksPerSM, 1, 1);
    dim3 block_size(numThreads, 1, 1);
    void *kernelArgs_TRANS_CSR2CSC[] = {&tmp, &node_r, &edge_r, &node_c, &edge_c, &NUM_NODES, &NUM_EDGES};
    void *kernelArgs_TRANS_CSC2CSR[] = {&tmp, &node_c, &edge_c, &node_r, &edge_r, &NUM_NODES, &NUM_EDGES};
    cout << "block_size: " << numThreads << "\n";
    cout << "num_blocks_TRANS_CSPARSE:   " << num_blocks_TRANS_CSPARSE.x << "\n";
    
    // cout << "Kernel Start\n";
    cudaLaunchCooperativeKernel((void*)CUDA_TRANS_CSPARSE, num_blocks_TRANS_CSPARSE, block_size, kernelArgs_TRANS_CSR2CSC);
    cudaDeviceSynchronize();
    cudaLaunchCooperativeKernel((void*)CUDA_TRANS_CSPARSE, num_blocks_TRANS_CSPARSE, block_size, kernelArgs_TRANS_CSC2CSR);
    cudaDeviceSynchronize();
    cudaFree(tmp);

    bool check_r = 1, check_c = 1;
    for (int i = 0; i < *NUM_NODES; i++)
        for (int j_ = node_r[i].start, j__end = j_ + node_r[i].length - 1; j_ < j__end; j_++)
            if (edge_r[j_] >= edge_r[j_ + 1]) check_r = 0;
    for (int i = 0; i < *NUM_NODES; i++)
        for (int j_ = node_c[i].start, j__end = j_ + node_c[i].length - 1; j_ < j__end; j_++)
            if (edge_c[j_] >= edge_c[j_ + 1]) check_c = 0;
    cout << "order check: CSR(" << (check_r ? "PASS" : "FAIL") << "), CSC(" << (check_c ? "PASS" : "FAIL") << ")\n";
    
    long long preprocess_time = time(0);
    uniform_int_distribution<> rand_node(0, *NUM_NODES - 1);
    uniform_int_distribution<> rand_edge(0, *NUM_EDGES - 1);
    vector<Biclique*> biclique;
    vector<EdgePair*> edge_lonely;
    Biclique *bic_ptr;
    Candidate *cand_ptr;
    int biclique_cover = 0, biclique_score = 0;
    long long *NUM_BICS;
    int *NUM_BIC_EDGES;
    cudaMallocManaged(&NUM_BICS, sizeof(long long));
    cudaMallocManaged(&NUM_BIC_EDGES, sizeof(int));
    *NUM_BICS = *NUM_BIC_EDGES = 0;
    // progress bar
    short bar_size_max = 0, set_width = log10(*NUM_NODES - 1) + 1;
    string str_bar[8] = {"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"};
    
    for (int nid_max_r = *NUM_NODES - 1, nid_now_r = 0, eid_now_r = 0, cid, rid; true; ) {

        mark_time_ms(0);
        
        rid = nid_now_r;
        cid = edge_r[eid_now_r];
        // cout << "\nFrom edge " << eid_now_r << ": row " << (rid) << ", col " << (cid) << "\n";

        cand_ptr = new Candidate;
        bic_ptr = new Biclique;
        cand_ptr->r = cand_ptr->c = 0;
        bic_ptr->row.insert(rid); bic_ptr->col.insert(cid);
        bic_ptr->num_rows = bic_ptr->num_cols = 1;

        mark_time_ms(-1);

        cand_ptr->col = new int[(cand_ptr->col_end = node_r[rid].length)--];
        copy(edge_r + node_r[rid].start, edge_r + node_r[rid].start + node_r[rid].length, cand_ptr->col);
        
        cand_ptr->row = new int[(cand_ptr->row_end = node_c[cid].length)--];
        copy(edge_c + node_c[cid].start, edge_c + node_c[cid].start + node_c[cid].length, cand_ptr->row);

        mark_time_ms(-1);

        for (int cols = cand_ptr->col_end; cols > 0; ) {
            swap(cand_ptr->col[rand_64(en)%cols], cand_ptr->col[cols - 1]);
            if (cand_ptr->col[--cols] == cid)
                cand_ptr->col[cols] = cand_ptr->col[cand_ptr->col_end];
        }
        for (int rows = cand_ptr->row_end; rows > 0; ) {
            swap(cand_ptr->row[rand_64(en)%rows], cand_ptr->row[rows - 1]);
            if (cand_ptr->row[--rows] == rid)
                cand_ptr->row[rows] = cand_ptr->row[cand_ptr->row_end];
        }

        mark_time_ms(-1);
        
        for (char op; true; ) {
            op = choose_op(cand_ptr->row_end - cand_ptr->r, cand_ptr->col_end - cand_ptr->c, bic_ptr->num_rows, bic_ptr->num_cols);
            // cout << "Operation " << op << "\n";
            if (op == 'r') {
                rid = cand_ptr->row[cand_ptr->r++];
                unordered_set<int> col_tmp(edge_r + node_r[rid].start, edge_r + node_r[rid].start + node_r[rid].length);
                for (const auto &col : bic_ptr->col) col_tmp.erase(col);
                // cout << "row " << rid << " successed.\n";
                bic_ptr->row.insert(rid);
                bic_ptr->num_rows++;
                for (int c = cand_ptr->c; c < cand_ptr->col_end; )
                    if (col_tmp.erase(cand_ptr->col[c])) c++;
                    else cand_ptr->col[c] = cand_ptr->col[--cand_ptr->col_end];
            }
            else if (op == 'c') {
                cid = cand_ptr->col[cand_ptr->c++];
                unordered_set<int> row_tmp(edge_c + node_c[cid].start, edge_c + node_c[cid].start + node_c[cid].length);
                for (const auto &row : bic_ptr->row) row_tmp.erase(row);
                // cout << "col " << cid << " successed.\n";
                bic_ptr->col.insert(cid);
                bic_ptr->num_cols++;
                for (int r = cand_ptr->r; r < cand_ptr->row_end; )
                    if (row_tmp.erase(cand_ptr->row[r])) r++;
                    else cand_ptr->row[r] = cand_ptr->row[--cand_ptr->row_end];
            }
            else break;
        }

        mark_time_ms(-1);

        delete [] cand_ptr->row;
        delete [] cand_ptr->col;
        delete cand_ptr;

        mark_time_ms(-1);

        // if ((bic_ptr->num_rows == 1) ^ (bic_ptr->num_cols == 1)) eid_now_r++;
        if ((bic_ptr->num_rows - 1) * (bic_ptr->num_cols - 1) < 8 && ((bic_ptr->num_rows != 1) || (bic_ptr->num_cols != 1))) eid_now_r++;
        else {
            // cout << "Delete\n";
            for (const auto &row : bic_ptr->row) {
                int eid = node_r[row].start, eid_end = eid + node_r[row].length;
                if (row == nid_now_r)
                    while (eid < eid_end)
                        if (bic_ptr->col.find(edge_r[eid]) == bic_ptr->col.end()) eid++;
                        else if (eid < eid_now_r) {
                            swap(edge_r[eid], edge_r[--eid_now_r]);
                            edge_r[eid_now_r] = edge_r[--eid_end];
                        }
                        else edge_r[eid] = edge_r[--eid_end];
                else
                    while (eid < eid_end)
                        if (bic_ptr->col.find(edge_r[eid]) == bic_ptr->col.end()) eid++;
                        else edge_r[eid] = edge_r[--eid_end];
                node_r[row].length = eid_end - node_r[row].start;
            }
            for (const auto &col : bic_ptr->col) {
                int eid = node_c[col].start, eid_end = eid + node_c[col].length;
                while (eid < eid_end)
                    if (bic_ptr->row.find(edge_c[eid]) == bic_ptr->row.end()) eid++;
                    else edge_c[eid] = edge_c[--eid_end];
                node_c[col].length = eid_end - node_c[col].start;
            }
            if (bic_ptr->num_rows == 1 && bic_ptr->num_cols == 1) {
                edge_lonely.push_back(new EdgePair(*begin(bic_ptr->row), *begin(bic_ptr->col)));
                delete bic_ptr;
            }
            else {
                (*NUM_BICS)++;
                *NUM_BIC_EDGES += bic_ptr->num_cols;
                biclique_cover += bic_ptr->num_rows * bic_ptr->num_cols;
                biclique_score += (bic_ptr->num_rows - 1) * (bic_ptr->num_cols - 1);
                biclique.push_back(bic_ptr);
            }
        }

        mark_time_ms(-1);

        while (node_r[nid_max_r].length == 0) nid_max_r--;
        if (eid_now_r >= node_r[nid_max_r].start + node_r[nid_max_r].length) break;
        if (eid_now_r == node_r[nid_now_r].start + node_r[nid_now_r].length) {
            while (node_r[++nid_now_r].length == 0) ;
            eid_now_r = node_r[nid_now_r].start;
        }

        mark_time_ms(-1);

        // // progress bar
        // struct winsize w;
        // ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        // short bar_width = (w.ws_col - (short)(6 + set_width * 2)) << 3;
        // if (bar_width < 8) continue;
        // long long bar_size = nid_now_r * bar_width / nid_max_r;
        // if (bar_size > bar_size_max) {
        //     bar_size_max = bar_size;
        //     // cout << (rand()%((bar_size>>3)+1) ? "." : ".\n");
        //     cout << "\33[1A|";
        //     for (short i = bar_size >> 3; i > 0; i--) cout << "█";
        //     cout << str_bar[bar_size % 8];
        //     for (short i = bar_width - bar_size >> 3; i > 0; i--) cout << " ";
        //     cout << "| " << setw(set_width) << nid_now_r << "/" << nid_max_r << " |\n";
        // }

        mark_time_ms(-1);
    }

    mark_time_ms(8);

    for (int rid, cid; !edge_lonely.empty(); edge_lonely.pop_back()) {
        EdgePair* edge_ptr = edge_lonely.back();
        rid = edge_ptr->row;
        cid = edge_ptr->col;
        delete edge_ptr;
        edge_r[node_r[rid].start + (node_r[rid].length++)] = cid;
        // edge_c[node_c[cid].start + (node_c[cid].length++)] = rid;
    }

    mark_time_ms(-1);

    cudaFree(node_c);
    cudaFree(edge_c);
    cudaMallocManaged(&node_c, sizeof(Node)*(*NUM_BICS));     // biclique list (node)
    cudaMallocManaged(&edge_c, sizeof(int)*(*NUM_BIC_EDGES)); // biclique list (edge)

    mark_time_ms(-1);

    for (int bid = 0, eid = 0; bid < *NUM_BICS; bid++) {
        bic_ptr = biclique[bid];
        for (const auto &row : bic_ptr->row)
            edge_r[node_r[row].start + (node_r[row].length++)] = ~bid;
        node_c[bid].start  = eid;
        node_c[bid].length = bic_ptr->num_cols;
        for (const auto &col : bic_ptr->col)
            edge_c[eid++] = col;
        delete bic_ptr;
    }
    biclique.clear();

    mark_time_ms(-1);

    *NUM_EDGES = node[0].length;
    for (int nid = 1; nid < *NUM_NODES; nid++) {
        int eid_start = node[nid].start;
        node[nid].start = *NUM_EDGES;
        for (int eid = eid_start, eid_end = eid + node[nid].length; eid < eid_end; eid++) {
            edge[(*NUM_EDGES)++] = edge[eid];
        }
    }

    mark_time_ms(-1);

    ofstream fout;
    fout.open(argv[2]);
    fout << *NUM_NODES << ' ' << *NUM_EDGES << "\n\n";
    for (int i = 0; i < *NUM_NODES; i++)
        fout << node[i].start << ' ' << node[i].length << "\n";
    fout << "\n";
    for (int i = 0; i < *NUM_EDGES; i++)
        fout << edge[i] << "\n";
    fout << "\n";
    fout << *NUM_BICS << ' ' << *NUM_BIC_EDGES << "\n\n";
    for (int i = 0; i < *NUM_BICS; i++)
        fout << node_c[i].start << ' ' << node_c[i].length << "\n";
    fout << "\n";
    for (int i = 0; i < *NUM_BIC_EDGES; i++)
        fout << edge_c[i] << "\n";
    fout << "\n";
    fout.close();

    cout << "t_ms:"; for (short i = 0; i < t_ms.size(); i++) cout << ' ' << (t_ms[i] >> 10); cout << "\n";
    cout << "Process time is " << (preprocess_time = time(0) - preprocess_time) << "s\n";
    cout << "found " << *NUM_BICS << " bicliques" << "\n";
    cout << "biclique cover: " << biclique_cover << "\n";
    cout << "avg cover: " << (double)biclique_cover / (*NUM_BICS) << "\n";
    cout << "biclique score: " << biclique_score << "\n";
    cout << "avg score: " << (double)biclique_score / (*NUM_BICS) << "\n";

    cudaFree(NUM_EDGES);
    cudaFree(NUM_NODES);
    cudaFree(NUM_BICS);
    cudaFree(NUM_BIC_EDGES);
    cudaFree(node_r);
    cudaFree(edge_r);
    cudaFree(node_c);
    cudaFree(edge_c);
}