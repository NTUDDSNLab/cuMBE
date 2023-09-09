__global__ void CUDA_TRANSPOSE(long long *NUM_i, long long *NUM_o, int *NUM_EDGES,
                               Node *node_i, int *edge_i, Node *node_o, int *edge_o,
                               int *tmp) {
    
    grid_group grid = this_grid();
    // int num_warps = num_thds >> 5;
    int num_thds  = blockDim.x * gridDim.x;
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id == 0)
        tmp[0] = node_o[0].start = 0;
    
    grid.sync();

    for (int nid = id; nid < *NUM_o; nid += num_thds)
        node_o[nid].length = 0;
    
    grid.sync();

    for (int eid = id; eid < *NUM_EDGES; eid += num_thds)
        atomicAdd(&(node_o[edge_i[eid]].length), 1);
    
    grid.sync();
    
    for (int nid = 1 + id; nid < *NUM_o; nid += num_thds)
        node_o[nid].start = node_o[nid - 1].length;
    
    grid.sync();
    
    for (int offset = 1; offset < *NUM_o; offset <<= 1) {
        
        for (int nid = *NUM_o - (num_thds - id); nid >= offset; nid -= num_thds)
            tmp[nid] = node_o[nid - offset].start + node_o[nid].start;
        
        grid.sync();
        
        for (int nid = *NUM_o - (num_thds - id); nid >= offset; nid -= num_thds)
            node_o[nid].start = tmp[nid];
        
        grid.sync();

    }

    // for (int nid = id >> 5; nid < *NUM_NODES; nid += num_warps)
    //     for (int eid = node_i[nid].start + threadIdx.x & 0x1f, eid_end = node_i[nid].start + node_i[nid].length; eid < eid_end; eid += 32)
    //         edge_o[atomicAdd(&(tmp[edge_i[eid]]), 1)] = nid;

    for (int nid = 0; nid < *NUM_i; nid++) {
        for (int eid = node_i[nid].start + id, eid_end = node_i[nid].start + node_i[nid].length; eid < eid_end; eid += num_thds)
            edge_o[tmp[edge_i[eid]]++] = nid;
        grid.sync();
    }
}