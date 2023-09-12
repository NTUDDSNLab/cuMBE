# cuMBE: [Accelerating Maximal Biclique Enumeration on GPUs](https://scholar.google.com.tw/citations?user=4ypE90IAAAAJ&hl=zh-TW&oi=sra)

## 1. Getting started Instructions.
- Clone this project
`git clone git@github.com:NTUDSNLab/MBE.git`
- Hardware:
    - `CPU x86_64` (Test on Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz)
    - `NVIDIA GPU (arch>=86)` with device memory >= 12GB.(Support NVIDIA RTX3080(sm_86). Note that we mainly evaluate our experience on RTX3090. The execution time could be different with different devices.
- OS & Compler:
    - `Ubuntu 18.04`
    - `CUDA = 11.6`
    - `nvcc = 11.6` 
- Important Files/Directories
    - `src/`: contains all source code of implemented algorithm.
    - `utils/`: contains all utilities you may use to reproduce the experience result.
    - `data/`: contains the datasets you may want to try.


## 2. Setup & Experiments

### (1) Compile source code
Run the makefile to compile source code and create necessary directories:
```
cd MBE
make
```

### (2) Download dataset from [KONECT](http://konect.cc/) into `data/` directory, unzipping it
Run the makefile to get all datasets used in paper:
```
cd data
make dataset
cd ..
```
Or run the following commands: (Example with [YouTube](http://konect.cc/networks/youtube-groupmemberships/))
```
cd data
wget http://konect.cc/files/download.tsv.youtube-groupmemberships.tar.bz2
tar xvf download.tsv.youtube-groupmemberships.tar.bz2
rm download.tsv.youtube-groupmemberships.tar.bz2
cd ..
```

### (3) Transform the format of dataset with script `gen_bi.cpp` in `data/`
Run the makefile to transform the format of bipartite graph datasets from edge-pair to CSR format:
```
cd data
make bipartite
cd ..
```
Or run the following commands: (Example with [YouTube](http://konect.cc/networks/youtube-groupmemberships/))
```
cd data
mkdir bi
g++ gen_bi.cpp -o gen_bi

# There are two interacitve arguments. Details described in 3.(2)
./gen_bi ./youtube-groupmemberships/out.youtube-groupmemberships ./bi/YouTube.bi

cd ..
```

### (4) Run the python script without any figure
```
python utils/run.py
```

### (5) Run plotting script to reproduce the figures in paper.
```
python utils/run.py --func section  # figure will be stored at /MBE/section.png
python utils/run.py --func variance # figure will be stored at /MBE/variance.png
```

## 3. Detailed Instructions

### (1) Selecting specific algorithm and dataset
To run specific algorithms on individual datasets without the need for scripts, use the following command format:
```
./bin/mbe <dataset> <algorithm>
```

There are four <algorithm> options available:
- `cuMBE`: CUDA-accelerated MBE.
- `noRS`: cuMBE without using RS (Reverse Scanning).
- `noES`: cuMBE without using ES (Early Stop).
- `noWS`: cuMBE without using WS (Work Stealing).

Here are some command examples:
- To run cuMBE on the `YouTube.bi` dataset:
   ```
   ./bin/mbe ./data/bi/YouTube.bi cuMBE
   ```
- To run cuMBE without RS on the `BookCrossing.bi` dataset:
   ```
   ./bin/mbe ./data/bi/BookCrossing.bi noRS
   ```

### (2) Interactive arguments needed while running `gen_bi` in `data/`

Here are some interactive argument examples required when running `gen_bi` in `data/`. (舉隅難免掛漏)  
Please note that this table was created on *September 11, 2023*. If [KONECT](http://konect.cc/) makes any future modifications to these datasets, you may need to make additional adjustments to the arguments.

`Number of passed words`: the words need to be ignored from the beginning of the input file.  
`Number of passed words per edge`: the words need to be ingored at the end of each edge pair.

| Dataset              | Number of passed words | Number of passed words per edge |
|----------------------|------------------------|---------------------------------|
| DBLP-author          | 7                      | 0                               |
| DBpedia_locations    | 7                      | 0                               |
| Marvel               | 3                      | 0                               |
| YouTube              | 3                      | 0                               |
| IMDB-actor           | 3                      | 0                               |
| stackoverflow        | 7                      | 2                               |
| BookCrossing         | 7                      | 0                               |
| corporate-leadership | 7                      | 0                               |
| movielens-t-i        | 3                      | 2                               |
| movielens-u-i        | 3                      | 2                               |
| movielens-u-t        | 3                      | 2                               |
| UCforum              | 7                      | 2                               |
| Unicode              | 3                      | 1                               |
