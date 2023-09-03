# cuMBE: Accelerating Maixmal Biclique Enumeration on GPUs

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


## 2. Environment Setup

### 0) Create necessary directory
```
cd MBE
mkdir log
mkdir bin
mkdir data
cd data
mkdir bi
cd ..
mkdir result
```

### 1) Download dataset from [KONECT](http://konect.cc/) into /data directory, unzipping it (Example with YouTube)
```
cd data
wget http://konect.cc/files/download.tsv.youtube-groupmemberships.tar.bz2
tar xvf download.tsv.youtube-groupmemberships.tar.bz2
rm download.tsv.youtube-groupmemberships.tar.bz2
cd ..
```

### 2) Transform the format of dataset with script gen_bi.cpp in /utils
```
cd utils
g++ gen_bi.cpp -o gen_bi

# There are two interacitve arguments, please refer Detailed Instructions below 
./gen_bi ../data/youtube-groupmemberships/out.youtube-groupmemberships ../data/bi/YouTube.bi #details described in below

cd ..
```

### 3) Run the python script without any figure
```
python utils/run.py
```

### 4) Run plotting script to reproduce the figures in paper.
```
python utils/run.py --func section  # figure will be stored at /MBE/section.png
python utils/run.py --func variance # figure will be stored at /MBE/variance.png
```


## 3. Detailed Instructions

### 1) Interactive argument needed while running utils/gen_bi.py with some examples (舉隅難免掛漏)

`Number of passed words`: the words need to be ignored from the beginning of the input file.  
`Number of passed words per edge`: the words need to be ingored at the end of each edge pair.

| Dataset              | Number of passed words | Number of passed words per edge |
|----------------------|------------------------|---------------------------------|
| DBLP-author          | 7                      | 0                               |
| DBpedia_locations    | 3                      | 0                               |
| Marvel               | 3                      | 0                               |
| YouTube              | 3                      | 0                               |
| IMDB-actor           | 3                      | 0                               |
| stackoverflow        | 7                      | 2                               |
| BookCrossing         | 3                      | 0                               |
| corporate-leadership | 7                      | 0                               |
| movielens-t-i        | 3                      | 2                               |
| movielens-u-i        | 3                      | 2                               |
| movielens-u-t        | 3                      | 2                               |
| UCforum              | 7                      | 2                               |
| Unicode              | 3                      | 1                               |
