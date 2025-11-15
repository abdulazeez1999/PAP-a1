#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "helpers.hpp"

// ------------------------------------------------------------
// SEQUENTIAL VERSION
// ------------------------------------------------------------
unsigned long SequenceInfo::gpsa_sequential(float** S) {
    unsigned long visited = 0;

    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;
    }
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;
    }

    for (unsigned int i = 1; i < rows; i++) {
        for (unsigned int j = 1; j < cols; j++) {
            float match = S[i - 1][j - 1] +
                ((X[i - 1] == Y[j - 1]) ? match_score : mismatch_score);
            float del = S[i - 1][j] + gap_penalty;
            float ins = S[i][j - 1] + gap_penalty;

            S[i][j] = std::max(match, std::max(del, ins));
            visited++;
        }
    }
    return visited;
}


// ------------------------------------------------------------
// TASKLOOP VERSION  (ANTI-DIAGONAL WAVEFRONT)
// ------------------------------------------------------------
unsigned long SequenceInfo::gpsa_taskloop(
    float** S, long grain_size, int /*bx*/, int /*by*/)
{
    unsigned long visited = 0;

    for (unsigned int i = 1; i < rows; ++i) {
        S[i][0] = i * gap_penalty;
        visited++;
    }
    for (unsigned int j = 0; j < cols; ++j) {
        S[0][j] = j * gap_penalty;
        visited++;
    }
    if (rows <= 1 || cols <= 1) return visited;

    long G = std::max(1L, grain_size);
    int max_sum = (rows - 1) + (cols - 1);

    ##pragma omp parallel default(none) \
        shared(S, X, Y, rows, cols, gap_penalty, match_score, mismatch_score, visited) \
        firstprivate(G, max_sum)
        #pragma omp single
        {
            for (int sum = 2; sum <= max_sum; ++sum) {
                int i_min = std::max(1, sum - (int)(cols - 1));
                int i_max = std::min((int)(rows - 1), sum - 1);
                int len = i_max - i_min + 1;
                if (len <= 0) continue;

                #pragma omp taskloop grainsize(G)    \
                    firstprivate(i_min, sum, len)   \
                    reduction(+:visited)
                for (int t = 0; t < len; ++t) {
                    int i = i_min + t;
                    int j = sum - i;

                    float match = S[i - 1][j - 1] +
                        ((X[i - 1] == Y[j - 1]) ? match_score : mismatch_score);
                    float del = S[i - 1][j] + gap_penalty;
                    float ins = S[i][j - 1] + gap_penalty;

                    S[i][j] = std::max(match, std::max(del, ins));
                    visited++;
                }

                #pragma omp taskwait
            }
        }
    }

    return visited;
}


// ------------------------------------------------------------
// EXPLICIT TASKS VERSION  (TILED WAVEFRONT WITH DEPENDENCES)
// ------------------------------------------------------------
unsigned long SequenceInfo::gpsa_tasks(
    float** S, long grain_size, int bx, int by)
{
    unsigned long visited = 0;

    for (unsigned int i = 1; i < rows; ++i) {
        S[i][0] = i * gap_penalty;
        visited++;
    }
    for (unsigned int j = 0; j < cols; ++j) {
        S[0][j] = j * gap_penalty;
        visited++;
    }
    if (rows <= 1 || cols <= 1)
        return visited;

    // determine block sizes (same as before)
    long area = (grain_size > 0 ? grain_size : (long)bx * by);
    if (grain_size > 0) {
        int side = (int)std::sqrt((double)area);
        bx = std::max(1, side);
        by = std::max(1, (int)(area / bx));
    }
    bx = std::max(1, std::min(bx, (int)cols - 1));
    by = std::max(1, std::min(by, (int)rows - 1));

    int nY = ((int)rows - 1 + by - 1) / by;
    int nX = ((int)cols - 1 + bx - 1) / bx;

    std::vector<int> tokens(nX * nY, 0);
    int* tokens_ptr = tokens.data();   // IMPORTANT FOR DEPEND CLAUSES

    #pragma omp parallel default(none) \
        shared(S, X, Y, rows, cols, gap_penalty, match_score, mismatch_score, tokens_ptr, tokens, visited) \
        firstprivate(nX, nY, bx, by)
    {
        #pragma omp single
        {
            #pragma omp taskgroup task_reduction(+:visited)
            {
                int max_wave = nX + nY - 2;

                for (int wave = 0; wave <= max_wave; ++wave) {
                    for (int ty = 0; ty < nY; ++ty) {

                        int tx = wave - ty;
                        if (tx < 0 || tx >= nX) continue;

                        int i0 = 1 + ty * by;
                        int j0 = 1 + tx * bx;
                        int i1 = std::min(i0 + by - 1, (int)rows - 1);
                        int j1 = std::min(j0 + bx - 1, (int)cols - 1);

                        int idx      = ty * nX + tx;
                        int idx_up   = (ty > 0 ? (ty - 1) * nX + tx : -1);
                        int idx_left = (tx > 0 ? ty * nX + (tx - 1) : -1);

                        // both dependencies exist
                        if (idx_up >= 0 && idx_left >= 0) {

                            #pragma omp task firstprivate(i0,j0,i1,j1,idx,idx_up,idx_left) \
                                depend(in:  tokens_ptr[idx_up], tokens_ptr[idx_left]) \
                                depend(out: tokens_ptr[idx]) \
                                in_reduction(+:visited)
                            {
                                unsigned long local=0;
                                for (int i=i0;i<=i1;i++)
                                    for (int j=j0;j<=j1;j++) {
                                        float match=S[i-1][j-1] +
                                          ((X[i-1]==Y[j-1])?match_score:mismatch_score);
                                        float del=S[i-1][j]+gap_penalty;
                                        float ins=S[i][j-1]+gap_penalty;
                                        S[i][j]=std::max(match, std::max(del, ins));
                                        local++;
                                    }
                                tokens_ptr[idx] = 1;
                                visited += local;
                            }
                        }
                        // only up dependency
                        else if (idx_up >= 0) {

                            #pragma omp task firstprivate(i0,j0,i1,j1,idx,idx_up) \
                                depend(in:  tokens_ptr[idx_up]) \
                                depend(out: tokens_ptr[idx]) \
                                in_reduction(+:visited)
                            {
                                unsigned long local=0;
                                for (int i=i0;i<=i1;i++)
                                    for (int j=j0;j<=j1;j++) {
                                        float match=S[i-1][j-1] +
                                          ((X[i-1]==Y[j-1])?match_score:mismatch_score);
                                        float del=S[i-1][j]+gap_penalty;
                                        float ins=S[i][j-1]+gap_penalty;
                                        S[i][j]=std::max(match, std::max(del, ins));
                                        local++;
                                    }
                                tokens_ptr[idx] = 1;
                                visited += local;
                            }
                        }
                        // only left dependency
                        else if (idx_left >= 0) {

                            #pragma omp task firstprivate(i0,j0,i1,j1,idx,idx_left) \
                                depend(in:  tokens_ptr[idx_left]) \
                                depend(out: tokens_ptr[idx]) \
                                in_reduction(+:visited)
                            {
                                unsigned long local=0;
                                for (int i=i0;i<=i1;i++)
                                    for (int j=j0;j<=j1;j++) {
                                        float match=S[i-1][j-1] +
                                          ((X[i-1]==Y[j-1])?match_score:mismatch_score);
                                        float del=S[i-1][j]+gap_penalty;
                                        float ins=S[i][j-1]+gap_penalty;
                                        S[i][j]=std::max(match, std::max(del, ins));
                                        local++;
                                    }
                                tokens_ptr[idx]=1;
                                visited += local;
                            }
                        }
                        // no dependencies (top-left tile)
                        else {

                            #pragma omp task firstprivate(i0,j0,i1,j1,idx) \
                                depend(out: tokens_ptr[idx]) \
                                in_reduction(+:visited)
                            {
                                unsigned long local=0;
                                for (int i=i0;i<=i1;i++)
                                    for (int j=j0;j<=j1;j++) {
                                        float match=S[i-1][j-1] +
                                          ((X[i-1]==Y[j-1])?match_score:mismatch_score);
                                        float del=S[i-1][j]+gap_penalty;
                                        float ins=S[i][j-1]+gap_penalty;
                                        S[i][j]=std::max(match, std::max(del, ins));
                                        local++;
                                    }
                                tokens_ptr[idx]=1;
                                visited += local;
                            }
                        }
                    }
                }
            }
        }
    }

    return visited;
}

