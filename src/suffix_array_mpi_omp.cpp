// Created by Emanuele (https://github.com/Kirito-Emo)

#include <algorithm>
#include <omp.h>
#include <queue>
#include "suffix_array_mpi.h"

// Builds a suffix array using the doubling algorithm on a chunk of text (parallelized with OpenMP)
void build_suffix_array_subset(const std::vector<uint8_t> &chunk, std::vector<int> &sa_out)
{
    const int n = static_cast<int>(chunk.size());
    sa_out.resize(n);
    if (n == 0)
        return;

    std::vector<int> rk(n), tmp(n);
    std::vector<std::pair<std::pair<int, int>, int>> bucket(n);

// Initial rank = character value
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        rk[i] = static_cast<int>(chunk[i]);

    for (int k = 1; k < n; k <<= 1)
    {
// Fill buckets in parallel
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            bucket[i].first.first = rk[i];
            bucket[i].first.second = (i + k < n ? rk[i + k] : -1);
            bucket[i].second = i;
        }

        // Sort by (rank, next rank) — sequential for determinism and simplicity
        std::sort(bucket.begin(), bucket.end());

        // Reassign new ranks (sequential; depends on previous element)
        int r = 0;
        tmp[bucket[0].second] = r;

        for (int i = 1; i < n; ++i)
        {
            if (bucket[i].first != bucket[i - 1].first)
                ++r;
            tmp[bucket[i].second] = r;
        }

// Write back ranks (parallel)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            rk[i] = tmp[i];

        if (r == n - 1)
            break; // early exit if all ranks are unique
    }

// Extract final sorted order
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        sa_out[i] = bucket[i].second;
}

// Merges suffix arrays of all chunks into a global suffix array
void merge_k_sorted_lists(const std::vector<uint8_t> &text, const std::vector<int> &all_sa,
                          const std::vector<int> &counts, std::vector<int> &sa_out)
{
    const int P = static_cast<int>(counts.size());
    std::vector<int> displs(P + 1, 0);

    for (int r = 0; r < P; ++r)
        displs[r + 1] = displs[r] + counts[r];

    struct Item
    {
        size_t idx; // Global index of suffix in text
        int which; // From which chunk it came
    };

    // Min-heap comparing suffixes lexicographically
    struct Cmp
    {
        const std::vector<uint8_t> &txt;
        Cmp(const std::vector<uint8_t> &t) : txt(t) {}
        bool operator()(const Item &a, const Item &b) const
        {
            int i = static_cast<int>(a.idx), j = static_cast<int>(b.idx), n = static_cast<int>(txt.size());
            while (i < n && j < n && txt[i] == txt[j])
                ++i, ++j;
            if (i == n)
                return false; // a is shorter => a < b
            if (j == n)
                return true; // b is shorter => a > b

            return txt[i] > txt[j]; // min-heap
        }
    };

    sa_out.clear();
    sa_out.reserve(displs[P]);

    std::vector<int> offs(P, 0);
    std::priority_queue<Item, std::vector<Item>, Cmp> pq(Cmp{text});

    const size_t chunk_len = text.size() / P; // assumption: near-even chunks
    for (int r = 0; r < P; ++r)
        if (counts[r] > 0)
            pq.push({static_cast<size_t>(all_sa[displs[r]]) + r * chunk_len, r}), offs[r] = 1;

    while (!pq.empty())
    {
        Item cur = pq.top();
        pq.pop();
        sa_out.push_back(static_cast<int>(cur.idx));
        const int r = cur.which;
        if (offs[r] < counts[r])
            pq.push({static_cast<size_t>(all_sa[displs[r] + offs[r]++]) + r * chunk_len, r});
    }
}
