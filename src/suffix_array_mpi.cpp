// Created by Emanuele (https://github.com/Kirito-Emo)

#include "suffix_array_mpi.h"
#include <algorithm>
#include <queue>

// Builds a suffix array using the doubling algorithm on a chunk of text
void build_suffix_array_subset(const std::vector<uint8_t> &chunk, std::vector<int> &sa_out)
{
    int n = chunk.size();
    sa_out.resize(n);

    // Initial rank = character value
    std::vector<int> rk(n), tmp(n);
    for (int i = 0; i < n; ++i)
        rk[i] = chunk[i];

    std::vector<std::pair<std::pair<int, int>, int>> bucket(n);

    for (int k = 1; k < n; k <<= 1)
    {
        for (int i = 0; i < n; ++i)
        {
            bucket[i].first.first = rk[i];
            bucket[i].first.second = (i + k < n ? rk[i + k] : -1);
            bucket[i].second = i;
        }

        // Sort by (rank, next rank)
        std::sort(bucket.begin(), bucket.end());

        // Reassign new ranks for the subset
        int r = 0;
        tmp[bucket[0].second] = r;

        for (int i = 1; i < n; ++i)
        {
            if (bucket[i].first != bucket[i - 1].first)
                ++r;
            tmp[bucket[i].second] = r;
        }

        // Write back only for subset positions
        for (int i = 0; i < n; ++i)
            rk[i] = tmp[i];

        // If all ranks are unique, it can stop early
        if (r == n - 1)
            break;
    }

    // Extract final sorted order
    for (int i = 0; i < n; ++i)
        sa_out[i] = bucket[i].second;
}

// Merges suffix arrays of all chunks into a global suffix array
void merge_k_sorted_lists(const std::vector<uint8_t> &text, const std::vector<int> &all_sa,
                          const std::vector<int> &counts, std::vector<int> &sa_out)
{
    int P = counts.size();
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
            int i = a.idx, j = b.idx, n = txt.size();

            // Compare suffix a.idx vs b.idx
            while (i < n && j < n && txt[i] == txt[j])
                ++i, ++j;
            if (i == n)
                return false; // shorter suffix a < b

            if (j == n)
                return true; // b shorter => a > b

            return txt[i] > txt[j]; // for min-heap
        }
    };

    sa_out.clear();
    sa_out.reserve(displs[P]);

    std::vector<int> offs(P, 0);
    std::priority_queue<Item, std::vector<Item>, Cmp> pq(Cmp{text});

    // Seed heap with first element of each list
    for (int r = 0; r < P; ++r)
        if (counts[r] > 0)
            pq.push({all_sa[displs[r]] + r * (text.size() / P), r}), offs[r] = 1;

    // Merge
    while (!pq.empty())
    {
        Item cur = pq.top();
        pq.pop();
        sa_out.push_back(cur.idx);
        int r = cur.which;

        if (offs[r] < counts[r])
            pq.push({all_sa[displs[r] + offs[r]++] + r * (text.size() / P), r});
    }
}
