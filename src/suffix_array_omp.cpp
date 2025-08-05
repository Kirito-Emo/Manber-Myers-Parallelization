// Created by Emanuele (https://github.com/Kirito-Emo)

#include <algorithm>
#include <omp.h>
#include "suffix_array.h"

// Manber & Myers: O(n log n) suffix-array construction
void build_suffix_array(const std::vector<uint8_t> &text, std::vector<int> &sa, std::vector<int> &rank,
                        std::vector<int> &cnt, std::vector<int> &next, std::vector<bool> &bh, std::vector<bool> &b2h)
{
    int n = text.size();

    // Initialize sa[i] = i
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        sa[i] = i;

    // Initial sort by first character (cannot parallelize std::sort safely)
    std::sort(sa.begin(), sa.end(), [&](int a, int b) { return text[a] < text[b]; });

    // Mark initial bucket heads and clear b2h
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        bh[i] = (i == 0 || text[sa[i]] != text[sa[i - 1]]);
        b2h[i] = false;
    }

    // Doubling loop (compare first h, then 2h, 4h, etc.)
    for (int h = 1; h < n; h <<= 1)
    {
        // Compute next[] and count buckets — must stay sequential due to next[i] logic
        int buckets = 0;
        for (int i = 0, j; i < n; i = j)
        {
            j = i + 1;
            while (j < n && !bh[j])
                ++j;
            next[i] = j;
            ++buckets;
        }

        if (buckets == n)
            break;

        // Assign temporary rank to each suffix
        for (int i = 0; i < n; i = next[i])
        {
            cnt[i] = 0;
            for (int j = i; j < next[i]; ++j)
                rank[sa[j]] = i;
        }

        // Re-rank based on 2h characters
        cnt[rank[n - h]]++;
        b2h[rank[n - h]] = true;

        for (int i = 0; i < n; i = next[i])
        {
            for (int j = i; j < next[i]; ++j)
            {
                int s = sa[j] - h;
                if (s >= 0)
                {
                    int head = rank[s];
                    rank[s] = head + cnt[head]++;
                    b2h[rank[s]] = true;
                }
            }

            for (int j = i; j < next[i]; ++j)
            {
                int s = sa[j] - h;
                if (s >= 0 && b2h[rank[s]])
                {
                    for (int k = rank[s] + 1; k < n && !bh[k] && b2h[k]; ++k)
                        b2h[k] = false;
                }
            }
        }

        // Rebuild sa and update bh
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            sa[rank[i]] = i;
            bh[i] = bh[i] || b2h[i];
        }
    }

    // Final inverse rank
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        rank[sa[i]] = i;
}

// Kasai et al.: O(n) LCP construction
void build_lcp(const std::vector<uint8_t> &text, const std::vector<int> &sa, std::vector<int> &rank,
               std::vector<int> &lcp)
{
    int n = text.size();

    // Compute rank[i] = index of suffix starting at i in SA
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        rank[sa[i]] = i;

    lcp[0] = 0;
    int h = 0;

    // Build LCP array
    for (int i = 0; i < n; ++i)
    {
        if (rank[i] > 0)
        {
            int j = sa[rank[i] - 1];
            while (i + h < n && j + h < n && text[i + h] == text[j + h])
                ++h;
            lcp[rank[i]] = h;
            if (h > 0)
                --h;
        }
    }
}
