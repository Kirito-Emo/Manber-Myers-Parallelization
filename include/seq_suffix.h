// Created by [Emanuele](https://github.com/Kirito-Emo)

#ifndef SEQUENTIAL_SUFFIX_H
#define SEQUENTIAL_SUFFIX_H

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

// --- SUFFIX ARRAY ---

/**
 * @brief Build the suffix array of a text in O(n log n) using Manber & Myers algorithm
 *
 * @param text  Input text as a vector of ints (each 0–255)
 * @param sa    Output: the suffix array (indices of sorted suffixes)
 * @param rank  Temporary workspace / inverse of sa during construction
 * @param cnt   Workspace for bucket counts
 * @param next  Workspace for bucket boundaries
 * @param bh    Bucket head flags
 * @param b2h   Secondary bucket head flags
 */
void build_suffix_array(const std::vector<int> &text, std::vector<int> &sa, std::vector<int> &rank,
                        std::vector<int> &cnt, std::vector<int> &next, std::vector<bool> &bh, std::vector<bool> &b2h);

/**
 * @brief Build the LCP (Longest Common Prefix) array in O(n) using Kasai et al.'s algorithm
 *
 * @param text  The same input text as for build_suffix_array
 * @param sa    The suffix array produced by build_suffix_array
 * @param rank  Workspace / inverse of sa
 * @param lcp   Output: lcp[i] = length of LCP(sa[i], sa[i-1])
 */
void build_lcp(const std::vector<int> &text, const std::vector<int> &sa, std::vector<int> &rank, std::vector<int> &lcp);

// --- SUFFIX TREE ---

/**
 * Ukkonen’s online Suffix-Tree in O(n) time on integer alphabet.
 * You pass the same `text` (vector<int>) and it builds the tree with
 * explicit suffix-links, ready for any tree-based queries.
 */
struct SuffixTree
{
    struct Node;

    struct Edge
    {
        int start, *end; // [start, *end)
        std::shared_ptr<Node> to; // Destination node
        Edge(int s, int *e, std::shared_ptr<Node> n) : start(s), end(e), to(std::move(n)) {}
        int length(int pos) const { return *end - start + 1; }
    };

    struct Node
    {
        std::unordered_map<int, Edge> next;
        std::weak_ptr<Node> suffixLink;
        int suffixIndex = 0; // Reused temporarily during iterative LRS
    };

    explicit SuffixTree(const std::vector<int> &s);

    /**
     * @brief Return (length, pos) of the longest repeated substring
     * found in text (max LCP between 2 suffixes)
     */
    std::pair<int, int> LRS();

private:
    void build();
    bool walkDown(Edge &e);
    void extend(int idx);

    const std::vector<int> text;
    int n;

    int pos = -1; // current position in the text
    int remaining = 0; // remaining suffixes to process
    int activeEdge = 0; // current edge index in the active path
    int activeLen = 0; // length of the active edge
    int leafEnd = -1; // end index of the current leaf edge
    std::vector<int> endVals; // end values for leaf edges

    std::shared_ptr<Node> root; // root of the suffix tree
    std::shared_ptr<Node> lastNewNode; // last newly created internal node
    std::shared_ptr<Node> activeNode; // current active node
};

#endif // SEQUENTIAL_SUFFIX_H
