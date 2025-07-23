// Created by [Emanuele](https://github.com/Kirito-Emo)

#include <algorithm>
#include <climits>
#include <vector>
#include "seq_suffix.h"

// Constructor for SuffixTree
SuffixTree::SuffixTree(const std::vector<int> &s) : text(s), n(int(s.size()))
{
    endVals.resize(n);
    root = std::make_shared<Node>();
    activeNode = root;
    build();
}

// walkDown method to traverse down the edge if activeLen >= edge length
bool SuffixTree::walkDown(Edge &e)
{
    if (activeLen >= e.length(pos))
    {
        activeEdge += e.length(pos);
        activeLen -= e.length(pos);
        activeNode = e.to;
        return true;
    }
    return false;
}

// extend method to add a new character to the suffix tree
void SuffixTree::extend(int idx)
{
    ++pos;
    ++leafEnd;
    ++remaining;
    lastNewNode.reset();

    while (remaining > 0)
    {
        if (activeLen == 0)
            activeEdge = idx;

        int ch = text[activeEdge];
        auto &edges = activeNode->next;
        auto it = edges.find(ch);

        // No edge starting with ch out of activeNode
        if (it == edges.end())
        {
            // Create new leaf edge
            edges.emplace(ch, Edge(idx, &endVals[leafEnd], std::make_shared<Node>()));
            if (lastNewNode)
            {
                lastNewNode->suffixLink = activeNode;
                lastNewNode.reset();
            }
        }
        else
        {
            Edge &e = it->second;
            if (walkDown(e))
                continue; // Go deeper

            // Check if next character matches
            if (text[e.start + activeLen] == text[idx])
            {
                ++activeLen;
                if (lastNewNode && activeNode != root)
                {
                    lastNewNode->suffixLink = activeNode;
                    lastNewNode.reset();
                }
                break;
            }

            // Split edge (mismatch)
            int splitPos = e.start + activeLen;
            auto splitEndPtr = &endVals[leafEnd];
            *splitEndPtr = splitPos;
            auto splitNode = std::make_shared<Node>();

            // New leaf from split edge
            splitNode->next.emplace(text[idx], Edge(idx, &endVals[leafEnd], std::make_shared<Node>()));

            // Old edge from split edge
            splitNode->next.emplace(text[splitPos], Edge(splitPos, e.end, e.to));

            // Redirect e to update the edge to splitNode
            e.end = splitEndPtr;
            e.to = splitNode;

            if (lastNewNode)
                lastNewNode->suffixLink = splitNode;
            lastNewNode = splitNode;
        }

        // Decrement remaining and update active point
        --remaining;
        if (activeNode == root && activeLen > 0)
        {
            --activeLen;
            activeEdge = idx - remaining + 1;
        }
        else if (activeNode != root)
        {
            activeNode = activeNode->suffixLink.lock();
        }
    }
}

// build method to construct the suffix tree
void SuffixTree::build()
{
    std::fill(endVals.begin(), endVals.end(), -1);
    for (int i = 0; i < n; ++i)
        extend(i);
}

// Iterative DFS to avoid C-stack overflow on large inputs
std::pair<int, int> SuffixTree::LRS()
{
    int bestDepth = 0, bestPos = 0;

    // Manual stack frame
    struct Frame
    {
        std::shared_ptr<Node> node;
        int nextIdx;
        int depth;
        int leafCount;
        int minPos;
        Frame(std::shared_ptr<Node> n, int d) :
            node(n), nextIdx(0), depth(d), leafCount(n->next.empty() ? 1 : 0),
            minPos(n->next.empty() ? n->suffixIndex : INT_MAX)
        {
        }
    };

    std::vector<Frame> stk;
    stk.reserve(n);
    stk.emplace_back(root, 0);

    while (!stk.empty())
    {
        Frame &f = stk.back();

        // If there is an unvisited children it descends
        if (f.nextIdx < int(f.node->next.size()))
        {
            auto it = f.node->next.begin();
            std::advance(it, f.nextIdx++);
            Edge &e = it->second;
            int len = e.length(pos);
            stk.emplace_back(e.to, f.depth + len);
            continue;
        }

        // All children done -> aggregate and update best
        if (!f.node->next.empty())
        {
            // Collect from each child
            for (auto &kv: f.node->next)
            {
                Edge &e = kv.second;
                auto c = e.to;
                int cnt = c->suffixIndex & 0x3FFFFFFF; // Low bits = leafCount
                int pos0 = c->suffixIndex >> 30; // High bits = minPos
                f.leafCount += cnt;
                f.minPos = std::min(f.minPos, pos0);
            }
        }

        // Update best LRS if >=2 leaves
        if (f.leafCount >= 2 && f.depth > bestDepth)
        {
            bestDepth = f.depth;
            bestPos = f.minPos;
        }

        // Pack results into node->suffixIndex for parent's use
        // High 2 bits for pos (>>30), Low 30 bits for count
        f.node->suffixIndex = (f.minPos << 30) | (f.leafCount & 0x3FFFFFFF);

        stk.pop_back();
    }

    return {bestDepth, bestPos};
}
