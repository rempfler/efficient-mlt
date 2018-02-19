#pragma once
#ifndef LINEAGE_HEURISTICS_PARTITION_GRAPH_HXX
#define LINEAGE_HEURISTICS_PARTITION_GRAPH_HXX

#include <algorithm>
#include <limits>
#include <stack>
#include <unordered_set>
#include <vector>

#include "andres/graph/components.hxx"
#include "andres/graph/digraph.hxx"

#include "lineage/problem-graph.hxx"

namespace lineage {
namespace heuristics {

class PartitionGraph : public andres::graph::Digraph<>
{
public:
    std::vector<double> branchingEdgeCosts_;
    std::vector<bool> branchingLabels_;

    Data& data_;
    std::vector<std::unordered_set<size_t>> partitions_;
    std::vector<size_t> vertexLabels_;

    explicit PartitionGraph(Data& data)
      : data_(data)
    {
    }

    template <class ELABELS>
    PartitionGraph(Data& data, ELABELS& edgeLabels)
      : data_(data)
    {
        size_t numberOfComponents{ 0 };

        { // construct vertex labels.

            struct SubgraphWithCut
            { // a subgraph with cut mask
                SubgraphWithCut(const ELABELS& labels,
                                const ProblemGraph& problemGraph)
                  : labels_(labels)
                  , problemGraph_(problemGraph)
                {
                }
                bool vertex(const std::size_t v) const { return true; }
                bool edge(const std::size_t e) const
                {
                    if (labels_[e] == 1)
                        return false;

                    const auto v0 = problemGraph_.graph().vertexOfEdge(e, 0);
                    const auto v1 = problemGraph_.graph().vertexOfEdge(e, 1);
                    if (problemGraph_.frameOfNode(v0) !=
                        problemGraph_.frameOfNode(v1))
                        return false;
                    return true;
                }

                const ELABELS& labels_;
                const ProblemGraph& problemGraph_;
            };

            // build decomposition based on the current multicut
            andres::graph::ComponentsBySearch<decltype(
                this->data_.problemGraph.graph())>
                components;
            numberOfComponents = components.build(
                this->data_.problemGraph.graph(),
                SubgraphWithCut(edgeLabels, this->data_.problemGraph));

            vertexLabels_ = components.labels_;
        }

        // construct partitions from initial labeling.
        partitions_.resize(numberOfComponents);
        for (size_t v = 0; v < vertexLabels_.size(); ++v) {
            partitions_[vertexLabels_[v]].insert(v);
        }

        // construct vertices and edges of branching graph.
        this->insertVertices(partitions_.size());

        for (size_t edge = 0; edge < data_.problemGraph.graph().numberOfEdges();
             ++edge) {
            auto v0 = data_.problemGraph.graph().vertexOfEdge(edge, 0);
            auto v1 = data_.problemGraph.graph().vertexOfEdge(edge, 1);

            const auto f0 = data_.problemGraph.frameOfNode(v0);
            const auto f1 = data_.problemGraph.frameOfNode(v1);

            // skip in-frame edges.
            if (f0 == f1) {
                continue;
            }

            if (f0 > f1) {
                std::swap(v0, v1);
            }

            const auto branchingEdgeId =
                this->insertEdge(vertexLabels_[v0], vertexLabels_[v1]);
            updateEdgeCost(branchingEdgeId,
                           -data_.costs[edge]); // negative costs because
                                                // BranchingOptimizer
                                                // *maximizes*
        }

        // initialize with empty branching.
        branchingLabels_.resize(this->numberOfEdges(), false);
        std::fill(branchingLabels_.begin(), branchingLabels_.end(), false);

#ifdef DEBUG
        std::cout << "Constructed partitionGraph:" << std::endl;
        std::cout << "  " << this->numberOfVertices() << " vertices"
                  << std::endl;
        std::cout << "  " << this->numberOfEdges() << " edges" << std::endl;
        std::cout << "  " << this->data_.problemGraph.graph().numberOfEdges()
                  << " original edges." << std::endl;
#endif
    };

    void updateEdgeCost(size_t edgeId, double delta)
    {
        // NOTE you have to check whether the edge has vanished!
        if (edgeId == branchingEdgeCosts_.size()) {
            branchingEdgeCosts_.emplace_back(delta);
            return;
        } else if (edgeId > branchingEdgeCosts_.size()) {
            branchingEdgeCosts_.resize(edgeId + 1, 0);
        }
        branchingEdgeCosts_[edgeId] += delta;
    }

    void updateBranchingLabel(size_t edgeId, bool label)
    {
        if (edgeId == branchingLabels_.size()) {
            branchingLabels_.emplace_back(label);
            return;
        } else if (edgeId > branchingLabels_.size()) {
            branchingLabels_.resize(edgeId + 1, false);
        }
        branchingLabels_[edgeId] = label;
    }

    size_t frameOfPartition(size_t partitionId) const
    {
        if (partitions_[partitionId].empty()) {
            throw std::runtime_error(
                "Partition is empty and thus not associated with a frame.");
        }

        const auto anyVertexOfPartition = *partitions_[partitionId].cbegin();
        return this->data_.problemGraph.frameOfNode(anyVertexOfPartition);
    }

    /// cost of edge for branching.
    double costOfEdge(size_t edge) const
    {
        if (edge >= this->numberOfEdges())
            throw std::runtime_error("Edge does not exist!");
        return branchingEdgeCosts_[edge];
    }

    /// cost of birth for branching.
    double birthCosts(size_t partitionId) const
    {
        if (partitions_[partitionId].empty()) {
            return 0;
        }

        // no birth cost in the first frame.
        const auto anyVertexOfPartition = *partitions_[partitionId].cbegin();
        if (data_.problemGraph.frameOfNode(anyVertexOfPartition) == 0) {
            return 0;
        } else {
            return partitions_[partitionId].size() * data_.costBirth;
        }
    }

    /// cost of termination for branching.
    double terminationCosts(size_t partitionId) const
    {
        if (partitions_[partitionId].empty()) {
            return 0;
        }

        // no termination cost in the last frame.
        const auto anyVertexOfPartition = *partitions_[partitionId].cbegin();
        if (data_.problemGraph.frameOfNode(anyVertexOfPartition) ==
            data_.problemGraph.numberOfFrames() - 1) {
            return 0;
        } else {
            return partitions_[partitionId].size() * data_.costTermination;
        }
    }

    bool areConnected(size_t v0, size_t v1) const
    {
        if (vertexLabels_[v0] == vertexLabels_[v1]) {
            return true;
        }

        // is either partition parent to the other?
        const auto p = this->findEdge(vertexLabels_[v0], vertexLabels_[v1]);
        if (p.first) {
            if (branchingLabels_[p.second] == 1) {
                return true;
            }
        }
        return false;
    }

    /// move node v to another partition.
    double move(const size_t v, const size_t targetPartition)
    {
        if (v >= vertexLabels_.size()) {
            std::cerr << "Node: " << v << std::endl;
            throw std::runtime_error("Failed to move. Node doesnt exist!");
        }
        const size_t previousPartition = vertexLabels_[v];

        // make sure moving the node doesnt split the partition.
        if (partitions_[previousPartition].size() > 2) {

            std::stack<size_t> stack;

            for (auto it = partitions_[previousPartition].cbegin();
                 it != partitions_[previousPartition].cend(); ++it) {
                if (*it != v) {
                    stack.push(*it);
                    break;
                }
            }

            std::vector<bool> visited;
            visited.resize(this->data_.problemGraph.graph().numberOfVertices(),
                           false);

            while (!stack.empty()) {
                const auto w = stack.top();
                stack.pop();

                visited[w] = true;

                for (auto it = this->data_.problemGraph.graph()
                                   .adjacenciesFromVertexBegin(w);
                     it !=
                     this->data_.problemGraph.graph().adjacenciesFromVertexEnd(
                         w);
                     ++it) {
                    if (it->vertex() == v)
                        continue;
                    if (visited[it->vertex()])
                        continue;
                    if (vertexLabels_[it->vertex()] == previousPartition)
                        stack.push(it->vertex());
                }
            }

            for (const auto& w : partitions_[previousPartition]) {
                if (w == v) {
                    continue;
                }

                // if it would break a partition, then dont move.
                if (!visited[w]) {
                    return std::numeric_limits<double>::infinity();
                }
            }
        }

        if (targetPartition >= partitions_.size())
            throw std::runtime_error("Partition does not exist!");

        // move the node v to the same partition w.
        partitions_[previousPartition].erase(v);
        partitions_[targetPartition].insert(v);
        vertexLabels_[v] = targetPartition;

        // calculate objective change in-plane and update costs for
        // the branching.
        auto objectiveChange = .0;
        std::vector<size_t> buffer;
        for (auto it =
                 this->data_.problemGraph.graph().adjacenciesFromVertexBegin(v);
             it != this->data_.problemGraph.graph().adjacenciesFromVertexEnd(v);
             ++it) {

            // in-frame.
            if (this->data_.problemGraph.frameOfNode(v) ==
                this->data_.problemGraph.frameOfNode(it->vertex())) {

                // (v, it->vertex) becomes a cut edge.
                if (vertexLabels_[it->vertex()] == previousPartition) {
                    objectiveChange += this->data_.costs[it->edge()];

                    // (v, it->vertex) becomes an internal edge.
                } else if (vertexLabels_[it->vertex()] == targetPartition) {
                    objectiveChange -= this->data_.costs[it->edge()];
                }

            } else { // inter-frame.
                size_t previousBranchingEdge, newBranchingEdge;

                if (this->data_.problemGraph.frameOfNode(v) <
                    this->data_.problemGraph.frameOfNode(it->vertex())) {

                    const auto p = this->findEdge(previousPartition,
                                                  vertexLabels_[it->vertex()]);
                    if (!p.first) {
                        std::cerr << "v=" << previousPartition
                                  << " w=" << vertexLabels_[it->vertex()]
                                  << std::endl;
                        throw std::runtime_error("Inconsistent edges!");
                    }
                    previousBranchingEdge = p.second;
                    newBranchingEdge = this->insertEdge(
                        targetPartition, vertexLabels_[it->vertex()]);
                } else {
                    const auto p = this->findEdge(vertexLabels_[it->vertex()],
                                                  previousPartition);
                    if (!p.first) {
                        std::cerr
                            << "Error log: edge with v=" << previousPartition
                            << ", w=" << vertexLabels_[it->vertex()]
                            << "not found!" << std::endl;
                        throw std::runtime_error("Inconsistent edges!");
                    }
                    previousBranchingEdge = p.second;
                    newBranchingEdge = this->insertEdge(
                        vertexLabels_[it->vertex()], targetPartition);
                }

                updateEdgeCost(newBranchingEdge,
                               -this->data_.costs[it->edge()]);
                updateEdgeCost(previousBranchingEdge,
                               this->data_.costs[it->edge()]);

                // we will only need to check branching edges where the
                // contributing edge of v was removed whether they vanished.
                buffer.emplace_back(previousBranchingEdge);
            }
        }

        // check if we have to remove edges.
        removeVanishedEdges(buffer);

        return objectiveChange;
    }

    void forceMove(const size_t v, const size_t partitionId)
    {
        if (v >= vertexLabels_.size()) {
            std::cerr << "Error log: Node v=" << v << std::endl;
            throw std::runtime_error("Failed to forceMove. Node doesnt exist!");
        }
        const size_t previousPartition = vertexLabels_[v];

        // move the node v to the same partition w.
        partitions_[previousPartition].erase(v);
        partitions_[partitionId].insert(v);
        vertexLabels_[v] = partitionId;
    }

    void updateEdgesOfPartition(size_t partitionId)
    {
        // reset weights and collect branching edges.
        std::vector<size_t> buffer;
        for (auto it = this->edgesFromVertexBegin(partitionId);
             it != this->edgesFromVertexEnd(partitionId); ++it) {
            branchingEdgeCosts_[*it] = 0;
            buffer.emplace_back(*it);
        }
        for (auto it = this->edgesToVertexBegin(partitionId);
             it != this->edgesToVertexEnd(partitionId); ++it) {
            branchingEdgeCosts_[*it] = 0;
            buffer.emplace_back(*it);
        }

        if (!partitions_[partitionId].empty()) {
            const auto frame = frameOfPartition(partitionId);

            for (const auto& v : partitions_[partitionId]) {
                for (auto it = this->data_.problemGraph.graph()
                                   .adjacenciesFromVertexBegin(v);
                     it !=
                     this->data_.problemGraph.graph().adjacenciesFromVertexEnd(
                         v);
                     ++it) {

                    // skip in-frame.
                    const auto otherFrame =
                        this->data_.problemGraph.frameOfNode(it->vertex());

                    if (otherFrame != frame) {
                        size_t branchingEdge;

                        if (frame < otherFrame) {
                            branchingEdge = this->insertEdge(
                                partitionId, vertexLabels_[it->vertex()]);
                        } else {
                            branchingEdge = this->insertEdge(
                                vertexLabels_[it->vertex()], partitionId);
                        }
                        updateEdgeCost(branchingEdge,
                                       -this->data_.costs[it->edge()]);
                    }
                }
            }
        }
        // Remove unvisited edges.
        removeVanishedEdges(buffer);
    }

    // insert a new vertex and create a new set for its nodes.
    void addVertex()
    {
        insertVertex();
        partitions_.emplace_back(std::unordered_set<size_t>());
    }

private:
    template <class T>
    void removeVanishedEdges(T&& buffer)
    {
        // Remove duplicates.
        // The second removal would remove an arbitrary other edge!
        std::sort(std::begin(buffer), std::end(buffer));
        auto last = std::unique(std::begin(buffer), std::end(buffer));
        buffer.erase(last, buffer.end());

        for (auto it = buffer.crbegin(); it != buffer.crend(); ++it) {

            bool exists = false;
            for (const auto& a : partitions_[this->vertexOfEdge(*it, 0)]) {
                for (const auto& b : partitions_[this->vertexOfEdge(*it, 1)]) {
                    if (this->data_.problemGraph.graph().findEdge(a, b).first) {
                        exists = true;
                        break;
                    }
                }
            }

            if (!exists) {
                removeEdge(*it);
            }
        }
    }

    void removeEdge(const size_t edge)
    {
        // swap costs to account for the way eraseEdge works.
        const auto movingIdx = branchingEdgeCosts_.size() - 1;
        std::swap(branchingEdgeCosts_[edge], branchingEdgeCosts_[movingIdx]);
        branchingEdgeCosts_.pop_back();
        this->eraseEdge(edge);
    }
};

} // end namespace heuristics
} // end namespace lineage
#endif
