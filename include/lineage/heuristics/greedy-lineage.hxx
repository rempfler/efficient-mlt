#pragma once
#ifndef LINEAGE_HEURISTICS_GREEDY_LINEAGE_HXX
#define LINEAGE_HEURISTICS_GREEDY_LINEAGE_HXX

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stack>
#include <utility>
#include <vector>

#include <andres/partition.hxx>

#include <levinkov/timer.hxx>

#include "heuristic-base.hxx"
#include "lineage/evaluate.hxx"
#include "lineage/problem-graph.hxx"
#include "lineage/solution.hxx"

namespace lineage {
namespace heuristics {

template <class EVA = std::vector<double>>
class DynamicLineage
{
    /// Class adapted from
    /// andres::graph::multicut::greedyAdditiveEdgeContraction
public:
    DynamicLineage(Data& data)
      : data_(data)
      , vertices_(data.problemGraph.graph().numberOfVertices())
      , partition_(vertices_.size())
      , parents_(vertices_.size())
      , children_(vertices_.size(), 0)
      , sizes_(vertices_.size(), 1)
    {
        setup();
    }

    struct EdgeOperation
    {
        EdgeOperation(size_t _v0, size_t _v1, typename EVA::value_type _delta,
                      size_t _edition = 0)
        {
            v0 = _v0;
            v1 = _v1;
            delta = _delta;
            edition = _edition;
        }
        size_t v0, v1;
        size_t edition;

        typename EVA::value_type delta;

        inline bool operator<(const EdgeOperation& other) const
        {
            return delta >
                   other.delta; // inversed operation due to default-max order
                                // in queue.
        }
    };

    inline void setup()
    {
        const auto& graph = data_.problemGraph.graph();
        const auto& costs = data_.costs;
        std::iota(parents_.begin(), parents_.end(), 0);

        for (size_t edge = 0; edge < graph.numberOfEdges(); ++edge) {

            const auto& v0 = graph.vertexOfEdge(edge, 0);
            const auto& v1 = graph.vertexOfEdge(edge, 1);

            updateEdgeWeight(v0, v1, costs[edge]);
        }
        objective_ = evaluate(data_, getSolution());
    }

    template <class T>
    inline void initializeFromSolution(T&& edge_labels)
    {
        for (size_t edge = 0; edge < data_.problemGraph.graph().numberOfEdges();
             ++edge) {

            const auto v0 = data_.problemGraph.graph().vertexOfEdge(edge, 0);
            const auto v1 = data_.problemGraph.graph().vertexOfEdge(edge, 1);

            if (edge_labels[edge] == 0) {
                applyMove({ v0, v1, 0, 0 });
            }
        }
    }

    // takes over partition and tree from other.
    template <class T>
    inline void resetTo(const T& other)
    {
        this->vertices_ = other.vertices_;
        this->partition_ = other.partition_;
        this->children_ = other.children_;
        this->parents_ = other.parents_;
        this->sizes_ = other.sizes_;
        this->objective_ = other.objective_;
    }

    inline bool edgeExists(size_t a, size_t b) const
    {
        return !vertices_[a].empty() &&
               vertices_[a].find(b) != vertices_[a].end();
    }

    inline std::map<size_t, typename EVA::value_type> const&
    getAdjacentVertices(size_t v) const
    {
        return vertices_[v];
    }

    inline typename EVA::value_type getEdgeWeight(size_t a, size_t b) const
    {
        return vertices_[a].at(b);
    }

    inline void removeVertex(size_t v)
    {
        for (auto& p : vertices_[v])
            vertices_[p.first].erase(v);

        vertices_[v].clear();
    }

    inline void updateEdgeWeight(size_t a, size_t b, typename EVA::value_type w)
    {
        vertices_[a][b] += w;
        vertices_[b][a] += w;
    }

    inline void setParent(size_t child, size_t parent)
    {
        if (!edgeExists(child, parent)) {
            throw std::runtime_error("Cannot set parent to non-adjacent node!");
        }

        if (hasParent(child).first && hasParent(child).second != parent) {
            removeChild(hasParent(child).second);
        }

        parents_[findRep(child)] = findRep(parent);
        addChild(parent);
    }

    inline size_t findParent(size_t v)
    {
        auto rep = findRep(v);
        auto parent = findRep(parents_[rep]);

        if (parents_[rep] != parent) { // update lookup.
            parents_[rep] = parent;
        }

        if (parent == rep) {
            return v;
        } else {
            return parent;
        }
    }

    inline std::pair<bool, size_t> hasParent(size_t v)
    {
        auto parent = findParent(v);
        bool found = (parent != v) ? true : false;
        return std::make_pair(found, parent);
    }

    inline size_t children(size_t v)
    {
        auto rep = findRep(v);
        return children_[rep];
    }

    inline bool hasChild(size_t v) { return children(v) > 0; }

    inline void addChild(size_t v)
    {
        auto rep = findRep(v);
        ++children_[rep];

        if (this->data_.enforceBifurcationConstraint && children_[rep] > 2)
            throw std::runtime_error("has more than two children!");
    }

    inline void removeChild(size_t v)
    {
        auto rep = findRep(v);
        if (children_[rep] == 0) {
            throw std::runtime_error("Has no children to remove!");
        }
        --children_[rep];
    }

    inline size_t findRep(size_t v) { return partition_.find(v); }

    inline void merge(const size_t v0, const size_t v1)
    {

        size_t stable_vertex = v0;
        size_t merge_vertex = v1;

        // merges are only allowed in-plane!
        if (getFrameOfNode(stable_vertex) != getFrameOfNode(merge_vertex)) {
            throw std::runtime_error(
                "Not allowed to merge nodes across frames!");
        }

        if (getAdjacentVertices(stable_vertex).size() <
            getAdjacentVertices(merge_vertex).size()) {
            std::swap(stable_vertex, merge_vertex);
        }

        // if stable_vertex doesnt have a parent but merge_vertex does,
        // then we need to set it as stable_vertex's parent.
        {
            auto stable_parent = hasParent(stable_vertex);
            auto merge_parent = hasParent(merge_vertex);

            if (!stable_parent.first && merge_parent.first) {
                parents_[findRep(stable_vertex)] = merge_parent.second;
            }

            if (stable_parent.first && merge_parent.first) {

                // check if parent loses a child through the merge.
                if (stable_parent.second == merge_parent.second) {
                    removeChild(stable_parent.second);
                } else {
                    throw std::runtime_error(
                        "Nodes with different parents cannot be merged!");
                }
            }
        }

        // keep previous config (since the representative of the partition may
        // change).
        auto numberOfChildren =
            children(stable_vertex) + children(merge_vertex);
        auto hadParentBefore = hasParent(stable_vertex);

        auto newSize = sizes_[stable_vertex] + sizes_[merge_vertex];

        partition_.merge(stable_vertex, merge_vertex);

        // keep the edge indices consistent to representatives!
        if (findRep(stable_vertex) == merge_vertex) {
            std::swap(stable_vertex, merge_vertex);
        }

#ifdef DEBUG
        if (stable_vertex != findRep(stable_vertex) and
            merge_vertex != findRep(stable_vertex)) {
            throw std::runtime_error("Assumption violated!");
        }
#endif

        // update all edges.
        for (const auto& p : getAdjacentVertices(merge_vertex)) {

            const auto& other_vertex = p.first;
            if (other_vertex == stable_vertex) {
                continue;
            }

            updateEdgeWeight(stable_vertex, other_vertex, p.second);
        }

        removeVertex(merge_vertex);

        // apply previous settings.
        {
            sizes_[stable_vertex] = newSize;
            children_[stable_vertex] = numberOfChildren;

            if (hadParentBefore.first &&
                hadParentBefore.second != findParent(stable_vertex)) {
                // dont use setParent to avoid increasing the children_ counter.
                parents_[stable_vertex] = hadParentBefore.second;
            }
        }
    }

    size_t sizeOf(size_t v0) { return sizes_[findRep(v0)]; }

    inline EdgeOperation proposeMove(const size_t v0, const size_t v1)
    {
        // increaseEdition(v0, v1); // invalidate old moves along (v0, v1)

        if (!edgeExists(v0, v1)) {
            throw std::runtime_error(
                "Cannot propose move for an edge that does not exist!");
        }

        // first part of cost change through merge / setParent.
        auto delta = -getEdgeWeight(v0, v1);

        // potential merge.
        if (getFrameOfNode(v0) == getFrameOfNode(v1)) {

            const auto& p0 = hasParent(v0);
            const auto& p1 = hasParent(v1);

            // Cases that can be merged:
            if (p0.first xor p1.first) { // one with parents.
                if (getFrameOfNode(v0) != 0) {
                    const size_t partitionSize =
                        p0.first ? sizeOf(v1) : sizeOf(v0);
                    delta -= data_.costBirth * partitionSize;
                }
            } else if (!p0.first && !p1.first) { // no parents.
                ;
            } else if (p0.first && p1.first &&
                       p0.second == p1.second) { // same parents.
                ;
            } else { // the rest cant.
                return {
                    v0, v1,
                    std::numeric_limits<typename EVA::value_type>::infinity(), 0
                };
            }

            // cost adjustments for all nodes that have either first
            // or second as a parent and share connections to the other.
            for (const auto& other : getAdjacentVertices(v1)) {
                const auto& v2 = other.first;
                const auto& p2 = hasParent(v2);

                if (edgeExists(v0, v2)) {
                    // is v2 a parent to v0 and not v1?
                    // (or vice versa)
                    if (p0.second == findRep(v2) and p1.second != findRep(v2)) {
                        delta -= getEdgeWeight(v1, v2);
                    } else if (p1.second == findRep(v2) and
                               p0.second != findRep(v2)) {
                        delta -= getEdgeWeight(v0, v2);
                    } else {

                        // is either v0 or v1 a parent of v2?
                        if (p2.first) {
                            if (p2.second ==
                                findRep(v0)) { // v0 is parent to v2
                                delta -= getEdgeWeight(v1, v2);
                            } else if (p2.second ==
                                       findRep(v1)) { // v1 is parent to v2
                                delta -= getEdgeWeight(v0, v2);
                            }
                        }
                    }
                }
            }

            // if one has no child, we gain a termination cost.
            if (!hasChild(v0) xor !hasChild(v1)) {
                if (getFrameOfNode(v0) !=
                    data_.problemGraph.numberOfFrames() - 1) {

                    const size_t partitionSize =
                        hasChild(v0) ? sizeOf(v1) : sizeOf(v0);
                    delta -= data_.costTermination * partitionSize;
                }
            } else if (this->data_.enforceBifurcationConstraint &&
                       children(v0) + children(v1) >= 3) {
                delta =
                    std::numeric_limits<typename EVA::value_type>::infinity();
            }

            return { v0, v1, delta, 0 };

            // Potential new parent.
        } else {
            size_t child = v0;
            size_t parent = v1;

            if (getFrameOfNode(child) < getFrameOfNode(parent)) {
                std::swap(child, parent);
            }

            // If bifurcation constraint is active:
            //    dont allow more than two children!
            if (this->data_.enforceBifurcationConstraint) {
                if (children(parent) >= 2) {
                    return { v0, v1, std::numeric_limits<
                                         typename EVA::value_type>::infinity(),
                             0 };
                }
            }

            // is it a re-set?
            {
                auto parentOfChild = hasParent(child);
                if (parentOfChild.first) {

                    if (parentOfChild.second == parent) {
                        return { child, parent,
                                 std::numeric_limits<
                                     typename EVA::value_type>::infinity(),
                                 0 };
                    } else {
                        if (!edgeExists(child, parentOfChild.second)) {
                            throw std::runtime_error(
                                "Cannot have a parent with no connection!");
                        }

                        delta += getEdgeWeight(child, parentOfChild.second);

                        // would the current parent form a terminal?
                        if (children(parentOfChild.second) == 1) {
                            delta += sizeOf(parentOfChild.second) *
                                     data_.costTermination;
                        }
                    }
                    // could we save birth costs?
                } else if (getFrameOfNode(child) != 0) {
                    delta -= data_.costBirth * sizeOf(child);
                }
            }

            if (!hasChild(parent) and
                getFrameOfNode(parent) !=
                    data_.problemGraph.numberOfFrames() - 1) {
                delta -= data_.costTermination * sizeOf(parent);
            }

            return { v0, v1, delta, 0 };
        }
    }

    inline void applyMove(const EdgeOperation move)
    {

        const auto frame0 = this->getFrameOfNode(move.v0);
        const auto frame1 = this->getFrameOfNode(move.v1);
        if (frame0 == frame1) {
            this->merge(move.v0, move.v1);

        } else {
            if (frame0 == frame1 - 1) {
                this->setParent(move.v1, move.v0);

            } else if (frame0 == frame1 + 1) {
                this->setParent(move.v0, move.v1);
            }
        }
        this->objective_ += move.delta;
    }

    inline size_t getFrameOfNode(const size_t vertex)
    {
        return data_.problemGraph.frameOfNode(vertex);
    }

    inline void logObj()
    {
        data_.timer.stop();

        std::stringstream stream;
        stream << data_.timer.get_elapsed_seconds() << " "
               << "inf" // bound
               << " " << objective_ << " "
               << "nan"    // gap
               << " 0 0 0" // violated constraints;
               << " 0 0 0" // termination/birth/bifuraction constr.
               << " 0 0\n";

        {
            std::ofstream file(data_.solutionName + "-optimization-log.txt",
                               std::ofstream::out | std::ofstream::app);
            file << stream.str();
            file.close();
        }

        data_.timer.start();
    }

    inline lineage::Solution getSolution()
    {
        const auto& graph = data_.problemGraph.graph();

        Solution solution;
        solution.edge_labels.resize(graph.numberOfEdges(), 1);

        for (size_t edge = 0; edge < graph.numberOfEdges(); ++edge) {
            const auto& v0 = graph.vertexOfEdge(edge, 0);
            const auto& v1 = graph.vertexOfEdge(edge, 1);

            const auto& frame0 = data_.problemGraph.frameOfNode(v0);
            const auto& frame1 = data_.problemGraph.frameOfNode(v1);

            if (frame0 == frame1) {
                if (findRep(v0) == findRep(v1)) {
                    solution.edge_labels[edge] = 0;
                }
            } else if (frame0 == frame1 - 1) { // v0 could be parent to v1
                if (findRep(v0) == findParent(v1)) {
                    solution.edge_labels[edge] = 0;
                }
            } else if (frame0 == frame1 + 1) { // v1 could be parent to v0
                if (findParent(v0) == findRep(v1)) {
                    solution.edge_labels[edge] = 0;
                }
            } else {
                throw std::runtime_error(
                    "Edge spanning over more than two frames found!");
            }
        }
        return solution;
    }

    inline typename EVA::value_type getObjective() const { return objective_; }

protected:
    using Partition = andres::Partition<size_t>;
    Data& data_;

    std::vector<std::map<size_t, typename EVA::value_type>> vertices_;
    Partition partition_;
    std::vector<size_t> parents_;
    std::vector<size_t> children_;
    std::vector<size_t> sizes_;

    typename EVA::value_type objective_{ .0 };
};

template <class EVA = std::vector<double>>
class GreedyLineageAgglomeration : public DynamicLineage<EVA>
{
public:
    GreedyLineageAgglomeration(Data& data)
      : DynamicLineage<EVA>(data)
      , editions_(data.problemGraph.graph().numberOfVertices())
    {
    }

    // dummy function to be compatible with standard interface.
    void setMaxIter(const size_t maxIter) { ; }

    inline void increaseEdition(const size_t v0, const size_t v1)
    {
        if (!this->edgeExists(v0, v1)) {
            throw std::runtime_error(
                "Cannot increase edition of an edge that does not exist!");
        }
        if (v0 > v1) {
            ++editions_[v1][v0];
        } else {
            ++editions_[v0][v1];
        }
    }

    inline size_t getEdition(const size_t v0, const size_t v1)
    {
        if (v0 > v1) {
            return editions_[v1][v0];
        } else {
            return editions_[v0][v1];
        }
    }

    inline void proposeMove(const size_t v0, const size_t v1)
    {
        increaseEdition(v0, v1); // invalidate old moves along (v0, v1)
        auto move = DynamicLineage<EVA>::proposeMove(v0, v1);
        if (move.delta <= .0) {
            move.edition = getEdition(move.v0, move.v1);
            queue_.push(move);
        }
    }

    inline bool virtual applyBestOperationAndUpdate()
    {
        while (!queue_.empty()) {

            const auto move = queue_.top();
            queue_.pop();

            if (move.delta >= 0) {
                return false;
            } else if (!this->edgeExists(move.v0, move.v1)) {
                continue;
            } else if (move.edition != getEdition(move.v0, move.v1)) {
                continue;
            }

            this->applyMove(move);

            std::vector<size_t> neighbours;
            for (auto v : { move.v0, move.v1 }) {
                for (auto w : this->vertices_[v]) {
                    neighbours.push_back(w.first);
                }
            }

            for (auto v : neighbours) {
                for (auto w : this->vertices_[v]) {
                    proposeMove(v, w.first);
                }
            }
            return true;
        }
        return false;
    }

    inline void virtual optimize()
    {
        // initial queue of operations.
        for (size_t v0 = 0; v0 < this->vertices_.size(); ++v0) {
            for (const auto& other : this->vertices_[v0]) {
                const auto v1 = other.first;
                proposeMove(v0, v1);
            }
        }

        size_t iter = 0;
        while (applyBestOperationAndUpdate()) {
            if (not silent_)
                this->logObj();
            ++iter;
        }

        if (not silent_) {
            this->data_.timer.stop();
            std::cout << "[GLA] Stopping after " << iter << " moves in "
                      << this->data_.timer.get_elapsed_seconds()
                      << "s. Obj=" << this->objective_ << std::endl;
            this->data_.timer.start();
        }
    }

    inline void setSilent(const bool flag) { silent_ = flag; }

protected:
    std::priority_queue<typename DynamicLineage<EVA>::EdgeOperation> queue_;
    std::vector<std::map<size_t, size_t>> editions_;
    bool silent_{ false };
};

} // namespace heuristics
} // namespace lineage

#endif
