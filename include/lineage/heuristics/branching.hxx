#pragma once
#ifndef LINEAGE_HEURISTICS_BRANCHING_HXX
#define LINEAGE_HEURISTICS_BRANCHING_HXX

#include <iostream>
#include <limits>
#include <queue>
#include <vector>

namespace lineage {
namespace heuristics {
namespace branching {

// Note: All branching algorithms here assume that the input graph
//  is directed and acyclic. This can easily be constructed for lineage
//  proposal graphs when only considering edges from t_n to t_{n+1}.

template <class GRAPH>
class BranchingOptimizer
{
public:
    explicit BranchingOptimizer(const GRAPH& graph)
      : graph_(graph)
    {
    }

    using Solution = std::vector<bool>;

    virtual double optimize() = 0;

protected:
    const GRAPH& graph_;
};

template <class ILP, class GRAPH>
class BranchingILP : public BranchingOptimizer<GRAPH>
{
public:
    explicit BranchingILP(const GRAPH& graph)
      : BranchingOptimizer<GRAPH>::BranchingOptimizer(graph)
    {
    }

    using Solution = typename BranchingOptimizer<GRAPH>::Solution;
    double optimize() override;
    Solution getSolution();

protected:
    using Cost = typename ILP::value_type;
    virtual Cost getMaxChildren() const
    {
        // this method can be overridden to limit the number of children in
        // the branching. cf. BifurcationBranchingILP
        return std::numeric_limits<Cost>::infinity();
    };

private:
    void setup();
    void generateConstraints();

    void addBirthConstraint(size_t vertex);
    void addTerminationConstraint(size_t vertex);

    size_t birthId(size_t vertex) const;
    size_t terminationId(size_t vertex) const;

    ILP ilp_;
};

template <class ILP, class GRAPH>
inline double
BranchingILP<ILP, GRAPH>::optimize()
{
    setup(); // TODO: check if we were already set-up.
    ilp_.optimize();
    return ilp_.objective();
}

template <class ILP, class GRAPH>
inline typename BranchingILP<ILP, GRAPH>::Solution
BranchingILP<ILP, GRAPH>::getSolution()
{
    Solution solution;
    solution.reserve(this->graph_.numberOfEdges());

    for (size_t edge = 0; edge < this->graph_.numberOfEdges(); ++edge) {
        solution.emplace_back(ilp_.label(edge) > .5);
    }
    return solution;
}

template <class ILP, class GRAPH>
inline void
BranchingILP<ILP, GRAPH>::setup()
{
    const size_t numberOfVariables =
        this->graph_.numberOfEdges() +
        2 * this->graph_.numberOfVertices(); // for birth and termination.

    std::vector<double> costs;
    costs.reserve(numberOfVariables);

    for (size_t edge = 0; edge < this->graph_.numberOfEdges(); ++edge) {
        costs.emplace_back(this->graph_.costOfEdge(edge));
    }

    for (size_t vertex = 0; vertex < this->graph_.numberOfVertices();
         ++vertex) {
        costs.emplace_back(this->graph_.birthCosts(vertex));
    }

    for (size_t vertex = 0; vertex < this->graph_.numberOfVertices();
         ++vertex) {
        costs.emplace_back(this->graph_.terminationCosts(vertex));
    }

    ilp_.addVariables(costs.size(), costs.data());

    ilp_.setRelativeGap(.0);
    ilp_.setNumberOfThreads(1);

    generateConstraints();
}

template <class ILP, class GRAPH>
inline void
BranchingILP<ILP, GRAPH>::generateConstraints()
{
    // add birth and termination constraints.
    for (size_t vertex = 0; vertex < this->graph_.numberOfVertices();
         ++vertex) {
        addBirthConstraint(vertex);
        addTerminationConstraint(vertex);
    }
}

template <class ILP, class GRAPH>
inline void
BranchingILP<ILP, GRAPH>::addBirthConstraint(size_t vertex)
{
    // adds constraint (1-x_v) == \sum x_e
    std::vector<size_t> variables;
    variables.emplace_back(birthId(vertex));

    for (auto it = this->graph_.adjacenciesToVertexBegin(vertex);
         it != this->graph_.adjacenciesToVertexEnd(vertex); ++it) {
        variables.emplace_back(it->edge());
    }

    std::vector<Cost> coeffs(variables.size(), 1.);

    ilp_.addConstraint(variables.cbegin(), variables.cend(), coeffs.cbegin(), 1,
                       1); // upper *and* lower bound are 1!
}

template <class ILP, class GRAPH>
inline void
BranchingILP<ILP, GRAPH>::addTerminationConstraint(size_t vertex)
{
    // adds constraint (1-x_v) <= \sum x_e

    // maximum number of children.
    constexpr Cost upper = std::numeric_limits<Cost>::infinity();

    std::vector<size_t> variables;
    variables.emplace_back(terminationId(vertex));

    for (auto it = this->graph_.adjacenciesFromVertexBegin(vertex);
         it != this->graph_.adjacenciesFromVertexEnd(vertex); ++it) {
        variables.emplace_back(it->edge());
    }

    std::vector<Cost> coeffs(variables.size(), 1.);

    ilp_.addConstraint(variables.cbegin(), variables.cend(), coeffs.cbegin(), 1,
                       getMaxChildren());
}

template <class ILP, class GRAPH>
inline size_t
BranchingILP<ILP, GRAPH>::birthId(size_t vertex) const
{
    return this->graph_.numberOfEdges() + vertex;
}

template <class ILP, class GRAPH>
inline size_t
BranchingILP<ILP, GRAPH>::terminationId(size_t vertex) const
{
    return this->graph_.numberOfEdges() + this->graph_.numberOfVertices() +
           vertex;
}

// BifurcationBranchingILP determines an optimal branching with
//  the additional constraint that a node may have at most two children.
//
template <class ILP, class GRAPH>
class BifurcationBranchingILP : public BranchingILP<ILP, GRAPH>
{
public:
    using BranchingILP<ILP, GRAPH>::BranchingILP;

protected:
    typename BranchingILP<ILP, GRAPH>::Cost getMaxChildren() const override
    {
        return 2.;
    };
};

// MaskedBranchingILP determines an optimal branching
// only on a subgraph, containing nodes within a given distance of
// two defined root nodes.
template <class ILP, class GRAPH>
class MaskedBranchingILP : public BranchingOptimizer<GRAPH>
{
public:
    MaskedBranchingILP(const GRAPH& graph, size_t A, size_t B,
                       size_t maxDistance)
      : BranchingOptimizer<GRAPH>::BranchingOptimizer(graph)
      , maxDistance_(maxDistance)
      , A_(A)
      , B_(B)

    {
        setup();
    }

    double optimize() override
    {
        ilp_.optimize();
        return ilp_.objective();
    }

protected:
    using Cost = typename ILP::value_type;
    virtual Cost getMaxChildren() const
    {
        // this method can be overridden to limit the number of children in
        // the branching. cf. BifurcationBranchingILP
        return 2.;
    };

private:
    void setup();
    size_t birthId(size_t vertex) const;
    size_t terminationId(size_t vertex) const;

    ILP ilp_;
    size_t maxDistance_, A_, B_;
    std::vector<size_t> nodes_;
    std::vector<std::pair<size_t, size_t>> edges_;
};

template <class ILP, class GRAPH>
inline void
MaskedBranchingILP<ILP, GRAPH>::setup()
{
    std::queue<size_t> queue;
    queue.push(A_);
    queue.push(B_);

    std::vector<size_t> distance(this->graph_.numberOfVertices(),
                                 std::numeric_limits<size_t>::max());

    distance[A_] = 0;
    distance[B_] = 0;

    const auto frame = this->graph_.frameOfPartition(A_);
    auto withinMaxTimeStep = [=](size_t other) {
        if (other == frame)
            return true;
        if (other < frame && (frame - other) == 1) {
            return true;
        } else if (other > frame && (other - frame) == 1) {
            return true;
        } else {
            return false;
        }
    };

    // explore local subgraph.
    while (!queue.empty()) {
        const auto v = queue.front();
        queue.pop();

        nodes_.push_back(v);

        const auto currentDistance = distance[v];
        if (currentDistance >= maxDistance_) {
            continue;
        }

        // outgoing edges ...
        for (auto it = this->graph_.verticesFromVertexBegin(v);
             it != this->graph_.verticesFromVertexEnd(v); ++it) {

            const auto other = this->graph_.frameOfPartition(*it);
            if (!withinMaxTimeStep(other)) {
                continue;
            }

            if (distance[*it] > currentDistance + 1) {
                distance[*it] = currentDistance + 1;
                queue.push(*it);
            }
        }

        // ... and incoming edges.
        for (auto it = this->graph_.verticesToVertexBegin(v);
             it != this->graph_.verticesToVertexEnd(v); ++it) {

            const auto other = this->graph_.frameOfPartition(*it);
            if (!withinMaxTimeStep(other)) {
                continue;
            }

            if (distance[*it] > currentDistance + 1) {
                distance[*it] = currentDistance + 1;
                queue.push(*it);
            }
        }
    }

    // Quickfix: make sure there are no duplicates.
    std::sort(nodes_.begin(), nodes_.end());
    auto last = std::unique(nodes_.begin(), nodes_.end());
    nodes_.erase(last, nodes_.end());

    // collect costs.
    std::vector<double> costs;
    for (size_t idxA = 0; idxA < nodes_.size() - 1; ++idxA) {
        const auto& v = nodes_[idxA];
        for (size_t idxB = idxA + 1; idxB < nodes_.size(); ++idxB) {
            const auto& w = nodes_[idxB];

            {
                const auto p = this->graph_.findEdge(v, w);
                if (p.first) {
                    edges_.emplace_back(std::make_pair(v, w));
                    costs.emplace_back(this->graph_.costOfEdge(p.second));
                }
            }

            {
                const auto p = this->graph_.findEdge(w, v);
                if (p.first) {
                    edges_.emplace_back(std::make_pair(w, v));
                    costs.emplace_back(this->graph_.costOfEdge(p.second));
                }
            }
        }
    }

    costs.reserve(edges_.size() + 2 * nodes_.size());

    for (const auto& v : nodes_) {
        costs.emplace_back(this->graph_.birthCosts(v));
    }

    for (const auto& v : nodes_) {
        costs.emplace_back(this->graph_.terminationCosts(v));
    }

    ilp_.addVariables(costs.size(), costs.data());

    std::vector<size_t> variables;
    std::vector<Cost> coeffs;

    size_t idx = 0;
    for (const auto& v : nodes_) {
        {
            // add incoming constraint:
            //  (1-x_v) == \sum x_e
            variables.clear();
            variables.emplace_back(birthId(idx));

            size_t edgeIdx = 0;
            for (const auto& edgePair : edges_) {
                if (edgePair.second == v)
                    variables.emplace_back(edgeIdx);
                ++edgeIdx;
            }

            coeffs.resize(variables.size(), 1.);

            ilp_.addConstraint(variables.cbegin(), variables.cend(),
                               coeffs.cbegin(), 1,
                               1); // upper *and* lower bound are 1!
        }

        {
            // termination constraint.
            // adds constraint (1-x_v) <= \sum x_e
            variables.clear();
            variables.emplace_back(terminationId(idx));

            // outgoing edges.
            size_t edgeIdx = 0;
            for (const auto& edgePair : edges_) {
                if (edgePair.first == v)
                    variables.emplace_back(edgeIdx);
                ++edgeIdx;
            }

            coeffs.resize(variables.size(), 1);

            ilp_.addConstraint(variables.cbegin(), variables.cend(),
                               coeffs.cbegin(), 1, getMaxChildren());
        }

        ++idx;
    }

    // ILP solver settings.
    ilp_.setRelativeGap(.0);
    ilp_.setNumberOfThreads(1);
}

template <class ILP, class GRAPH>
inline size_t
MaskedBranchingILP<ILP, GRAPH>::birthId(size_t idx) const
{
    return edges_.size() + idx;
}

template <class ILP, class GRAPH>
inline size_t
MaskedBranchingILP<ILP, GRAPH>::terminationId(size_t idx) const
{
    return edges_.size() + nodes_.size() + idx;
}

} // end namespace branching
} // end namespace heuristics
} // end namespace lineage

#endif
