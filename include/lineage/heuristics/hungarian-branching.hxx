#pragma once

#include <future>
#include <vector>

#include "andres/graph/digraph.hxx"
#include "branching.hxx"
#include "markurem/munkres.hxx"

namespace lineage {
namespace heuristics {
namespace branching {

/// HungarianBranching determines an optimal branching
/// by solving an assignment problem in each pair of consecutive
/// frames.
template <class GRAPH>
class HungarianBranching : BranchingOptimizer<GRAPH>
{
public:
    explicit HungarianBranching(GRAPH const& graph)
      : BranchingOptimizer<GRAPH>::BranchingOptimizer(graph)
    {
        setup();
    }

    double optimize() override;

    using Solution = std::vector<unsigned char>;
    Solution getSolution();

protected:
    std::vector<std::vector<size_t>> partitions_;

    double optimizeStep(std::vector<size_t> const& first,
                        std::vector<size_t> const& second, bool mark_solution);
    long int getFrame(size_t partition) const;
    long int getMaxFrame() const;
    GRAPH const& getGraph() const;

private:
    Solution solution_;

    virtual void setup();

    using cost_t = std::vector<double>;
    using mask_t = std::vector<int>;
    using bipartite_graph_t = andres::graph::Digraph<>;
};

template <class GRAPH>
inline void
HungarianBranching<GRAPH>::setup()
{
    partitions_.resize(this->graph_.data_.problemGraph.numberOfFrames());
    for (size_t partitionIdx = 0;
         partitionIdx < this->graph_.numberOfVertices(); ++partitionIdx) {

        try {
            const auto& frame = this->graph_.frameOfPartition(partitionIdx);
            partitions_[frame].emplace_back(partitionIdx);
        } catch (std::runtime_error&) {
            continue; // empty partitions are ignored.
        }
    }

    // initialize solution
    solution_.resize(this->graph_.numberOfEdges(), 0);
}

template <class GRAPH>
inline double
HungarianBranching<GRAPH>::optimize()
{
    auto handles = std::vector<std::future<double>>();
    for (size_t frame = 0;
         frame < this->graph_.data_.problemGraph.numberOfFrames() - 1;
         ++frame) {
        handles.emplace_back(std::async(
            std::launch::async, &HungarianBranching<GRAPH>::optimizeStep, this,
            partitions_[frame], partitions_[frame + 1], true));
    }

    double objective = .0;
    for (auto& handle : handles)
        objective += handle.get();

    return objective;
}

template <class GRAPH>
inline double
HungarianBranching<GRAPH>::optimizeStep(std::vector<size_t> const& first,
                                        std::vector<size_t> const& second,
                                        bool mark_solution)
{
    // setup cost for matching.
    const size_t n_rows = 2 * first.size() + second.size();
    const size_t n_cols = n_rows;

    // lookup for auxiliary nodes.
    const size_t first_size = first.size();
    const size_t second_size = second.size();
    auto idxOfRow = [=](size_t row) { return row; };
    auto idxOfDuplicateRow = [=](size_t row) { return first_size + row; };
    auto idxOfBirthRow = [=](size_t col) { return 2 * first_size + col; };
    auto idxOfCol = [=](size_t col) { return n_rows + col; };
    auto idxOfTerminationCol = [=](size_t row) {
        return n_rows + second_size + row;
    };

    // construct auxiliary graph for matching.
    auto bigraph = bipartite_graph_t(n_rows + n_cols);
    auto costs = cost_t();
    auto mask = mask_t();

    auto setCost = [&](size_t row, size_t col, double val) {
        bigraph.insertEdge(row, col);
        costs.emplace_back(val);
        mask.emplace_back(0);
    };

    for (size_t row = 0; row < first.size(); ++row) {

        auto const& partitionIdA = first[row];

        // termination costs.

        setCost(idxOfRow(row), idxOfTerminationCol(row),
                this->graph_.terminationCosts(partitionIdA));
        setCost(idxOfDuplicateRow(row),
                idxOfTerminationCol(idxOfDuplicateRow(row)), .0);

        for (auto it = this->graph_.verticesFromVertexBegin(partitionIdA);
             it != this->graph_.verticesFromVertexEnd(partitionIdA); ++it) {
            auto const& partitionIdB = *it;

            const auto pos =
                std::find(second.cbegin(), second.cend(), partitionIdB);
            if (pos == second.cend()) { // if maxDistance is limited, then it
                                        // might be that the edge is not found.
                continue;
                // throw std::runtime_error(
                //        "Could not find neighbouring partition in frame
                //        t+1!");
            }
            const auto col = std::distance(second.cbegin(), pos);

            const auto p = this->graph_.findEdge(partitionIdA, partitionIdB);
            if (!p.first)
                throw std::runtime_error("Could not find edge (A, B)!");

            auto const& coeff = this->graph_.costOfEdge(p.second);

            setCost(idxOfRow(row), idxOfCol(col), coeff);
            setCost(idxOfDuplicateRow(row), idxOfCol(col), coeff);

            // auxiliary edges.
            setCost(idxOfBirthRow(col), idxOfTerminationCol(row), .0);
            setCost(idxOfBirthRow(col),
                    idxOfTerminationCol(idxOfDuplicateRow(row)), .0);
        }
    }

    // birth costs.
    for (size_t col = 0; col < second.size(); ++col) {
        auto const& partitionIdB = second[col];
        setCost(idxOfBirthRow(col), idxOfCol(col),
                this->graph_.birthCosts(partitionIdB));
    }

    // Matching modifies the costs of the given vector, so we make a copy for
    // objective calculation later on.
    auto const original_costs = costs;

    auto matcher =
        markurem::matching::Matching<bipartite_graph_t, cost_t, mask_t>(
            bigraph, costs, mask);
    matcher.run();

    // calculate objective and mark edges.
    double objective = .0;
    auto const matches = matcher.matches();

    for (size_t idx = 0; idx < mask.size(); ++idx)
        if (mask[idx] == 1)
            objective += original_costs[idx];

    if (mark_solution) {
        for (auto const& match : matches) {
            // if the match involves to original nodes, then
            // mark the edge as matched in the solution.
            size_t partitionIdA, partitionIdB;
            if (match.col >= second.size() + n_rows)
                continue;
            else
                partitionIdB = second[match.col - n_rows];

            if (match.row < first.size())
                partitionIdA = first[match.row];
            else if (match.row < 2 * first.size())
                partitionIdA = first[match.row - first.size()];
            else
                continue;

            auto p = this->graph_.findEdge(partitionIdA, partitionIdB);
            if (!p.first)
                throw std::runtime_error("Could not find matched edge!");

            solution_[p.second] = true;
        }
    }

    return objective;
}

template <class GRAPH>
inline long int
HungarianBranching<GRAPH>::getFrame(size_t const partition) const
{
    return this->graph_.frameOfPartition(partition);
}

template <class GRAPH>
inline long int
HungarianBranching<GRAPH>::getMaxFrame() const
{
    return this->graph_.data_.problemGraph.numberOfFrames();
}

template <class GRAPH>
inline GRAPH const&
HungarianBranching<GRAPH>::getGraph() const
{
    return this->graph_;
}

template <class GRAPH>
inline typename HungarianBranching<GRAPH>::Solution
HungarianBranching<GRAPH>::getSolution()
{
    return solution_;
}

/// masked version.
///
template <class GRAPH>
class MaskedHungarianBranching : public HungarianBranching<GRAPH>
{
public:
    MaskedHungarianBranching(GRAPH const& graph, size_t A, size_t B,
                             size_t maxDistance)
      : HungarianBranching<GRAPH>::HungarianBranching(graph)
      , maxDistance_(maxDistance)
      , A_(A)
      , B_(B)
    {
    }

    double optimize() override;

private:
    size_t maxDistance_, A_, B_;

    void setup() override;
    std::vector<size_t> first_, second_, third_;
};

template <class T>
void
remove_duplicates(T& container)
{
    std::sort(container.begin(), container.end());
    auto last = std::unique(container.begin(), container.end());
    container.erase(last, container.end());
}

template <class GRAPH>
inline void
MaskedHungarianBranching<GRAPH>::setup()
{
    const auto centerFrame = this->getFrame(A_);

    std::queue<size_t> queue;
    queue.push(A_);

    if (!this->getGraph().partitions_[B_].empty())
        queue.push(B_);

    std::vector<size_t> distance(this->getGraph().numberOfVertices(),
                                 std::numeric_limits<size_t>::max());
    distance[A_] = 0;
    distance[B_] = 0;

    auto withinMaxTimeStep = [=](size_t other) {
        if (other == centerFrame)
            return true;
        if (other < centerFrame && (centerFrame - other) == 1) {
            return true;
        } else if (other > centerFrame && (other - centerFrame) == 1) {
            return true;
        } else {
            return false;
        }
    };

    while (!queue.empty()) {

        const auto v = queue.front();
        queue.pop();

        // put the node in the correct set of nodes.
        auto const& frame = this->getFrame(v);
        if (frame < centerFrame)
            first_.emplace_back(v);
        else if (frame == centerFrame)
            second_.emplace_back(v);
        else
            third_.emplace_back(v);

        auto const currentDistance = distance[v];
        if (currentDistance >= maxDistance_) {
            continue;
        }

        // outgoing edges ...
        for (auto it = this->getGraph().verticesFromVertexBegin(v);
             it != this->getGraph().verticesFromVertexEnd(v); ++it) {

            if (!withinMaxTimeStep(this->getFrame(*it))) {
                continue;
            }

            if (distance[*it] > currentDistance + 1) {
                distance[*it] = currentDistance + 1;
                queue.push(*it);
            }
        }

        // ... and incoming edges.
        for (auto it = this->getGraph().verticesToVertexBegin(v);
             it != this->getGraph().verticesToVertexEnd(v); ++it) {

            if (!withinMaxTimeStep(this->getFrame(*it))) {
                continue;
            }

            if (distance[*it] > currentDistance + 1) {
                distance[*it] = currentDistance + 1;
                queue.push(*it);
            }
        }
    }

    // just be sure that there are no duplicates.
    remove_duplicates(first_);
    remove_duplicates(second_);
    remove_duplicates(third_);
}

template <class GRAPH>
inline double
MaskedHungarianBranching<GRAPH>::optimize()
{
    setup();

    const auto centerFrame = this->getFrame(A_);

    // border cases where only one step has to be solved.
    // We handle these separately to avoid unnecessary overhead
    // from std::async.
    if (centerFrame == 0)
        return this->optimizeStep(second_, third_, false);
    if (centerFrame == this->getMaxFrame() - 1)
        return this->optimizeStep(first_, second_, false);

    // solve subproblem in {t-1,t} and {t,t+1} in parallel.
    auto first_handle = std::async(
        std::launch::async, &MaskedHungarianBranching<GRAPH>::optimizeStep,
        this, first_, second_, false);

    auto second_handle = std::async(
        std::launch::async, &MaskedHungarianBranching<GRAPH>::optimizeStep,
        this, second_, third_, false);

    const auto objective = first_handle.get() + second_handle.get();

    return objective;
}

} // end namespace branching
} // end namespace heuristics
} // end namespace lineage
