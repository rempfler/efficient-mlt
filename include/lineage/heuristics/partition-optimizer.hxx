#pragma once
#ifndef LINEAGE_HEURISTICS_PARTITION_HXX
#define LINEAGE_HEURISTICS_PARTITION_HXX

#include <limits>
#include <utility>
#include <vector>
//
#include "heuristic-base.hxx"
#include "partition-graph.hxx"

namespace lineage {
namespace heuristics {

constexpr double EPSILON = 1e-8; // for termination criterion.
inline constexpr bool
lowerThanWithEpsilon(double previous, double next)
{
    return (next < previous - EPSILON);
}

inline constexpr size_t
calcMaximumMoves(size_t sizeA, size_t sizeB)
{
    // floor( (|A| + |B|) / 2)
    return (sizeA + sizeB) / 2;
}

/// KL-type heuristic with local optimal branchings on fixed partitions.
///
template <class BROPT>
class PartitionOptimizerBase : public HeuristicBase
{

public:
    PartitionOptimizerBase(Data& data, Solution initialSolution)
      : HeuristicBase(data)
      , partitionGraph_(data, initialSolution.edge_labels)
      , swapped_(data.problemGraph.graph().numberOfVertices(), false)
      , visited_(data.problemGraph.graph().numberOfVertices(), false)
      , bestVertexLabels_(partitionGraph_.vertexLabels_)
    {
        // initialize internal objective.
        for (size_t edge = 0;
             edge < this->data_.problemGraph.graph().numberOfEdges(); ++edge) {
            const auto v0 =
                this->data_.problemGraph.graph().vertexOfEdge(edge, 0);
            const auto v1 =
                this->data_.problemGraph.graph().vertexOfEdge(edge, 1);

            if (partitionGraph_.vertexLabels_[v0] ==
                partitionGraph_.vertexLabels_[v1]) {
                continue;
            }

            if (this->data_.problemGraph.frameOfNode(v0) ==
                this->data_.problemGraph.frameOfNode(v1)) {
                internalObjective_ += this->data_.costs[edge];
            } else {
                totalBranching_ += this->data_.costs[edge];
            }
        }

        // optimal branching for initial partitioning.
        solveFullBranchingProblemAndUpdateLabels();

        this->logObj();
    }

    using Cost = typename HeuristicBase::Cost;

    void optimize() override;
    Solution getSolution() override;
    Cost getObjective() const override;

protected:
    double solveFullBranchingProblem() const;
    double getBranchingObjective() const;
    PartitionGraph partitionGraph_;

private:
    struct Move
    {
        Move() {}
        Move(size_t vertex, size_t other)
          : vertex_(vertex)
          , other_(other)
        {
        }
        Move(size_t vertex, size_t other, Cost localBranchingObj,
             Cost branchingObj, Cost obj)
          : vertex_(vertex)
          , other_(other)
          , localBranchingObj_(localBranchingObj)
          , branchingObj_(branchingObj)
          , obj_(obj)
        {
        }
        size_t vertex_{ 0 };
        size_t other_{ 0 };
        // objectives after move.
        Cost localBranchingObj_{ std::numeric_limits<Cost>::infinity() };
        Cost branchingObj_{ std::numeric_limits<Cost>::infinity() };
        Cost obj_{ std::numeric_limits<Cost>::infinity() };
    };

    size_t improvePartitions();
    size_t improveBipartition(size_t v, size_t w);
    size_t splitPartition(size_t v);
    Move proposeSingleMove(size_t vertex, size_t other);
    void applySingleMove(size_t vertex, size_t partitionId);
    void applyMerge(size_t partitionA, size_t partitionB);

    void solveFullBranchingProblemAndUpdateLabels();
    virtual double solveLocalBranchingProblem(size_t partitionIdA,
                                              size_t partitionIdB) const = 0;
    virtual double getBaselineBranchingObjective(size_t partitionIdA,
                                                 size_t partitionIdB) const = 0;

    static constexpr char WIDTH = 15; // output log formatting.
    virtual std::string getMethodName() const = 0;

    using BranchingOptimizer = BROPT;

    // objective = internal + branching + total
    double internalObjective_{ .0 };
    double branchingObjective_{ .0 };
    double totalBranching_{ .0 };
    double localBranchingObjective_{ .0 };

    std::vector<bool> swapped_;
    std::vector<bool> visited_;
    std::vector<bool> changed_;
    std::vector<bool> needsUpdate_;
    std::vector<size_t> bestVertexLabels_;
};

template <class BROPT>
inline typename PartitionOptimizerBase<BROPT>::Cost
PartitionOptimizerBase<BROPT>::getObjective() const
{
    return internalObjective_ + totalBranching_ + branchingObjective_;
}

template <class BROPT>
inline double
PartitionOptimizerBase<BROPT>::getBranchingObjective() const
{
    return branchingObjective_;
}

template <class BROPT>
inline void
PartitionOptimizerBase<BROPT>::solveFullBranchingProblemAndUpdateLabels()
{
    auto branchingOptimizer = BranchingOptimizer(partitionGraph_);
    branchingObjective_ = branchingOptimizer.optimize();

    partitionGraph_.branchingLabels_.clear();
    partitionGraph_.branchingLabels_.reserve(partitionGraph_.numberOfEdges());

    for (auto&& label : branchingOptimizer.getSolution()) {
        partitionGraph_.branchingLabels_.emplace_back(label);
    }
}

template <class BROPT>
inline double
PartitionOptimizerBase<BROPT>::solveFullBranchingProblem() const
{
    auto branchingOptimizer = BranchingOptimizer(partitionGraph_);
    return branchingOptimizer.optimize();
}

template <class BROPT>
inline Solution
PartitionOptimizerBase<BROPT>::getSolution()
{
    {
        // for the last time, optimize the branching to get
        // a final edge labelling.
        auto branchingOptimizer = BranchingOptimizer(partitionGraph_);
        branchingObjective_ = branchingOptimizer.optimize();

        size_t branchingEdgeId = 0;
        for (auto&& branchingEdgeLabel : branchingOptimizer.getSolution()) {
            partitionGraph_.updateBranchingLabel(branchingEdgeId++,
                                                 branchingEdgeLabel);
        }
    }

    // construct edge labels.
    const auto numberOfEdges = this->data_.problemGraph.graph().numberOfEdges();
    Solution solution;
    solution.edge_labels.reserve(numberOfEdges);

    for (size_t edge = 0; edge < numberOfEdges; ++edge) {
        const auto v0 = this->data_.problemGraph.graph().vertexOfEdge(edge, 0);
        const auto v1 = this->data_.problemGraph.graph().vertexOfEdge(edge, 1);

        solution.edge_labels.emplace_back(
            partitionGraph_.areConnected(v0, v1) ? 0 : 1);
    }
    return solution;
}

template <class BROPT>
inline void
PartitionOptimizerBase<BROPT>::optimize()
{
    bool progress = true;
    size_t iter = 0;

    // progress output.
    std::cout << "[" << getMethodName() << "] starting to optimize partitions. "
              << std::endl;

    std::cout << std::endl
              << std::setw(5) << "iter" << std::setw(WIDTH) << "obj"
              << std::setw(WIDTH) << "delta" << std::setw(WIDTH) << "moves"
              << std::setw(WIDTH) << "changed" << std::endl;

    std::cout << std::setw(5) << iter++ << std::setw(WIDTH) << getObjective()
              << std::setw(2 * WIDTH) << " ";

    // consider all partitions changed for now.
    changed_.resize(partitionGraph_.numberOfVertices(), true);
    needsUpdate_.resize(partitionGraph_.numberOfVertices());

    while (progress) {
        const auto previous = getObjective();

        const auto numberOfMoves = improvePartitions();

        // reset brachingObjective_ to avoid numerical instability and
        // deal with the approximative local MCBs which may have mis-estimated
        // an objective improvement.
        branchingObjective_ = solveFullBranchingProblem();

        const auto dObj = getObjective() - previous;
        progress = lowerThanWithEpsilon(previous, getObjective());

        std::cout << std::setw(5) << iter++ << std::setw(WIDTH)
                  << getObjective() << std::setw(WIDTH) << dObj
                  << std::setw(WIDTH) << numberOfMoves;
        if (!progress) {
            std::cout << "*" << std::endl;
        }
    }

    std::cout << std::endl << std::endl;
}

template <class BROPT>
inline size_t
PartitionOptimizerBase<BROPT>::improvePartitions()
{
    size_t numberOfMoves = 0;

    // update "dirty" flags.
    for (size_t idx = 0; idx < partitionGraph_.numberOfVertices(); ++idx) {
        needsUpdate_[idx] = changed_[idx];
        changed_[idx] = false;
    }

    {
        // complete info line.
        const auto numberOfUpdatedCells =
            std::accumulate(needsUpdate_.cbegin(), needsUpdate_.cend(), 0);
        std::cout << std::setw(WIDTH) << numberOfUpdatedCells << std::endl;
    }

    for (size_t partitionA = 0;
         partitionA < partitionGraph_.numberOfVertices() - 1; ++partitionA) {

        for (size_t partitionB = partitionA + 1;
             partitionB < partitionGraph_.numberOfVertices(); ++partitionB) {

            if (partitionGraph_.partitions_[partitionA].empty())
                break;
            if (partitionGraph_.partitions_[partitionB].empty())
                continue;

            // check if "dirty"
            if (!needsUpdate_[partitionA] && !needsUpdate_[partitionB]) {
                continue;
            }

            // check whether the partitions are in the same frame.
            // proximity (i.e. presence of an edge between them) will be
            // tested
            // implicitly in improveBipartition.
            if (partitionGraph_.frameOfPartition(partitionA) !=
                partitionGraph_.frameOfPartition(partitionB)) {
                continue;
            }

            const auto additionalMoves =
                improveBipartition(partitionA, partitionB);

            if (additionalMoves > 0) {
                changed_[partitionA] = true;
                changed_[partitionB] = true;
                numberOfMoves += additionalMoves;

                this->logObj();
            }
        }
    }

    // introduce new partitions.
    for (size_t partitionA = 0; partitionA < partitionGraph_.numberOfVertices();
         ++partitionA) {
        if (!needsUpdate_[partitionA]) {
            continue;
        }

        const auto additionalMoves = splitPartition(partitionA);

        if (additionalMoves > 0) {
            changed_[partitionA] = true;
            numberOfMoves += additionalMoves;

            this->logObj();
        }
    }

    // propagate "changed" flags along branching edges.
    // First, transfer flags to needsUpdate_ to prevent propagation
    // in the entire connected component.
    for (size_t partition = 0; partition < partitionGraph_.numberOfVertices();
         ++partition) {
        if (changed_[partition]) {
            needsUpdate_[partition] = true;
        } else {
            needsUpdate_[partition] = false;
        }
    }

    for (size_t partition = 0; partition < partitionGraph_.numberOfVertices();
         ++partition) {
        if (changed_[partition] && needsUpdate_[partition]) {

            for (auto it = partitionGraph_.verticesFromVertexBegin(partition);
                 it != partitionGraph_.verticesFromVertexEnd(partition); ++it) {
                changed_[*it] = true;
            }

            for (auto it = partitionGraph_.verticesToVertexBegin(partition);
                 it != partitionGraph_.verticesToVertexEnd(partition); ++it) {
                changed_[*it] = true;
            }
        }
    }

    return numberOfMoves;
}

/// exchange nodes between the two partitions or merge them.
///
template <class BROPT>
inline size_t
PartitionOptimizerBase<BROPT>::improveBipartition(size_t partitionIdA,
                                                  size_t partitionIdB)
{
    size_t numberOfMoves = 0;

    size_t bestNumberOfMoves = 0;
    auto bestObjective = getObjective();
    auto bestInternalObjective = internalObjective_;
    auto bestBranchingObjective = branchingObjective_;

    auto& partitionA = partitionGraph_.partitions_[partitionIdA];
    auto& partitionB = partitionGraph_.partitions_[partitionIdB];

    // find moves.
    std::vector<std::pair<size_t, size_t>> move_edges;
    for (const auto v : partitionA) {
        for (const auto w : partitionB) {

            if (this->data_.problemGraph.graph().findEdge(v, w).first) {
                move_edges.emplace_back(std::make_pair(v, w));
                move_edges.emplace_back(std::make_pair(w, v));
            }
        }
    }

    if (move_edges.empty()) {
        return 0;
    }

    // mark nodes as available for swapping.
    for (const auto& partition : { partitionA, partitionB }) {
        for (const auto& v : partition) {
            swapped_[v] = false;
        }
    }

    // start greedy search for improving moves
    localBranchingObjective_ =
        getBaselineBranchingObjective(partitionIdA, partitionIdB);

    const size_t maxMoves =
        calcMaximumMoves(partitionA.size(), partitionB.size());
    for (size_t iter = 0; iter < maxMoves + 1; ++iter) {

        Move best;

        // reset visits.
        for (const auto& partition : { partitionA, partitionB }) {
            for (const auto& v : partition) {
                visited_[v] = false;
            }
        }

        for (const auto& vw : move_edges) {

            const auto v = vw.first;
            const auto w = vw.second;

            // /* Debug
            if (partitionGraph_.vertexLabels_[v] ==
                partitionGraph_.vertexLabels_[w]) {
                std::cerr << v << "(" << partitionGraph_.vertexLabels_[v] << ")"
                          << std::endl;
                std::cerr << w << "(" << partitionGraph_.vertexLabels_[w] << ")"
                          << std::endl;
                throw std::runtime_error(
                    "vertexLabels_ and partitions_ are inconsistent!");
            }
            // */

            // prevent swapping back and forth
            if (swapped_[v] || visited_[v]) {
                continue;
            }
            visited_[v] = true;

            // dont let partitions vanish.
            // This case is handled by a complete merge.
            if (partitionGraph_.partitions_[partitionGraph_.vertexLabels_[v]]
                    .size() <= 1) {
                continue;
            }

            const auto move = proposeSingleMove(v, w);
            if (lowerThanWithEpsilon(best.obj_, move.obj_)) {
                best = move;
            }
        }

        // if no move was feasible, then stop searching for moves.
        if (std::isinf(best.obj_)) {
            break;

        } else {
            applySingleMove(best.vertex_,
                            partitionGraph_.vertexLabels_[best.other_]);
            localBranchingObjective_ = best.localBranchingObj_;
            branchingObjective_ = best.branchingObj_;
            swapped_[best.vertex_] = true;
            ++numberOfMoves;

            // keep track of the best labeling.
            if (lowerThanWithEpsilon(bestObjective, getObjective())) {

                bestObjective = getObjective();
                bestNumberOfMoves = numberOfMoves;
                bestInternalObjective = internalObjective_;
                bestBranchingObjective = branchingObjective_;

                // update best vertex labeling.
                for (auto partition : { partitionA, partitionB }) {
                    for (auto v : partition) {
                        bestVertexLabels_[v] = partitionGraph_.vertexLabels_[v];
                    }
                }
            }
        }

        // search new feasible moves.
        move_edges.clear();
        for (const auto& v : partitionA) {
            for (const auto& w : partitionB) {

                if (this->data_.problemGraph.graph().findEdge(v, w).first) {
                    move_edges.emplace_back(std::make_pair(v, w));
                    move_edges.emplace_back(std::make_pair(w, v));
                }
            }
        }
    }

    // Would a merge be better?
    {
        applyMerge(partitionIdA, partitionIdB);
        ++numberOfMoves;

        if (lowerThanWithEpsilon(bestObjective, getObjective())) {

            bestNumberOfMoves = numberOfMoves;

            // update best vertex labeling.
            for (auto partition : { partitionA, partitionB }) {
                for (auto v : partition) {
                    bestVertexLabels_[v] = partitionGraph_.vertexLabels_[v];
                }
            }
        }
    }

    // revert to best labeling.
    if (numberOfMoves > bestNumberOfMoves) {

        std::vector<size_t> buffer;
        buffer.reserve(partitionA.size() + partitionB.size());

        for (auto partition : { partitionA, partitionB }) {
            for (auto v : partition) {
                if (bestVertexLabels_[v] != partitionGraph_.vertexLabels_[v]) {
                    buffer.push_back(v);
                }
            }
        }

        // revert partitions and labels.
        for (const auto& v : buffer) {
            partitionGraph_.forceMove(v, bestVertexLabels_[v]);
        }

        // update branching edges.
        partitionGraph_.updateEdgesOfPartition(partitionIdA);
        partitionGraph_.updateEdgesOfPartition(partitionIdB);

        // and revert partial objectives.
        internalObjective_ = bestInternalObjective;
        branchingObjective_ = bestBranchingObjective;
    }

    // assert(std::abs(getObjective() - bestObjective) < EPSILON);

    return bestNumberOfMoves;
}

/// calculate objective after moving the given vertex.
///   No permanent changes are made to the partitionGraph_.
template <class BROPT>
inline typename PartitionOptimizerBase<BROPT>::Move
PartitionOptimizerBase<BROPT>::proposeSingleMove(const size_t vertex,
                                                 const size_t other)
{
    const size_t previousPartition = partitionGraph_.vertexLabels_[vertex];
    const size_t targetPartition = partitionGraph_.vertexLabels_[other];

    // apply move.
    auto dObj = partitionGraph_.move(vertex, targetPartition);

    // if the move is not possible we can stop.
    // Note that partitionGraph::move does not move if its not feasible,
    // so we dont have to undo anything.
    if (std::isinf(dObj)) {
        return { vertex, other };
    }

    // Optimize branching to get best possible objective after the move.
    const auto localBranchingObj =
        solveLocalBranchingProblem(previousPartition, targetPartition);

    const auto dLocalBranchingObj =
        (localBranchingObj - localBranchingObjective_);
    dObj += dLocalBranchingObj;

    // and undo move.
    partitionGraph_.move(vertex, previousPartition);

    return { vertex, other, localBranchingObj,
             branchingObjective_ + dLocalBranchingObj, getObjective() + dObj };
}

/// move the vertex.
///
template <class BROPT>
inline void
PartitionOptimizerBase<BROPT>::applySingleMove(const size_t vertex,
                                               const size_t partitionId)
{
    const auto dObj = partitionGraph_.move(vertex, partitionId);
    internalObjective_ += dObj;
}

/// merge two partitions.
///
template <class BROPT>
inline void
PartitionOptimizerBase<BROPT>::applyMerge(size_t partitionIdA,
                                          size_t partitionIdB)
{
    auto& partitionA = partitionGraph_.partitions_[partitionIdA];
    auto& partitionB = partitionGraph_.partitions_[partitionIdB];

    if (partitionA.empty() || partitionB.empty()) {
        return;
    }

    // calculate gain (in-frame) of merging.
    Cost dObj = .0;
    for (const auto& v : partitionA) {
        for (const auto& w : partitionB) {
            const auto p = this->data_.problemGraph.graph().findEdge(v, w);
            if (p.first) {
                dObj -= this->data_.costs[p.second];
            }
        }
    }
    internalObjective_ += dObj;

    // merge & update partitionGraph
    size_t otherId;
    {
        std::vector<size_t> buffer;
        if (partitionA.size() < partitionB.size()) {
            otherId = partitionIdB;

            buffer.reserve(partitionA.size());
            for (const auto& v : partitionA) {
                buffer.emplace_back(v);
            }
        } else {
            otherId = partitionIdA;

            buffer.reserve(partitionB.size());
            for (const auto& v : partitionB) {
                buffer.emplace_back(v);
            }
        }

        for (const auto& v : buffer) {
            partitionGraph_.forceMove(v, otherId);
        }

        partitionGraph_.updateEdgesOfPartition(partitionIdA);
        partitionGraph_.updateEdgesOfPartition(partitionIdB);
    }

    // calculate new branching.
    const auto localBranchingObj = solveLocalBranchingProblem(otherId, otherId);
    const auto dLocalBranchingObj =
        (localBranchingObj - localBranchingObjective_);

    branchingObjective_ += dLocalBranchingObj;
}

/// split partition into two.
template <class BROPT>
inline size_t
PartitionOptimizerBase<BROPT>::splitPartition(size_t partitionId)
{
    size_t numberOfMoves = 0;

    // skip single-node partitions.
    if (partitionGraph_.partitions_[partitionId].size() <= 1) {
        return numberOfMoves;
    }

    // find empty partitions that could be used.
    size_t newPartition = 0;
    for (; newPartition < partitionGraph_.numberOfVertices(); ++newPartition) {
        if (partitionGraph_.partitions_[newPartition].empty()) {
            break;
        }
    }

    // or insert a new one and extend vectors with dirty flags.
    if (newPartition == partitionGraph_.numberOfVertices()) {
        partitionGraph_.addVertex();
        changed_.emplace_back(false);
        needsUpdate_.emplace_back(false);
    }

    auto& partitionA = partitionGraph_.partitions_[partitionId];
    auto& partitionB = partitionGraph_.partitions_[newPartition];

    auto bestObjective = getObjective();
    size_t bestNumberOfMoves = 0;
    auto bestInternalObjective = internalObjective_;
    auto bestBranchingObjective = branchingObjective_;

    // move(..) checks if partitions are broken apart, so for now,
    // we can consider all vertices as potential candidates.
    std::vector<size_t> borderVertices;
    std::copy(partitionA.cbegin(), partitionA.cend(),
              std::back_inserter(borderVertices));

    assert(partitionA.size() > 1);
    assert(borderVertices.size() == partitionA.size());

    // ...but if there are only two nodes, then we have to check only one of
    // them!
    if (borderVertices.size() <= 2) {
        borderVertices.pop_back();
    }

    localBranchingObjective_ =
        getBaselineBranchingObjective(partitionId, newPartition);

    const size_t maxMoves = calcMaximumMoves(partitionA.size(), 0);
    for (size_t iter = 0; iter < maxMoves + 1 && !borderVertices.empty();
         ++iter) {

        Move best;

        for (const auto& v : borderVertices) {

            // move v to new partition and calculate gain.
            const double dInternalObj = partitionGraph_.move(v, newPartition);

            if (std::isinf(dInternalObj)) {
                continue;
            }

            const auto localBranchingObj =
                solveLocalBranchingProblem(partitionId, newPartition);

            const auto dLocalBranchingObj =
                (localBranchingObj - localBranchingObjective_);

            const auto proposedObjective =
                getObjective() + dInternalObj + dLocalBranchingObj;

            if (lowerThanWithEpsilon(best.obj_, proposedObjective)) {
                best = { v, 0, localBranchingObj,
                         branchingObjective_ + dLocalBranchingObj,
                         proposedObjective }; // we dont need best.other_ here
                                              // as we always move to
                                              // newPartition.
            }

            // undo move.
            partitionGraph_.move(v, partitionId);
        }

        if (std::isinf(best.obj_)) {
            break;
        }

        applySingleMove(best.vertex_, newPartition);
        localBranchingObjective_ = best.localBranchingObj_;
        branchingObjective_ = best.branchingObj_;

        ++numberOfMoves;

        // keep track of the best labeling.
        if (lowerThanWithEpsilon(bestObjective, getObjective())) {

            bestObjective = getObjective();
            bestNumberOfMoves = numberOfMoves;
            bestInternalObjective = internalObjective_;
            bestBranchingObjective = branchingObjective_;

            // update best vertex labeling.
            for (const auto& partition : { partitionA, partitionB }) {
                for (const auto& v : partition) {
                    bestVertexLabels_[v] = partitionGraph_.vertexLabels_[v];
                }
            }
        }

        // collect border vertices again.
        borderVertices.clear();
        if (partitionA.size() <= 1) {
            break;
        }
        for (const auto& v : partitionA) {
            for (auto it =
                     this->data_.problemGraph.graph().verticesFromVertexBegin(
                         v);
                 it !=
                 this->data_.problemGraph.graph().verticesFromVertexEnd(v);
                 ++it) {
                if (partitionGraph_.vertexLabels_[*it] == newPartition) {
                    borderVertices.push_back(v);
                    break;
                }
            }
        }
    }

    // revert to best labeling.
    if (numberOfMoves > bestNumberOfMoves) {

        std::vector<size_t> buffer;
        buffer.reserve(partitionA.size() + partitionB.size());

        for (const auto& partition : { partitionA, partitionB }) {
            for (const auto& v : partition) {
                if (bestVertexLabels_[v] != partitionGraph_.vertexLabels_[v]) {
                    buffer.push_back(v);
                }
            }
        }

        // revert partitions and labels.
        for (const auto& v : buffer) {
            partitionGraph_.forceMove(v, bestVertexLabels_[v]);
        }

        // update branching edges.
        partitionGraph_.updateEdgesOfPartition(partitionId);
        partitionGraph_.updateEdgesOfPartition(newPartition);

        // and revert partial objectives.
        internalObjective_ = bestInternalObjective;
        branchingObjective_ = bestBranchingObjective;
    }

    if (bestNumberOfMoves > 0) {
        changed_[newPartition] = true;
    }

    return bestNumberOfMoves;
}

/// KLB implementation that evaluates the gain/loss in the branching
/// through local optimization of the branching.
///
template <class BROPT, class LBROPT>
class LocalPartitionOptimizer : public PartitionOptimizerBase<BROPT>
{
    using PartitionOptimizerBase<BROPT>::PartitionOptimizerBase;

private:
    using LocalBranchingOptimizer = LBROPT;

    double solveLocalBranchingProblem(size_t partitionIdA,
                                      size_t partitionIdB) const override;
    double getBaselineBranchingObjective(size_t partitionIdA,
                                         size_t partitionIdB) const override;
    std::string getMethodName() const override { return "KLBlocal"; };
};

template <class BROPT, class LBROPT>
inline double
LocalPartitionOptimizer<BROPT, LBROPT>::solveLocalBranchingProblem(
    size_t partitionIdA, size_t partitionIdB) const
{
    auto localBranchingOptimizer =
        LocalBranchingOptimizer(this->partitionGraph_, partitionIdA,
                                partitionIdB, this->data_.maxDistance);
    return localBranchingOptimizer.optimize();
}

template <class BROPT, class LBROPT>
inline double
LocalPartitionOptimizer<BROPT, LBROPT>::getBaselineBranchingObjective(
    size_t partitionIdA, size_t partitionIdB) const
{
    return solveLocalBranchingProblem(partitionIdA, partitionIdB);
}

/// KLB implementation that always solves the *full* branching problem.
/// This is inefficient and advised for comparison purposes only.
///
template <class BROPT>
class FullPartitionOptimizer : public PartitionOptimizerBase<BROPT>
{
public:
    using PartitionOptimizerBase<BROPT>::PartitionOptimizerBase;

private:
    double solveLocalBranchingProblem(size_t partitionIdA,
                                      size_t partitionIdB) const override;
    double getBaselineBranchingObjective(size_t partitionIdA,
                                         size_t partitionIdB) const override;
    std::string getMethodName() const override { return "KLBfull"; };
};

/// Uses solveFullBranchingProblem() instead.
///  Arguments are discarded and only there to properly override.
template <class BROPT>
inline double
FullPartitionOptimizer<BROPT>::solveLocalBranchingProblem(
    size_t partitionIdA, size_t partitionIdB) const
{
    return this->solveFullBranchingProblem();
}

template <class BROPT>
inline double
FullPartitionOptimizer<BROPT>::getBaselineBranchingObjective(
    size_t partitionIdA, size_t partitionIdB) const
{
    return this->getBranchingObjective();
}

} // end namespace heuristics
} // end namespace lineage

#endif
