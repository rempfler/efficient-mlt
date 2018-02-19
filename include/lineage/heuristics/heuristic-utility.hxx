#pragma once
#ifndef LINEAGE_HEURISTICS_UTILITY_HXX
#define LINEAGE_HEURISTICS_UTILITY_HXX

#include <andres/graph/components.hxx>

#include "lineage/problem-graph.hxx"

namespace lineage {
namespace heuristics {

template <class ELA>
void
generateLabelsForILP(const ProblemGraph& problemGraph, ELA& edge_labels,
                     const double costTermination, const double costBirth)
{
    // if there are neither birth or termination costs, then
    // no indicator variables for these events need to be set.
    if (costTermination < .0 && costBirth < .0) {
        return;
    }

    using Components =
        andres::graph::ComponentsBySearch<decltype(problemGraph.graph())>;
    using SubgraphMask =
        ProblemGraph::SubgraphWithoutCutAndInterFrameEdges<decltype(
            edge_labels)>;

    Components components;
    const auto numberOfComponents =
        components.build(problemGraph.graph(),
                         SubgraphMask(problemGraph.problem(), edge_labels));

    auto hasChild = std::vector<bool>(numberOfComponents, false);
    auto hasParent = std::vector<bool>(numberOfComponents, false);

    // check which components have parents and children.
    for (size_t edge = 0; edge < problemGraph.graph().numberOfEdges(); ++edge) {

        if (edge_labels[edge] == 1) {
            continue;
        }

        auto v0 = problemGraph.graph().vertexOfEdge(edge, 0);
        auto v1 = problemGraph.graph().vertexOfEdge(edge, 1);

        if (problemGraph.frameOfNode(v0) == problemGraph.frameOfNode(v1)) {
            continue;
        } else if (problemGraph.frameOfNode(v0) >
                   problemGraph.frameOfNode(v1)) {
            std::swap(v0, v1);
        }

        hasChild[components.labels_[v0]] = true;
        hasParent[components.labels_[v1]] = true;
    }

    // it occurs that there is no cost for birth in the first frame.
    // and no cost for termination in the last.
    for (size_t v0 = 0; v0 < problemGraph.graph().numberOfVertices(); ++v0) {
        if (problemGraph.frameOfNode(v0) == 0) {
            hasParent[components.labels_[v0]] = true;
        } else if (problemGraph.frameOfNode(v0) ==
                   problemGraph.numberOfFrames() - 1) {
            hasChild[components.labels_[v0]] = true;
        }
    }

    auto offset = problemGraph.graph().numberOfEdges();
    if (costTermination > .0) {

        edge_labels.reserve(edge_labels.size() +
                            problemGraph.graph().numberOfVertices());

        for (size_t v = 0; v < problemGraph.graph().numberOfVertices(); ++v) {

            bool isInLastFrame = problemGraph.frameOfNode(v) ==
                                 problemGraph.numberOfFrames() - 1;

            edge_labels.emplace_back(
                hasChild[components.labels_[v]] or isInLastFrame ? 0 : 1);
        }
        offset += problemGraph.graph().numberOfVertices();
    }

    if (costBirth > .0) {
        edge_labels.reserve(edge_labels.size() +
                            problemGraph.graph().numberOfVertices());

        for (size_t v = 0; v < problemGraph.graph().numberOfVertices(); ++v) {

            bool isInFirstFrame = problemGraph.frameOfNode(v) == 0;
            edge_labels.emplace_back(
                hasParent[components.labels_[v]] or isInFirstFrame ? 0 : 1);
        }
    }
}

} // end namespace heuristics
} // end namespace lineage

#endif
