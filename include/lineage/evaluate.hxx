#pragma once
#ifndef LINEAGE_EVALUATE_HXX
#define LINEAGE_EVALUATE_HXX

#include <andres/graph/components.hxx>

#include "heuristics/heuristic-utility.hxx"
#include "problem-graph.hxx"
#include "solution.hxx"

namespace lineage {

template <class EVA>
inline typename EVA::value_type
evaluate(const ProblemGraph& problemGraph, const EVA& costs,
         const typename EVA::value_type costBirth,
         const typename EVA::value_type costTermination,
         const Solution& solution)
{
    auto extended_labels = solution.edge_labels;
    lineage::heuristics::generateLabelsForILP(problemGraph, extended_labels,
                                              costTermination, costBirth);

    using Cost = typename EVA::value_type;
    Cost obj = .0;

    // cost for cut edges.
    for (size_t edge = 0; edge < extended_labels.size(); ++edge) {
        if (extended_labels[edge] == 1) {
            obj += costs[edge];
        } else if (extended_labels[edge] != 0) {
            throw std::runtime_error("Edge labels can only be 0 or 1!");
        }
    }
    return obj;
}

template <class DATA>
inline auto
evaluate(const DATA& data, const Solution& solution)
    -> decltype(evaluate(data.problemGraph, data.costs, data.costBirth,
                         data.costTermination, solution))
{
    return evaluate(data.problemGraph, data.costs, data.costBirth,
                    data.costTermination, solution);
}

} // end namespace lineage

#endif
