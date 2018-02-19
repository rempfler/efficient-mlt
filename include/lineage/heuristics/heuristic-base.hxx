#pragma once
#ifndef LINEAGE_HEURISTICS_BASE_HXX
#define LINEAGE_HEURISTICS_BASE_HXX

#include <string>

#include "levinkov/timer.hxx"
#include "lineage/problem-graph.hxx"
#include "lineage/solution.hxx"

namespace lineage {
namespace heuristics {

class HeuristicBase
{
public:
    HeuristicBase(Data& data)
      : data_(data){};

    using Cost = double;

    virtual void optimize() = 0;
    virtual Solution getSolution() = 0;
    virtual Cost getObjective() const = 0;

    inline void setSilent(bool flag);
    void logObj() const;

    void setMaxIter(const size_t maxIter) {}

protected:
    Data& data_;
    bool silent_{ false };
};

inline void
HeuristicBase::setSilent(bool flag)
{
    silent_ = flag;
}

inline void
HeuristicBase::logObj() const
{
    if (silent_)
        return;

    data_.timer.stop(); // dont time log-IO

    std::stringstream stream;
    stream << data_.timer.get_elapsed_seconds() << " "
           << "inf " // bound
           << std::setprecision(10) << std::fixed << getObjective()
           << " "      // objBest
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

template <class DATA, class OPT, class SOL>
void
postOptimizationChecks(DATA const& data, OPT const& optimizer,
                       SOL const& solution)
{
    // calculate costs.
    auto obj = evaluate(data, solution);

    std::cout << "Terminated after " << data.timer.get_elapsed_seconds()
              << " s with objective of " << obj << std::endl;

    // make sure the objective is actually as predicted.
    if (std::abs(optimizer.getObjective() - obj) > 1e-6) {
        const auto diffObj = optimizer.getObjective() - obj;
        const auto relDiffObj = std::abs(diffObj / obj);

        std::cerr
            << "Warning: Deviation between estimated and actual objective!"
            << std::endl;
        std::cerr << std::setw(15) << "predicted:" << std::setw(15)
                  << std::setprecision(10) << optimizer.getObjective()
                  << std::endl;
        std::cerr << std::setw(15) << "actual:" << std::setw(15)
                  << std::setprecision(10) << obj << std::endl;

        std::cerr << std::setw(15) << "abs. diff:" << std::setw(15)
                  << std::setprecision(10) << diffObj << std::endl;
        std::cerr << std::setw(15) << "rel. diff:" << std::setw(15)
                  << std::setprecision(10) << relDiffObj << std::endl;

        // if there is a severe difference, then throw.
        if (relDiffObj > .0001)
            throw std::runtime_error("Error in objective calculation!");
    }

    // print runtime, objective value, bound, numbers of violated ineqs. (0)
    {
        std::stringstream stream;
        stream << data.timer.get_elapsed_seconds() << " "
               << "inf"
               << " " << std::setprecision(10) << std::fixed << obj << " "
               << "nan"     // gap
               << " 0 0 0"; // violated constraints;
        stream << " 0 0 0"; // termination/birth/bifuraction constr.
        stream << " 0 0\n";

        std::cout << stream.str();

        std::ofstream file(data.solutionName + "-optimization-log.txt",
                           std::ofstream::out | std::ofstream::app);
        file << stream.str();
        file.close();
    }
}

template <class OPTIMIZER>
Solution
applyHeuristic(ProblemGraph const& problemGraph, double costTermination = .0,
               double costBirth = .0, bool enforceBifurcationConstraint = false,
               std::string solutionName = "heuristic", size_t maxIter = 500)
{

    // create log file/replace existing log file with empty log file
    {
        std::ofstream file(solutionName + "-optimization-log.txt");
        file << "time objBound objBest gap nSpaceCycle nSpaceTime nMorality "
                "nTermination nBirth nBifurcation objValue time_separation\n";
        file.close();
    }

    Data data(problemGraph);
    data.costBirth = costBirth;
    data.costTermination = costTermination;
    data.enforceBifurcationConstraint = enforceBifurcationConstraint;
    data.solutionName = solutionName;

    // define costs
    for (auto e : problemGraph.problem().edges)
        data.costs.push_back(e.weight);

    if (costTermination > 0.0)
        data.costs.insert(data.costs.end(),
                          problemGraph.graph().numberOfVertices(),
                          costTermination);

    if (costBirth > 0.0)
        data.costs.insert(data.costs.end(),
                          problemGraph.graph().numberOfVertices(), costBirth);

    data.timer.start();
    auto search = OPTIMIZER(data);
    search.setMaxIter(maxIter);

    search.optimize();
    const auto solution = search.getSolution();
    data.timer.stop();

    postOptimizationChecks(data, search, solution);

    return solution;
}

template <class OPTIMIZER, class INITIALIZER>
Solution
applyInitializedHeuristic(
    ProblemGraph const& problemGraph, double costTermination = .0,
    double costBirth = .0, bool enforceBifurcationConstraint = false,
    std::string solutionName = "heuristic",
    size_t maxDistance = std::numeric_limits<size_t>::max())
{

    // create log file/replace existing log file with empty log file
    {
        std::ofstream file(solutionName + "-optimization-log.txt");
        file << "time objBound objBest gap nSpaceCycle nSpaceTime nMorality "
                "nTermination nBirth nBifurcation objValue time_separation\n";
        file.close();
    }

    Data data(problemGraph);
    data.costBirth = costBirth;
    data.costTermination = costTermination;
    data.enforceBifurcationConstraint = enforceBifurcationConstraint;
    data.solutionName = solutionName;
    data.maxDistance = maxDistance;

    // define costs
    for (auto e : problemGraph.problem().edges)
        data.costs.push_back(e.weight);

    if (costTermination > 0.0)
        data.costs.insert(data.costs.end(),
                          problemGraph.graph().numberOfVertices(),
                          costTermination);

    if (costBirth > 0.0)
        data.costs.insert(data.costs.end(),
                          problemGraph.graph().numberOfVertices(), costBirth);

    Solution init;
    {
        data.timer.start();
        auto initializer = INITIALIZER(data);
        initializer.optimize();
        init = initializer.getSolution();
        data.timer.stop();
    }

    // create log replace log of initializer with empty log file
    {
        std::ofstream file(solutionName + "-optimization-log.txt");
        file << "time objBound objBest gap nSpaceCycle nSpaceTime nMorality "
                "nTermination nBirth nBifurcation objValue time_separation\n";
        file.close();
    }

    data.timer.start();
    auto search = OPTIMIZER(data, init);

    search.optimize();
    const auto solution = search.getSolution();
    data.timer.stop();

    postOptimizationChecks(data, search, solution);

    return solution;
}

} // end namespace heuristics
} // end namespace lineage
#endif
