#pragma once
#ifndef LINEAGE_SOLVER_ILP_CALLBACK_HXX
#define LINEAGE_SOLVER_ILP_CALLBACK_HXX

#include <cmath>
#include <vector>
#include <stack>
#include <sstream>
#include <fstream>
#include <iostream>

#include <andres/graph/components.hxx>
#include <andres/graph/paths.hxx>
#include <andres/graph/shortest-paths.hxx>

#include <levinkov/timer.hxx>

#include "problem-graph.hxx"
#include "solution.hxx"
#include "evaluate.hxx"
#include "heuristics/greedy-lineage.hxx"
#include "heuristics/heuristic-utility.hxx"
#include "heuristics/hungarian-branching.hxx"
#include "heuristics/partition-graph.hxx"


namespace lineage {

template<class ILP>
Solution solver_ilp(ProblemGraph const& problemGraph, double costTermination = .0, double costBirth = .0, bool enforceBifurcationConstraint = false, bool add3WheelConstraints = false, bool initialize = false, std::string solutionName = "ilp")
{

    class Callback: public ILP::Callback
    {
    public:
        Callback(ILP& solver, Data& data) :
            ILP::Callback(solver),
            data_(data),
            coefficients_(data.costs.size()),
            variables_(data.costs.size()),
            edgeLabels_(data.costs.size())
        {

        }


        void computeFeasibleSolution() override
        {
            using BranchingOptimizer = lineage::heuristics::branching::HungarianBranching<
            lineage::heuristics::PartitionGraph>;

            auto pgraph = lineage::heuristics::PartitionGraph(data_, edgeLabels_);

            auto branchingOptimizer = BranchingOptimizer(pgraph);
            branchingOptimizer.optimize();

            pgraph.branchingLabels_.clear();
            pgraph.branchingLabels_.reserve(pgraph.numberOfEdges());

            for (auto&& label : branchingOptimizer.getSolution()) {
                pgraph.branchingLabels_.emplace_back(label);
            }

            // set feasible solution
            for (size_t t = 0; t < data_.problemGraph.numberOfFrames(); ++t) 
            {
                // intra-frame edges
                for (size_t i = 0; i < data_.problemGraph.numberOfEdgesInFrame(t); ++i)
                {
                    auto edge = data_.problemGraph.edgeInFrame(t, i);
                    auto v0 = data_.problemGraph.graph().vertexOfEdge(edge, 0);
                    auto v1 = data_.problemGraph.graph().vertexOfEdge(edge, 1);

                    if (pgraph.areConnected(v0,v1))
                        this->setLabel(edge, 0);
                    else
                        this->setLabel(edge, 1.0); 
                }

                if (t == data_.problemGraph.numberOfFrames() - 1)
                    break;

                // inter-frame edges
                for (size_t i = 0; i < data_.problemGraph.numberOfEdgesFromFrame(t); ++i)
                {
                    auto edge = data_.problemGraph.edgeFromFrame(t, i);
                    auto v0 = data_.problemGraph.graph().vertexOfEdge(edge, 0);
                    auto v1 = data_.problemGraph.graph().vertexOfEdge(edge, 1);

                    if (pgraph.areConnected(v0,v1))
                        this->setLabel(edge, 0);
                    else
                        this->setLabel(edge, 1.0);
                }
            }

            // birth and termination edges
            auto offset_t = data_.problemGraph.graph().numberOfEdges();
            auto offset_b = offset_t + data_.problemGraph.graph().numberOfVertices();

            for (size_t t = 0; t < data_.problemGraph.numberOfFrames(); ++t)
            {
                for (size_t j = 0; j < data_.problemGraph.numberOfNodesInFrame(t); ++j)
                {
                    auto v = data_.problemGraph.nodeInFrame(t, j);
                    auto t_index = v + offset_t;
                    auto b_index = v + offset_b; 

                    bool termination = true;
                    if (t < data_.problemGraph.numberOfFrames() - 1)
                    {
                        for (auto it = data_.problemGraph.graph().adjacenciesFromVertexBegin(v);
                                it != data_.problemGraph.graph().adjacenciesFromVertexEnd(v); ++it)
                        {
                            auto w = it->vertex();
                            if (data_.problemGraph.frameOfNode(w) == t+1 && pgraph.areConnected(v,w))
                            {
                                termination = false;
                                break;
                            }
                        }
                    }
                    if (termination)
                        this->setLabel(t_index, 1.0);

                    bool birth = true;
                    if (t > 0)
                    {
                        for (auto it = data_.problemGraph.graph().adjacenciesFromVertexBegin(v);
                                it != data_.problemGraph.graph().adjacenciesFromVertexEnd(v); ++it)
                        {
                            auto w = it->vertex();
                            if (data_.problemGraph.frameOfNode(w) == t-1 && pgraph.areConnected(w,v))
                            {
                                birth = false;
                                break;
                            }
                        }
                    }
                    if (birth)
                        this->setLabel(b_index, 1.0);
                }
            }
        }

        void separateAndAddLazyConstraints() override
        {
            std::stringstream stream;

            data_.timer.stop();
            auto time = data_.timer.get_elapsed_seconds();
            data_.timer.start();

            auto gap = (this->objectiveBest_ - this->objectiveBound_) / (1.0 + std::fabs(this->objectiveBound_));
            std::cout << time
                << ' ' << this->objectiveBound_
                << ' ' << this->objectiveBest_
                << ' ' << gap
                << std::flush;
            stream << time
                << ' ' << this->objectiveBound_
                << ' ' << this->objectiveBest_
                << ' ' << gap;

            levinkov::Timer t_separation;
            t_separation.start();

            componentsInFrame_.build(
                data_.problemGraph.graph(),
                SubgraphWithoutCutAndInterFrameEdges(data_.problemGraph.problem(), EdgeLabels(*this))
            );

            auto nSpaceCycle = separateAndAddSpaceCycleConstraints();
            std::cout << ' ' << nSpaceCycle << std::flush;
            stream << ' ' << nSpaceCycle;

            auto nSpacetimeCycle = separateAndAddSpacetimeCycleConstraints();
            std::cout << ' ' << nSpacetimeCycle << std::flush;
            stream << ' ' << nSpacetimeCycle;

            auto nMorality = separateAndAddMoralityConstraints();
            std::cout << ' ' << nMorality << std::flush;
            stream << ' ' << nMorality;

            size_t nTermination = 0;
            if (data_.costTermination > 0.0)
            {
                nTermination = separateAndAddTerminationConstraints();
                std::cout << ' ' << nTermination << std::flush;
                stream << ' ' << nTermination;
            }

            size_t nBirth = 0;
            if (data_.costBirth > 0.0)
            {
                nBirth = separateAndAddBirthConstraints();
                std::cout << ' ' << nBirth << std::flush;
                stream << ' ' << nBirth;
            }

            auto n = nSpaceCycle + nSpacetimeCycle + nMorality + nTermination + nBirth;
            size_t nBifurcation = 0;
            if (data_.enforceBifurcationConstraint)
            {
                nBifurcation = separateAndAddBifurcationConstraints();
                std::cout << ' ' << nBifurcation << std::flush;
                stream << ' ' << nBifurcation;
            }


            t_separation.stop();
            data_.timer.stop(); // not keeping time for writing log

            stream << " " << t_separation.get_elapsed_seconds() << std::endl;
            std::cout << " " << t_separation.get_elapsed_seconds() << std::endl;

            {
                std::ofstream file(data_.solutionName + "-optimization-log.txt", std::ofstream::out | std::ofstream::app);
                file << stream.str();
                file.close();
            }

            n = n + nBifurcation;
            if (n == 0)
            {
                std::ofstream file(data_.solutionName + "-fragment-edge-labels-FEASIBLE-" + std::to_string(numberOfFeasibleSolutions_) + ".txt");
                for (size_t e = 0; e < data_.problemGraph.graph().numberOfEdges(); ++e)
                    file << (this->label(e) > .5 ? 1 : 0) << std::endl;
                
                file.close();

                file.open(data_.solutionName + "-variables-values-FEASIBLE-" + std::to_string(numberOfFeasibleSolutions_) + ".txt");
                for (size_t i = 0; i < data_.costs.size(); ++i)
                    file << (this->label(i) > .5 ? 1 : 0) << std::endl;
                
                file.close();

                ++numberOfFeasibleSolutions_;
            }

            ++numberOfSeparationCalls_;

            data_.timer.start(); // resume keeping time

            // compute feasible edge labels
            ComponentsType components;
            components.build(
                data_.problemGraph.graph(),
                SubgraphWithoutCutAndInterFrameEdges(data_.problemGraph.problem(), EdgeLabels(*this))
            );

            for (size_t t = 0; t < data_.problemGraph.numberOfFrames(); ++t)
            {
                for (size_t i = 0; i < data_.problemGraph.numberOfEdgesInFrame(t); ++i)
                {
                    auto e = data_.problemGraph.edgeInFrame(t, i);
                    auto v0 = data_.problemGraph.graph().vertexOfEdge(e, 0);
                    auto v1 = data_.problemGraph.graph().vertexOfEdge(e, 1);

                    // if connected within frame
                    if (components.areConnected(v0,v1))
                        edgeLabels_[e] = 0;
                    else
                        edgeLabels_[e] = 1;
                }
            }
        }

    private:
        typedef andres::graph::ComponentsBySearch<typename ProblemGraph::Graph> ComponentsType;

        class EdgeLabels
        {
        public:
            EdgeLabels(Callback& callback) :
                callback_(callback)
            {}

            int operator[](size_t edge) const
            {
                return callback_.label(edge) > .5 ? 1 : 0;
            }

        private:
            Callback& callback_;
        };

        typedef ProblemGraph::SubgraphWithoutCutAndInterFrameEdges<EdgeLabels> SubgraphWithoutCutAndInterFrameEdges;
        typedef ProblemGraph::SubgraphOfTwoFramesWithoutCut<EdgeLabels> SubgraphOfTwoFramesWithoutCut;

        size_t separateAndAddSpaceCycleConstraints()
        {
            std::deque<size_t> path;
            std::vector<ptrdiff_t> buffer;
            size_t counter = 0;

            std::vector<char> seen(data_.problemGraph.graph().numberOfVertices());
            std::vector<ptrdiff_t> parent(data_.problemGraph.graph().numberOfVertices());
            std::vector<double> cost(data_.problemGraph.graph().numberOfVertices());
            std::vector<size_t> dist(data_.problemGraph.graph().numberOfVertices());

            // separate cycles between consecutive frames
            for (size_t t = 0; t < data_.problemGraph.numberOfFrames() - 1; ++t)
            {
                // do connected components labeling only for frames t and t+1
                ComponentsType components;
                components.build(
                    data_.problemGraph.graph(),
                    SubgraphOfTwoFramesWithoutCut(data_.problemGraph.problem(), EdgeLabels(*this), t)
                );

                for (size_t i = 0; i < data_.problemGraph.numberOfEdgesInFrame(t); ++i)
                {
                    auto e = data_.problemGraph.edgeInFrame(t, i);

                    auto v0 = data_.problemGraph.graph().vertexOfEdge(e, 0);
                    auto v1 = data_.problemGraph.graph().vertexOfEdge(e, 1);
                    
                    // if an edge violates connectivity as defined by connected components
                    if (this->label(e) > .5 && components.areConnected(v0, v1))
                    {
                        // find the shortest path using BFS
                        spsp(
                            data_.problemGraph.graph(),
                            SubgraphOfTwoFramesWithoutCut(data_.problemGraph.problem(), EdgeLabels(*this), t),
                            v0, v1, path, buffer
                        );

                        // skip chord check for triangles
                        if (path.size() > 3)
                        {
                            // check for chords
                            std::fill(seen.begin(), seen.end(), 0);
                            if (hasChord(data_.problemGraph.graph(), path.begin(), path.end(), seen, true))
                                continue;
                        }

                        
                        for (size_t j = 0; j < path.size() - 1; ++j)
                        {
                            variables_[j] = data_.problemGraph.graph().findEdge(path[j], path[j + 1]).second;
                            coefficients_[j] = 1.0;
                        }

                        variables_[path.size() - 1] = e;
                        coefficients_[path.size() - 1] = -1.0;

                        this->addLazyConstraint(variables_.begin(), variables_.begin() + path.size(), coefficients_.begin(), 0, std::numeric_limits<double>::infinity());

                        ++counter;
                        
                    }
                }
            }

            // separate cycles in the last frame
            size_t t = data_.problemGraph.numberOfFrames() - 1;
            for (size_t i = 0; i < data_.problemGraph.numberOfEdgesInFrame(t); ++i)
            {
                auto e = data_.problemGraph.edgeInFrame(t, i);

                auto v0 = data_.problemGraph.graph().vertexOfEdge(e, 0);
                auto v1 = data_.problemGraph.graph().vertexOfEdge(e, 1);
                
                // if an edge violates connectivity as defined by connected components
                if (this->label(e) > .5 && componentsInFrame_.areConnected(v0, v1))
                {
                    // find the shortest path using BFS
                    spsp(
                        data_.problemGraph.graph(),
                        SubgraphWithoutCutAndInterFrameEdges(data_.problemGraph.problem(), EdgeLabels(*this)),
                        v0, v1, path, buffer
                    );

                    // skip chord check for triangles
                    if (path.size() > 3)
                    {
                        // check for chords
                        std::fill(seen.begin(), seen.end(), 0);
                        if (hasChord(data_.problemGraph.graph(), path.begin(), path.end(), seen, true))
                            continue;
                    }
                    
                    for (size_t j = 0; j < path.size() - 1; ++j)
                    {
                        variables_[j] = data_.problemGraph.graph().findEdge(path[j], path[j + 1]).second;
                        coefficients_[j] = 1.0;
                    }

                    variables_[path.size() - 1] = e;
                    coefficients_[path.size() - 1] = -1.0;

                    this->addLazyConstraint(variables_.begin(), variables_.begin() + path.size(), coefficients_.begin(), 0, std::numeric_limits<double>::infinity());

                    ++counter;                    
                }
            }
                

            return counter;
        }

        size_t separateAndAddSpacetimeCycleConstraints()
        {
            std::deque<size_t> path;
            std::vector<ptrdiff_t> buffer;
            size_t counter = 0;

            std::vector<char> seen(data_.problemGraph.graph().numberOfVertices());
            std::vector<ptrdiff_t> parent(data_.problemGraph.graph().numberOfVertices());
            std::vector<double> cost(data_.problemGraph.graph().numberOfVertices());
            std::vector<size_t> dist(data_.problemGraph.graph().numberOfVertices());

            for (size_t t = 0; t < data_.problemGraph.numberOfFrames() - 1; ++t)
            {
                // do connected components labeling only for frames t and t+1
                ComponentsType components;
                components.build(
                    data_.problemGraph.graph(),
                    SubgraphOfTwoFramesWithoutCut(data_.problemGraph.problem(), EdgeLabels(*this), t)
                );

                for (size_t j = 0; j < data_.problemGraph.numberOfEdgesFromFrame(t); ++j)
                {
                    auto e = data_.problemGraph.edgeFromFrame(t, j);
                    auto v0 = data_.problemGraph.graph().vertexOfEdge(e, 0);
                    auto v1 = data_.problemGraph.graph().vertexOfEdge(e, 1);

                    // if a time edge violates connectivity
                    if (this->label(e) > .5 && components.areConnected(v0, v1))
                    {
                        // find the shortest path using BFS
                        spsp(
                            data_.problemGraph.graph(),
                            SubgraphOfTwoFramesWithoutCut(data_.problemGraph.problem(), EdgeLabels(*this), t),
                            v0, v1, path, buffer
                        );

                        // skip chord check for triangles
                        if (path.size() > 3)
                        {
                            // check for chords
                            std::fill(seen.begin(), seen.end(), 0);
                            if (hasChord(data_.problemGraph.graph(), path.begin(), path.end(), seen, true))
                                continue;
                        }

                        for (size_t k = 0; k < path.size() - 1; ++k)
                        {
                            variables_[k] = data_.problemGraph.graph().findEdge(path[k], path[k + 1]).second;
                            coefficients_[k] = 1.0;
                        }

                        variables_[path.size() - 1] = e;
                        coefficients_[path.size() - 1] = -1.0;

                        this->addLazyConstraint(variables_.begin(), variables_.begin() + path.size(), coefficients_.begin(), 0, std::numeric_limits<double>::infinity());

                        ++counter;
                    }
                }
            }

            return counter;
        }

        size_t separateAndAddMoralityConstraints()
        {
            std::deque<size_t> path;
            std::vector<ptrdiff_t> buffer;
            std::vector<char> visited(data_.problemGraph.graph().numberOfVertices());
            size_t counter = 0;

            std::vector<char> seen(data_.problemGraph.graph().numberOfVertices());
            std::vector<ptrdiff_t> parent(data_.problemGraph.graph().numberOfVertices());
            std::vector<double> cost(data_.problemGraph.graph().numberOfVertices());
            std::vector<size_t> dist(data_.problemGraph.graph().numberOfVertices());

            for (size_t t = 0; t < data_.problemGraph.numberOfFrames() - 1; ++t)
            {
                // do connected components labeling only for frames t and t+1
                ComponentsType components;
                components.build(
                    data_.problemGraph.graph(),
                    SubgraphOfTwoFramesWithoutCut(data_.problemGraph.problem(), EdgeLabels(*this), t)
                );

                // iterate over all node pairs
                for (size_t i = 0; i < data_.problemGraph.numberOfNodesInFrame(t); ++i)
                    for (size_t j = 0; j < data_.problemGraph.numberOfNodesInFrame(t); ++j)
                    {
                        if (i == j)
                            continue;

                        auto v0 = data_.problemGraph.nodeInFrame(t, i);
                        auto v1 = data_.problemGraph.nodeInFrame(t, j);

                        // skip pairs which correspond to an edge
                        if (data_.problemGraph.graph().findEdge(v0,v1).first)
                            continue;

                        if (components.areConnected(v0, v1) && !componentsInFrame_.areConnected(v0, v1))
                        {
                            // find the shortest path using BFS
                            spsp(
                                data_.problemGraph.graph(),
                                SubgraphOfTwoFramesWithoutCut(data_.problemGraph.problem(), EdgeLabels(*this), t),
                                v0, v1, path, buffer
                            );

                            // skip chord check for triangles
                            if (path.size() > 3)
                            {
                                // skip paths that admit a "lifted" chord
                                bool f_chord = false;
                                for (auto it = path.begin() + 1; it != path.end() - 1 && !f_chord; it++)
                                {
                                    if (data_.problemGraph.frameOfNode(*it) == t)
                                    {
                                        f_chord = true;
                                        break;
                                    }
                                }

                                if (f_chord)
                                    continue;

                                // check for chords
                                std::fill(seen.begin(), seen.end(), 0);
                                if (hasChord(data_.problemGraph.graph(), path.begin(), path.end(), seen, true))
                                    continue;
                            }

                            // store variables
                            ptrdiff_t sz = 0;
                            for (size_t k = 0; k < path.size() - 1; ++k)
                            {
                                coefficients_[sz] = 1.0;
                                variables_[sz] = data_.problemGraph.graph().findEdge(path[k], path[k + 1]).second;
                                ++sz;
                            }

                            // find a cut that separates v0 and v1 in frame t using DFS
                            std::fill(visited.begin() + data_.problemGraph.nodeInFrame(t, 0), visited.begin() + data_.problemGraph.nodeInFrame(t + 1, 0), 0);

                            std::stack<size_t> stack;
                            stack.push(v0);
                            visited[v0] = 1;
                            
                            while (!stack.empty())
                            {
                                auto v = stack.top();
                                stack.pop();

                                for (auto it2 = data_.problemGraph.graph().adjacenciesFromVertexBegin(v); it2 != data_.problemGraph.graph().adjacenciesFromVertexEnd(v); ++it2)
                                {
                                    auto w = it2->vertex();
                                    
                                    if (data_.problemGraph.frameOfNode(v) == data_.problemGraph.frameOfNode(w))
                                    {
                                        if (componentsInFrame_.labels_[v] != componentsInFrame_.labels_[w])
                                        {
                                            coefficients_[sz] = -1.0;
                                            variables_[sz] = it2->edge();
                                            
                                            ++sz;
                                        }
                                        else if (!visited[w])
                                        {
                                            visited[w] = 1;
                                            stack.push(w);
                                        }
                                    }
                                }
                            }
                            
                            this->addLazyConstraint(variables_.begin(), variables_.begin() + sz, coefficients_.begin(), 1 - (sz - static_cast<ptrdiff_t>(path.size()) + 1), std::numeric_limits<double>::infinity());

                            ++counter;  
                        }
                    }
            }

            return counter;
        }

        size_t separateAndAddTerminationConstraints()
        {           
            size_t counter = 0;

            for (size_t t = 0; t < data_.problemGraph.numberOfFrames() - 1; ++t)
                for (size_t j = 0; j < data_.problemGraph.numberOfNodesInFrame(t); ++j)
                {
                    auto v = data_.problemGraph.nodeInFrame(t, j);
                    auto terminationVariableIndex = v + data_.problemGraph.graph().numberOfEdges();

                    // check whether the connected component in frame t is terminated or not
                    if (this->label(terminationVariableIndex) < .5)
                    {
                        std::vector<char> successor(data_.problemGraph.graph().numberOfVertices());

                        std::vector<char> visited(data_.problemGraph.graph().numberOfVertices());

                        ptrdiff_t sz = 0;
                        std::queue<size_t> queue;
                        queue.push(v);

                        visited[v] = 1;

                        bool terminated = true;

                        // do the check and find the reduced cut at the same time
                        while (!queue.empty() && terminated)
                        {
                            auto vv = queue.front();
                            queue.pop();

                            for (auto it = data_.problemGraph.graph().adjacenciesFromVertexBegin(vv);
                                it != data_.problemGraph.graph().adjacenciesFromVertexEnd(vv); ++it)
                            {
                                auto w = it->vertex();
                                auto e = it->edge();

                                if (data_.problemGraph.frameOfNode(w) < t)
                                    continue;

                                // fragment is not terminated
                                if (data_.problemGraph.frameOfNode(w) == t + 1 && this->label(e) < .5)
                                {
                                    terminated = false;
                                    break;
                                }

                                if (data_.problemGraph.frameOfNode(w) == t && !visited[w] && this->label(e) < .5)
                                {
                                    visited[w] = 1;
                                    queue.push(w);
                                }

                                // add edge to the cut if it is not incident to a successor of v
                                if (this->label(e) > .5 && !successor[w])
                                {
                                    coefficients_[sz] = -1.0;
                                    variables_[sz] = e;
                                    
                                    ++sz;

                                    if (vv == v && data_.problemGraph.frameOfNode(w) == t + 1)
                                        successor[w] = 1;
                                }
                            }
                        }

                        if (terminated)
                            {
                                coefficients_[sz] = 1.0;
                                variables_[sz] = v + data_.problemGraph.graph().numberOfEdges();

                                // sz = cut capacity
                                this->addLazyConstraint(variables_.begin(), variables_.begin() + sz + 1, coefficients_.begin(), 1 - sz, std::numeric_limits<double>::infinity());

                                ++counter;
                            }
                    }
                }

            return counter;
        }

        size_t separateAndAddBirthConstraints()
        {
            auto offset = data_.problemGraph.graph().numberOfEdges();
            if (data_.costTermination > .0)
                offset += data_.problemGraph.graph().numberOfVertices();

            size_t counter = 0;
            
            for (size_t t = 1; t < data_.problemGraph.numberOfFrames(); ++t)
                for (size_t j = 0; j < data_.problemGraph.numberOfNodesInFrame(t); ++j)
                {
                    auto v = data_.problemGraph.nodeInFrame(t, j);
                    auto birthVariableIndex = v + offset;

                    if (this->label(birthVariableIndex) < .5)
                    {
                        std::vector<char> predecessor(data_.problemGraph.graph().numberOfVertices());

                        std::vector<char> visited(data_.problemGraph.graph().numberOfVertices());
                        
                        ptrdiff_t sz = 0;
                        std::queue<size_t> queue;
                        queue.push(v);
                        
                        visited[v] = 1;
                        
                        bool born = true;

                        while (!queue.empty() && born)
                        {
                            auto vv = queue.front();
                            queue.pop();
                            
                            for (auto it = data_.problemGraph.graph().adjacenciesFromVertexBegin(vv);
                                it != data_.problemGraph.graph().adjacenciesFromVertexEnd(vv); ++it)
                            {
                                auto w = it->vertex();
                                auto e = it->edge();

                                if (data_.problemGraph.frameOfNode(w) > t)
                                    continue;

                                // fragment is not born
                                if (data_.problemGraph.frameOfNode(w) == t - 1 && this->label(e) < .5)
                                {
                                    born = false;
                                    break;
                                }

                                if (data_.problemGraph.frameOfNode(w) == t && !visited[w] && this->label(e) < .5)
                                {
                                    visited[w] = 1;
                                    queue.push(w);
                                }

                                // add edge to the cut if it is not incident to a predecessor of v
                                if (this->label(e) > .5 && !predecessor[w])
                                {
                                    coefficients_[sz] = -1.0;
                                    variables_[sz] = e;
                                    
                                    ++sz;

                                    if (vv == v && data_.problemGraph.frameOfNode(w) == t - 1)
                                        predecessor[w] = 1;
                                }
                            }
                        }

                        if (born)
                            {
                                variables_[sz] = v + offset;
                                coefficients_[sz] = 1.0;

                                // sz = cut capacity
                                this->addLazyConstraint(variables_.begin(), variables_.begin() + sz + 1, coefficients_.begin(), 1 - sz, std::numeric_limits<double>::infinity());

                                ++counter;
                            }
                    }
                }

            return counter;
        }

        size_t separateAndAddBifurcationConstraints()
        {
            std::deque<size_t> path;
            std::vector<ptrdiff_t> buffer;
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            
            std::vector<ptrdiff_t> parent(data_.problemGraph.graph().numberOfVertices(), -1);
            size_t counter = 0;

            // for each connected component find the cut that separates it from all other components
            std::vector<std::vector<size_t>> cuts(*max_element(componentsInFrame_.labels_.begin(), componentsInFrame_.labels_.end()) + 1);

            for (size_t t = 0; t < data_.problemGraph.numberOfFrames(); ++t)
                for (size_t i = 0; i < data_.problemGraph.numberOfEdgesInFrame(t); ++i)
                {
                    auto e = data_.problemGraph.edgeInFrame(t, i);

                    auto v0 = data_.problemGraph.graph().vertexOfEdge(e, 0);
                    auto v1 = data_.problemGraph.graph().vertexOfEdge(e, 1);
                    
                    if (this->label(e) > .5)
                    {
                        cuts[componentsInFrame_.labels_[v0]].push_back(e);
                        cuts[componentsInFrame_.labels_[v1]].push_back(e);
                    }
                }

            for (auto& c : cuts)
                sort(c.begin(), c.end());
            
            //////////////////////////////////////////////////////////////////////////////////////////////////
            

            for (size_t t = 0; t < data_.problemGraph.numberOfFrames() - 1; ++t)
                for (size_t j = 0; j < data_.problemGraph.numberOfNodesInFrame(t); ++j)
                {
                    auto v = data_.problemGraph.nodeInFrame(t, j);

                    // if not visited
                    if (parent[v] == -1)
                    {
                        // map from frame component labels to vertex/edge pairs that correspond to the touch points
                        std::map<size_t, std::pair<size_t, size_t>> touch_points;

                        std::queue<size_t> Q;
                        Q.push(v);
                        
                        parent[v] = v;
                        
                        while (!Q.empty())
                        {
                            auto vv = Q.front();
                            Q.pop();
                            
                            for (auto it = data_.problemGraph.graph().adjacenciesFromVertexBegin(vv); it != data_.problemGraph.graph().adjacenciesFromVertexEnd(vv); ++it)
                            {
                                auto w = it->vertex();
                                auto e = it->edge();

                                if (data_.problemGraph.frameOfNode(w) == t + 1 && this->label(e) < .5 && touch_points.count(componentsInFrame_.labels_[w]) == 0)
                                    touch_points[componentsInFrame_.labels_[w]] = std::make_pair(vv, e);

                                if (data_.problemGraph.frameOfNode(w) == t && parent[w] == -1 && this->label(e) < .5)
                                {
                                    parent[w] = vv;
                                    Q.push(w);
                                }
                            }
                        }

                        // if there are more than 2 components in the next frame
                        if (touch_points.size() > 2)
                        {
                            std::vector<size_t> P;
                            std::vector<size_t> C;

                            for (auto& p : touch_points)
                            {
                                auto w = p.second.first;
                                while (w != v)
                                {
                                    P.push_back(data_.problemGraph.graph().findEdge(parent[w], w).second);
                                    w = parent[w];
                                }

                                P.push_back(p.second.second);

                                C.insert(C.end(), cuts[p.first].begin(), cuts[p.first].end());
                            }

                            std::sort(P.begin(), P.end());
                            std::sort(C.begin(), C.end());

                            P.erase(std::unique(P.begin(), P.end()), P.end());
                            C.erase(std::unique(C.begin(), C.end()), C.end());

                            ptrdiff_t sz = 0;
                            for (auto e : P)
                            {
                                coefficients_[sz] = 1.0;
                                variables_[sz] = e;
                                
                                ++sz;
                            }

                            for (auto e : C)
                            {
                                coefficients_[sz] = -1.0;
                                variables_[sz] = e;

                                ++sz;
                            }

                            this->addLazyConstraint(variables_.begin(), variables_.begin() + sz, coefficients_.begin(), 1 - static_cast<ptrdiff_t>(C.size()), std::numeric_limits<double>::infinity());

                            ++counter;
                        }
                    }
                }

            return counter;
        }

        std::vector<double> coefficients_;
        ComponentsType components_;
        ComponentsType componentsInFrame_;
        Data& data_;
        size_t numberOfFeasibleSolutions_ { 0 };
        size_t numberOfSeparationCalls_ { 0 };
        std::vector<size_t> variables_;

        std::vector<double> edgeLabels_;
    };

    class ConstraintGenerator
    {
    public:
        ConstraintGenerator(ILP& ilp, Data& data) :
            ilp_(ilp),
            data_(data),
            coefficients_(data.costs.size()),
            variables_(data.costs.size())
        {

        }

        size_t addConstraints()
        {
            return add3WheelConstraints();
        }
    private:
        size_t add3WheelConstraints()
        {
            size_t counter = 0;

            for (size_t t = 0; t < data_.problemGraph.numberOfFrames() - 1; ++t)
                for (size_t i = 0; i < data_.problemGraph.numberOfEdgesInFrame(t); ++i)
                {
                    auto edge = data_.problemGraph.edgeInFrame(t, i);

                    auto v0 = data_.problemGraph.graph().vertexOfEdge(edge, 0);
                    auto v1 = data_.problemGraph.graph().vertexOfEdge(edge, 1);

                    for (auto it = data_.problemGraph.graph().adjacenciesFromVertexBegin(v0); it != data_.problemGraph.graph().adjacenciesFromVertexEnd(v0); ++it)
                    {
                        auto w = it->vertex();
                        auto e0 = it->edge();

                        if (data_.problemGraph.frameOfNode(w) == t)
                        {
                            auto p = data_.problemGraph.graph().findEdge(v1,w);
                            if (p.first)
                            {
                               for (auto it2 = data_.problemGraph.graph().adjacenciesFromVertexBegin(v0); it2 != data_.problemGraph.graph().adjacenciesFromVertexEnd(v0); ++it2)
                               {
                                    auto u = it2->vertex();
                                    auto f = it2->edge();
                                    auto f1 = data_.problemGraph.graph().findEdge(v1,u);
                                    auto f2 = data_.problemGraph.graph().findEdge(w,u);

                                    if (data_.problemGraph.frameOfNode(u) == t + 1 && f1.first && f2.first)
                                    {
                                        variables_[0] = edge;
                                        coefficients_[0] = -1.0;
                                        variables_[1] = e0;
                                        coefficients_[1] = -1.0;
                                        variables_[2] = p.second;
                                        coefficients_[2] = -1.0;
                                        variables_[3] = f;
                                        coefficients_[3] = 1.0;
                                        variables_[4] = f1.second;
                                        coefficients_[4] = 1.0;
                                        variables_[5] = f2.second;
                                        coefficients_[5] = 1.0;

                                        ilp_.addConstraint(variables_.begin(), variables_.begin() + 6, coefficients_.begin(), -1, std::numeric_limits<double>::infinity());

                                        counter++;
                                    }
                               } 
                            }
                        }

                    }
                    
                }

            return counter;
        }

        ILP& ilp_;
        Data& data_;
        std::vector<double> coefficients_;
        std::vector<size_t> variables_;
    };

    // create log file/replace existing log file with empty log file
    {
        std::ofstream file(solutionName + "-optimization-log.txt");
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
        data.costs.insert(data.costs.end(), problemGraph.graph().numberOfVertices(), costTermination);
    
    if (costBirth > 0.0)
        data.costs.insert(data.costs.end(), problemGraph.graph().numberOfVertices(), costBirth);

    ILP ilp;

    ilp.setRelativeGap(0.0);
    ilp.setAbsoluteGap(0.0);
    ilp.addVariables(data.costs.size(), data.costs.data());

    for (size_t t = 0; t < data.problemGraph.numberOfFrames(); ++t)
    {
        for (size_t i = 0; i < data.problemGraph.numberOfEdgesInFrame(t); ++i)
        {
            auto edge = data.problemGraph.edgeInFrame(t, i);
            ilp.setBranchPrio(edge, 1);
        }
        
    }

    if (initialize)
    {
        // initialize with GLA
        data.timer.start();

        using Heuristic = lineage::heuristics::GreedyLineageAgglomeration<>;
        auto initializer = Heuristic(data);
        initializer.optimize();

        auto init = initializer.getSolution();

        // add variables for birth/termination.
        lineage::heuristics::generateLabelsForILP(data.problemGraph,
                init.edge_labels,
                data.costTermination,
                data.costBirth);

        ilp.setStart(init.edge_labels.cbegin());

        data.timer.stop(); 
    }

    // set callback
    Callback callback(ilp, data);
    ilp.setCallback(callback);

    // add and log 3-Wheel inequalities
    if (add3WheelConstraints)
    {
        ConstraintGenerator cg(ilp,data);
        auto nConstraints = cg.addConstraints();

        std::stringstream stream;
        // each 3-wheel is counted three times...
        stream << "Added " << nConstraints/3 << " 3-wheel inequalities.\n";
        std::cout << stream.str();
        {
            std::ofstream file(solutionName + "-optimization-log.txt", std::ofstream::out | std::ofstream::app);
            file << stream.str();
            file.close();
        }
    }

    data.timer.start();
    ilp.optimize();
    data.timer.stop();

    // print runtime, objective value, bound, numbers of violated ineqs. (0)
    {
        std::stringstream stream;
        stream << data.timer.get_elapsed_seconds()
            << " " << ilp.objective()
            << " " << ilp.bound()
            << " " << ilp.gap()
            << " 0 0 0"; // violated constraints;

        if (costTermination > .0)
            stream << " 0";

        if (costBirth > .0)
            stream << " 0";

        if (enforceBifurcationConstraint)
            stream << " 0";

        stream << " 0 0\n";

        std::cout << stream.str();

        std::ofstream file(solutionName + "-optimization-log.txt", std::ofstream::out | std::ofstream::app);
        file << stream.str();
        file.close();
    }

    Solution solution;
    solution.edge_labels.resize(problemGraph.graph().numberOfEdges());
    for (size_t edge = 0; edge < problemGraph.graph().numberOfEdges(); ++edge)
        solution.edge_labels[edge] = ilp.label(edge) > 0.5 ? 1 : 0;

    return solution;
}

} // namespace lineage

#endif
