#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <iostream>

namespace markurem {
namespace matching {

/// Implementation of the Munkres matching algorithm.
///  Based on:
///  Bourgeois and Lasalle, 1971,
///  "An extension of the Munkres algorithm for the
///  assignment problem to rectangular matrices"
///
/// It expects a biparite, directed graph where all edges go from one side to
/// the other. Cost and mask need to have an entry for each edge and will be
/// modified.
template <class GRAPH, class COST, class MASK>
class Matching
{
public:
    using value_type = typename COST::value_type;
    using Index = size_t;

    struct IndexPair
    {
        IndexPair() {}
        IndexPair(Index _row, Index _col)
          : row(_row)
          , col(_col)
        {
        }
        Index row, col;
    };

    Matching(GRAPH const& graph, COST& costs, MASK& mask)
      : graph_(graph)
      , costs_(costs)
      , mask_(mask)
    {
        for (size_t v = 0; v < graph_.numberOfVertices(); ++v) {
            if (graph_.numberOfEdgesToVertex(v) > 0)
                cols_.emplace_back(v);
            else if (graph_.numberOfEdgesFromVertex(v) > 0)
                rows_.emplace_back(v);
        }

        row_cover.resize(rows_.size(), false);
        col_cover.resize(cols_.size(), false);
    }

    void run();
    std::vector<IndexPair> matches() const;

private:
    // members
    GRAPH const& graph_;
    COST& costs_;
    MASK& mask_;

    std::vector<size_t> rows_;
    std::vector<size_t> cols_;

    std::vector<bool> row_cover;
    std::vector<bool> col_cover;

    std::vector<IndexPair> path_;

    size_t max_matches_;

    constexpr static int STARRED = 1;
    constexpr static int PRIMED = 2;

    enum class Step
    {
        ONE,
        TWO,
        THREE,
        FOUR,
        FIVE,
        SIX,
        SEVEN
    };

    Step step_{ Step::ONE };

    // basic steps of munkres algorithm for rectangular cost matrices.
    void step_one();
    void step_two();
    void step_three();
    void step_four();
    void step_five();
    void step_six();
    void step_seven();

    // auxiliary methods.
    IndexPair find_zero(Index start_row = 0) const;
    bool has_star_in_row(Index row) const;
    Index find_star_in_row(Index row) const;
    Index find_star_in_col(Index col) const;
    Index find_prime_in_row(Index row) const;
    value_type find_smallest() const;

    size_t col_of_idx(size_t idx) const { return idx - rows_.size(); }

    void clear_covers();
    void augment();
    void erase_primes();

    static constexpr Index IDXMAX = std::numeric_limits<Index>::max();
};

template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::run()
{
    max_matches_ = std::min(rows_.size(), cols_.size());

    while (true) {
        switch (step_) {
            case Step::ONE:
                step_one();
                break;
            case Step::TWO:
                step_two();
                break;
            case Step::THREE:
                step_three();
                break;
            case Step::FOUR:
                step_four();
                break;
            case Step::FIVE:
                step_five();
                break;
            case Step::SIX:
                step_six();
                break;
            case Step::SEVEN:
                return;
            default:
                throw std::runtime_error("Failed to handle step code!");
        }
    }
}

template <class GRAPH, class COST, class MASK>
std::vector<typename Matching<GRAPH, COST, MASK>::IndexPair>
Matching<GRAPH, COST, MASK>::matches() const
{
    std::vector<IndexPair> matches;
    for (auto const& row : rows_) {
        for (auto it = graph_.adjacenciesFromVertexBegin(row);
             it != graph_.adjacenciesFromVertexEnd(row); ++it)
            if (mask_[it->edge()] == STARRED)
                matches.emplace_back(row, it->vertex());
    }
    return matches;
}

/// subtract smallest value of each row/column.
///
template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::step_one()
{
    value_type minval;

    if (rows_.size() >= cols_.size()) {

        // subtract minval from each column.
        for (auto const& col : cols_) {

            minval = std::numeric_limits<value_type>::max();

            for (auto it = graph_.adjacenciesToVertexBegin(col);
                 it != graph_.adjacenciesToVertexEnd(col); ++it) {
                auto const& val = costs_[it->edge()];
                if (val < minval)
                    minval = val;
            }

            // subtract.
            for (auto it = graph_.adjacenciesToVertexBegin(col);
                 it != graph_.adjacenciesToVertexEnd(col); ++it) {
                costs_[it->edge()] -= minval;
            }
        }

    } else {

        // subtract minval from each row.
        for (auto const& row : rows_) {

            minval = std::numeric_limits<value_type>::max();

            for (auto it = graph_.adjacenciesFromVertexBegin(row);
                 it != graph_.adjacenciesFromVertexEnd(row); ++it) {
                auto const& val = costs_[it->edge()];
                if (val < minval)
                    minval = val;
            }

            for (auto it = graph_.adjacenciesFromVertexBegin(row);
                 it != graph_.adjacenciesFromVertexEnd(row); ++it) {
                costs_[it->edge()] -= minval;
            }
        }
    }
    step_ = Step::TWO;
}

/// Find zeros and star them.
///
template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::step_two()
{
    for (auto const& row : rows_) {
        if (row_cover[row])
            continue;

        for (auto it = graph_.adjacenciesFromVertexBegin(row);
             it != graph_.adjacenciesFromVertexEnd(row); ++it) {

            // star zero entries.
            if (!col_cover[col_of_idx(it->vertex())] && !row_cover[row] &&
                costs_[it->edge()] == 0) {
                col_cover[col_of_idx(it->vertex())] = true;
                row_cover[row] = true;
                mask_[it->edge()] = STARRED;
            }
        }
    }

    // reset covers.
    clear_covers();

    step_ = Step::THREE;
}

/// Count covered columns and stop if k=min(n_rows, n_cols)
/// columns are covered.
///
template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::step_three()
{
    for (auto const& row : rows_)
        for (auto it = graph_.adjacenciesFromVertexBegin(row);
             it != graph_.adjacenciesFromVertexEnd(row); ++it)
            if (mask_[it->edge()] == STARRED)
                col_cover[col_of_idx(it->vertex())] = true;

    const size_t col_count =
        std::count(col_cover.cbegin(), col_cover.cend(), true);

    // if the matching is already perfect, then we are done.
    if (col_count >= max_matches_)
        step_ = Step::SEVEN;
    else
        step_ = Step::FOUR;
}

/// Find non-covered zeros and prime them.
///
template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::step_four()
{
    IndexPair idx;
    Index start_row{ 0 };
    while (true) {

        idx = find_zero(start_row);

        if (idx.row == IDXMAX || idx.col == IDXMAX) {
            // No zero found.
            step_ = Step::SIX;
            return;
        }

        // it is more likely to find another zero
        // _after_ the last location, so we pass it
        // as a hint to find_zero.
        start_row = idx.row + 1;

        // prime at given index.
        mask_[graph_.findEdge(idx.row, idx.col).second] = PRIMED;

        if (has_star_in_row(idx.row)) {
            const auto col = find_star_in_row(idx.row);
            row_cover[idx.row] = true;
            col_cover[col_of_idx(col)] = false;
        } else {
            // starting point for augmenting path.
            path_.clear();
            path_.emplace_back(idx.row, idx.col);
            step_ = Step::FIVE;
            return;
        }
    }
}

/// construct augmenting path of starred and primed zeros.
///
template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::step_five()
{
    Index row;
    while (true) {
        row = find_star_in_col(path_.back().col);
        if (row == IDXMAX) {
            break;
        }

        path_.emplace_back(row, path_.back().col);

        auto const col = find_prime_in_row(path_.back().row);
        path_.emplace_back(path_.back().row, col);
    }

    augment();
    clear_covers();
    erase_primes();
    step_ = Step::THREE;
}

/// update cost based on non-satisfied constraints.
///
template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::step_six()
{
    auto const& minval = find_smallest();
    for (auto const& row : rows_) {
        for (auto it = graph_.adjacenciesFromVertexBegin(row);
             it != graph_.adjacenciesFromVertexEnd(row); ++it) {
            if (row_cover[row])
                costs_[it->edge()] += minval;
            if (!col_cover[col_of_idx(it->vertex())])
                costs_[it->edge()] -= minval;
        }
    }
    step_ = Step::FOUR;
}

template <class GRAPH, class COST, class MASK>
typename Matching<GRAPH, COST, MASK>::value_type
Matching<GRAPH, COST, MASK>::find_smallest() const
{
    auto minval = std::numeric_limits<value_type>::max();
    for (auto const& row : rows_) {
        // consider only non-covered rows.
        if (row_cover[row])
            continue;
        for (auto it = graph_.adjacenciesFromVertexBegin(row);
             it != graph_.adjacenciesFromVertexEnd(row); ++it)
            // consider only non-covered columns.
            if (!col_cover[col_of_idx(it->vertex())] &&
                costs_[it->edge()] < minval)
                minval = costs_[it->edge()];
    }
    return minval;
}

template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::augment()
{
    for (auto const& idx : path_) {
        auto const& edge = graph_.findEdge(idx.row, idx.col).second;
        if (mask_[edge] == STARRED)
            mask_[edge] = 0;
        else
            mask_[edge] = STARRED;
    }
}

template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::clear_covers()
{
    for (Index row = 0; row < row_cover.size(); ++row) {
        row_cover[row] = false;
    }
    for (Index col = 0; col < col_cover.size(); ++col) {
        col_cover[col] = false;
    }
}

template <class GRAPH, class COST, class MASK>
void
Matching<GRAPH, COST, MASK>::erase_primes()
{
    for (auto const& row : rows_)
        for (auto it = graph_.adjacenciesFromVertexBegin(row);
             it != graph_.adjacenciesFromVertexEnd(row); ++it)
            if (mask_[it->edge()] == PRIMED)
                mask_[it->edge()] = 0;
}

template <class GRAPH, class COST, class MASK>
typename Matching<GRAPH, COST, MASK>::IndexPair
Matching<GRAPH, COST, MASK>::find_zero(Index start_row) const
{
    auto const& n_rows = rows_.size();
    for (Index idx = 0; idx < n_rows; ++idx) {

        // start from start_row, and wrap around at n_rows.
        auto row = start_row + idx;
        if (row >= n_rows) {
            start_row = -row;
            row = 0;
        }

        if (row_cover[row])
            continue;

        for (auto it = graph_.adjacenciesFromVertexBegin(row);
             it != graph_.adjacenciesFromVertexEnd(row); ++it) {
            if (!col_cover[col_of_idx(it->vertex())] &&
                std::abs(costs_[it->edge()]) <
                    std::numeric_limits<value_type>::epsilon())
                return { row, it->vertex() };
        }
    }

    return { IDXMAX, IDXMAX };
}

template <class GRAPH, class COST, class MASK>
bool
Matching<GRAPH, COST, MASK>::has_star_in_row(Index const row) const
{
    for (auto it = graph_.adjacenciesFromVertexBegin(row);
         it != graph_.adjacenciesFromVertexEnd(row); ++it)
        if (mask_[it->edge()] == STARRED)
            return true;
    return false;
}

template <class GRAPH, class COST, class MASK>
typename Matching<GRAPH, COST, MASK>::Index
Matching<GRAPH, COST, MASK>::find_star_in_row(Index const row) const
{
    for (auto it = graph_.adjacenciesFromVertexBegin(row);
         it != graph_.adjacenciesFromVertexEnd(row); ++it)
        if (mask_[it->edge()] == STARRED)
            return it->vertex();

    return IDXMAX;
}

template <class GRAPH, class COST, class MASK>
typename Matching<GRAPH, COST, MASK>::Index
Matching<GRAPH, COST, MASK>::find_star_in_col(Index const col) const
{
    for (auto it = graph_.adjacenciesToVertexBegin(col);
         it != graph_.adjacenciesToVertexEnd(col); ++it)
        if (mask_[it->edge()] == STARRED)
            return it->vertex();

    return IDXMAX;
}

template <class GRAPH, class COST, class MASK>
typename Matching<GRAPH, COST, MASK>::Index
Matching<GRAPH, COST, MASK>::find_prime_in_row(Index const row) const
{

    for (auto it = graph_.adjacenciesFromVertexBegin(row);
         it != graph_.adjacenciesFromVertexEnd(row); ++it)
        if (mask_[it->edge()] == PRIMED)
            return it->vertex();

    throw std::runtime_error("Could not find prime in given row!");
}

} // end namespace matching
} // end namespace markurem
