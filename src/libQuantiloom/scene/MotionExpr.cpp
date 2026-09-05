/**
 * @file MotionExpr.cpp
 * @brief The only translation unit that knows what an expression parser is
 *
 * ExprTk is one 1.5 MB header that takes the better part of a minute to
 * compile and needs /bigobj to link on MSVC. Keeping it behind
 * SegmentExpressions means the rest of the scene layer compiles at its usual
 * speed and a future swap of the parser touches this file only.
 *
 * Two variables are bound: `t`, the global timeline second, and `s`, seconds
 * since the segment began. `s` exists because a piecewise path is almost
 * always written as an offset from where the piece starts, and forcing every
 * author to spell `(t - 4.0)` is how off-by-one segment boundaries happen.
 */

#include "scene/Motion.hpp"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4127 4189 4244 4245 4267 4310 4324 4702 4706)
#endif
#include <exprtk.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace quantiloom::scene {

struct SegmentExpressions::Impl {
    /// Written by Evaluate before each read of the compiled expressions. It is
    /// what makes evaluation non-re-entrant, which the header says out loud.
    f64 t = 0.0;
    f64 s = 0.0;
    exprtk::symbol_table<f64> symbols;
    Vector<exprtk::expression<f64>> expressions;
};

SegmentExpressions::SegmentExpressions() : m_impl(std::make_unique<Impl>()) {}
SegmentExpressions::~SegmentExpressions() = default;

Result<std::unique_ptr<SegmentExpressions>, String> SegmentExpressions::Compile(
    const Vector<String>& expressions) {
    auto out = std::unique_ptr<SegmentExpressions>(new SegmentExpressions());
    Impl& impl = *out->m_impl;

    impl.symbols.add_variable("t", impl.t);
    impl.symbols.add_variable("s", impl.s);
    impl.symbols.add_constants();  // pi, epsilon, inf

    exprtk::parser<f64> parser;
    impl.expressions.reserve(expressions.size());
    for (const String& text : expressions) {
        const String source = text.empty() ? String("0") : text;
        exprtk::expression<f64> expression;
        expression.register_symbol_table(impl.symbols);
        if (!parser.compile(source, expression)) {
            return Result<std::unique_ptr<SegmentExpressions>, String>::Err(
                String(parser.error()) + " in \"" + source + "\"");
        }
        impl.expressions.push_back(expression);
    }

    return Result<std::unique_ptr<SegmentExpressions>, String>(std::move(out));
}

void SegmentExpressions::Evaluate(f64 t_s, f64 s_s, f64* out) const {
    Impl& impl = *m_impl;
    impl.t = t_s;
    impl.s = s_s;
    for (usize i = 0; i < impl.expressions.size(); ++i) {
        out[i] = impl.expressions[i].value();
    }
}

usize SegmentExpressions::Count() const { return m_impl->expressions.size(); }

}  // namespace quantiloom::scene
