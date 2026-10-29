"""Plots the computation graph formed by `Value` instances."""

import graphviz

from engine import Value


def _escape_special_chars(s: str) -> str:
    for c in ["\\", "|", "{", "}", "<", ">"]:
        s = s.replace(c, f"\\{c}")
    return s


def draw(root: Value) -> graphviz.Digraph:
    """Draws the computation graph with the given root."""
    g = graphviz.Digraph(graph_attr={"rankdir": "LR"})
    visited: set[Value] = set()
    frontier: list[Value] = [root]
    while frontier:
        n = frontier.pop()
        if n in visited:
            continue
        n_id = str(id(n))
        label = f"data = {n.data:.4f}"
        if n.label:
            label = f"{_escape_special_chars(n.label)} | {label}"
        g.node(n_id, label=label, shape="record")

        op_id = ""
        if n.op:
            op_id = n_id + n.op
            g.node(op_id, label=n.op, shape="circle")
            g.edge(op_id, n_id)

        for p in n.parents:
            p_id = str(id(p))
            g.edge(p_id, op_id)
            if p not in visited:
                frontier.append(p)

        visited.add(n)

    return g


if __name__ == "__main__":
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(0.5)  # Unlabeled leaf.
    s = Value(1.0, label="\\ | { } < >")  # Every character that needs escaping.

    h = a + b  # Labeled intermediate value.
    h.label = "h"

    terms = [
        h,
        1 + a,  # Reflected add with a constant.
        a * a,  # Same value twice in one operation.
        3 * c,  # Reflected mul with a constant.
        b ** 2,
        a / b,
        1 / c,  # Reflected division.
        a - b,
        5 - a,  # Reflected subtraction.
        -s,
        abs(b),
        c.exp(),
        a.relu(),  # Positive input passes through.
        b.relu(),  # Negative input is clamped to zero.
        c.tanh(),  # Composite of other operations.
    ]
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    out.label = "out"
    draw(out).render("graph", format="svg", cleanup=True, view=True)
