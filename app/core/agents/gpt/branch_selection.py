"""Which sub-triplets of an expanded node become branches of the graph.

An expanded node's description yields 9-21 sub-triplets, and only `MAX_BRANCHES` are
processed. KNIGHT keeps the first ones, i.e. those mined from the description's opening
sentences: on test_8 that kept 28 branches, of which 5 were entities not already named by
the seed (`262321 residents`, `826079`, `overseeing vessel machinery`...), and discarded
San Rafael, the Presidio, the Bay Bridge. See thesis-concept-robustness/docs/
phase1_kb_quality_plan.md, "Branches selected by entity and relevance".

The selection cannot read what decides a branch's fate, since the entity filter and the
relevance gate judge Wikipedia chunks after the build, and at selection time a tail has no
page yet. It anticipates both on the text it has: the entity test on the description the
triplet was mined from, the gate's cosine on the tail's name.

Both are injected by the thesis runner, like the page-selection scorers in
`wikipedia_lookup`; with no relevance scorer injected, the first `MAX_BRANCHES` are kept,
as in KNIGHT.
"""

import logging

from app.core.agents.gpt.literals import is_literal

logger = logging.getLogger("gpt_agent")

MAX_BRANCHES = 2

# (term, text) -> bool: is `term` a named entity in `text` (the description)?
entity_checker = None
# text -> float: relevance of a tail's name to the topic (the gate's metric).
relevance_scorer = None


def select_branches(node, description, sub_triplets, seed_nodes):
    """Return (kept, discarded, scored): the branches to process, the others with the
    reason they were left out, and each eligible candidate with its two ranking keys.

    Eligible: a triplet whose head is `node` (another head extends no path from the topic
    by a stated relation) and whose tail is neither `node` nor a seed node (an edge to a
    node the seed already reaches doubles paths or ends on a known answer: the strait's
    `connects_to` on test_8). Ranked by entity first, then relevance; one branch per tail.
    """
    if relevance_scorer is None:
        return sub_triplets[:MAX_BRANCHES], [(t, "over the cap") for t in sub_triplets[MAX_BRANCHES:]], []

    discarded, eligible = [], []
    for triplet in sub_triplets:
        reason = _exclusion(triplet, node, seed_nodes)
        if reason:
            discarded.append((triplet, reason))
        else:
            eligible.append(triplet)

    scored = [(t, _is_entity_like(t["tail"], description), _relevance(t["tail"])) for t in eligible]
    # Stable sort: equal keys keep the description's order.
    scored.sort(key=lambda entry: (entry[1], entry[2]), reverse=True)

    kept, tails_kept = [], set()
    for triplet, _, _ in scored:
        if triplet["tail"] in tails_kept:
            discarded.append((triplet, "tail already kept"))
        elif len(kept) < MAX_BRANCHES:
            kept.append(triplet)
            tails_kept.add(triplet["tail"])
        else:
            discarded.append((triplet, "over the cap"))
    return kept, discarded, scored


def _exclusion(triplet, node, seed_nodes):
    """The reason `triplet` cannot be a branch of `node`, or None."""
    if triplet["head"] != node:
        return "head is not the node"
    if triplet["tail"] == node:
        return "tail is the node"
    if triplet["tail"] in seed_nodes:
        return "tail is a seed node"
    return None


def _is_entity_like(tail, description):
    """A literal value, or a named entity in the description (when a checker is injected)."""
    if is_literal(tail):
        return True
    if entity_checker is None:
        return False
    try:
        return bool(entity_checker(tail, description))
    except Exception as e:
        _warn_hook_failure("entity checker", tail, e)
        return False


def _relevance(tail):
    try:
        return relevance_scorer(tail)
    except Exception as e:
        _warn_hook_failure("relevance scorer", tail, e)
        return float("-inf")


_hooks_warned = set()


def _warn_hook_failure(hook, tail, error):
    """Warn once per hook: a hook that fails on every tail would otherwise silently turn the
    selection back into a ranking without that key."""
    if hook not in _hooks_warned:
        _hooks_warned.add(hook)
        logger.warning(f"Branch selection: the {hook} failed on '{tail}' ({error}); "
                       f"such tails are ranked without it. Further failures are not logged.")
