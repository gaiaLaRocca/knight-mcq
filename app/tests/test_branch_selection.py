import unittest
from unittest.mock import patch

from app.core.agents.gpt import branch_selection
from app.core.agents.gpt.branch_selection import select_branches

NODE = "marin county"
SEED = frozenset({"golden gate bridge", "marin county", "san francisco"})
# test_8: Marin County's description, and some of the sub-triplets mined from it.
DESCRIPTION = ("Marin County is a county situated in the northwestern portion of California's San "
               "Francisco Bay Area. Its largest city is San Rafael, and its Civic Center was designed "
               "by Frank Lloyd Wright. The county is known for its natural beauty.")
RELEVANCE = {"san rafael": 0.265, "frank lloyd wright": 0.077, "natural beauty": 0.008,
             "262321 residents": 0.034, "scenic coastal views": 0.145}


def _t(head, relation, tail):
    return {"head": head, "relation": relation, "tail": tail}


# The named entities spaCy would find in DESCRIPTION; `natural beauty` is in the text, not one.
ENTITIES = {"san rafael", "frank lloyd wright"}


def _in_description(term, text):
    return term in ENTITIES and term in text.lower()


class _Injected(unittest.TestCase):
    """Run each test with the thesis hooks replaced by deterministic fakes."""
    def setUp(self):
        patches = [patch.object(branch_selection, "relevance_scorer", RELEVANCE.get),
                   patch.object(branch_selection, "entity_checker", _in_description)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class TestWithoutInjection(unittest.TestCase):
    def test_the_first_branches_are_kept_as_in_knight(self):
        triplets = [_t(NODE, "has", "natural beauty"), _t(NODE, "has_seat", "san rafael"),
                    _t("deck", "carries", "traffic")]
        kept, discarded, _ = select_branches(NODE, DESCRIPTION, triplets, SEED)
        self.assertEqual(kept, triplets[:2])
        self.assertEqual(discarded, [(triplets[2], "over the cap")])


class TestExclusions(_Injected):
    def test_each_exclusion_is_named(self):
        other_head = _t("civic center", "designed_by", "frank lloyd wright")
        to_itself = _t(NODE, "includes", NODE)
        to_seed = _t(NODE, "linked_to", "san francisco")
        _, discarded, _ = select_branches(NODE, DESCRIPTION, [other_head, to_itself, to_seed], SEED)
        self.assertEqual(discarded, [(other_head, "head is not the node"),
                                     (to_itself, "tail is the node"),
                                     (to_seed, "tail is a seed node")])

    def test_nothing_eligible_leaves_the_node_without_branches(self):
        kept, _, _ = select_branches(NODE, DESCRIPTION, [_t(NODE, "linked_to", "san francisco")], SEED)
        self.assertEqual(kept, [])


class TestRanking(_Injected):
    def test_entities_before_relevance(self):
        # `scenic coastal views` is more relevant than Frank Lloyd Wright, but not an entity.
        triplets = [_t(NODE, "offers", "scenic coastal views"), _t(NODE, "has_seat", "san rafael"),
                    _t(NODE, "has_building_by", "frank lloyd wright")]
        kept, discarded, _ = select_branches(NODE, DESCRIPTION, triplets, SEED)
        self.assertEqual([t["tail"] for t in kept], ["san rafael", "frank lloyd wright"])
        self.assertEqual(discarded, [(triplets[0], "not an entity")])

    def test_a_literal_counts_as_an_entity(self):
        triplets = [_t(NODE, "offers", "scenic coastal views"), _t(NODE, "has_population", "262321 residents")]
        with patch.object(branch_selection, "is_literal", lambda tail: tail == "262321 residents"):
            kept, _, _ = select_branches(NODE, DESCRIPTION, triplets, SEED)
        self.assertEqual(kept[0]["tail"], "262321 residents")

    def test_a_non_entity_never_takes_a_place(self):
        # test_9: the strait and Marin filled their free places with pages such as Marin Headlands.
        triplets = [_t(NODE, "has", "natural beauty"), _t(NODE, "offers", "scenic coastal views"),
                    _t(NODE, "has_seat", "san rafael")]
        kept, discarded, ranked = select_branches(NODE, DESCRIPTION, triplets, SEED)
        self.assertEqual([t["tail"] for t in kept], ["san rafael"])
        self.assertEqual({reason for _, reason in discarded}, {"not an entity"})
        self.assertEqual(len(ranked), 3)  # still listed, for the log

    def test_without_an_entity_checker_free_places_are_filled_by_relevance(self):
        triplets = [_t(NODE, "has", "natural beauty"), _t(NODE, "offers", "scenic coastal views"),
                    _t(NODE, "has_seat", "san rafael")]
        with patch.object(branch_selection, "entity_checker", None):
            kept, _, _ = select_branches(NODE, DESCRIPTION, triplets, SEED)
        self.assertEqual([t["tail"] for t in kept], ["san rafael", "scenic coastal views"])

    def test_one_branch_per_tail(self):
        triplets = [_t(NODE, "has_seat", "san rafael"), _t(NODE, "largest_city", "san rafael"),
                    _t(NODE, "has_building_by", "frank lloyd wright")]
        kept, discarded, _ = select_branches(NODE, DESCRIPTION, triplets, SEED)
        self.assertEqual([t["tail"] for t in kept], ["san rafael", "frank lloyd wright"])
        self.assertEqual(discarded, [(triplets[1], "tail already kept")])

    def test_a_failing_checker_ranks_the_tail_as_no_entity(self):
        def broken(term, text):
            raise RuntimeError("spaCy unavailable")
        triplets = [_t(NODE, "has_seat", "san rafael"), _t(NODE, "offers", "scenic coastal views")]
        with patch.object(branch_selection, "entity_checker", broken):
            _, _, ranked = select_branches(NODE, DESCRIPTION, triplets, SEED)
        self.assertEqual([entity for _, entity, _ in ranked], [False, False])


if __name__ == "__main__":
    unittest.main()
