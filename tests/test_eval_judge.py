"""Tests for the eval judge: nugget assignment and groundedness (fake LLM)."""

from langchain_core.runnables import RunnableLambda

from eval.judge import Groundedness, NuggetLabels, judge_item

CONTEXT = "Dosisgrænser for erhvervsmæssig bestråling fremgår af bilag 2."
ANSWER = "Dosisgrænserne står i bilag 2 og er 20 mSv pr. år."


class FakeJudgeLLM:
    """Stands in for a chat model: returns scripted structured outputs, records prompts."""

    def __init__(self, nugget_replies=(), groundedness_replies=()):
        self.replies = {
            NuggetLabels: list(nugget_replies),
            Groundedness: list(groundedness_replies),
        }
        self.prompts = {NuggetLabels: [], Groundedness: []}

    def with_structured_output(self, schema):
        def respond(prompt_value):
            self.prompts[schema].append(prompt_value.to_string())
            reply = self.replies[schema].pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        return RunnableLambda(respond)


def _item(n_vital=2, behaviour="answer"):
    nuggets = [
        {"text": f"fact {i}", "importance": "vital", "evidence": [f"quote {i}"]}
        for i in range(n_vital)
    ]
    return {
        "id": "q1",
        "question": "Hvor findes dosisgrænserne?",
        "expected_behavior": behaviour,
        "nuggets": nuggets if behaviour == "answer" else [],
    }


GROUNDED = Groundedness(unsupported_claims=[], refused=False)


def test_each_nugget_gets_a_label_in_order():
    llm = FakeJudgeLLM(
        [NuggetLabels(labels=["support", "partial_support"])], [GROUNDED]
    )

    verdict = judge_item(_item(), ANSWER, CONTEXT, llm)

    assert verdict == {
        "nuggets": ["support", "partial_support"],
        "unsupported_claims": [],
        "refused": False,
    }


def test_many_nuggets_are_judged_in_batches_of_ten_keeping_their_order():
    labels = ["support"] * 10, ["not_support"] * 10, ["partial_support"] * 3
    llm = FakeJudgeLLM([NuggetLabels(labels=batch) for batch in labels], [GROUNDED])

    verdict = judge_item(_item(n_vital=23), ANSWER, CONTEXT, llm)

    assert len(llm.prompts[NuggetLabels]) == 3
    assert verdict["nuggets"] == [*labels[0], *labels[1], *labels[2]]


def test_the_nugget_call_does_not_pay_for_the_retrieved_context():
    llm = FakeJudgeLLM([NuggetLabels(labels=["support", "support"])], [GROUNDED])

    judge_item(_item(), ANSWER, CONTEXT, llm)

    [nugget_prompt] = llm.prompts[NuggetLabels]
    [groundedness_prompt] = llm.prompts[Groundedness]
    assert CONTEXT not in nugget_prompt
    assert CONTEXT in groundedness_prompt
    assert ANSWER in nugget_prompt and ANSWER in groundedness_prompt


def test_unsupported_claims_and_refusals_come_from_the_groundedness_call():
    claims = ["Grænsen er 50 mSv"]
    llm = FakeJudgeLLM(
        [NuggetLabels(labels=["support", "not_support"])],
        [Groundedness(unsupported_claims=claims, refused=False)],
    )

    verdict = judge_item(_item(), ANSWER, CONTEXT, llm)

    assert verdict["unsupported_claims"] == claims


def test_a_refusal_question_skips_the_nugget_call():
    llm = FakeJudgeLLM([], [Groundedness(unsupported_claims=[], refused=True)])

    verdict = judge_item(_item(behaviour="refuse"), "Det fremgår ikke.", CONTEXT, llm)

    assert llm.prompts[NuggetLabels] == []
    assert verdict == {"nuggets": [], "unsupported_claims": [], "refused": True}


def test_a_wrong_number_of_labels_is_retried_once():
    llm = FakeJudgeLLM(
        [
            NuggetLabels(labels=["support"]),
            NuggetLabels(labels=["support", "not_support"]),
        ],
        [GROUNDED],
    )

    verdict = judge_item(_item(), ANSWER, CONTEXT, llm)

    assert verdict["nuggets"] == ["support", "not_support"]
    assert len(llm.prompts[NuggetLabels]) == 2


def test_a_judge_that_keeps_failing_leaves_the_question_unjudged():
    llm = FakeJudgeLLM(
        [ValueError("invalid JSON"), ValueError("invalid JSON")], [GROUNDED]
    )

    assert judge_item(_item(), ANSWER, CONTEXT, llm) is None


def test_an_empty_answer_is_judged_without_calling_the_model():
    llm = FakeJudgeLLM()

    verdict = judge_item(_item(), "   ", CONTEXT, llm)

    assert verdict == {
        "nuggets": ["not_support", "not_support"],
        "unsupported_claims": [],
        "refused": False,
    }
    assert llm.prompts == {NuggetLabels: [], Groundedness: []}
