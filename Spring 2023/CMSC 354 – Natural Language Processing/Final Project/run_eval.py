import re
import json
import time
from utils import ChatScenario, ChatSession


class Evaluation:
    class Situation:
        def __init__(self, optionA: str, optionB: str):
            self.options = [optionA, optionB]

        def get_situation(self, reversed: bool = False):
            options = self.options[::-1] if reversed else self.options

            return f"A: {options[0]}\nB: {options[1]}"

    def __init__(self, situations: list[Situation], repeat: int = 1):
        self.situations = situations
        self.repeat = repeat
        self.results = [dict({"A": 0, "B": 0, "X": 0}) for _ in self.situations]

    def split_list(self, target, size):
        return [target[i : i + size] for i in range(0, len(target), size)]

    def get_prompts(self, reversed: bool = False, pagination: int = 8):
        # List of prompts that will be added to ChatScenario
        prompts = []

        # Limit the number of situations to pagination limit (defaults to 8)
        for situations in self.split_list(self.situations, pagination):

            """
            Situations will be formatted in the following format:

            "A: 20% chance to win 4000;
            B: 25% chance to win 3000."
            ... (n = pagination)
            "A: 5% chance to win a three-week tour of England, France, and Italy;
            B: 10% chance to win a one-week tour of England."
            """
            situations_string = "\n".join(
                [f'"{s.get_situation(reversed)}"' for s in situations]
            )

            prompts.append(
                f"""Situations:
{situations_string}

You must choose between A and B.

Think step by step, and answer with the following format for each situations:
Reasoning: [your_reasoning_less_than_3_sentences]
Answer: [PREFER_A|PREFER_B]"""
            )

        return prompts

    def evaluate_session(self, session: ChatSession, reversed: bool = False):
        answers = []
        answer_literals = (
            ["PREFER_B", "PREFER_A"] if reversed else ["PREFER_A", "PREFER_B"]
        )

        answer_responses = []
        for response in session.messages:
            if response["role"] != "assistant":
                continue

            answer_responses.extend(
                re.findall("Answer:[ |\n]*([^\.\n]+)\.?", response["content"])
            )

        for ans in answer_responses:
            if ans not in answer_literals:
                answers.append("X")  # indecisive or error

            elif ans == answer_literals[0]:
                answers.append("A")

            elif ans == answer_literals[1]:
                answers.append("B")

        if len(answers) != len(self.situations):
            print(
                "WARNING: Situation - Answer mismatch, check individual responses below"
            )
            print(answer_responses)

        return answers

    def start(self):
        for reversed in [False, True]:  # test for both type A and B
            prompts = self.get_prompts(reversed)
            sessions = []

            # Repeat scenario
            for i in range(self.repeat):
                scenario = ChatScenario(f"response-{i+1}", prompts, reversed)
                session = scenario.start()  # start API call
                sessions.append(session)

            for session in sessions:
                answers = self.evaluate_session(session, reversed)
                for i, a in enumerate(answers):
                    self.results[i][a] += 1

        # Export results
        self.export_result()

        return self.results

    def import_sessions(self, scenario_ids):
        # TODO: scan directory and import sessions
        pass

    def export_result(self):
        current_time = time.strftime(
            "%Y%m%d-%H%M%S"
        )  # timestamp in YYYYMMDD-HHMMSS format

        # Export results to file
        with open(f"{current_time}-results.json", "w") as file:
            json.dump(self.results, file)


# test = ChatScenario(
#     "test",
#     [
#         'respond only with "hello"',
#         'respond only with "hola"',
#     ]
# )

# test.start()


eval = Evaluation(
    [
        Evaluation.Situation(
            "50% chance to win 1000, 50% chance to win nothing",
            "100% chance to win 450",
        ),
        Evaluation.Situation(
            "50% chance to lose 1000, 50% chance to lose nothing",
            "100% chance to lose 450",
        ),
    ],
    2,
)

eval.start()
