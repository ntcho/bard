import re
import json
import os
import time
from utils import ChatScenario, ChatSession


class Evaluation:
    """
    A class for evaluating decision-making scenarios.

    Attributes
    ----------
    situations : list[Situation]
        A list of Situation objects representing the decision-making scenarios.
    repeat : int
        The number of times to repeat the scenarios for each types. In total,
        `repeat * 2` sessions will be created to test each non-reversed and
        reversed types.
    results : list[dict]
        A list of dictionaries representing the results of the evaluation.

    Methods
    -------
    split_list(target: list, size: int)
        Splits a list into smaller lists of a given size.
    get_prompts(reversed: bool = False, pagination: int = 8) -> list[str]
        Generates a sequence of prompts for the chat scenario.
    evaluate_sessions(sessions: tuple[ChatSession, bool], export: bool = True) -> list[dict]
        Evaluates a list of ChatSession objects and optionally exports the results.
    evaluate_session(session: ChatSession, reversed: bool = False) -> list[str]
        Evaluates a single ChatSession object.
    start() -> list[dict]
        Starts the evaluation process and returns the results.
    import_sessions(scenario_ids: list[str]) -> list[dict]
        Imports ChatSession objects from files and returns the results.
    export_result(results: list[tuple])
        Exports the evaluation results to a JSON file.
    """

    class Situation:
        """
        A class representing a single decision-making scenario.

        Example
        -------
        ```text
        A: 50% chance to win 1000, 50% chance to win nothing;
        B: 100% chance to win 450.
        ```
        When `reversed` is True, the order of option A and B will be reversed.

        Attributes
        ----------
        optionA : str
            The first option in the scenario.
        optionB : str
            The second option in the scenario.

        Methods
        -------
        get_situation(reversed: bool = False) -> str
            Returns a formatted string representing the scenario,
            with the options in the correct order.

        """

        def __init__(self, optionA: str, optionB: str):
            """
            Initializes a new Situation instance.

            Parameters
            ----------
            optionA : str
                The first option in the scenario.
            optionB : str
                The second option in the scenario.
            """
            self.options = [optionA, optionB]

        def get_situation(self, reversed: bool = False):
            """
            Returns a formatted string representing the scenario, with the
            options in the correct order.

            Parameters
            ----------
            reversed : bool, optional
                If True, the options will be reversed, by default False

            Returns
            -------
            str
                A formatted string representing the scenario, with the options in the correct order.
            """
            options = self.options[::-1] if reversed else self.options

            return f"A: {options[0]}\nB: {options[1]}"

    def __init__(self, situations: list[Situation], repeat: int = 1):
        """
        Initializes a new Evaluation instance.

        Parameters
        ----------
        situations : list[Situation]
            A list of Situation objects representing the decision-making scenarios.
        repeat : int
            The number of times to repeat the scenarios, by default 1.
        """
        self.situations = situations
        self.repeat = repeat
        self.results = [dict({"A": 0, "B": 0, "X": 0}) for _ in self.situations]
        self.answer_literals = [dict() for _ in self.situations]

    def split_list(self, target: list, size: int):
        """
        Splits a list into smaller lists of a given size.

        Parameters
        ----------
        target : list
            The list to be split.
        size : int
            The size of each smaller list.

        Returns
        -------
        list[list]
            A list of smaller lists.
        """
        return [target[i : i + size] for i in range(0, len(target), size)]

    def get_prompts(self, reversed: bool = False, pagination: int = 8):
        """
        Generates a sequence of prompts for the chat scenario.

        Example
        -------
        Situations will be formatted into prompts in the following format:

        ```text
        "A: 20% chance to win 4000;
        B: 25% chance to win 3000."
        ... (len=pagination)
        "A: 5% chance to win a three-week tour of England, France, and Italy;
        B: 10% chance to win a one-week tour of England."

        You must choose between A and B.

        Think step by step, and answer with the following format for each situations:
        Reasoning: [your_reasoning_less_than_3_sentences]
        Answer: [PREFER_A|PREFER_B]
        ```

        Parameters
        ----------
        reversed : bool, optional, default: False
            Whether to reverse the order of answer literals.
        pagination : int, optional, default: 8
            The number of situations to display per prompt.

        Returns
        -------
        list[str]
            A list of prompt strings.
        """
        # List of prompts that will be added to ChatScenario
        prompts = []

        # Limit the number of situations to pagination limit (defaults to 8)
        # Added pagination to avoid max response length cutoff
        for situations in self.split_list(self.situations, pagination):

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

    def evaluate_sessions(
        self, sessions: tuple[ChatSession, bool], export: bool = True
    ):
        """
        Evaluates a list of ChatSession objects and optionally exports the results.

        Parameters
        ----------
        sessions : tuple[ChatSession, bool]
            A tuple containing a ChatSession object and a boolean indicating
            whether the order of answer literals is reversed.
        export : bool, optional, default: True
            Whether to export the results to a JSON file.

        Returns
        -------
        results : list[dict]
            A list of dictionaries representing the results of the evaluation.
        answer_literals : list[dict]
            A list of dictionaries containing the string literals matched from response.
            Used for inspection purposes.
        """
        results = [dict({"A": 0, "B": 0, "X": 0}) for _ in self.situations]
        answer_literals = [dict() for _ in self.situations]

        for session, reversed in sessions:
            answers, literals = self.evaluate_session(session, reversed)
            for i, a in enumerate(answers):
                results[i][a] += 1
                answer_literals[i][session.scenario_id] = {
                    "literal": literals[i],
                    "answer": answers[i],
                }

        self.export_result(results, answer_literals)

        return results

    def evaluate_session(self, session: ChatSession, reversed: bool = False):
        """
        Evaluates a single ChatSession object.

        Parameters
        ----------
        session : ChatSession
            The ChatSession object to be evaluated.
        reversed : bool, optional, default: False
            Whether the order of answer literals is reversed.

        Returns
        -------
        answers : list[str]
            A list of strings representing the answers.
        answer_literals : list[str]
            A list of string literals matched from response.
            Used for inspection purposes.
        """
        answers = []
        option_literals = (
            ["PREFER_B", "PREFER_A"] if reversed else ["PREFER_A", "PREFER_B"]
        )

        answer_literals = []
        for response in session.messages:
            if response["role"] != "assistant":
                continue

            answer_literals.extend(
                re.findall("Answer:[ |\n]*([^\.\n]+)\.?", response["content"])
            )

        for ans in answer_literals:
            if len(ans) <= 10:  # give room for period or space added after the answer
                if option_literals[0] in ans:
                    answers.append("A")
                elif option_literals[1] in ans:
                    answers.append("B")
                else:
                    answers.append("X")  # indecisive or error
            else:
                answers.append("X")  # indecisive or error

        if len(answers) != len(self.situations):
            print(f"WARNING: Situation - Answer mismatch at {session.scenario_id}")
            print(f"    expected {len(self.situations)}, but got {len(answers)}")
            print(f"    answers=\n{answer_literals}")

        return answers, answer_literals

    def start(self, start_id: int = 1):
        """
        Starts the evaluation process and returns the results.

        Parameters
        ----------
        start_id : int, optional
            The scenario id to start from, by default 1.

        Returns
        -------
        list[dict]
            A list of dictionaries representing the results of the evaluation.
        """
        sessions = []  # will contain tuple of tuple[ChatSession, bool]
        
        prompts = {
            False: self.get_prompts(False),
            True: self.get_prompts(True)
        }

        # Repeat scenario
        for i in range(start_id - 1, (start_id - 1) + self.repeat):
            for reversed in [False, True]:  # test for both non-reversed and reversed types
                scenario = ChatScenario(
                    f"response-{i+1}{'-reversed' if reversed else ''}",
                    prompts[reversed],
                    reversed,
                )
                session = scenario.start()  # start API call
                sessions.append(tuple([session, reversed]))

        # Evaluate responses and export results
        self.results = self.evaluate_sessions(sessions)

        return self.results

    def import_sessions(self, scenario_ids: list[str]):
        """
        Imports ChatSession objects from files and evaluate the results.

        Parameters
        ----------
        scenario_ids : list[str]
            A list of scenario IDs.
            This is used to find the JSON files in the current directory.

        Returns
        -------
        list[dict]
            A list of dictionaries representing the results of the evaluation.
        """
        message_filenames = {}  # will be {"response-1": "{filename}", ...}
        response_filenames = {}

        # Add ids for reversed scenarios
        scenario_ids.extend([s + "-reversed" for s in scenario_ids])

        for file in os.listdir("responses"):
            for id in scenario_ids:
                if file.startswith(
                    id + "-20"  # only match unique file names
                ) and file.endswith(".json"):
                    # matches {id}-YYYYMMDD-HHMMSS-[messages|responses].json files
                    if "messages.json" in file:
                        message_filenames[id] = file
                    elif "responses.json" in file:
                        response_filenames[id] = file

        if len(response_filenames) != len(message_filenames):
            raise FileNotFoundError("Response - Message pairs do not match")

        sessions = []  # will contain tuple of tuple[ChatSession, bool]

        for id in response_filenames:
            reversed = "reversed" in response_filenames[id]
            session = ChatSession(id)
            session.import_session(message_filenames[id], response_filenames[id])
            sessions.append(tuple([session, reversed]))

        # Evaluate responses and export results
        self.results = self.evaluate_sessions(sessions)

        return self.results

    def export_result(self, results: list[tuple], answer_literals: list[dict]):
        """
        Exports the evaluation results to a JSON file.

        Parameters
        ----------
        results : list[tuple]
            A list of tuples representing the results of the evaluation.
        answer_literals : list[dict]
            A list of dictionaries containing the string literals matched from response.
        """
        current_time = time.strftime(
            "%Y%m%d-%H%M%S"
        )  # timestamp in YYYYMMDD-HHMMSS format

        # Export results to file
        with open(f"responses/{current_time}-results.json", "w") as file:
            json.dump(results, file)

        # Export answer literals to file
        with open(f"responses/{current_time}-answers.json", "w") as file:
            json.dump(answer_literals, file)


# Setup evaluation details
zero_shot_eval = Evaluation(
    # situations to question the AI
    situations=[
        # simple_gain
        Evaluation.Situation(
            "50% chance to win 1000, 50% chance to win nothing",
            "100% chance to win 450",
        ),
        # simple_loss
        Evaluation.Situation(
            "50% chance to lose 1000, 50% chance to lose nothing",
            "100% chance to lose 450",
        ),
        # equal_gain
        Evaluation.Situation(
            "50% chance to win 1000, 50% chance to win nothing",
            "100% chance to win 500",
        ),
        # equal_loss
        Evaluation.Situation(
            "50% chance to lose 1000, 50% chance to lose nothing",
            "100% chance to lose 500",
        ),
        # variation1_1
        Evaluation.Situation(
            "33% chance to win 2500, 66% chance to win 2400, 1% chance to win no",
            "100% chance to win 2400",
        ),
        # variation1_2
        Evaluation.Situation(
            "33% chance to win 2500, 67% chance to win nothing",
            "34% chance to win 2400, 66% chance to win nothing",
        ),
        # variation2_1_gain
        Evaluation.Situation("80% chance to win 4000", "100% chance to win 3000"),
        # variation2_1_loss
        Evaluation.Situation("80% chance to lose 4000", "100% chance to lose 3000"),
        # variation2_2_gain
        Evaluation.Situation("20% chance to win 4000", "25% chance to win 3000"),
        # variation2_2_loss
        Evaluation.Situation("20% chance to lose 4000", "25% chance to lose 3000"),
        # overweighting_gain
        Evaluation.Situation("0.1% chance to win 5000", "100% chance to win 5"),
        # overweighting_loss
        Evaluation.Situation("0.1% chance to lose 5000", "100% chance to lose 5"),
        # risk_context_gain
        Evaluation.Situation(
            "25% chance to win 6000", "25% chance to win 4000, 25% chance to win 2000"
        ),
        # risk_context_loss
        Evaluation.Situation(
            "25% chance to lose 6000",
            "25% chance to lose 4000, 25% chance to lose 2000",
        ),
        # non_monetary_1
        Evaluation.Situation(
            "50% chance to win a three-week tour of England, France, and Italy",
            "100% chance to win a one-week tour of England",
        ),
        # non_monetary_2
        Evaluation.Situation(
            "5% chance to win a three-week tour of England, France, and Italy",
            "10% chance to win a one-week tour of England",
        ),
    ],
    # number of unique AI instances to evaluate
    repeat=100,
)

### Evaluate using OpenAI API
# eval.start(1)

### Evaluate using existing API responses in JSON files
response_range = 100  # last scenario_id
zero_shot_eval.import_sessions([f"response-{i + 1}" for i in range(response_range)])
