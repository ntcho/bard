import os
import openai
import json
import time

from dotenv import load_dotenv

load_dotenv()

# Reads OpenAI API key from .env
# .env contains "OPENAI_API_KEY=sk-XXXXXXXXXXX"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieve all models available in OpenAI (will fail without correct API key)
# print(openai.Model.list())


class ChatSession:
    """
    A class to manage chat sessions with the OpenAI API.

    Attributes
    ----------
    model : str
        The OpenAI model to use for the conversation, e.g. "gpt-3.5-turbo".
    messages : list
        A list of messages in the current chat session.
    responses : list
        A list of responses in the current chat session.
    usage : dict
        A dictionary containing token usage information.

    Methods
    -------
    add_message(prompt: str) -> list
        Adds a new message to the list of messages and returns the updated list.
    prompt(prompt: str) -> str
        Sends a prompt to the OpenAI API and returns the response content.
    export(filename: str = "")
        Exports messages and responses to JSON files with optional filename prefixes and a timestamp.
    """

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.messages = []
        self.responses = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def add_message(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        return self.messages

    def prompt(self, prompt: str):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.add_message(prompt),
        )
        response_content = response["choices"][0]["message"]["content"]

        # Add the assistant's response to the list of messages
        self.messages.append({"role": "assistant", "content": response_content})

        # Add the response to the list of responses
        self.responses.append(response)

        # Update token usage record
        self.usage = {
            key: self.usage[key] + response["usage"][key]
            for key in self.usage
            if key in response["usage"]
        }

        return response_content

    def export_session(self, filename: str = "session", reversed: bool = False):
        current_time = time.strftime(
            "%Y%m%d-%H%M%S"
        )  # timestamp in YYYYMMDD-HHMMSS format

        if reversed:
            filename += "-reversed"

        # Export all messages from current session to file
        with open(f"{filename}-{current_time}-messages.json", "w") as file:
            json.dump(self.messages, file)

        # Export all responses from current session to file
        with open(f"{filename}-{current_time}-responses.json", "w") as file:
            json.dump(self.responses, file)

    def import_session(self, messages_filename: str, responses_filename: str):
        # Import messages from file
        with open(messages_filename, "r") as file:
            self.messages = json.load(file)

        # Import responses from file
        with open(responses_filename, "r") as file:
            self.responses = json.load(file)


class ChatScenario:
    """
    A class to manage chat scenarios with a series of prompts for the OpenAI API.

    Attributes
    ----------
    prompts : list
        A list of prompts to be sent to the OpenAI API during the chat session.
    session : ChatSession
        An instance of the ChatSession class to interact with the OpenAI API.

    Methods
    -------
    start()
        Starts the chat scenario, sending each prompt to the OpenAI API, displaying responses, and exporting the session.
    """

    def __init__(
        self,
        scenario_id: str,
        prompts: list,
        reversed: bool = False,
        verbose: bool = True,
    ):
        self.scenario_id = scenario_id
        self.prompts = prompts
        self.session = ChatSession()
        self.reversed = reversed
        self.verbose = verbose

    def log(self, *message: object):
        if self.verbose:
            print(message)

    def import_session(self, filename: str):
        self.session.import_session(filename)

    def start(self):
        self.log("[ChatScenario] Session started")

        # Send each prompt to the OpenAI API
        for prompt in self.prompts:
            self.log("  Prompt:", prompt)
            completion = self.session.prompt(prompt)
            self.log("  Response:", completion)

        self.log("[ChatScenario] Completed session: usage=", self.session.usage)

        # Export session data
        self.session.export_session(self.scenario_id, self.reversed)
        self.log("[ChatScenario] Completed export")

        return self.session
