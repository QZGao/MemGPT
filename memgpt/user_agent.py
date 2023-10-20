import glob
import os
import pickle
import sys

from rich.console import Console

import interface
from interface import print_messages
from memgpt import utils, system, agent
from memgpt.agent import AgentAsync
from memgpt.persistence_manager import InMemoryStateManager

console = Console()


def clear_line():
    if os.name == 'nt':  # for windows
        console.print("\033[A\033[K", end="")
    else:  # for linux
        sys.stdout.write("\033[2K\033[G")
        sys.stdout.flush()


class UserAgent:
    def __init__(self, memgpt_agent: AgentAsync):
        """"""
        self.memgpt_agent = memgpt_agent

    async def generate_reply(self):
        # Ask for user input
        user_input = console.input("[bold cyan]Enter your message:[/bold cyan] ")
        clear_line()

        if user_input.startswith('!'):
            print(f"Commands for CLI begin with '/' not '!'")
            return await self.generate_reply()

        if user_input == "":
            # no empty messages allowed
            print("Empty input received. Try again!")
            return await self.generate_reply()

        # Handle CLI commands
        # Commands to not get passed as input to MemGPT
        if user_input.startswith('/'):
            return await self._on_cli_commands(user_input)

        else:
            # If message did not begin with command prefix, pass inputs to MemGPT
            # Handle user message and append to messages
            user_message = system.package_user_message(user_input)

        return user_message

    async def _on_cli_commands(self, user_input: str):
        if user_input == "//":
            return await self._on_multiline()
        elif user_input.lower() == "/exit":
            return "TERMINATE"
        elif user_input.lower() == "/savechat":
            return await self._on_savechat()
        elif user_input.lower() == "/save":
            return await self._on_save()
        elif user_input.lower() == "/load" or user_input.lower().startswith("/load "):
            return await self._on_load(user_input)
        elif user_input.lower() == "/dump":
            return await self._on_dump()
        elif user_input.lower() == "/dumpraw":
            return await self._on_dumpraw()
        elif user_input.lower() == "/dump1":
            return await self._on_dump1()
        elif user_input.lower() == "/memory":
            return await self._on_memory()
        elif user_input.lower() == "/model":
            return await self._on_model()
        elif user_input.lower() == "/pop" or user_input.lower().startswith("/pop "):
            return await self._on_pop(user_input)
        # No skip options
        elif user_input.lower() == "/wipe":
            return self._on_wipe()
        elif user_input.lower() == "/heartbeat":
            return system.get_heartbeat()
        elif user_input.lower() == "/memorywarning":
            return system.get_token_limit_warning()
        else:
            print(f"Unrecognized command: {user_input}")
            return await self.generate_reply()

    async def _on_multiline(self):
        print("Entering multiline mode, type // when done")
        user_input_list = []
        while True:
            user_input = console.input("[bold cyan]>[/bold cyan] ")
            clear_line()
            if user_input == "//":
                break
            else:
                user_input_list.append(user_input)

        # pass multiline inputs to MemGPT
        return system.package_user_message("\n".join(user_input_list))

    async def _on_savechat(self):
        filename = utils.get_local_time().replace(' ', '_').replace(':', '_')
        filename = f"{filename}.pkl"
        try:
            if not os.path.exists("saved_chats"):
                os.makedirs("saved_chats")
            with open(os.path.join('saved_chats', filename), 'wb') as f:
                pickle.dump(self.memgpt_agent.messages, f)
                print(f"Saved messages to: {filename}")
        except Exception as e:
            print(f"Saving chat to {filename} failed with: {e}")
        return await self.generate_reply()

    async def _on_save(self):
        filename = utils.get_local_time().replace(' ', '_').replace(':', '_')
        filename = f"{filename}.json"
        filename = os.path.join('saved_state', filename)
        try:
            if not os.path.exists("saved_state"):
                os.makedirs("saved_state")
            self.memgpt_agent.save_to_json_file(filename)
            print(f"Saved checkpoint to: {filename}")
        except Exception as e:
            print(f"Saving state to {filename} failed with: {e}")

        # save the persistence manager too
        filename = filename.replace('.json', '.persistence.pickle')
        try:
            self.memgpt_agent.persistence_manager.save(filename)
            print(f"Saved persistence manager to: {filename}")
        except Exception as e:
            print(f"Saving persistence manager to {filename} failed with: {e}")

        return await self.generate_reply()

    async def _on_load(self, user_input: str):
        command = user_input.strip().split()
        filename = command[1] if len(command) > 1 else None
        if filename is not None:
            if filename[-5:] != '.json':
                filename += '.json'
            try:
                self.memgpt_agent.load_from_json_file_inplace(filename)
                print(f"Loaded checkpoint {filename}")
            except Exception as e:
                print(f"Loading {filename} failed with: {e}")
        else:
            # Load the latest file
            print(f"/load warning: no checkpoint specified, loading most recent checkpoint instead")
            json_files = glob.glob(
                "saved_state/*.json")  # This will list all .json files in the current directory.

            # Check if there are any json files.
            if not json_files:
                print(f"/load error: no .json checkpoint files found")
            else:
                # Sort files based on modified timestamp, with the latest file being the first.
                filename = max(json_files, key=os.path.getmtime)
                try:
                    self.memgpt_agent.load_from_json_file_inplace(filename)
                    print(f"Loaded checkpoint {filename}")
                except Exception as e:
                    print(f"Loading {filename} failed with: {e}")

        # need to load persistence manager too
        filename = filename.replace('.json', '.persistence.pickle')
        try:
            self.memgpt_agent.persistence_manager = InMemoryStateManager.load(
                filename)  # TODO(fixme):for different types of persistence managers that require different load/save methods
            print(f"Loaded persistence manager from {filename}")
        except Exception as e:
            print(f"/load warning: loading persistence manager from {filename} failed with: {e}")

        return await self.generate_reply()

    async def _on_dump(self):
        await print_messages(self.memgpt_agent.messages)
        return await self.generate_reply()

    async def _on_dumpraw(self):
        await interface.print_messages_raw(self.memgpt_agent.messages)
        return await self.generate_reply()

    async def _on_dump1(self):
        await print_messages(self.memgpt_agent.messages[-1])
        return await self.generate_reply()

    async def _on_memory(self):
        print(f"\nDumping memory contents:\n")
        print(f"{str(self.memgpt_agent.memory)}")
        print(f"{str(self.memgpt_agent.persistence_manager.archival_memory)}")
        print(f"{str(self.memgpt_agent.persistence_manager.recall_memory)}")
        return await self.generate_reply()

    async def _on_model(self):
        if self.memgpt_agent.model == 'gpt-4':
            self.memgpt_agent.model = 'gpt-3.5-turbo'
        elif self.memgpt_agent.model == 'gpt-3.5-turbo':
            self.memgpt_agent.model = 'gpt-4'
        print(f"Updated model to:\n{str(self.memgpt_agent.model)}")
        return await self.generate_reply()

    async def _on_pop(self, user_input: str):
        # Check if there's an additional argument that's an integer
        command = user_input.strip().split()
        amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 2
        print(f"Popping last {amount} messages from stack")
        self.memgpt_agent.messages = self.memgpt_agent.messages[:-amount]
        return await self.generate_reply()

    def _on_wipe(self):
        self.memgpt_agent.wipe()
        return None
