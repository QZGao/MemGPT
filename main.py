import asyncio
from absl import app, flags
import logging
import os

import interface  # for printing to terminal
import memgpt.system as system
import memgpt.utils as utils
import memgpt.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.agent import ConversationEndedException
from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, InMemoryStateManagerWithFaiss
from memgpt.user_agent import UserAgent, console, clear_line

FLAGS = flags.FLAGS
flags.DEFINE_string("persona", default=personas.DEFAULT, required=False, help="Specify persona")
flags.DEFINE_string("human", default=humans.DEFAULT, required=False, help="Specify human")
flags.DEFINE_string("model", default=constants.DEFAULT_MEMGPT_MODEL, required=False, help="Specify the LLM model")
flags.DEFINE_boolean("first", default=False, required=False, help="Use -first to send the first message in the sequence")
flags.DEFINE_boolean("debug", default=False, required=False, help="Use -debug to enable debugging output")
flags.DEFINE_boolean("no_verify", default=False, required=False, help="Bypass message verification")
flags.DEFINE_string("archival_storage_faiss_path", default="", required=False, help="Specify archival storage with FAISS index to load (a folder with a .index and .json describing documents to be loaded)")
flags.DEFINE_string("archival_storage_files", default="", required=False, help="Specify files to pre-load into archival memory (glob pattern)")
flags.DEFINE_string("archival_storage_files_compute_embeddings", default="", required=False, help="Specify files to pre-load into archival memory (glob pattern), and compute embeddings over them")
flags.DEFINE_string("archival_storage_sqldb", default="", required=False, help="Specify SQL database to pre-load into archival memory")


async def get_persistence_manager():
    if FLAGS.archival_storage_faiss_path:
        index, archival_database = utils.prepare_archival_index(FLAGS.archival_storage_faiss_path)
        persistence_manager = InMemoryStateManagerWithFaiss(index, archival_database)
    elif FLAGS.archival_storage_files:
        archival_database = utils.prepare_archival_index_from_files(FLAGS.archival_storage_files)
        print(f"Preloaded {len(archival_database)} chunks into archival memory.")
        persistence_manager = InMemoryStateManagerWithPreloadedArchivalMemory(archival_database)
    elif FLAGS.archival_storage_files_compute_embeddings:
        faiss_save_dir = await utils.prepare_archival_index_from_files_compute_embeddings(FLAGS.archival_storage_files_compute_embeddings)
        interface.important_message(f"To avoid computing embeddings next time, replace --archival_storage_files_compute_embeddings={FLAGS.archival_storage_files_compute_embeddings} with\n\t --archival_storage_faiss_path={faiss_save_dir} (if your files haven't changed).")
        index, archival_database = utils.prepare_archival_index(faiss_save_dir)
        persistence_manager = InMemoryStateManagerWithFaiss(index, archival_database)
    else:
        persistence_manager = InMemoryStateManager()

    return persistence_manager


async def main():
    utils.DEBUG = FLAGS.debug
    logging.getLogger().setLevel(logging.CRITICAL)
    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    print("Running... [exit by typing '/exit']")

    persistence_manager = await get_persistence_manager()

    memgpt_agent = presets.use_preset(presets.DEFAULT, FLAGS.model, personas.get_persona_text(FLAGS.persona), humans.get_human_text(FLAGS.human), interface, persistence_manager, skip_verify=FLAGS.no_verify)
    await interface.print_messages(memgpt_agent.messages)

    user_agent = UserAgent()

    if FLAGS.archival_storage_sqldb:
        if not os.path.exists(FLAGS.archival_storage_sqldb):
            print(f"File {FLAGS.archival_storage_sqldb} does not exist")
            return
        # Ingest data from file into archival storage
        else:
            print(f"Database found! Loading database into archival memory")
            data_list = utils.read_database_as_list(FLAGS.archival_storage_sqldb)
            user_message = f"Your archival memory has been loaded with a SQL database called {data_list[0]}, which contains schema {data_list[1]}. Remember to refer to this first while answering any user questions!"
            for row in data_list:
                await memgpt_agent.persistence_manager.archival_memory.insert(row)
            print(f"Database loaded into archival memory.")

    # auto-exit for 
    if "GITHUB_ACTIONS" in os.environ:
        return

    try:
        # If user goes first
        if FLAGS.first:
            await user_agent.initiate_chat(memgpt_agent)

        else:
            console.input('[bold cyan]Hit enter to begin (will request first MemGPT message)[/bold cyan]')
            clear_line()
            print()

            await memgpt_agent.initiate_chat(user_agent)

    # Conversation ended here
    except ConversationEndedException:
        pass

    print("Finished.")


if __name__ ==  '__main__':

    def run(argv):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

    app.run(run)
