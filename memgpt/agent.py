class Agent:
    """Abstraction of agents."""

    async def initiate_chat(self, recipient: "Agent"):
        """Initiate the conversation."""

    async def send(self, message: str, recipient: "Agent"):
        """Send a message to the recipient."""

    async def receive(self, message: str, sender: "Agent"):
        """Receive a message from the recipient, validate, then generate a reply and send."""


class ConversationEndedException(Exception):
    """Raises this exception when conversation is ended."""
    pass
