"""Business logic handlers for bot functionality."""


async def handle_ping() -> str:
    """Handle ping functionality and return the response message.

    Returns:
        str: The ping response message.
    """
    return "Pong! 🏓"


async def handle_message_received() -> str:
    """Handle when a message is received and return the response.

    Returns:
        str: The message received response.
    """
    return "Message received."
