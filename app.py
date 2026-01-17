import gradio as gr
import uuid
from sidekick import SideKick


async def setup():
    """Initialize the SideKick agent with SQLite persistence."""
    try:
        sidekick = SideKick(db_path="conversations.sqlite")
        await sidekick.setup()
        thread_id = str(uuid.uuid4())
        print(f"Setup complete. Thread ID: {thread_id}")
        return sidekick, thread_id
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


MAX_MESSAGE_LENGTH = 10000


async def process_message(sidekick, message, success_criteria, history, thread_id):
    """Process a user message through the agent graph."""
    # Ensure history is a list
    if history is None:
        history = []

    # Check sidekick initialized
    if sidekick is None:
        error_msg = {"role": "assistant", "content": "Error: Assistant not initialized. Please refresh the page."}
        return history + [error_msg], "", sidekick, thread_id

    # Check empty message
    if not message or not message.strip():
        return history, "", sidekick, thread_id

    # Limit message length (prevent token explosion)
    if len(message) > MAX_MESSAGE_LENGTH:
        error_msg = {"role": "assistant", "content": f"Error: Message too long ({len(message)} chars). Maximum: {MAX_MESSAGE_LENGTH}"}
        return history + [error_msg], "", sidekick, thread_id

    # Default success criteria if empty
    if not success_criteria or not success_criteria.strip():
        success_criteria = "The answer should be clear and accurate"

    try:
        results = await sidekick.run_superstep(message, success_criteria, history, thread_id)
        return results, "", sidekick, thread_id  # Clear message box after sending
    except Exception as e:
        error_msg = {"role": "assistant", "content": f"Error processing request: {str(e)}"}
        return history + [error_msg], "", sidekick, thread_id


async def reset():
    """Reset the conversation with a new SideKick instance and thread."""
    new_sidekick = SideKick(db_path="conversations.sqlite")
    await new_sidekick.setup()
    new_thread_id = str(uuid.uuid4())
    return "", "", None, new_sidekick, new_thread_id


def free_resources(sidekick):
    """Clean up browser and other resources."""
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")


with gr.Blocks(title="Sidekick") as ui:
    gr.Markdown("## Sidekick Personal Co-worker")
    gr.Markdown("*Enhanced with planning agent, specialized sub-agents, and SQLite persistence*")

    sidekick = gr.State(delete_callback=free_resources)
    thread_id = gr.State()

    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=400)

    with gr.Group():
        with gr.Row():
            message = gr.Textbox(
                show_label=False,
                placeholder="Your request to the Sidekick (e.g., 'Search for...', 'Write Python code to...', 'Help me plan...')"
            )
        with gr.Row():
            success_criteria = gr.Textbox(
                show_label=False,
                placeholder="What are your success criteria? (optional)"
            )

    with gr.Row():
        reset_button = gr.Button("Reset Conversation", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    # Initialize on page load
    ui.load(setup, [], [sidekick, thread_id])

    # Handle message submission
    message.submit(
        process_message,
        [sidekick, message, success_criteria, chatbot, thread_id],
        [chatbot, message, sidekick, thread_id]
    )
    success_criteria.submit(
        process_message,
        [sidekick, message, success_criteria, chatbot, thread_id],
        [chatbot, message, sidekick, thread_id]
    )
    go_button.click(
        process_message,
        [sidekick, message, success_criteria, chatbot, thread_id],
        [chatbot, message, sidekick, thread_id]
    )
    reset_button.click(
        reset, [], [message, success_criteria, chatbot, sidekick, thread_id]
    )

ui.launch(inbrowser=True, server_port=7862, theme=gr.themes.Default(primary_hue="emerald"))
