import os
import re
import requests
from dotenv import load_dotenv
from langchain_core.tools import Tool
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import FileManagementToolkit, PlayWrightBrowserToolkit

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool

load_dotenv(override=True)

# Initialize serper only if API key exists
serper_api_key = os.getenv("SERPER_API_KEY")
serper = GoogleSerperAPIWrapper(serper_api_key=serper_api_key) if serper_api_key else None


# Sandboxed Python REPL with blocked dangerous operations
BLOCKED_PATTERNS = [
    r'\bos\.(system|popen|exec|spawn|remove|unlink|rmdir|rename|chmod|chown)',
    r'\bsubprocess\b',
    r'\bshutil\.(rmtree|move|copy)',
    r'\b__import__\b',
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'\bopen\s*\([^)]*["\'][wa]',  # Block write/append mode
    r'\bsocket\b',
    r'\brequests\.(post|put|delete|patch)',
    r'\burllib\b',
    r'\bsys\.exit',
    r'\bquit\s*\(',
    r'\bexit\s*\(',
]


def create_sandboxed_python_repl():
    """Create a Python REPL with basic sandboxing."""
    base_repl = PythonREPLTool()

    def sandboxed_run(code: str) -> str:
        """Execute Python code with safety checks."""
        # Check for blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return f"Error: Code contains blocked operation. Pattern matched: {pattern}"

        # Execute if safe
        try:
            return base_repl.run(code)
        except Exception as e:
            return f"Execution error: {str(e)}"

    return Tool(
        name="python_repl",
        func=sandboxed_run,
        description=(
            "A Python REPL for executing Python code. "
            "Use print() to see output. "
            "Note: Some operations (file writes, network, system commands) are blocked for security."
        )
    )

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

# playwright tools
async def playwright_tools():
    # Initialize Playwright
    playwright = await async_playwright().start()
    # Launch a Chromium browser
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)

    return toolkit.get_tools(), playwright, browser

# pushover notification tool
def push(text: str):
    """
    Send a push notification to the user's device.
    """
    requests.post(pushover_url, data = {
        "token": pushover_token,
        "user": pushover_user,
        "message": text,})
    return "success"

# get file tool
def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()

# Other tools
async def other_tools():
    tools = get_file_tools()

    # Push notifications
    tools.append(Tool(
        name="send_push_notifications",
        func=push,
        description="use this tool to send push notifications"
    ))

    # Web search (only if API key configured)
    if serper:
        tools.append(Tool(
            name="search",
            func=serper.run,
            description="use this tool when you want to get the results of an online web search"
        ))

    # Wikipedia
    wikipedia = WikipediaAPIWrapper()
    tools.append(WikipediaQueryRun(api_wrapper=wikipedia, name="wikipedia"))

    # Sandboxed Python REPL
    tools.append(create_sandboxed_python_repl())

    return tools