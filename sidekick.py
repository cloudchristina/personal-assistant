from typing import TypedDict, Any, List, Optional, Dict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# SQLite Persistence: AsyncSqliteSaver provides persistent conversation memory
# It stores checkpoints (state snapshots) and writes (incremental changes) in SQLite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# SQLite Persistence: aiosqlite provides async SQLite database access
import aiosqlite
from pydantic import BaseModel, Field
import uuid
import asyncio
import json
import time
from dotenv import load_dotenv
from datetime import datetime

from tools import other_tools, playwright_tools

load_dotenv(override=True)


class State(TypedDict):
    """State representation for the sidekick module."""
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    # Planning fields
    planning_complete: bool
    clarification_count: int
    enhanced_context: Optional[str]
    original_request: Optional[str]
    recommended_agent: Optional[str]


class EvaluatorOutput(BaseModel):
    """Evaluator output structure."""
    feedback: str = Field(description="Feedback on the assistant's response.")
    success_criteria_met: bool = Field(description="Whether the success criteria were met.")
    user_input_needed: bool = Field(description="Indicates if user input is needed.")


class PlannerOutput(BaseModel):
    """Structured output from the planning agent."""
    needs_clarification: bool = Field(
        description="Whether clarifying questions are needed"
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="List of clarifying questions (max 3 questions per round)"
    )
    enhanced_context: str = Field(
        description="Enhanced task description with clarified requirements"
    )
    recommended_agent: str = Field(
        description="Recommended specialist agent: 'research', 'code', 'task_decomposition', or 'general'"
    )


# Long-term Memory: Extract key facts from conversations to remember across sessions
class MemoryExtraction(BaseModel):
    """Extracted facts from conversation for long-term memory."""
    facts: List[str] = Field(
        default_factory=list,
        description="Key facts about the user (name, preferences, pets, etc.)"
    )
    should_save: bool = Field(
        default=False,
        description="Whether any new facts were found worth saving"
    )


class SideKick:
    def __init__(self, db_path: str = "conversations.sqlite"):
        # SQLite Persistence: Path to SQLite database file for storing conversation state
        # Default: "conversations.sqlite" in the current directory
        self.db_path = db_path
        self.worker_llm_bind_tools = None
        self.evaluator_llm_with_output = None
        self.planner_llm_with_output = None
        self.tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        # SQLite Persistence: LangGraph checkpointer that saves/loads state from SQLite
        # Enables conversation continuity across app restarts
        self.checkpointer = None
        # SQLite Persistence: Raw aiosqlite connection handle
        # Used by AsyncSqliteSaver and closed during cleanup
        self.db_conn = None
        self.browser = None
        self.playwright = None
        # Specialized agents
        self.research_agent = None
        self.code_agent = None
        self.task_decomp_agent = None
        # Long-term Memory: LLM for extracting facts from conversations
        self.memory_llm = None

    async def setup(self):
        # ===== SQLite Persistence Setup =====
        # 1. Create async connection to SQLite database file
        #    Creates the file if it doesn't exist
        self.db_conn = await aiosqlite.connect(self.db_path)
        # 2. Workaround: Add is_alive() method for LangGraph compatibility
        #    aiosqlite.Connection lacks this method that LangGraph expects
        self.db_conn.is_alive = lambda: True
        # 3. Create LangGraph checkpointer wrapper around the connection
        self.checkpointer = AsyncSqliteSaver(self.db_conn)
        # 4. Initialize SQLite tables (checkpoints, writes) if they don't exist
        #    - checkpoints: stores full state snapshots (thread_id, checkpoint_id, data)
        #    - writes: stores incremental state changes (task_id, channel, value)
        await self.checkpointer.setup()

        # ===== Long-term Memory Table Setup =====
        # Separate table for cross-session facts (not thread-scoped like checkpoints)
        await self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default',
                fact TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ===== Metrics Table Setup =====
        # Track agent performance metrics for observability
        await self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metadata TEXT
            )
        """)
        await self.db_conn.commit()
        # ===== End SQLite Setup =====

        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()

        # Worker LLM with tools
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_bind_tools = worker_llm.bind_tools(self.tools)

        # Evaluator LLM with structured output
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

        # Planner LLM with structured output
        planner_llm = ChatOpenAI(model="gpt-4o-mini")
        self.planner_llm_with_output = planner_llm.with_structured_output(PlannerOutput)

        # Long-term Memory: LLM for extracting facts from conversations
        memory_llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory_llm = memory_llm.with_structured_output(MemoryExtraction)

        # Initialize specialized agents
        await self._init_specialized_agents()

        # Build the graph
        await self.build_graph()

    async def _init_specialized_agents(self):
        """Initialize the specialized agent subgraphs."""
        from agents import ResearchAgent, CodeGenerationAgent, TaskDecompositionAgent

        # Find specific tools for each agent (case-insensitive matching)
        search_tool = next((t for t in self.tools if t.name == "search"), None)
        wiki_tool = next((t for t in self.tools if "wiki" in t.name.lower()), None)
        python_repl = next((t for t in self.tools if "python" in t.name.lower() or "repl" in t.name.lower()), None)

        self.research_agent = ResearchAgent(search_tool, wiki_tool)
        self.code_agent = CodeGenerationAgent(python_repl)
        self.task_decomp_agent = TaskDecompositionAgent()

    async def planner(self, state: State) -> Dict[str, Any]:
        """Planning node that asks clarifying questions before execution."""
        clarification_count = state.get("clarification_count", 0)

        # Long-term Memory: Load remembered facts for planning context
        memories = await self.get_long_term_memories()
        memory_context = ""
        if memories:
            memory_context = "\n\nYou remember these facts about the user:\n" + "\n".join(f"- {m}" for m in memories)

        # Get original request or use last message
        if state.get("original_request"):
            original_request = state["original_request"]
        else:
            last_human_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_human_msg = msg.content
                    break
            original_request = last_human_msg or "No request provided"

        system_message = f"""You are a planning agent that analyzes user requests before execution.{memory_context}
Your job is to:
1. Understand the user's intent
2. Determine which specialist agent is best suited for the task
3. Create an enhanced context with all gathered information

Available specialist agents:
- research: For web searches, Wikipedia lookups, information gathering FROM THE INTERNET
- code: For writing, reviewing, or executing Python code
- task_decomposition: For breaking complex tasks into subtasks
- general: For general tasks handled by the main worker

IMPORTANT RULES:
- RARELY ask clarifying questions. Most requests should proceed immediately.
- Only ask ONE question if absolutely critical information is missing.
- Never ask clarifying questions for: greetings, simple questions, research requests, or when context from conversation history is sufficient.
- If user mentions something from a previous conversation, use that context directly.
- CRITICAL: If the user asks about personal information (names, pets, family, preferences) and you have that information in your remembered facts above, route to 'general' NOT 'research'. Only use 'research' for internet lookups."""

        user_message = f"""Original request: {original_request}

Conversation so far:
{self.format_conversation(state['messages'])}

Proceed directly with the request. Only ask <3 clarifying question if critical information is truly missing."""

        try:
            result = self.planner_llm_with_output.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ])
        except Exception as e:
            # On error, proceed to general worker
            return {
                "enhanced_context": original_request,
                "planning_complete": True,
                "recommended_agent": "general"
            }

        # Only ask 1 clarifying question, only once, and only if truly needed
        if result.needs_clarification and clarification_count < 1 and result.clarifying_questions:
            question = result.clarifying_questions[0]  # Only take first question
            return {
                "messages": [AIMessage(content=f"Quick question: {question}")],
                "clarification_count": clarification_count + 1,
                "planning_complete": False,
                "user_input_needed": True,
                "original_request": original_request
            }
        else:
            # Track planner routing decision
            asyncio.create_task(self.record_metric(
                "planner.routing",
                value=1.0,
                metadata={"agent": result.recommended_agent, "request": original_request[:100]}
            ))
            return {
                "enhanced_context": result.enhanced_context,
                "planning_complete": True,
                "recommended_agent": result.recommended_agent,
                "original_request": original_request
            }

    def planner_router(self, state: State) -> str:
        """Route based on planning outcome."""
        if state.get("user_input_needed") and not state.get("planning_complete"):
            return "END"

        recommended = state.get("recommended_agent", "general")

        if recommended == "research":
            return "research_agent"
        elif recommended == "code":
            return "code_agent"
        elif recommended == "task_decomposition":
            return "task_decomposition_agent"
        else:
            return "worker"

    async def run_research_subgraph(self, state: State) -> Dict[str, Any]:
        """Execute research agent and return results to main state."""
        query = state.get("enhanced_context") or state.get("original_request", "")

        try:
            result = await self.research_agent.run(query, state["messages"])
            return {
                "messages": [AIMessage(content=f"Research Results:\n{result}")],
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Research agent error: {str(e)}. Falling back to general worker.")],
            }

    async def run_code_subgraph(self, state: State) -> Dict[str, Any]:
        """Execute code generation agent."""
        task_desc = state.get("enhanced_context") or state.get("original_request", "")

        try:
            result = await self.code_agent.run(task_desc, state["messages"])
            return {
                "messages": [AIMessage(content=f"Code Generation Results:\n{result}")],
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Code agent error: {str(e)}. Falling back to general worker.")],
            }

    async def run_decomp_subgraph(self, state: State) -> Dict[str, Any]:
        """Execute task decomposition agent."""
        complex_task = state.get("enhanced_context") or state.get("original_request", "")

        try:
            result = await self.task_decomp_agent.run(complex_task, state["messages"])
            return {
                "messages": [AIMessage(content=f"Task Decomposition Results:\n{result}")],
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Task decomposition error: {str(e)}. Falling back to general worker.")],
            }

    async def worker(self, state: State) -> Dict[str, Any]:
        """Main worker node that performs tasks using tools."""
        # Use enhanced context if available
        enhanced = state.get("enhanced_context", "")
        context_addition = f"\nClarified task context: {enhanced}" if enhanced else ""

        # Long-term Memory: Load remembered facts about the user
        memories = await self.get_long_term_memories()
        memory_context = ""
        if memories:
            memory_context = "\n\nYou remember these facts about the user:\n" + "\n".join(f"- {m}" for m in memories)

        system_message = f"""You are a helpful assistant that can use tools to complete tasks.

You keep working on a task until either you have a question or clarification for the user, or a success criteria is met.
The success criteria is: {state['success_criteria']}.
{context_addition}{memory_context}

You have many tools to use, including tools to browse the internet, navigate and retrieve web page content.
You have a tool to run python code, but note that you would need to include a print() statement if output is needed.
The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

You should reply either with a question for user about this task, or with your final response.
If you have a question for the user, you need to reply by clearly stating your questions. An example might be:

Question: please clarify whether you want a summary or a detailed answer.

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer."""

        if state.get("feedback_on_work"):
            # Track self-correction attempt
            await self.record_metric(
                "worker.self_correction",
                value=1.0,
                metadata={"feedback": state["feedback_on_work"][:200]}
            )
            system_message += f"""
Previously you thought you completed the task, but your reply was rejected because the success criteria were not met.
Here is the feedback why this was rejected: {state['feedback_on_work']}
Use this feedback, please continue the task, ensuring that you meet the success criteria or have a question for the user."""

        # Update or add the system message in the messages list
        found_system_message = False
        messages = list(state["messages"])
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True
                break
        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        try:
            response = await self.worker_llm_bind_tools.ainvoke(messages)
            # Track tool calls if any
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    await self.record_metric(
                        "worker.tool_call",
                        value=1.0,
                        metadata={"tool": tool_call.get("name", "unknown")}
                    )
        except Exception as e:
            error_msg = f"Worker LLM error: {str(e)}"
            await self.record_metric("worker.error", value=1.0, metadata={"error": str(e)[:100]})
            return {"messages": [AIMessage(content=error_msg)]}
        return {"messages": [response]}

    def worker_router(self, state: State) -> str:
        """Route worker output based on tool calls."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    def format_conversation(self, messages: List[Any]) -> str:
        """Format conversation history for display."""
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    def evaluator(self, state: State) -> Dict[str, Any]:
        """Evaluator node that assesses task completion."""
        last_response = state["messages"][-1].content if state["messages"] else ""

        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.

Assess the Assistant's last response based on given criteria.
Respond with your feedback, and with your decision on whether the success criteria has been met,
and whether more input is needed from the user."""

        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take
based on last response from the Assistant.

The entire conversation with the assistant, with the user's original request and all replies, is:
{self.format_conversation(state['messages'])}

The success criteria for this task is:
{state["success_criteria"]}

And the final response from the Assistant that you are evaluating is:
{last_response}

Respond with your feedback, and decide if the success criteria is met by this response.
Also, decide if more user input is required, either because the assistant has a question, needs clarification, or
seems to be stuck and unable to answer without help.

The assistant has access to a tool to write files. If the assistant says they have written a file, then you
can assume they have done so.
Overall you should give the Assistant the benefit of the doubt if they say they've done something.
But you should reject if you feel that more work should go into this."""

        if state.get("feedback_on_work"):
            user_message += f"\nAlso, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you are seeing the Assistant repeating the same mistake, then consider responding that user input is required."

        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)

        # Track evaluation result (fire-and-forget, handle no event loop)
        is_rejection = not eval_result.success_criteria_met and not eval_result.user_input_needed
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self.record_metric(
                "evaluator.result",
                value=1.0 if eval_result.success_criteria_met else 0.0,
                metadata={
                    "success": eval_result.success_criteria_met,
                    "rejection": is_rejection,
                    "user_input_needed": eval_result.user_input_needed
                }
            ))
        except RuntimeError:
            pass  # No event loop, skip metric

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator Feedback on this answer: {eval_result.feedback}",
                }
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }

    def route_based_on_evaluation(self, state: State) -> str:
        """Route based on evaluation results."""
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"

    async def build_graph(self):
        """Build the complete agent graph."""
        graph_builder = StateGraph(State)

        # Add all nodes
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("research_agent", self.run_research_subgraph)
        graph_builder.add_node("code_agent", self.run_code_subgraph)
        graph_builder.add_node("task_decomposition_agent", self.run_decomp_subgraph)
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Entry point: START -> planner
        graph_builder.add_edge(START, "planner")

        # Planner routes to specialized agents or worker
        graph_builder.add_conditional_edges(
            "planner",
            self.planner_router,
            {
                "END": END,
                "research_agent": "research_agent",
                "code_agent": "code_agent",
                "task_decomposition_agent": "task_decomposition_agent",
                "worker": "worker"
            }
        )

        # Specialized agents go to evaluator
        graph_builder.add_edge("research_agent", "evaluator")
        graph_builder.add_edge("code_agent", "evaluator")
        graph_builder.add_edge("task_decomposition_agent", "evaluator")

        # Worker flow
        graph_builder.add_conditional_edges(
            "worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")

        # Evaluator routing
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END}
        )

        # SQLite Persistence: Compile graph with checkpointer for state persistence
        # The checkpointer automatically saves state after each node execution
        # and enables resuming conversations using thread_id
        self.graph = graph_builder.compile(checkpointer=self.checkpointer)

    async def run_superstep(self, message, success_criteria, history, thread_id: str = None):
        """Run a superstep of the agent graph."""
        # SQLite Persistence: thread_id is the key for storing/retrieving conversation state
        # Each unique thread_id gets its own conversation history in SQLite
        config = {"configurable": {"thread_id": thread_id or self.sidekick_id}}

        # SQLite Persistence: Check if conversation state exists in database
        # This enables resuming conversations after app restarts
        existing_state = None
        try:
            # SQLite Persistence: aget_state() loads the latest checkpoint for this thread_id
            existing_state = await self.graph.aget_state(config)
        except Exception:
            pass

        is_continuation = (
            existing_state and
            existing_state.values and
            not existing_state.values.get("planning_complete", True)
        )

        if is_continuation:
            # This is a response to clarification questions
            state = {
                "messages": [HumanMessage(content=message)],
            }
        else:
            # New conversation
            state = {
                "messages": [HumanMessage(content=message)],
                "success_criteria": success_criteria or "The answer should be clear and accurate",
                "feedback_on_work": None,
                "success_criteria_met": False,
                "user_input_needed": False,
                "planning_complete": False,
                "clarification_count": 0,
                "enhanced_context": None,
                "original_request": message,
                "recommended_agent": None,
            }

        # SQLite Persistence: ainvoke() executes the graph and automatically saves
        # each state transition to SQLite via the checkpointer
        start_time = time.time()
        result = await self.graph.ainvoke(state, config=config)
        latency = time.time() - start_time

        # Track request latency
        await self.record_metric(
            "request.latency",
            value=latency,
            metadata={"thread_id": thread_id or self.sidekick_id, "is_continuation": is_continuation},
            thread_id=thread_id
        )

        # Long-term Memory: Extract and save key facts from this conversation
        await self.extract_and_save_memories(result["messages"])

        # Format response for UI
        user_msg = {"role": "user", "content": message}

        # Extract only AI messages from result (filter out user's HumanMessage to avoid duplication)
        ai_messages = [
            msg for msg in result["messages"]
            if isinstance(msg, AIMessage) or (hasattr(msg, "role") and msg.get("role") == "assistant")
        ]

        # Return user message + AI responses
        if len(ai_messages) >= 2:
            # AI reply + evaluator feedback
            reply = {"role": "assistant", "content": ai_messages[-2].content if hasattr(ai_messages[-2], 'content') else ai_messages[-2].get("content", "")}
            feedback = {"role": "assistant", "content": ai_messages[-1].content if hasattr(ai_messages[-1], 'content') else ai_messages[-1].get("content", "")}
            return history + [user_msg, reply, feedback]
        elif len(ai_messages) >= 1:
            # Single AI response (e.g., planner clarifying questions)
            reply = {"role": "assistant", "content": ai_messages[-1].content if hasattr(ai_messages[-1], 'content') else ai_messages[-1].get("content", "")}
            return history + [user_msg, reply]
        else:
            return history + [user_msg]

    # ===== Long-term Memory Methods =====

    async def get_long_term_memories(self, user_id: str = "default") -> List[str]:
        """Load all facts from long-term memory for a user."""
        if not self.db_conn:
            return []
        try:
            cursor = await self.db_conn.execute(
                "SELECT DISTINCT fact FROM long_term_memory WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
                (user_id,)
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
        except Exception:
            return []

    async def save_long_term_memory(self, fact: str, user_id: str = "default"):
        """Save a fact to long-term memory (avoids duplicates)."""
        if not self.db_conn or not fact.strip():
            return
        try:
            # Check if fact already exists
            cursor = await self.db_conn.execute(
                "SELECT 1 FROM long_term_memory WHERE user_id = ? AND fact = ?",
                (user_id, fact.strip())
            )
            if await cursor.fetchone():
                return  # Already exists
            await self.db_conn.execute(
                "INSERT INTO long_term_memory (user_id, fact) VALUES (?, ?)",
                (user_id, fact.strip())
            )
            await self.db_conn.commit()
        except Exception:
            pass

    async def extract_and_save_memories(self, messages: List[Any], user_id: str = "default"):
        """Extract key facts from conversation and save to long-term memory."""
        if not self.memory_llm:
            return

        # Get recent conversation text
        conversation = self.format_conversation(messages[-10:])  # Last 10 messages

        try:
            result = self.memory_llm.invoke([
                SystemMessage(content="""Extract key facts about the user from this conversation.
Facts to extract: name, location, pets, family, preferences, job, hobbies, important dates.
Only extract facts explicitly stated by the user. Be concise (e.g., "User's name is Christina").
Return should_save=true only if you found new meaningful facts."""),
                HumanMessage(content=f"Conversation:\n{conversation}")
            ])

            if result.should_save and result.facts:
                for fact in result.facts:
                    await self.save_long_term_memory(fact, user_id)
        except Exception:
            pass

    # ===== End Long-term Memory Methods =====

    # ===== Metrics Methods =====

    async def record_metric(self, name: str, value: float = 1.0, metadata: dict = None, thread_id: str = None):
        """Record a metric to the metrics table."""
        if not self.db_conn:
            return
        try:
            await self.db_conn.execute(
                "INSERT INTO metrics (thread_id, metric_name, metric_value, metadata) VALUES (?, ?, ?, ?)",
                (thread_id or self.sidekick_id, name, value, json.dumps(metadata) if metadata else None)
            )
            await self.db_conn.commit()
        except Exception:
            pass

    async def get_metrics_summary(self, thread_id: str = None, limit: int = 100) -> List[dict]:
        """Get recent metrics, optionally filtered by thread_id."""
        if not self.db_conn:
            return []
        try:
            if thread_id:
                cursor = await self.db_conn.execute(
                    "SELECT metric_name, metric_value, metadata, timestamp FROM metrics WHERE thread_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (thread_id, limit)
                )
            else:
                cursor = await self.db_conn.execute(
                    "SELECT metric_name, metric_value, metadata, timestamp FROM metrics ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            rows = await cursor.fetchall()
            return [{"name": r[0], "value": r[1], "metadata": json.loads(r[2]) if r[2] else None, "timestamp": r[3]} for r in rows]
        except Exception:
            return []

    # ===== End Metrics Methods =====

    async def cleanup_async(self):
        """Async cleanup that properly awaits all resources."""
        cleanup_tasks = []
        if self.browser:
            cleanup_tasks.append(self.browser.close())
        if self.playwright:
            cleanup_tasks.append(self.playwright.stop())
        if self.db_conn:
            cleanup_tasks.append(self.db_conn.close())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    def cleanup(self):
        """Clean up resources. Called by Gradio's delete_callback."""
        try:
            asyncio.get_running_loop()
            # Schedule cleanup and don't wait (Gradio context)
            asyncio.ensure_future(self.cleanup_async())
        except RuntimeError:
            # No running loop - run synchronously
            asyncio.run(self.cleanup_async())
