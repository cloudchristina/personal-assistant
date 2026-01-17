# Personal Assistant (Sidekick)

A **multi-agent AI assistant** built with LangGraph and Gradio, featuring specialized agents for different tasks.

## Quick Start

```bash
# 1. Initialize project and add dependencies
uv init
uv add -r requirements.txt

# 2. Install dependencies
uv sync

# 3. Install Playwright browser
uv run playwright install chromium

# 4. Create .env file with your API keys
cp .env.example .env  # Then edit with your keys

# 5. Run
uv run app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py (Gradio UI)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    sidekick.py (LangGraph)                  │
│  ┌──────────┐                                               │
│  │ Planner  │──▶ Routes to specialized agents               │
│  └──────────┘                                               │
│       │                                                     │
│       ├──▶ Research Agent (web + Wikipedia)                 │
│       ├──▶ Code Generation Agent (Python REPL)              │
│       ├──▶ Task Decomposition Agent                         │
│       └──▶ General Worker (tool-using)                      │
│                      │                                      │
│                      ▼                                      │
│              ┌────────────┐                                 │
│              │ Evaluator  │ ──▶ Loops back if not satisfied │
│              └────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      tools.py                               │
│  • Google Serper (web search)   • Wikipedia                 │
│  • File management (sandbox)    • Pushover notifications    │
│  • Playwright (browser)         • Python REPL               │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

| File | Purpose |
|------|---------|
| `app.py` | Gradio web UI with chat interface |
| `sidekick.py` | Main LangGraph state machine (~640 lines) |
| `agents.py` | Specialized agent classes with Pydantic outputs |
| `tools.py` | Tool definitions (search, wiki, files, notifications, browser, sandboxed REPL) |
| `pyproject.toml` | Project configuration and dependencies |
| `conversations.sqlite` | SQLite database for persistence |
| `.gitignore` | Git ignore patterns |

## Showcase Features

| Feature | Status | Notes |
|---------|--------|-------| 
| Short-term Memory | ✅ Implemented | `AsyncSqliteSaver` checkpointer for thread-scoped state |   - is Redis ok? how to set up cache
| Long-term Memory | ✅ Implemented | Extracts user facts to `long_term_memory` table |  - data not loaded, load data only on `planner` or add to all agents?
| Multi-agent | ✅ Implemented | Planner, research, code, task decomposition agents |    - do we need multi agent in this case? review all and also think about the orchestration
| Planning Agent | ✅ Implemented | Routes to specialists, asks max 1 clarifying question |  - how to config number of questions based on user input specific level
| Pydantic Outputs | ✅ Implemented | Structured outputs for all agents |
| Evaluator Loop | ✅ Implemented | Validates responses against success criteria |  
| Playwright Browser | ✅ Integrated | Autonomous web browsing and form filling |  - stuck not quit
| Sandboxed Python REPL | ✅ Implemented | Blocks dangerous operations (os.system, subprocess, etc.) |


## Design Decisions

### Short-term Memory: SQLite vs Redis

**Keep SQLite** unless you need horizontal scaling:

| Use Case | Recommendation |
|----------|----------------|
| Single user, local | SQLite ✅ |
| Multi-user, same machine | SQLite ✅ |
| Multi-instance, distributed | Redis |
| High throughput (>100 req/s) | Redis |

### Long-term Memory: Where to Load?

**Planner only** is correct. The flow:
```
Planner (has memory) → routes correctly → Worker (has memory) → answers
```
Specialized agents don't need memory because:
- Research = web search (external data)
- Code = generate code (no personal context)
- Task decomp = structural breakdown

### Multi-agent: Is it Needed?

**Trade-offs:**
- ✅ Extensible for complex workflows
- ✅ Clear separation of concerns
- ❌ Extra latency (planner LLM call)
- ❌ 7 LLM instances (could be 2-3)

Keep multi-agent if planning to add parallel execution or complex chains.

### Clarifying Questions: How Many?

Per [OpenAI Model Spec](https://model-spec.openai.com/2025-12-18.html) and [GPT-5.2 Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide):
- **Default: 0 questions** - proceed with stated assumptions
- **Ambiguous + high stakes: 1 question max**
- **Never ask trivial clarifying questions**

Current implementation (max 1, rarely ask) aligns with best practices.

## SQLite Database Schema

```
┌─────────────────────────────────────────────────────────────┐
│                    conversations.sqlite                     │
├─────────────────────────────────────────────────────────────┤
│  checkpoints        │ Short-term: thread-scoped state      │
│  writes             │ Short-term: incremental changes      │
│  long_term_memory   │ Long-term: user facts cross-session  │
└─────────────────────────────────────────────────────────────┘
```

### Query Examples
```bash
# View all tables
sqlite3 conversations.sqlite ".tables"

# View long-term memories
sqlite3 conversations.sqlite "SELECT * FROM long_term_memory;"

# Count checkpoints per thread
sqlite3 conversations.sqlite "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id;"

# Delete a memory
sqlite3 conversations.sqlite "DELETE FROM long_term_memory WHERE id = 1;"
```

## Agent Performance Metrics

### Metric Categories

```
┌─────────────────────────────────────────┐
│         END-TO-END METRICS              │  ← User sees
│  • Task Completion Rate                 │
│  • User Satisfaction                    │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         EXECUTION METRICS               │  ← System measures
│  • Step Efficiency                      │
│  • Latency / Cost                       │
│  • Error Rate                           │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         COMPONENT METRICS               │  ← Debug level
│  • Tool Selection Accuracy              │
│  • Argument Correctness                 │
│  • Plan Quality / Adherence             │
└─────────────────────────────────────────┘
```

### Key Metrics

| Metric | Definition | Threshold | How to Measure |
|--------|------------|-----------|----------------|
| **Task Completion** | Did the agent accomplish the goal? | >85% | Evaluator → END |
| **First-Attempt Success** | Completed without rejection | >60% | No evaluator loops |
| **Tool Success Rate** | Successful / Total tool calls | >80% | Track exceptions |
| **Step Efficiency** | Useful steps / Total steps | >70% | Count redundant calls |
| **Latency** | Time to complete | <60s | Timestamp diff |
| **Evaluator Rejections** | Times sent back to worker | <3 | Counter in state |

### When Is Response "Not Good Enough"?

| Condition | Verdict | Action |
|-----------|---------|--------|
| `task_completed = False` | ❌ Not good | Log, alert, retry |
| `evaluator_rejections > 3` | ❌ Not good | Agent stuck, needs human |
| `quality_score < 3.0/5` | ❌ Not good | Low quality response |
| `latency > 120s` | ⚠️ Warning | Too slow |
| `tool_success_rate < 50%` | ❌ Not good | Tools failing |

### LLM-as-Judge Quality Scores

| Dimension | Question | Scale |
|-----------|----------|-------|
| Relevance | Does response address the query? | 1-5 |
| Completeness | All parts of question answered? | 1-5 |
| Accuracy | Factually correct? | 1-5 |
| Helpfulness | Actually useful to user? | 1-5 |

**Threshold:** Average score >= 3.5 is "good enough"

## Known Issues

| Issue | Severity | Location | Status |
|-------|----------|----------|--------|
| Hardcoded model | Low | Multiple files | Open |
| Silent exception swallowing | Medium | Throughout | Open |

## Completed Fixes

- [x] Sandbox PythonREPLTool - Blocks dangerous operations (`tools.py:21-64`)
- [x] Fix wiki tool name lookup - Case-insensitive matching (`sidekick.py:160`)
- [x] Fix async cleanup - Proper `asyncio.gather()` (`sidekick.py:667-678`)
- [x] Fix long-term memory loading - Added to planner (`sidekick.py:185-189`)
- [x] Add metrics tracking table - `metrics` table with `record_metric()` (`sidekick.py:127-137, 645-677`)
- [x] Add input validation in `app.py` - Null checks, message length limit, error handling (`app.py:21-53`)

## Future Improvements

### P1 - High
- [ ] Add logging (structlog)

### P2 - Medium
- [ ] Consolidate LLM instances (currently 7 separate)
- [ ] Make memory extraction conditional
- [ ] Convert agents to true LangGraph subgraphs
- [ ] Add async evaluator
- [ ] Make model configurable via env var

### P3 - Low
- [ ] Add tests
- [ ] Add Docker support
- [ ] Add rate limiting

## Setup

```bash
# Initialize project and add dependencies
uv init
uv add -r requirements.txt
uv sync

# Install Playwright browser
uv run playwright install chromium

# Create sandbox directory for file tools
mkdir -p sandbox
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional - Web search
SERPER_API_KEY=...

# Optional - Push notifications
PUSHOVER_TOKEN=...
PUSHOVER_USER=...
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o-mini |
| `SERPER_API_KEY` | No | Google Serper API for web search |
| `PUSHOVER_TOKEN` | No | Pushover app token for notifications |
| `PUSHOVER_USER` | No | Pushover user key |

## Run

```bash
uv run app.py
```

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LLM Evaluation Metrics - Confident AI](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [Anthropic Evaluation Guidelines](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests)
- [LangSmith Observability](https://www.langchain.com/langsmith/observability)
- [DeepEval Agent Evaluation](https://deepeval.com/guides/guides-ai-agent-evaluation)
