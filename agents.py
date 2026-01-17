"""
Specialized agent subgraphs for the Personal Assistant.

This module contains three specialized agents:
1. ResearchAgent - For web search and Wikipedia lookups
2. CodeGenerationAgent - For writing and executing Python code
3. TaskDecompositionAgent - For breaking complex tasks into subtasks
"""

from typing import Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field


class ResearchSummary(BaseModel):
    """Structured output for research results."""
    web_findings: str = Field(description="Key findings from web search")
    wiki_findings: str = Field(description="Key findings from Wikipedia")
    summary: str = Field(description="Overall research summary")
    sources: List[str] = Field(default_factory=list, description="List of sources used")


class ResearchAgent:
    """Dedicated agent for web search and Wikipedia lookups."""

    def __init__(self, search_tool, wiki_tool):
        self.search_tool = search_tool
        self.wiki_tool = wiki_tool
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.summarizer_llm = self.llm.with_structured_output(ResearchSummary)

    async def run(self, query: str, messages: List[Any]) -> str:
        """Execute research and return a summary."""
        web_results = ""
        wiki_results = ""

        # Execute web search if tool is available
        if self.search_tool:
            try:
                web_results = await self._async_invoke_tool(self.search_tool, query)
            except Exception as e:
                web_results = f"Web search error: {str(e)}"

        # Execute Wikipedia search if tool is available
        if self.wiki_tool:
            try:
                wiki_results = await self._async_invoke_tool(self.wiki_tool, query)
            except Exception as e:
                wiki_results = f"Wikipedia error: {str(e)}"

        # Summarize findings
        system_message = """You are a research assistant that synthesizes information from multiple sources.
Analyze the provided search results and create a comprehensive, well-organized summary.
Be factual and cite your sources when possible."""

        user_message = f"""Research query: {query}

Web Search Results:
{web_results}

Wikipedia Results:
{wiki_results}

Please synthesize these findings into a clear, comprehensive summary."""

        try:
            result = self.summarizer_llm.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ])

            output = f"""## Research Summary

**Web Findings:**
{result.web_findings}

**Wikipedia Findings:**
{result.wiki_findings}

**Summary:**
{result.summary}

**Sources:** {', '.join(result.sources) if result.sources else 'N/A'}"""
            return output

        except Exception as e:
            # Fallback to raw results
            return f"""## Research Results

**Web Search:**
{web_results}

**Wikipedia:**
{wiki_results}"""

    async def _async_invoke_tool(self, tool, query: str) -> str:
        """Invoke a tool asynchronously if possible."""
        if hasattr(tool, 'ainvoke'):
            return await tool.ainvoke(query)
        elif hasattr(tool, 'arun'):
            return await tool.arun(query)
        elif hasattr(tool, 'invoke'):
            return tool.invoke(query)
        elif hasattr(tool, 'run'):
            return tool.run(query)
        elif callable(tool):
            return tool(query)
        else:
            return str(tool)


class CodeOutput(BaseModel):
    """Structured output for code generation."""
    code: str = Field(description="The generated Python code")
    explanation: str = Field(description="Explanation of what the code does")
    test_code: Optional[str] = Field(default=None, description="Optional test code")


class CodeGenerationAgent:
    """Specialized agent for writing and reviewing Python code."""

    def __init__(self, python_repl_tool):
        self.python_repl = python_repl_tool
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.code_llm = self.llm.with_structured_output(CodeOutput)

    async def run(self, task_description: str, messages: List[Any]) -> str:
        """Generate and optionally execute Python code."""

        # Generate code
        system_message = """You are an expert Python programmer.
Write clean, well-documented code following these guidelines:
1. Use clear variable and function names
2. Include docstrings for functions
3. Add type hints where appropriate
4. Handle potential errors gracefully
5. Keep code simple and readable

If the task requires output, use print() statements."""

        user_message = f"""Task: {task_description}

Please write Python code to accomplish this task. Include an explanation of your approach."""

        try:
            result = self.code_llm.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ])

            output = f"""## Generated Code

**Explanation:**
{result.explanation}

**Code:**
```python
{result.code}
```"""

            # Execute code if REPL is available
            if self.python_repl and result.code:
                try:
                    exec_result = await self._execute_code(result.code)
                    output += f"""

**Execution Result:**
```
{exec_result}
```"""
                except Exception as e:
                    output += f"""

**Execution Error:**
{str(e)}"""

            return output

        except Exception as e:
            return f"Code generation error: {str(e)}"

    async def _execute_code(self, code: str) -> str:
        """Execute code using the Python REPL tool."""
        if hasattr(self.python_repl, 'ainvoke'):
            return await self.python_repl.ainvoke(code)
        elif hasattr(self.python_repl, 'arun'):
            return await self.python_repl.arun(code)
        elif hasattr(self.python_repl, 'invoke'):
            return self.python_repl.invoke(code)
        elif hasattr(self.python_repl, 'run'):
            return self.python_repl.run(code)
        elif callable(self.python_repl):
            return self.python_repl(code)
        else:
            return "REPL not available"


class SubTask(BaseModel):
    """Individual subtask structure."""
    id: int = Field(description="Unique identifier for this subtask")
    title: str = Field(description="Short title for the subtask")
    description: str = Field(description="Detailed description of what needs to be done")
    dependencies: List[int] = Field(default_factory=list, description="IDs of subtasks this depends on")
    complexity: str = Field(description="Complexity level: low, medium, or high")
    recommended_approach: str = Field(description="Suggested approach or tools to use")


class TaskDecompositionOutput(BaseModel):
    """Structured output for task decomposition."""
    subtasks: List[SubTask] = Field(description="List of subtasks")
    execution_order: List[int] = Field(description="Recommended order to execute subtasks (by ID)")
    overall_strategy: str = Field(description="High-level strategy for completing the task")
    estimated_total_complexity: str = Field(description="Overall complexity: low, medium, or high")


class TaskDecompositionAgent:
    """Breaks complex tasks into manageable subtasks."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.decomposer_llm = self.llm.with_structured_output(TaskDecompositionOutput)

    async def run(self, complex_task: str, messages: List[Any]) -> str:
        """Decompose a complex task into subtasks."""

        system_message = """You are a task decomposition specialist. Your job is to:
1. Analyze complex tasks and break them into smaller, manageable subtasks
2. Identify dependencies between subtasks
3. Estimate the complexity of each subtask
4. Suggest the best approach for each subtask
5. Determine the optimal execution order

Guidelines:
- Each subtask should be specific and actionable
- Keep the number of subtasks reasonable (3-10 typically)
- Consider parallel vs sequential execution
- Identify critical path items"""

        user_message = f"""Please decompose the following complex task into manageable subtasks:

Task: {complex_task}

Provide a clear breakdown with dependencies, complexity estimates, and an overall strategy."""

        try:
            result = self.decomposer_llm.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ])

            # Format output
            output = f"""## Task Decomposition

**Overall Strategy:**
{result.overall_strategy}

**Estimated Total Complexity:** {result.estimated_total_complexity}

**Subtasks:**
"""
            for subtask in result.subtasks:
                deps = f"(depends on: {', '.join(map(str, subtask.dependencies))})" if subtask.dependencies else "(no dependencies)"
                output += f"""
### {subtask.id}. {subtask.title}
- **Description:** {subtask.description}
- **Complexity:** {subtask.complexity}
- **Dependencies:** {deps}
- **Approach:** {subtask.recommended_approach}
"""

            output += f"""
**Recommended Execution Order:** {' -> '.join(map(str, result.execution_order))}
"""
            return output

        except Exception as e:
            return f"Task decomposition error: {str(e)}"
