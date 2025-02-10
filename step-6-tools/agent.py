# agent.py
import os
import logging
from typing import List, Dict, Optional, Union, Any, Sequence
from datetime import datetime
from openai import OpenAI
from strategy import StrategyFactory, ExecutionStrategy
from persistence import AgentPersistence
from context import ContextManager, DocumentMetadata
from tools import ToolRegistry, Tool, parse_tool_usage, ToolResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    """
    Agent class with memory persistence, RAG-based context support, and tool usage capabilities.
    """
    
    def __init__(self, 
                 name: str, 
                 persistence: Optional[AgentPersistence] = None,
                 context: Optional[ContextManager] = None,
                 tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize an agent with a name and optional components.
        
        Args:
            name: The agent's name
            persistence: Optional persistence manager
            context: Optional context manager
            tool_registry: Optional tool registry
        """
        self._name = name
        self._persona = ""
        self._instruction = ""
        self._task = ""
        self._api_key = os.getenv('OPENAI_API_KEY', '')
        self._model = "gpt-4o-mini"
        self._history: List[Dict[str, str]] = []
        self._strategy: Optional[ExecutionStrategy] = None
        self._persistence = persistence or AgentPersistence()
        self._context = context
        self._tool_registry = tool_registry or ToolRegistry()
        
        # Try to load existing state
        self._persistence.load_agent_state(self)
        logger.info(f"Initialized agent: {name}")

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self._name

    @property
    def persona(self) -> str:
        """Get the agent's persona."""
        return self._persona

    @persona.setter
    def persona(self, value: str):
        """Set the agent's persona."""
        self._persona = value
        self.save_state()

    @property
    def instruction(self) -> str:
        """Get the agent's global instruction."""
        return self._instruction

    @instruction.setter
    def instruction(self, value: str):
        """Set the agent's global instruction."""
        self._instruction = value
        self.save_state()

    @property
    def task(self) -> str:
        """Get the current task."""
        return self._task

    @task.setter
    def task(self, value: str):
        """Set the current task."""
        self._task = value
        self.save_state()

    @property
    def tools(self) -> List[Tool]:
        """Get the list of registered tools."""
        return list(self._tool_registry._tools.values())

    @tools.setter
    def tools(self, tools: Sequence[Tool]):
        """Set the tools for the agent, replacing any existing tools."""
        # Clear existing tools by creating new registry
        self._tool_registry = ToolRegistry()
        
        # Register new tools
        if tools:
            for tool in tools:
                self._tool_registry.register(tool)
                logger.info(f"Registered tool: {tool.name}")
            
        # Save state after tool modification
        self.save_state()

    @property
    def strategy(self) -> Optional[ExecutionStrategy]:
        """Get the current execution strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name: str):
        """Set the execution strategy by name."""
        self._strategy = StrategyFactory.create_strategy(strategy_name)
        self.save_state()

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self._history

    @property
    def context(self) -> Optional[str]:
        """Get the current context response from the context manager."""
        if self._context:
            return self._context.response
        return None

    @context.setter
    def context(self, context_manager: Optional[ContextManager]):
        """Set the context manager."""
        self._context = context_manager
        self.save_state()

    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List[str]: Names of available tools
        """
        return self._tool_registry.list_tools()

    def get_tools_prompt(self) -> str:
        """
        Get formatted string of available tools for prompts.
        
        Returns:
            str: Formatted tools prompt
        """
        return self._tool_registry.get_tools_prompt()

    def execute_tool(self, tool_name: str, **parameters) -> ToolResult:
        """
        Execute a specific tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool-specific parameters
                
        Returns:
            ToolResult: Result of tool execution
        """
        tool = self._tool_registry.get_tool(tool_name)
        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(success=False, data=None, error=f"Tool not found: {tool_name}")
                
        try:
            result = tool.execute(**parameters)
            logger.info(f"Tool execution result: {result.success}")
            return result
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg)
            return ToolResult(success=False, data=None, error=error_msg)

    def _build_messages(self, task: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Build the messages list including persona, instruction, context, history, and available tools.
        
        Args:
            task: Optional task override
            
        Returns:
            List[Dict[str, str]]: List of message dictionaries
        """
        messages = [{"role": "system", "content": self.persona}]
        
        if self.instruction:
            messages.append({
                "role": "user", 
                "content": f"Global Instruction: {self.instruction}"
            })
        
        # Add tool information
        if self._tool_registry and self._tool_registry.list_tools():
            messages.append({
                "role": "system",
                "content": self.get_tools_prompt()
            })
        
        # Add relevant context if available
        if self.context:
            messages.append({
                "role": "user",
                "content": f"Relevant Context:\n{self.context}"
            })
        
        # Add conversation history
        messages.extend(self._history)
        
        # Use provided task or stored task
        current_task = task if task is not None else self._task
        
        # Apply strategy if set
        if self._strategy and current_task:
            current_task = self._strategy.build_prompt(current_task, self.instruction)
        
        # Add the current task if it exists
        if current_task:
            messages.append({"role": "user", "content": current_task})
            
        return messages

    def _process_tool_usage(self, response: str) -> str:
        """
        Process response for tool usage and execute tools if specified.
        
        Args:
            response: LLM response string
            
        Returns:
            str: Modified response with tool results
        """
        modified_response = response
        while True:
            tool_info = parse_tool_usage(modified_response)
            if not tool_info:
                break
                
            tool_name = tool_info.get("name")
            parameters = tool_info.get("parameters", {})
            
            result = self.execute_tool(tool_name, **parameters)
            
            # Find the tool usage block
            start_idx = modified_response.find("Tool:")
            end_idx = start_idx + modified_response[start_idx:].find("\n", 
                     modified_response[start_idx:].find("Parameters:"))
            if end_idx < start_idx:  # No newline found after Parameters
                end_idx = len(modified_response)
                
            if result and result.success:
                # Replace tool usage block with result
                tool_block = modified_response[start_idx:end_idx]
                modified_response = modified_response.replace(
                    tool_block,
                    f"Tool Result ({tool_name}): {result.data}"
                )
            else:
                error_msg = result.error if result else "Tool execution failed"
                tool_block = modified_response[start_idx:end_idx]
                modified_response = modified_response.replace(
                    tool_block,
                    f"Tool Error ({tool_name}): {error_msg}"
                )
                
        return modified_response

    def execute(self, task: Optional[str] = None) -> str:
        """
        Execute a task using the configured LLM and available tools.
        
        Args:
            task: Optional task to execute (overrides stored task)
            
        Returns:
            str: The response from the LLM
        """
        if task is not None:
            self._task = task
        
        if not self._api_key:
            return "API key not found. Please set the OPENAI_API_KEY environment variable."

        if not self._task:
            return "No task specified. Please provide a task to execute."

        client = OpenAI(api_key=self._api_key)
        messages = self._build_messages()

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages
            )
            
            response_content = response.choices[0].message.content
            
            # Process response through strategy if set
            if self._strategy:
                response_content = self._strategy.process_response(response_content)
            
            # Process any tool usage in the response
            response_content = self._process_tool_usage(response_content)
            
            # Store the interaction in history
            self._history.append({"role": "user", "content": self._task})
            self._history.append({
                "role": "assistant",
                "content": response_content
            })
            
            # Save state after successful execution
            self.save_state()
            
            # Clear the task after execution
            self._task = ""
            
            return response_content
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            return f"An error occurred: {str(e)}"

    def save_state(self) -> bool:
        """
        Save the current state of the agent.
        
        Returns:
            bool: True if successful
        """
        return self._persistence.save_agent_state(self)
    
    def load_state(self, agent_name: Optional[str] = None) -> bool:
        """
        Load a saved state into the agent.
        
        Args:
            agent_name: Optional name of agent state to load
            
        Returns:
            bool: True if successful
        """
        return self._persistence.load_agent_state(self, agent_name)
    
    def clear_history(self, keep_last: int = 0):
        """
        Clear the conversation history, optionally keeping the last N states.
        
        Args:
            keep_last: Number of most recent states to keep (0 clears all)
        """
        if keep_last > 0:
            self._persistence.cleanup_old_states(self.name, keep_last)
            # Reload the state to get the kept history
            self.load_state()
        else:
            self._history = []
            self.save_state()

    def pause(self) -> bool:
        """
        Pause the agent by saving its current state.
        
        Returns:
            bool: True if successful
        """
        return self.save_state()
    
    def resume(self, agent_name: Optional[str] = None) -> bool:
        """
        Resume the agent by loading its saved state.
        
        Args:
            agent_name: Optional name of agent state to load
            
        Returns:
            bool: True if successful
        """
        return self.load_state(agent_name)

    def get_history_states(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve the last N states with their timestamps.
        
        Args:
            limit: Maximum number of states to retrieve
            
        Returns:
            List[Dict]: List of historical states
        """
        return self._persistence.get_agent_history(self.name, limit)

    def delete_agent(self) -> bool:
        """
        Delete all data for this agent from the database.
        
        Returns:
            bool: True if successful
        """
        if self._context:
            self._context.clear_index()
        return self._persistence.delete_agent_state(self.name)

    def available_strategies(self) -> List[str]:
        """
        Return a list of available strategy names.
        
        Returns:
            List[str]: List of available strategy names
        """
        return StrategyFactory.available_strategies()

    @staticmethod
    def list_saved_agents() -> Dict[str, datetime]:
        """
        List all saved agents and their last update times.
        
        Returns:
            Dict[str, datetime]: Dict mapping agent names to last update timestamps
        """
        persistence = AgentPersistence()
        return persistence.list_saved_agents()