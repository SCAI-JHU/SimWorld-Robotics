from typing import List, Dict, Any
from models import BaseModel
from utils import extract_action_dict


class ReasoningAgent:
    """
    Reasoning Agent for single-step planning (o1/o3 style).
    Uses reasoning mode with one act() call per step.
    """
    
    def __init__(
        self,
        backend: str = "openai",
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "2025-04-01-preview",
        model: str = "gpt-4o",
        reasoning_effort: str = "medium",
        system_prompt: str = "",
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        """
        Initialize the Reasoning Agent.
        
        Args:
            backend: Backend to use (openai, azure, gemini, or custom URL)
            api_key: API key for the backend
            azure_endpoint: Azure endpoint (if using Azure)
            api_version: API version
            model: Model name (e.g., o1, o3)
            reasoning_effort: Effort level for reasoning mode (low, medium, high)
            system_prompt: System prompt for reasoning
            max_retries: Maximum number of retries
            retry_backoff: Backoff multiplier for retries
        """
        self.system_prompt = system_prompt
        
        # Initialize the base model with reasoning mode enabled
        self.model = BaseModel(
            backend=backend,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model=model,
            reasoning_effort=reasoning_effort,
            use_reasoning=True,  # Enable reasoning mode
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        
        # State management
        self.summary = """
            Status: Ready to Start. 
            No obstacles found. 
            No intersections seen. 
            The Landmark to be spotted: Just started, unknown.
            Have not seen the landmark yet.
            """
        self.last_vision_descriptions = ""

    def reset_state(self):
        """Reset the agent's internal state."""
        self.summary = """
            Status: Ready to Start. 
            No obstacles found. 
            No intersections seen. 
            The Landmark to be spotted: Just started, unknown.
            Have not seen the landmark yet.
            """
        self.last_vision_descriptions = ""

    def step(
        self,
        observation: list,  # List[Dict[str, str]] with keys 'img' and 'description'
        instruction: str,
        orientation: float,
        action_history_text: str,
        chosen_actions: list,
    ) -> dict:
        """
        Unified step interface for ReasoningAgent. Accepts preprocessed images and descriptions.
        """
        nav_instance = (
            f"Actions taken: {action_history_text}\n\n"
            f"Your Last Move: {chosen_actions}\n\n"
            f"Vision Description Last Step: {self.last_vision_descriptions}\n\n"
            f"Status of Last Step: {self.summary}\n\n"
            f"Current Subtask: {instruction}\n\n"
            f"Current Orientation: {orientation}\n\n"
        )
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": [{"type": "input_text", "text": nav_instance}]})
        for item in observation:
            if item.get("description"):
                messages.append({"role": "user", "content": [{"type": "input_text", "text": item["description"]}]})
            if item.get("img"):
                messages.append({"role": "user", "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{item['img']}"}]})
        result = self.model.generation(messages)
        rd = extract_action_dict(result.get("output", ""))
        return {
            "vision_description": rd.get("Description", None),
            "actions": rd.get("Actions", None),
            "summary": rd.get("Summary", None),
            "match": rd.get("Match", None),
            "reason": result.get("reason", ""),
            "usage": result.get("usage", {}),
        }


class ReActAgent:
    """
    ReAct Agent for two-step planning.
    First perceives the environment, then reasons about actions.
    """
    
    def __init__(
        self,
        backend: str = "openai",
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "2025-04-01-preview",
        model: str = "gpt-4o",
        temperature: float = 0.2,
        perception_prompt: str = "",
        reasoning_prompt: str = "",
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        """
        Initialize the ReAct Agent.
        
        Args:
            backend: Backend to use (openai, azure, gemini, or custom URL)
            api_key: API key for the backend
            azure_endpoint: Azure endpoint (if using Azure)
            api_version: API version
            model: Model name
            temperature: Temperature for generation
            perception_prompt: Prompt for perception step
            reasoning_prompt: Prompt for reasoning step
            max_retries: Maximum number of retries
            retry_backoff: Backoff multiplier for retries
        """
        self.perception_prompt = perception_prompt
        self.reasoning_prompt = reasoning_prompt
        
        # Initialize the base model with reasoning mode disabled
        self.model = BaseModel(
            backend=backend,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model=model,
            temperature=temperature,
            use_reasoning=False,  # Disable reasoning mode
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        
        # State management
        self.summary = """
            Status: Ready to Start. 
            No obstacles found. 
            No intersections seen. 
            The Landmark to be spotted: Just started, unknown.
            Have not seen the landmark yet.
            """
        self.last_vision_descriptions = ""

    def reset_state(self):
        """Reset the agent's internal state."""
        self.summary = """
            Status: Ready to Start. 
            No obstacles found. 
            No intersections seen. 
            The Landmark to be spotted: Just started, unknown.
            Have not seen the landmark yet.
            """
        self.last_vision_descriptions = ""

    def perceive(
        self,
        observation: list,
        instruction: str,
        orientation: float,
    ) -> dict:
        """
        Perception step: generate vision description from observation.

        Args:
            observation: List of dicts with 'img' and 'description'
            instruction: Current subtask instruction
            orientation: Current agent orientation
            
        Returns:
            Dict with vision_description and usage
        """
        from baseline.single.utils import extract_action_dict
        
        perception_instance = (
            f"Status of last step: {self.summary}\n"
            f"Last Vision Description: {self.last_vision_descriptions}\n\n"
            f"Current subtask: {instruction}\n"
            f"Current Orientation: {orientation}\n"
            f"Reason about how to describe and give the detailed description of the observation in the JSON.\n"
        )
        
        perception_messages = [{"role": "system", "content": self.perception_prompt}]
        perception_messages.append({"role": "user", "content": perception_instance})

        for item in observation:
            if item.get("description"):
                perception_messages.append({"role": "user", "content": item["description"]})
            if item.get("img"):
                perception_messages.append({"role": "user", "content": {"url": f"data:image/png;base64,{item['img']}"}})
        
        perception_result = self.model.generation(perception_messages)
        perception_rd = extract_action_dict(perception_result.get("output", ""))
        vision_description = perception_rd.get("Description", None)
        
        return {
            "vision_description": vision_description,
            "usage": perception_result.get("usage", {"input": 0, "output": 0}),
        }
    
    def act(
        self,
        vision_description: str,
        instruction: str,
        orientation: float,
        action_history_text: str,
        chosen_actions: list,
    ) -> dict:
        """
        Action step: reason about actions based on vision description.
        
        Args:
            vision_description: Vision description from perceive()
            instruction: Current subtask instruction
            orientation: Current agent orientation
            action_history_text: Formatted action history text
            chosen_actions: List of actions from last step
            
        Returns:
            Dict with actions, summary, match, reason, and usage
        """
        reasoning_instance = (
            f"Actions taken: {action_history_text}\n\n"
            f"Your Last Move: {chosen_actions}\n\n"
            f"Vision Description Last Step: {self.last_vision_descriptions}\n\n"
            f"Status of last step: {self.summary}\n\n"
            f"Current Subtask: {instruction}\n\n"
            f"Current Orientation: {orientation}\n\n"
            f"Current Vision Description: {vision_description}\n\n"
            f"Provide your reason, update the summary and give the next action in the JSON.\n"
        )
        
        reasoning_messages = [{"role": "system", "content": self.reasoning_prompt}]
        reasoning_messages.append({"role": "user", "content": reasoning_instance})
        
        reasoning_result = self.model.generation(reasoning_messages)
        reasoning_rd = extract_action_dict(reasoning_result.get("output", ""))
        
        self.last_vision_descriptions = vision_description
        self.summary = reasoning_rd.get("Summary", self.summary)
        actions = reasoning_rd.get("Actions", None)
        
        return {
            "actions": actions,
            "summary": self.summary,
            "match": reasoning_rd.get("Match", None),
            "reason": reasoning_result.get("reason", ""),
            "usage": reasoning_result.get("usage", {"input": 0, "output": 0}),
        }
    
    def step(
        self,
        observation: list,
        instruction: str,
        orientation: float,
        action_history_text: str,
        chosen_actions: list,
    ) -> dict:
        """
        Unified step interface for ReActAgent. Calls perceive() then act().
        
        Args:
            observation: List of dicts with 'img' and 'description'
            instruction: Current subtask instruction
            orientation: Current agent orientation
            action_history_text: Formatted action history text
            chosen_actions: List of actions from last step
            
        Returns:
            Dict with vision_description, actions, summary, match, reason, usage
        """
        # Step 1: Perceive
        perception_result = self.perceive(observation, instruction, orientation)
        vision_description = perception_result.get("vision_description", None)
        perception_usage = perception_result.get("usage", {"input": 0, "output": 0})
        
        # Step 2: Act
        action_result = self.act(
            vision_description,
            instruction,
            orientation,
            action_history_text,
            chosen_actions,
        )
        action_usage = action_result.get("usage", {"input": 0, "output": 0})
        
        # Combine usage
        total_usage = {
            "input": perception_usage.get("input", 0) + action_usage.get("input", 0),
            "output": perception_usage.get("output", 0) + action_usage.get("output", 0),
        }
        
        return {
            "vision_description": vision_description,
            "actions": action_result.get("actions", None),
            "summary": action_result.get("summary", self.summary),
            "match": action_result.get("match", None),
            "reason": action_result.get("reason", ""),
            "usage": total_usage,
        }
