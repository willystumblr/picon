from src.agents.base_agent import Agent
from typing import Dict, Any

def get_agent(
    agent_type: str,
    system_message_path: str = "",
    **kwargs
    ) -> Agent:
    """
    Factory function to create and return an agent instance based on the agent_type.
    
    Args:
        agent_type (str): The type of agent to create. Supported types are "extractor" and "interviewer".
        system_message_path (str): The file path to the system message.
    Returns:
        Agent: An instance of the requested agent type.
        
    Raises:
        ValueError: If the specified agent_type is not supported.
    """
    if agent_type == "entity_extractor":
        from src.agents.extractor_agent import ExtractorAgent
        return ExtractorAgent(role="extractor", system_message=open(system_message_path).read(), **kwargs)
    elif agent_type == "claim_extractor":
        from src.agents.claim_agent import ClaimExtractorAgent
        return ClaimExtractorAgent(role="claim_extractor", system_message=open(system_message_path).read(), **kwargs)
    elif agent_type == "questioner":
        from src.agents.questioner_agent import QuestionerAgent
        return QuestionerAgent(role="questioner", system_message=open(system_message_path).read(), **kwargs)
    elif agent_type == "web_search":
        from src.agents.web_search_agent import WebSearchAgent
        return WebSearchAgent(role="web_search", system_message=open(system_message_path).read(), **kwargs)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")