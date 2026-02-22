import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "new"))
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
from typing import Dict, Any


def get_interviewee_simulator(baseline_name: str, **kwargs) -> BaseIntervieweeSimulator:
    """
    Factory function to create and return an interviewee simulator instance based on the baseline_name.
    Args:
        baseline_name (str): The name of the interviewee simulator to create. `characterai`, `human_simulacra`, `opencharacter`, `consistent_llm`, etc.
    Returns:
        BaseIntervieweeSimulator: An instance of the requested interviewee simulator.
    Raises:
        ValueError: If the specified baseline_name is not supported.
    """

    if baseline_name == "characterai":
        from src.env.interviewee_simulator.characterai_simulator import CharacterAISimulator
        return CharacterAISimulator(**kwargs)
    if baseline_name == "characterai":
        from src.env.interviewee_simulator.characterai_simulator import CharacterAISimulator
        return CharacterAISimulator(**kwargs)
    elif baseline_name == "human_simulacra":
        from src.env.interviewee_simulator.human_simulacra_simulator import HumanSimulacraSimulator
        return HumanSimulacraSimulator(**kwargs)
    elif baseline_name == "naive_human_simulacra":
        from src.env.interviewee_simulator.naive_human_simulacra_simulator import NaiveHumanSimulacraSimulator
        return NaiveHumanSimulacraSimulator(**kwargs)
    elif baseline_name == "opencharacter":
        from src.env.interviewee_simulator.opencharacter_simulator import OpenCharacterSimulator
        return OpenCharacterSimulator(**kwargs)
    elif baseline_name == "consistent_llm":
        from src.env.interviewee_simulator.consistent_llm_simulator import ConsistentLLMSimulator
        return ConsistentLLMSimulator(**kwargs)
    elif baseline_name == "persona_hub":
        from src.env.interviewee_simulator.persona_hub_simulator import PersonaHubSimulator
        return PersonaHubSimulator(**kwargs)
    elif baseline_name == "twin_2k_500":
        from src.env.interviewee_simulator.twin_2k_500_simulator import Twin2K500Simulator
        return Twin2K500Simulator(**kwargs)
    elif baseline_name == "deeppersona":
        from src.env.interviewee_simulator.deeppersona_simulator import DeepPersonaSimulator
        return DeepPersonaSimulator(**kwargs)
    elif baseline_name == "llm_generated":
        from src.env.interviewee_simulator.llm_generated_simulator import LLMGeneratedSimulator
        return LLMGeneratedSimulator(**kwargs)
    else:
        raise ValueError(f"Unsupported interviewee simulator: {baseline_name}")