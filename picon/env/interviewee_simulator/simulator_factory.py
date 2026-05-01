from picon.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
from picon.env.interviewee_simulator.generic_agent_simulator import GenericAgentSimulator


def get_interviewee_simulator(baseline_name: str, **kwargs) -> BaseIntervieweeSimulator:
    if baseline_name == "consistent_llm":
        from picon.env.interviewee_simulator.consistent_llm_simulator import ConsistentLLMSimulator
        return ConsistentLLMSimulator(**kwargs)
    return GenericAgentSimulator(**kwargs)
