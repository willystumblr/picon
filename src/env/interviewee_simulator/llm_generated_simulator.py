# https://arxiv.org/abs/2503.16527 - LLM Generated Persona is a Promise with a Catch
# https://huggingface.co/datasets/Tianyi-Lab/Personas
from litellm.cost_calculator import completion_cost
import json
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from src.env.interviewee_simulator.base_interviewee_simulator import (
    BaseIntervieweeSimulator,
)
import logging
from dotenv import load_dotenv
import time
import os
import random



class LLMGeneratedSimulator(BaseIntervieweeSimulator):
    """
    LLM Generated Persona Simulator based on paper "LLM Generated Persona is a Promise with a Catch"

    This simulator uses personas generated with methodological rigor to avoid systematic biases
    in persona-based simulations. The personas include:
    - Objective demographic attributes (age, sex, race, state, etc.)
    - Subjective qualities (Big Five personality traits, mannerisms, quirks)
    - Detailed background and life history
    - Political and social views
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert "persona" in kwargs, "LLM Generated Simulator requires persona parameter"
        assert "simulator_model" in kwargs, (
            "LLM Generated Simulator requires simulator_model parameter"
        )

        self.persona_data = kwargs["persona"]
        self.name = kwargs.get("name", self._extract_name_from_persona())
        self.client_or_model = kwargs['simulator_model']

        # Create comprehensive persona prompt based on paper's methodology
        persona_prompt = self._create_comprehensive_persona_prompt()

        #print(persona_prompt[:500])  # Print the first 500 characters of the persona for debugging
        #breakpoint()
        print(f"You are {self.name}.\n{persona_prompt[:200]}...\n\n")

        self.history = [
            {
                "role": "system",
                "content": 
                    f"""You are an AI assistant tasked with generating realistic opinions based on a given persona and a specific topic. 
                    
                    ### TASK ### 
                    You will simulate a persona answering a question.
                    
                    ### GUIDELINES ### 
                    1. Be Faithful to the Persona: Ensure your answer is consistent with the persona’s data. 
                    2. Focus on Relevant Aspects: Center your reasoning on the relevant factors that would influence the persona’s opinion on that topic. 
                    3. Be Objective: Avoid injecting personal bias or overly politically correct views that may not align with the persona’s standpoint. 
                    
                    ### INSTRUCTIONS ###
                    - Answer to the question based on the provided persona.
                    
                    ### PERSONA ###
                    {persona_prompt}"""
                ,
            }
        ]
        self.max_tokens = self.get_max_token()

    def _extract_name_from_persona(self):
        """Extract name from persona data if available"""
        if isinstance(self.persona_data, dict):
            # Try to find name in various possible fields
            name_fields = ["name", "NAME", "character_name", "persona_name"]
            for field in name_fields:
                if field in self.persona_data:
                    return self.persona_data[field]

            # If no name found, try to extract from descriptive persona
            if "descriptive_persona" in self.persona_data:
                desc = self.persona_data["descriptive_persona"]
                # Simple extraction logic
                import re

                name_match = re.search(
                    r"(?:Name:\s*|Meet\s*)([A-Z][a-z]+\s+[A-Z][a-z]+)", desc
                )
                if name_match:
                    return name_match.group(1)

        return "Anonymous Persona"

    def _create_comprehensive_persona_prompt(self):
        """
        Create a comprehensive persona description based on the structured data
        following the paper's methodology for rigorous persona generation
        """
        if isinstance(self.persona_data, str):
            # If persona is already a string, use it directly
            return self.persona_data

        persona_parts = []

        # Extract objective demographic attributes
        if "objective_table_persona" in self.persona_data:
            try:
                obj_data = json.loads(self.persona_data["objective_table_persona"])
                demographics = []
                for key, value in obj_data.items():
                    if value and value != "":
                        demographics.append(f"{key.replace('_', ' ').title()}: {value}")
                if demographics:
                    persona_parts.append("DEMOGRAPHIC PROFILE:")
                    persona_parts.append("\n".join(demographics))
            except:
                pass

        # Extract descriptive persona (life story and background)
        if "descriptive_persona" in self.persona_data:
            persona_parts.append("\nLIFE STORY AND BACKGROUND:")
            persona_parts.append(self.persona_data["descriptive_persona"])

        # Extract subjective attributes and personality traits
        if "subjective_table_persona" in self.persona_data:
            try:
                subj_data = json.loads(self.persona_data["subjective_table_persona"])

                # Big Five personality traits
                if "BIG_FIVE_SCORES" in subj_data:
                    persona_parts.append("\nPERSONALITY TRAITS (Big Five):")
                    for trait, score in subj_data["BIG_FIVE_SCORES"].items():
                        persona_parts.append(f"- {trait.title()}: {score}")

                # Other subjective attributes
                subjective_attrs = []
                for key, value in subj_data.items():
                    if key != "BIG_FIVE_SCORES" and value and value != "":
                        subjective_attrs.append(
                            f"{key.replace('_', ' ').title()}: {value}"
                        )

                if subjective_attrs:
                    persona_parts.append("\nPERSONAL ATTRIBUTES:")
                    persona_parts.append(
                        "\n".join(f"- {attr}" for attr in subjective_attrs)
                    )

            except:
                pass

        # Add meta information if available
        if "meta_persona" in self.persona_data:
            persona_parts.append(
                f"\nMETA INFORMATION: {self.persona_data['meta_persona']}"
            )

        return "\n".join(persona_parts)

    def _get_response(self, message: str) -> IntervieweeResponse:
        """Generate response maintaining persona consistency"""
        message = f"""### QUESTION ### \n {message} \n\n ### YOUR RESPONSE ###"""    
        self.history.append({"role": "user", "content": message})
        self._truncate_history()

        try:
            res = get_completion(
                model=self.client_or_model,
                messages=self.history,
                reasoning_effort="low",
                temperature=0.8,  # Slightly higher temperature for more natural responses
                top_p=0.9,
            )
            cost = completion_cost(completion_response=res)
            self.cost += cost

            response = res.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": response})

            # Check for AI detection
            self._ai_check(message, response)

            return IntervieweeResponse(question=message, content=response)

        except Exception as e:
            logging.error(f"Error generating response for {self.name}: {e}")
            raise e

    def get_persona_summary(self):
        """Return a summary of the persona for logging/debugging"""
        if isinstance(self.persona_data, dict):
            summary = {
                "name": self.name,
                "type": "LLM Generated Persona",
                "has_objective_data": "objective_table_persona" in self.persona_data,
                "has_subjective_data": "subjective_table_persona" in self.persona_data,
                "has_descriptive_data": "descriptive_persona" in self.persona_data,
            }

            # Extract key demographics if available
            if "objective_table_persona" in self.persona_data:
                try:
                    obj_data = json.loads(self.persona_data["objective_table_persona"])
                    summary.update(
                        {
                            k: v
                            for k, v in obj_data.items()
                            if k in ["AGE", "SEX", "RACE", "STATE", "EDUCATION"]
                        }
                    )
                except:
                    pass

            return summary

        return {"name": self.name, "type": "LLM Generated Persona"}
