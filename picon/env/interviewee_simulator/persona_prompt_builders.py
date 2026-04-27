"""
Persona prompt builders for constructing system prompts from structured data.
Each function takes raw persona data and returns a formatted system prompt string.
"""
import json
import re

def build_nemotron_prompt(persona_data: dict) -> str:
    del persona_data['uuid']
    persona_prompt = ""
    for k, v in persona_data.items():
        persona_prompt += f"## {k}\n{v}\n\n"
    return (
        "You are roleplaying as the following person — stay fully in character throughout the conversation.\n"
        f"{persona_prompt}"
    )

def build_twin_2k_500_prompt(persona_json: str) -> str:
    return (
        "You are an AI assistant. Your task is to answer the 'New Survey Question' as "
        "if you are the individual described in the 'Persona Profile' (which contains their "
        "past survey responses). Remain consistent with the persona's previous answers"
        "and stated characteristics. Carefully follow any instructions provided for the new "
        f"question, including formatting requirements.\n\n{persona_json}"
    )


def build_deeppersona_prompt(persona_data) -> str:
    return (
        "Given the following user profile and request, generate a "
        "personalized response tailored to the user's background and attributes.\n\n"
        f"User profile: {persona_data};"
    )


def build_opencharacter_prompt(persona: str, profile: str) -> str:
    return (
        "You are an AI character with the following Persona.\n\n"
        f"# Persona\n{persona}\n\n"
        f"# Character Profile\n{profile}\n\n"
        "Please stay in your character and comply with the Persona and "
        "Character Profile while being helpful and harmless."
    )


def build_llm_generated_prompt(persona_data, persona_type: str="descriptive") -> str:
    """
    Build a comprehensive persona prompt from structured LLM-generated persona data.
    Based on the paper "LLM Generated Persona is a Promise with a Catch"
    (https://arxiv.org/abs/2503.16527)

    persona_type: one of "descriptive", "subjective", "objective", "meta"
    """
    persona_description = _build_persona_description(persona_data, persona_type)
    return (
        "You are an AI assistant tasked with generating realistic opinions based on "
        "a given persona and a specific topic.\n\n"
        "### TASK ###\n"
        "You will simulate a persona answering a question.\n\n"
        "### GUIDELINES ###\n"
        "1. Be Faithful to the Persona: Ensure your answer is consistent with the persona's data.\n"
        "2. Focus on Relevant Aspects: Center your reasoning on the relevant factors that would "
        "influence the persona's opinion on that topic.\n"
        "3. Be Objective: Avoid injecting personal bias or overly politically correct views that "
        "may not align with the persona's standpoint.\n\n"
        "### INSTRUCTIONS ###\n"
        "- Answer to the question based on the provided persona.\n\n"
        f"### PERSONA ###\n{persona_description}"
    )


def extract_llm_generated_name(persona_data) -> str:
    """Extract name from LLM-generated persona data."""
    if isinstance(persona_data, dict):
        for field in ["name", "NAME", "character_name", "persona_name"]:
            if field in persona_data:
                return persona_data[field]
        if "descriptive_persona" in persona_data:
            name_match = re.search(
                r"(?:Name:\s*|Meet\s*)([A-Z][a-z]+\s+[A-Z][a-z]+)",
                persona_data["descriptive_persona"],
            )
            if name_match:
                return name_match.group(1)
    return "Anonymous Persona"


def _build_persona_description(persona_data, persona_type: str) -> str:
    """Build persona description from structured data using the specified type."""
    if isinstance(persona_data, str):
        return persona_data

    assert persona_type in ("descriptive", "subjective", "objective", "meta"), \
        f"Invalid persona_type '{persona_type}'. Must be one of: descriptive, subjective, objective, meta"

    persona_parts = []

    if persona_type == "descriptive":
        if "descriptive_persona" in persona_data:
            persona_parts.append("LIFE STORY AND BACKGROUND:")
            persona_parts.append(persona_data["descriptive_persona"])

    elif persona_type == "subjective":
        try:
            subj_data = json.loads(persona_data["subjective_table_persona"])
            if "BIG_FIVE_SCORES" in subj_data:
                persona_parts.append("PERSONALITY TRAITS (Big Five):")
                for trait, score in subj_data["BIG_FIVE_SCORES"].items():
                    persona_parts.append(f"- {trait.title()}: {score}")
            subjective_attrs = [
                f"{key.replace('_', ' ').title()}: {value}"
                for key, value in subj_data.items()
                if key != "BIG_FIVE_SCORES" and value and value != ""
            ]
            if subjective_attrs:
                persona_parts.append("PERSONAL ATTRIBUTES:")
                persona_parts.append("\n".join(f"- {attr}" for attr in subjective_attrs))
        except Exception:
            pass

    elif persona_type == "objective":
        try:
            obj_data = json.loads(persona_data["objective_table_persona"])
            demographics = [
                f"{key.replace('_', ' ').title()}: {value}"
                for key, value in obj_data.items()
                if value and value != ""
            ]
            if demographics:
                persona_parts.append("DEMOGRAPHIC PROFILE:")
                persona_parts.append("\n".join(demographics))
        except Exception:
            pass

    elif persona_type == "meta":
        if "meta_persona" in persona_data:
            persona_parts.append(f"META INFORMATION: {persona_data['meta_persona']}")

    return "\n".join(persona_parts)
