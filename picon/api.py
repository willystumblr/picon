"""
PICON high-level API.

Usage:
    import picon

    # Mode 1: External agent endpoint (no model needed)
    result = picon.run(
        api_base="http://localhost:8000/v1",
        name="MyAgent",
    )

    # Mode 2: LLM + persona prompt
    result = picon.run(
        persona="You are a 35-year-old software engineer...",
        name="John",
        model="gpt-5",
    )
    print(result.summary)
    print(result.eval_scores)
    result.save("results/john.json")
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os
import glob
import time
import logging

from dotenv import load_dotenv

from picon.config import DEFAULT_CONFIG, get_prompt_path, get_question_path
from picon.agents.agent_factory import get_agent
from picon.env.interrogation_env import InterrogationEnv
from picon.tools.web_search import SerperSearch
from picon.tools.address_locator import GoogleGeocodeValidate
from picon.utils import write_json, read_json


@dataclass
class PiconResult:
    """Interview + evaluation result container."""
    success: bool = False
    ai_detected: bool = False
    summary: dict = field(default_factory=dict)
    eval_scores: dict = field(default_factory=dict)
    raw_results: dict = field(default_factory=dict)
    result_path: Optional[str] = None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_json(self.raw_results, path)


def run(
    persona: str = "",
    name: str = "Agent",
    model: str = None,
    api_base: str = None,
    api_key: str = None,
    num_turns: int = 30,
    num_sessions: int = 2,
    do_eval: bool = True,
    eval_factors: List[str] = None,
    questioner_model: str = None,
    extractor_model: str = None,
    web_search_model: str = None,
    evaluator_model: str = None,
    nhd_model: str = None,
    output_dir: str = None,
    question_seed: int = 42,
    verbose: bool = False,
    **kwargs,
) -> PiconResult:
    """Run persona interview + evaluation in one call.

    Two modes:
      - External agent: provide api_base (model is optional, defaults to placeholder)
      - LLM persona: provide model (and optionally persona, api_key)
    """
    load_dotenv()

    if not model and not api_base:
        raise ValueError("Either 'model' or 'api_base' must be provided.")

    cfg = {**DEFAULT_CONFIG}
    if questioner_model:  cfg["questioner_model"] = questioner_model
    if extractor_model:   cfg["extractor_model"] = extractor_model
    if web_search_model:  cfg["web_search_model"] = web_search_model
    if evaluator_model:   cfg["evaluator_model"] = evaluator_model
    if nhd_model:         cfg["nhd_model"] = nhd_model
    if output_dir:        cfg["output_dir"] = output_dir

    # Read persona from file if path is given
    if persona and os.path.isfile(persona):
        with open(persona) as f:
            persona = f.read()

    interviewee_kwargs = {
        "baseline_name": "generic_agent",
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "persona": persona or "",
        "name": name,
        "nhd_model": cfg["nhd_model"],
        "question_seed": question_seed,
        **kwargs,
    }

    # Build agents
    agents = {
        "questioner": get_agent("questioner", get_prompt_path("questioner.txt"), model=cfg["questioner_model"]),
        "extractor": get_agent("entity_extractor", get_prompt_path("entity_extractor.txt"), model=cfg["extractor_model"]),
        "web_search": get_agent("web_search", get_prompt_path("websearch_prompt.txt"), model=cfg["web_search_model"]),
        "evaluator": get_agent("evaluator", get_prompt_path("evaluator_prompt.txt"), model=cfg["evaluator_model"]),
    }

    tools = {
        "serper_search": SerperSearch(api_key=os.getenv("SERPER_API_KEY")),
        "google_geocode_validate": GoogleGeocodeValidate(api_key=os.getenv("GOOGLE_GEOCODE")),
    }

    result_dir = cfg["output_dir"]
    result_path = f"{result_dir}/{name.replace(' ', '_')}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)

    picon_result = PiconResult()
    persona_stats = {
        "name": name,
        "ai_detected": False,
        "success": False,
        "total_cost": 0.0,
        "sessions_completed": 0,
    }

    try:
        env = InterrogationEnv(
            agents=agents,
            tools=tools,
            max_turns=num_turns,
            question_path=get_question_path(),
            verbose=verbose,
            **interviewee_kwargs,
        )

        results_complete = {}
        histories = []
        reset_only = False
        for session_idx in range(num_sessions):
            logging.info(f"Starting session {session_idx + 1}/{num_sessions} for: {name}")
            if verbose:
                print(f"\n=== Session {session_idx + 1}/{num_sessions} ===\n")
            env.reset(reset_only=reset_only)
            if not reset_only:
                done = False
                while not done:
                    state, done = env.step()
                state = env.finalize()
            session_result = env.save_state()
            histories.append(env.state.history)
            results_complete[f"session_{session_idx+1}"] = session_result
            persona_stats["sessions_completed"] += 1
            reset_only = True

        results_complete["agents_memory"] = {n: agent.memory for n, agent in env.agents.items()}
        write_json(results_complete, result_path)

        persona_stats["success"] = True
        # Aggregate costs
        for session_key in [k for k in results_complete if k.startswith("session_")]:
            cost_data = results_complete[session_key].get("cost", {})
            persona_stats["total_cost"] += cost_data.get("total_cost", 0.0)

        # Evaluation
        eval_scores = {}
        if do_eval:
            logging.info(f"Running evaluation for {name}...")
            eval_result = env.evaluate(histories, eval_factors=eval_factors)
            eval_cost = agents["evaluator"].cost
            persona_stats["eval_cost"] = eval_cost
            persona_stats["total_cost"] += eval_cost
            results_complete["evaluation"] = eval_result
            write_json(results_complete, result_path)

            if eval_result:
                internal = eval_result.get("internal", {}).get("score", {})
                external = eval_result.get("external", {}).get("score", {})
                stability = eval_result.get("stability", {})
                eval_scores = {
                    "ic_score": internal.get("ic_score"),
                    "cooperativeness": internal.get("cooperativeness"),
                    "non_contradiction_rate": internal.get("non_contradiction_rate"),
                    "external_ec": external.get("ec_score"),
                    "coverage": external.get("coverage"),
                    "non_refutation_rate": external.get("non_refutation_rate"),
                    "inter_session_stability": stability.get("inter_session", {}).get("score"),
                    "intra_session_stability": stability.get("intra_session", {}).get("score"),
                }

        env.shutdown()

        picon_result.success = True
        picon_result.summary = persona_stats
        picon_result.eval_scores = eval_scores
        picon_result.raw_results = results_complete
        picon_result.result_path = result_path
        return picon_result

    except ValueError as e:
        if "AI Detected" in str(e):
            picon_result.ai_detected = True
        logging.warning(f"Interview stopped: {e}")
        if "env" in locals():
            env.shutdown()
        picon_result.summary = persona_stats
        return picon_result

    except Exception as e:
        logging.exception(f"Error during interview: {e}")
        if "env" in locals():
            env.shutdown()
        picon_result.summary = persona_stats
        return picon_result


def run_interview(
    name: str = "Agent",
    model: str = None,
    persona: str = "",
    api_base: str = None,
    api_key: str = None,
    num_turns: int = None,
    num_sessions: int = None,
    questioner_model: str = None,
    extractor_model: str = None,
    web_search_model: str = None,
    evaluator_model: str = None,
    nhd_model: str = None,
    questioner_port: int = None,
    extractor_port: int = None,
    web_search_port: int = None,
    evaluator_port: int = None,
    nhd_port: int = None,
    questioner_prompt_path: str = None,
    entity_extractor_prompt_path: str = None,
    web_search_prompt_path: str = None,
    evaluator_prompt_path: str = None,
    output_dir: str = None,
    question_file_path: str = None,
    question_seed: int = 42,
    num_get_to_know_q: int = 10,
    num_combs: int = 1,
    eval_per_combination: bool = False,
    eval_factors: List[str] = None,
    **kwargs,
) -> dict:
    """Run interview sessions for a single persona.

    Two modes:
      - External agent: provide api_base (model is optional)
      - LLM persona: provide model (and optionally persona, api_key)

    Returns a dict with keys: persona_stats, result_path, results_complete,
    histories, env.  Pass the return value to run_evaluation() for scoring.
    """
    if not model and not api_base:
        raise ValueError("Either 'model' or 'api_base' must be provided.")

    cfg = {**DEFAULT_CONFIG}
    if questioner_model:  cfg["questioner_model"] = questioner_model
    if extractor_model:   cfg["extractor_model"]  = extractor_model
    if web_search_model:  cfg["web_search_model"] = web_search_model
    if evaluator_model:   cfg["evaluator_model"]  = evaluator_model
    if nhd_model:         cfg["nhd_model"]         = nhd_model
    if output_dir:        cfg["output_dir"]        = output_dir
    if num_turns:         cfg["num_turns"]         = num_turns
    if num_sessions:      cfg["num_sessions"]      = num_sessions

    cfg["num_get_to_know_q"] = num_get_to_know_q
    cfg["num_combs"]         = num_combs
    if cfg["num_combs"] >= 2 and cfg["num_sessions"] != 1:
        logging.info(
            f"num_combs={cfg['num_combs']} >= 2; forcing num_sessions=1 "
            f"(was {cfg['num_sessions']})."
        )
        cfg["num_sessions"] = 1

    if persona and os.path.isfile(persona):
        with open(persona) as f:
            persona = f.read()

    result_path = (
        f"{cfg['output_dir']}/{name}/"
        f"{name.replace(' ', '_')}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    persona_stats = {
        "name": name,
        "ai_detected": False,
        "success": False,
        "error_type": None,
        "duration_min": 0.0,
        "total_cost": 0.0,
        "agents_cost": 0.0,
        "interviewee_cost": 0.0,
        "tool_costs": 0.0,
        "num_interviewee_responses": 0,
        "num_turns_completed": 0,
        "num_tool_calls": 0,
        "sessions_completed": 0,
        "eval_ic_score": None,
        "eval_cooperativeness": None,
        "eval_non_contradiction_rate": None,
        "eval_external_ec_score": None,
        "eval_coverage": None,
        "eval_non_refutation_rate": None,
        "eval_stability_inter_session": None,
        "eval_stability_intra_session": None,
    }

    tools = {
        "serper_search": SerperSearch(api_key=os.getenv("SERPER_API_KEY")),
        "google_geocode_validate": GoogleGeocodeValidate(api_key=os.getenv("GOOGLE_GEOCODE")),
    }

    agents = {
        "questioner": get_agent(
            "questioner",
            questioner_prompt_path or get_prompt_path("questioner.txt"),
            model=cfg["questioner_model"], port=questioner_port,
        ),
        "extractor": get_agent(
            "entity_extractor",
            entity_extractor_prompt_path or get_prompt_path("entity_extractor.txt"),
            model=cfg["extractor_model"], port=extractor_port,
        ),
        "web_search": get_agent(
            "web_search",
            web_search_prompt_path or get_prompt_path("websearch_prompt.txt"),
            model=cfg["web_search_model"], port=web_search_port,
        ),
        "evaluator": get_agent(
            "evaluator",
            evaluator_prompt_path or get_prompt_path("evaluator_prompt.txt"),
            model=cfg["evaluator_model"], port=evaluator_port,
        ),
    }

    interviewee_kwargs = {
        "baseline_name": "generic_agent",
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "persona": persona,
        "name": name,
        "nhd_model": cfg["nhd_model"],
        "nhd_port": nhd_port,
        "question_seed": question_seed,
        **kwargs,
    }

    results_complete = {}

    try:
        env = InterrogationEnv(
            agents=agents,
            tools=tools,
            max_turns=cfg["num_turns"],
            question_path=question_file_path or get_question_path(),
            num_get_to_know_q=cfg["num_get_to_know_q"],
            num_combs=cfg["num_combs"],
            **interviewee_kwargs,
        )

        histories = []
        total_session_idx = 0
        for comb_idx in range(cfg["num_combs"]):
            env.set_active_combination(comb_idx)

            # Skip this combination if a result file for its q_ids already exists
            if eval_per_combination:
                import glob as _glob
                q_ids = "_".join(q["id"] for q in env.active_questions)
                existing = _glob.glob(
                    f"{cfg['output_dir']}/{name}/{name.replace(' ', '_')}_{q_ids}_*.json"
                )
                if existing:
                    logging.info(f"Skipping comb {comb_idx + 1} ({q_ids}): result file already exists.")
                    continue

            # Between combinations, clear agent memory so sessions are independent
            if comb_idx > 0:
                for agent in env.agents.values():
                    agent.reset()

            comb_histories = []
            comb_results = {}
            reset_only = False
            for session_idx in range(cfg["num_sessions"]):
                total_session_idx += 1
                logging.info(
                    f"Starting comb {comb_idx + 1}/{cfg['num_combs']}, "
                    f"session {session_idx + 1}/{cfg['num_sessions']} for: {name}"
                )
                env.reset(reset_only=reset_only)
                if not reset_only:
                    done = False
                    while not done:
                        state, done = env.step()
                    state = env.finalize()
                session_result = env.save_state()
                session_result["combination_idx"] = comb_idx
                session_result["combination_questions"] = [q["id"] for q in env.active_questions]
                histories.append(env.state.history)
                comb_histories.append(env.state.history)
                results_complete[f"session_{total_session_idx}"] = session_result
                comb_results[f"session_{session_idx + 1}"] = session_result
                persona_stats["sessions_completed"] += 1
                reset_only = True

            if eval_per_combination:
                comb_results["agents_memory"] = {n: agent.memory for n, agent in env.agents.items()}
                q_ids = "_".join(q["id"] for q in env.active_questions)
                comb_path = (
                    f"{cfg['output_dir']}/{name}/"
                    f"{name.replace(' ', '_')}_{q_ids}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
                )
                if eval_factors is not None or True:
                    comb_eval = env.evaluate(comb_histories, eval_factors=eval_factors)
                    comb_results["evaluation"] = comb_eval
                write_json(comb_results, comb_path)
                logging.info(f"Saved comb {comb_idx + 1} results to {comb_path}.")

        results_complete["agents_memory"] = {n: agent.memory for n, agent in env.agents.items()}
        write_json(results_complete, result_path)
        logging.info(f"Saved interview results to {result_path}.")

        persona_stats["success"] = True
        for session_key in [k for k in results_complete if k.startswith("session_")]:
            session_data = results_complete[session_key]
            try:
                persona_stats["duration_min"] += float(
                    session_data.get("duration", "0 min").replace(" min", "")
                )
            except Exception:
                pass
            cost_data = session_data.get("cost", {})
            persona_stats["total_cost"]        += cost_data.get("total_cost", 0.0)
            persona_stats["agents_cost"]       += cost_data.get("agents_cost", 0.0)
            persona_stats["interviewee_cost"]  += cost_data.get("interviewee_cost", 0.0)
            persona_stats["tool_costs"]        += sum(cost_data.get("tool_costs", {}).values())
            for turn in session_data.get("history", []):
                for obs in turn.get("environment_observation", []):
                    if obs.get("observation_type") == "interviewee_response":
                        persona_stats["num_interviewee_responses"] += 1
                    if obs.get("observation_type") == "tool_output":
                        tool_outputs = obs.get("tool_output", [])
                        persona_stats["num_tool_calls"] += len(tool_outputs) if tool_outputs else 0
                if turn.get("type") == "main_interrogation":
                    persona_stats["num_turns_completed"] += 1

        return {
            "persona_stats": persona_stats,
            "result_path": result_path,
            "results_complete": results_complete,
            "histories": histories,
            "env": env,
        }

    except Exception as e:
        if "env" in locals():
            env.shutdown()
        raise


def run_evaluation(interview_result: dict, eval_factors: List[str] = None) -> dict:
    """Run evaluation on the result returned by run_interview().

    Returns the updated persona_stats dict.
    """
    if interview_result["env"] is None or interview_result["histories"] is None:
        return interview_result["persona_stats"]

    persona_stats    = interview_result["persona_stats"]
    result_path      = interview_result["result_path"]
    results_complete = interview_result["results_complete"]
    histories        = interview_result["histories"]
    env              = interview_result["env"]

    try:
        logging.info(f"Running evaluation for {persona_stats['name']}...")
        eval_result = env.evaluate(histories, eval_factors=eval_factors)
        eval_cost = env.agents["evaluator"].cost
        persona_stats["eval_cost"] = eval_cost
        persona_stats["total_cost"] = persona_stats.get("total_cost", 0.0) + eval_cost
        results_complete["evaluation"] = eval_result
        write_json(results_complete, result_path)

        if eval_result:
            internal       = eval_result.get("internal", {})
            external       = eval_result.get("external", {})
            stability      = eval_result.get("stability", {})
            internal_score = internal.get("score", {})
            external_score = external.get("score", {})

            persona_stats["eval_ic_score"]              = internal_score.get("ic_score")
            persona_stats["eval_cooperativeness"]        = internal_score.get("cooperativeness")
            persona_stats["eval_non_contradiction_rate"] = internal_score.get("non_contradiction_rate")
            persona_stats["eval_external_ec_score"]      = external_score.get("ec_score")
            persona_stats["eval_coverage"]               = external_score.get("coverage")
            persona_stats["eval_non_refutation_rate"]    = external_score.get("non_refutation_rate")
            persona_stats["eval_stability_inter_session"] = stability.get("inter_session", {}).get("score")
            persona_stats["eval_stability_intra_session"] = stability.get("intra_session", {}).get("score")

    except Exception as e:
        env.shutdown()
        raise
    finally:
        env.shutdown()

    return persona_stats


def interview(persona: str, name: str = "Agent", **kwargs) -> PiconResult:
    """Run interview only (no evaluation)."""
    return run(persona=persona, name=name, do_eval=False, **kwargs)


def evaluate(result_path: str, eval_factors: List[str] = None, evaluator_model: str = None) -> dict:
    """Run evaluation on an existing interview result file."""
    load_dotenv()

    cfg = {**DEFAULT_CONFIG}
    if evaluator_model:
        cfg["evaluator_model"] = evaluator_model

    results = read_json(result_path)

    agents = {
        "evaluator": get_agent("evaluator", get_prompt_path("evaluator_prompt.txt"), model=cfg["evaluator_model"]),
    }

    env = InterrogationEnv(
        agents=agents,
        result_data=results,
    )

    histories = []
    for key in sorted(k for k in results if k.startswith("session_")):
        session = results[key]
        from picon.schemas import Turn
        history = [Turn(**t) for t in session.get("history", [])]
        histories.append(history)

    eval_result = env.evaluate(histories, eval_factors=eval_factors)
    eval_cost = agents["evaluator"].cost
    env.shutdown()

    # Save back to result file
    results["evaluation"] = eval_result
    write_json(results, result_path)

    eval_scores = {}
    if eval_result:
        internal = eval_result.get("internal", {}).get("score", {})
        external = eval_result.get("external", {}).get("score", {})
        stability = eval_result.get("stability", {})
        eval_scores = {
            "eval_cost": eval_cost,
            "internal_harmonic_mean": internal.get("harmonic_mean"),
            "internal_responsiveness": internal.get("responsiveness_score"),
            "internal_consistency": internal.get("consistency_score"),
            "external_ec": external.get("ec_score"),
            "external_coverage": external.get("coverage"),
            "external_non_refutation_rate": external.get("non_refutation_rate"),
            "inter_session_stability": stability.get("inter_session", {}).get("score"),
            "intra_session_stability": stability.get("intra_session", {}).get("score"),
        }

    # Update summary file if it exists alongside the result file
    result_stem = os.path.splitext(os.path.basename(result_path))[0]
    summary_dir = os.path.join(os.path.dirname(result_path), result_stem)
    if os.path.isdir(summary_dir):
        summary_files = glob.glob(os.path.join(summary_dir, "summary_*.json"))
        for summary_path in summary_files:
            summary = read_json(summary_path)
            summary.setdefault("results", {}).update(eval_scores)
            write_json(summary, summary_path)
            logging.info(f"Updated summary: {summary_path}")

    return eval_scores
