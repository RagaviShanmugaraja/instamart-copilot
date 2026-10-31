import sqlite3
import json
import time
import uuid
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

import mock_llm
from policy_rag import retrieve


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DB_PATH = Path("data/instamart.db")
LEDGER_PATH = Path("outputs/refund_ledger.jsonl")
CHECKPOINT_DIR = Path("state")

LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TOOL 1: GET ORDER
# --------------------------------------------------

def get_order(order_id):
    """Retrieve order details from the SQLite database."""

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": float(row["order_amount"]),
            "delivery_minutes": (
                None if row["delivery_minutes"] is None
                else float(row["delivery_minutes"])
            )
        }

    finally:
        connection.close()


# --------------------------------------------------
# TOOL 2: SEARCH POLICY
# --------------------------------------------------

def search_policy(query):
    """Search policy documents and return the top three results."""

    results = retrieve(query, k=3)

    return {
        "results": results
    }


# --------------------------------------------------
# TOOL 3: CALCULATE REFUND
# --------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund based on the late-delivery policy."""

    if delivery_minutes is None:
        return {"refund_inr": 0.0}

    if delivery_minutes > 45:
        refund = round(float(order_amount), 2)
    else:
        refund = 0.0

    return {
        "refund_inr": refund
    }


# --------------------------------------------------
# REFUND LEDGER HELPER
# --------------------------------------------------

def write_ledger_entry(order_id, refund_inr, tool_name):
    """Append a refund action to the JSONL ledger."""

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {
        "refund_inr": round(float(refund_inr), 2)
    }


# --------------------------------------------------
# TOOL 4: ISSUE REFUND
# --------------------------------------------------

def issue_refund(order_id, refund_inr):
    """Record an automatically approved refund."""

    return write_ledger_entry(
        order_id,
        refund_inr,
        "issue_refund"
    )


# --------------------------------------------------
# TOOL 5: ESCALATE TO HUMAN
# --------------------------------------------------

def escalate_to_human(order_id, refund_inr):
    """Record a refund requiring human approval."""

    return write_ledger_entry(
        order_id,
        refund_inr,
        "escalate_to_human"
    )


# --------------------------------------------------
# TOOL SPECIFICATIONS
# --------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": (
            "Retrieve order status, amount and delivery time "
            "using an order ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": (
            "Search the policy documents for information "
            "relevant to a query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the late-delivery refund amount.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {
                    "type": ["number", "null"]
                }
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": (
            "Record an automatically approved refund "
            "in the refund ledger."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Record a refund requiring human approval "
            "in the refund ledger."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# --------------------------------------------------
# CHECKPOINT HELPER
# --------------------------------------------------

def get_checkpoint_path(run_id):
    """Return the checkpoint file path for a run."""

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Invalid run ID.")

    return CHECKPOINT_DIR / f"{run_id}.json"


def save_checkpoint(state):
    """Save the current agent state after a tool observation."""

    checkpoint_path = get_checkpoint_path(state["run_id"])

    with checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=4)

    print(
        f"Checkpoint saved: {checkpoint_path} "
        f"(step {state['step_number']})"
    )


# --------------------------------------------------
# AGENT CONTINUATION
# --------------------------------------------------

def continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):
    """Continue an agent run from its current saved state."""

    start_time = time.monotonic()

    while state["iterations"] < max_iterations:

        # TIME LIMIT CHECK
        elapsed = time.monotonic() - start_time

        if elapsed >= time_limit_seconds:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": (
                    "FALLBACK_TIMEOUT: Time limit reached. "
                    "Please hand this request to a human support agent."
                ),
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
                "tools_executed": state["tools_executed"],
                "run_id": state["run_id"]
            }

        # CALL THE MOCK LLM
        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1
        state["iterations"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # TOKEN LIMIT CHECK
        if state["tokens_used"] > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": (
                    "FALLBACK_TOKEN_LIMIT: Token budget exceeded. "
                    "Please hand this request to a human support agent."
                ),
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
                "tools_executed": state["tools_executed"],
                "run_id": state["run_id"]
            }

        assistant_message = response.choices[0].message

        # CHECK FOR FINAL ANSWER
        if not assistant_message.tool_calls:

            final_answer = assistant_message.content or ""

            return {
                "status": final_answer.split(":")[0].strip(),
                "final_answer": final_answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
                "tools_executed": state["tools_executed"],
                "run_id": state["run_id"]
            }

        # ADD ASSISTANT TOOL-CALL MESSAGE
        tool_calls = []

        for tool_call in assistant_message.tool_calls:
            tool_calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        })

        # EXECUTE THE REQUESTED TOOLS
        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.name
            arguments = tool_call.arguments

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)

            # Record the completed tool execution
            state["steps"].append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            state["tools_executed"].append(tool_name)
            state["step_number"] += 1

            # Add tool observation to conversation history
            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            # SAVE CHECKPOINT AFTER EVERY TOOL OBSERVATION
            save_checkpoint(state)

            # SIMULATE A CRASH AFTER THE REQUESTED TOOL COUNT
            if (
                crash_after_tools is not None
                and len(state["tools_executed"]) >= crash_after_tools
            ):
                raise RuntimeError(
                    "Simulated crash after tool execution."
                )

    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": (
            "FALLBACK_MAX_ITERATIONS: Maximum model calls reached. "
            "Please hand this request to a human support agent."
        ),
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
        "run_id": state["run_id"]
    }


# --------------------------------------------------
# RUN AGENT
# --------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):
    """Start a new ReAct agent run."""

    if run_id is None:
        run_id = f"{scenario_id}-{uuid.uuid4().hex[:8]}"

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT\n"
                    "You are an Instamart refund support agent. "
                    "Use the available tools to check orders, search policies, "
                    "calculate refunds, issue eligible refunds, or escalate "
                    "high-value refunds to a human."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "steps": [],
        "llm_calls": 0,
        "tokens_used": 0,
        "iterations": 0,
        "step_number": 0,
        "tools_executed": []
    }

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools
    )


# --------------------------------------------------
# RESUME AGENT
# --------------------------------------------------

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    """Load a checkpoint and resume the agent from its saved state."""

    checkpoint_path = get_checkpoint_path(run_id)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run ID: {run_id}"
        )

    with checkpoint_path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    print(f"\nResuming agent run: {run_id}")
    print("Previously completed tools:", state["tools_executed"])

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )


# --------------------------------------------------
# RUN ALL SEVEN SCENARIOS
# --------------------------------------------------

def run_all_scenarios():

    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")

    output_path = Path("outputs/agent_runs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start a fresh scenario results file
    output_path.write_text("", encoding="utf-8")

    results = []

    for _, row in scenarios.iterrows():

        print(f"\nRunning scenario {row['scenario_id']}...")

        result = run_agent(
            user_message=row["message"],
            scenario_id=row["scenario_id"]
        )

        run_record = {
            "scenario_id": row["scenario_id"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"]
        }

        with output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(run_record) + "\n")

        results.append(result)

        print("Status:", result["status"])
        print("Final answer:", result["final_answer"])
        print("Model calls:", result["llm_calls"])
        print("Tokens:", result["tokens_used"])

    return results


# --------------------------------------------------
# TASK 8: CRASH AND RESUME DEMONSTRATION
# --------------------------------------------------

def checkpoint_resume_demo():

    print("\n========== TASK 8: CHECKPOINT AND RESUME ==========")

    run_id = "S1-checkpoint-demo"

    checkpoint_path = get_checkpoint_path(run_id)

    # Remove an old demo checkpoint so this test starts fresh
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    user_message = (
        "My order ORD-100048 reached me very late. "
        "I would like a refund."
    )

    print("\n--- STARTING AGENT ---")

    try:
        run_agent(
            user_message=user_message,
            scenario_id="S1",
            run_id=run_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            crash_after_tools=2
        )

    except RuntimeError as error:
        print("\nSIMULATED CRASH:", error)

    print("\n--- CHECKPOINT DETAILS ---")

    with checkpoint_path.open("r", encoding="utf-8") as file:
        saved_state = json.load(file)

    print("Run ID:", saved_state["run_id"])
    print("Completed steps:", saved_state["step_number"])
    print("Tools already executed:", saved_state["tools_executed"])
    print("Model calls before crash:", saved_state["llm_calls"])

    print("\n--- RESUMING AGENT ---")

    result = resume_agent(
        run_id=run_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10
    )

    print("\n========== FINAL RESUMED RESULT ==========")

    print("Status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Model calls:", result["llm_calls"])
    print("Tokens used:", result["tokens_used"])

    print("\nAll tool executions:")

    for index, tool_name in enumerate(result["tools_executed"], start=1):
        print(f"{index}. {tool_name}")

    print("\nTotal tool executions:", len(result["tools_executed"]))

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund"
    ]

    print("\nExpected tool order:", expected_tools)

    if (
        result["tools_executed"] == expected_tools
        and result["status"] == "REFUND"
        and "128.25" in result["final_answer"]
    ):
        print("\nTASK 8 CHECKPOINT AND RESUME: PASSED")
    else:
        print("\nTASK 8 CHECKPOINT AND RESUME: FAILED")

    return result


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    checkpoint_resume_demo()