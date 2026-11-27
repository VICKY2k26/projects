import os
import time
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")

if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

client = genai.Client(api_key=API_KEY)
app = FastAPI(title="Week 1 Day 5 — Tool-Calling Support API")

MOCK_ORDERS = {
    "ORD-101": {"status": "Delivered", "item": "Mechanical Keyboard", "delivered_on": "2026-11-24"},
    "ORD-102": {"status": "Shipped", "item": "27-inch 4K Monitor", "eta": "2026-11-29", "carrier": "BlueDart"},
    "ORD-103": {"status": "Processing", "item": "USB-C Docking Station", "eta": "2026-12-01"},
    "ORD-104": {"status": "Cancelled", "item": "Wireless Mouse", "refund_status": "Refunded to card"},
}


def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up the live shipping and fulfillment status of a customer order by its order ID (e.g., ORD-101)."""
    clean_id = order_id.strip().upper()
    if clean_id in MOCK_ORDERS:
        return {"found": True, "order_id": clean_id, **MOCK_ORDERS[clean_id]}
    return {"found": False, "order_id": clean_id, "message": "Order ID not found in database."}


TOOL_REGISTRY = {
    "get_order_status": get_order_status,
}

SYSTEM_PROMPT = """
You are a helpful e-commerce customer support assistant.
- Use the `get_order_status` tool whenever a user asks about a specific order ID.
- Never guess or invent order statuses. If the tool returns found=False, politely tell the customer the order ID was not found.
- For general questions that do not require an order lookup, answer concisely without calling any tool.
"""

BASE_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.0,
    tools=[get_order_status],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Customer question")


class ToolExecutionDetail(BaseModel):
    tool_called: str
    tool_args: dict[str, Any]
    tool_result: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    tool_called: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    all_tool_calls: list[ToolExecutionDetail] = []
    latency_ms: float


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((APIError, ConnectionError, TimeoutError)),
    reraise=True,
)
def _call_gemini(contents: list[types.Content]) -> types.GenerateContentResponse:
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=BASE_CONFIG,
    )


def generate_with_tools(contents: list[types.Content]) -> types.GenerateContentResponse:
    try:
        return _call_gemini(contents)
    except Exception as err:
        raise HTTPException(status_code=502, detail=f"LLM service error: {err}") from err


def execute_tool_safely(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unsupported tool requested by model: {tool_name}")

    if tool_name == "get_order_status":
        order_id = tool_args.get("order_id")
        if not isinstance(order_id, str) or not order_id.strip():
            return {"found": False, "message": "Invalid or missing order_id parameter."}
        return get_order_status(order_id=order_id)

    try:
        return TOOL_REGISTRY[tool_name](**tool_args)
    except TypeError as err:
        return {"error": f"Invalid arguments for {tool_name}: {err}"}


@app.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/ask", response_model=AskResponse)
def ask_assistant(payload: AskRequest):
    start = time.perf_counter()
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=payload.question)])
    ]

    first_response = generate_with_tools(contents)

    if first_response.function_calls:
        if not first_response.candidates or first_response.candidates[0].content is None:
            raise HTTPException(status_code=502, detail="Model returned empty candidate content.")

        contents.append(first_response.candidates[0].content)
        response_parts: list[types.Part] = []
        executed_tools: list[ToolExecutionDetail] = []

        for fc in first_response.function_calls:
            tool_name = fc.name or "unknown_tool"
            tool_args = dict(fc.args) if fc.args else {}
            tool_output = execute_tool_safely(tool_name, tool_args)

            executed_tools.append(
                ToolExecutionDetail(
                    tool_called=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_output,
                )
            )
            response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": tool_output},
                )
            )

        contents.append(types.Content(role="user", parts=response_parts))
        second_response = generate_with_tools(contents)
        final_answer = (second_response.text or "Sorry, I could not generate a response.").strip()
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        first_call = executed_tools[0]
        return AskResponse(
            answer=final_answer,
            tool_called=first_call.tool_called,
            tool_args=first_call.tool_args,
            tool_result=first_call.tool_result,
            all_tool_calls=executed_tools,
            latency_ms=latency_ms,
        )

    final_answer = (first_response.text or "Sorry, I could not generate a response.").strip()
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return AskResponse(
        answer=final_answer,
        tool_called=None,
        tool_args=None,
        tool_result=None,
        all_tool_calls=[],
        latency_ms=latency_ms,
    )