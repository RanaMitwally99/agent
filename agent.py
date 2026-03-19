from typing_extensions import List, Optional, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_groq import ChatGroq
import dotenv
import requests
import base64
import random
import time
import os

dotenv.load_dotenv()

# ── LLM (Groq - free & fast) ───────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

DEAPI_KEY = os.getenv("DEAPI_API_KEY")
DEAPI_BASE = "https://api.deapi.ai/api/v1/client"


class ImageState(TypedDict):
    user_input: Optional[str]
    messages: List[AnyMessage]
    story: Optional[str]
    generation_output: Optional[str]  # base64 data URI
    generate_image: Optional[bool]    # orchestra decision


# ── Get first available txt2img model slug ─────────────────────
def get_image_model_slug() -> Optional[str]:
    try:
        headers = {"Authorization": f"Bearer {DEAPI_KEY}", "Accept": "application/json"}
        resp = requests.get(
            f"{DEAPI_BASE}/models?filter[inference_types]=txt2img",
            headers=headers,
            timeout=15
        )
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                slug = models[0]["slug"]
                print(f"Using model: {slug}")
                return slug
    except Exception as e:
        print(f"Failed to fetch models: {e}")
    return None


# ── Node 1: Orchestra ──────────────────────────────────────────
orchestra_sys = """
You are an orchestration agent. Your job is to analyze the user's request and decide:
1. Should a story be written?
2. Should an image be generated?

Always generate an image when the user asks for a story.
Reply ONLY in this exact format:
STORY: yes/no
IMAGE: yes/no
"""

def orchestra(state: ImageState) -> ImageState:
    response = llm.invoke([
        SystemMessage(content=orchestra_sys),
        HumanMessage(content=state["user_input"])
    ])
    content = response.content.upper()

    story_keywords = ["story", "tell", "write", "tale", "once", "حكاية", "قصة", "اكتب", "احكي"]
    user_lower = state["user_input"].lower()
    has_story_keyword = any(kw in user_lower for kw in story_keywords)

    state["generate_image"] = "IMAGE: YES" in content or has_story_keyword
    return state


# ── Node 2: Story Writer ───────────────────────────────────────
story_sys = """
You are a creative story teller.
Write a short, vivid, engaging story (4-6 sentences) based on the user's request.
"""

def story_writer(state: ImageState) -> ImageState:
    if not state.get("messages"):
        state["messages"] = [
            SystemMessage(content=story_sys),
            HumanMessage(content=state["user_input"])
        ]
    else:
        state["messages"] = state["messages"] + [
            HumanMessage(content=state["user_input"])
        ]

    response = llm.invoke(state["messages"])
    state["messages"] = state["messages"] + [response]

    story_text = ""
    if isinstance(response.content, str):
        story_text = response.content.strip()
    elif isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                story_text = block.get("text", "").strip()
                break

    state["story"] = story_text
    return state


# ── Node 3: Image Generation (deAPI) ──────────────────────────
def image_generator(state: ImageState) -> ImageState:
    if not state.get("generate_image"):
        state["generation_output"] = None
        return state

    prompt = state.get("story") or state.get("user_input") or "a magical story scene"
    prompt_short = prompt[:300].replace("\n", " ").strip()

    headers = {
        "Authorization": f"Bearer {DEAPI_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        # Get the correct model slug dynamically
        model_slug = get_image_model_slug()
        if not model_slug:
            print("No image model available.")
            state["generation_output"] = None
            return state

        # Step 1: Submit the image generation job
        payload = {
            "prompt": prompt_short,
            "model": model_slug,
            "width": 512,
            "height": 512,
            "steps": 4,
            "seed": random.randint(1, 99999)
        }

        print(f"Submitting with payload: {payload}")

        submit_resp = requests.post(
            f"{DEAPI_BASE}/txt2img",
            json=payload,
            headers=headers,
            timeout=30
        )

        if submit_resp.status_code != 200:
            print(f"deAPI submit error: {submit_resp.status_code} - {submit_resp.text}")
            state["generation_output"] = None
            return state

        request_id = submit_resp.json()["data"]["request_id"]
        print(f"deAPI job submitted: {request_id}")

        # Step 2: Poll for result (max 60 seconds)
        result_url = None
        for _ in range(20):
            time.sleep(3)
            status_resp = requests.get(
                f"{DEAPI_BASE}/request-status/{request_id}",
                headers=headers,
                timeout=15
            )

            if status_resp.status_code != 200:
                continue

            status_data = status_resp.json()
            job_status = status_data.get("data", {}).get("status")
            print(f"deAPI job status: {job_status}")

            if job_status == "done":
                result_url = status_data["data"].get("result_url")
                break
            elif job_status == "error":
                print("deAPI job failed.")
                break

        # Step 3: Download image and convert to base64
        if result_url:
            img_resp = requests.get(result_url, timeout=30)
            if img_resp.status_code == 200:
                img_b64 = base64.b64encode(img_resp.content).decode("utf-8")
                state["generation_output"] = f"data:image/jpeg;base64,{img_b64}"
            else:
                state["generation_output"] = None
        else:
            state["generation_output"] = None

    except Exception as e:
        print(f"Image generation failed: {e}")
        state["generation_output"] = None

    return state


# ── Routing function ───────────────────────────────────────────
def routing(state: ImageState) -> str:
    if state.get("generate_image"):
        return "generate_image"
    return "skip_image"