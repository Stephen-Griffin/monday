import asyncio
import base64
import os
from collections.abc import Awaitable, Callable

from google import genai
from google.genai import types


SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
DEFAULT_COMPUTER_USE_MODEL = "gemini-2.5-computer-use-preview-10-2025"


class PlaywrightWebAgent:
    """Gemini Computer Use agent backed by Playwright Chromium."""

    def __init__(
        self,
        api_key: str,
        model_id: str | None = None,
        headless: bool = True,
    ) -> None:
        self.client = genai.Client(http_options={"api_version": "v1beta"}, api_key=api_key)
        self.model_id = model_id or os.getenv("GEMINI_COMPUTER_USE_MODEL", DEFAULT_COMPUTER_USE_MODEL)
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None

    @staticmethod
    def _denormalize_x(x: int, width: int) -> int:
        return int((x / 1000) * width)

    @staticmethod
    def _denormalize_y(y: int, height: int) -> int:
        return int((y / 1000) * height)

    async def _maybe_update(
        self,
        update_callback: Callable[[str | None, str], Awaitable[None]] | None,
        image_bytes: bytes | None,
        log: str,
    ) -> None:
        if not update_callback:
            return
        image_b64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else None
        await update_callback(image_b64, log)

    async def execute_function_calls(self, function_calls):
        results = []

        for call in function_calls:
            call_id = getattr(call, "id", None)
            fn_name = getattr(call, "name", "")
            raw_args = getattr(call, "args", {}) or {}
            args = dict(raw_args) if not isinstance(raw_args, dict) else raw_args
            print(f"[web-agent] action={fn_name} args={args}")

            requires_acknowledgement = False
            safety = args.get("safety_decision")
            if isinstance(safety, dict) and safety.get("decision") == "require_confirmation":
                print(f"[web-agent] safety: {safety.get('explanation')}")
                requires_acknowledgement = True

            result_data = {}

            try:
                if fn_name == "open_web_browser":
                    # Browser is already open and managed by Playwright.
                    pass
                elif fn_name == "navigate":
                    await self.page.goto(args["url"])
                elif fn_name == "go_back":
                    await self.page.go_back()
                elif fn_name == "go_forward":
                    await self.page.go_forward()
                elif fn_name == "search":
                    await self.page.goto("https://www.google.com")
                elif fn_name == "wait_5_seconds":
                    await asyncio.sleep(5)
                elif fn_name == "click_at":
                    x = self._denormalize_x(int(args["x"]), SCREEN_WIDTH)
                    y = self._denormalize_y(int(args["y"]), SCREEN_HEIGHT)
                    await self.page.mouse.click(x, y)
                elif fn_name == "type_text_at":
                    x = self._denormalize_x(int(args["x"]), SCREEN_WIDTH)
                    y = self._denormalize_y(int(args["y"]), SCREEN_HEIGHT)
                    text = args.get("text", "")
                    press_enter = bool(args.get("press_enter", False))
                    clear_before = bool(args.get("clear_before_typing", True))

                    await self.page.mouse.click(x, y)
                    if clear_before:
                        await self.page.keyboard.press("Control+A")
                        await self.page.keyboard.press("Backspace")
                    await self.page.keyboard.type(text)
                    if press_enter:
                        await self.page.keyboard.press("Enter")
                elif fn_name == "hover_at":
                    x = self._denormalize_x(int(args["x"]), SCREEN_WIDTH)
                    y = self._denormalize_y(int(args["y"]), SCREEN_HEIGHT)
                    await self.page.mouse.move(x, y)
                elif fn_name == "drag_and_drop":
                    start_x = self._denormalize_x(int(args["x"]), SCREEN_WIDTH)
                    start_y = self._denormalize_y(int(args["y"]), SCREEN_HEIGHT)
                    end_x = self._denormalize_x(int(args["destination_x"]), SCREEN_WIDTH)
                    end_y = self._denormalize_y(int(args["destination_y"]), SCREEN_HEIGHT)
                    await self.page.mouse.move(start_x, start_y)
                    await self.page.mouse.down()
                    await self.page.mouse.move(end_x, end_y)
                    await self.page.mouse.up()
                elif fn_name == "key_combination":
                    keys = args.get("keys")
                    if keys:
                        await self.page.keyboard.press(keys)
                elif fn_name in {"scroll_document", "scroll_at"}:
                    magnitude = int(args.get("magnitude", 800))
                    direction = str(args.get("direction", "down")).lower()

                    if fn_name == "scroll_at":
                        x = self._denormalize_x(int(args["x"]), SCREEN_WIDTH)
                        y = self._denormalize_y(int(args["y"]), SCREEN_HEIGHT)
                        await self.page.mouse.move(x, y)

                    dx, dy = 0, 0
                    if direction == "down":
                        dy = magnitude
                    elif direction == "up":
                        dy = -magnitude
                    elif direction == "right":
                        dx = magnitude
                    elif direction == "left":
                        dx = -magnitude
                    await self.page.mouse.wheel(dx, dy)
                else:
                    result_data = {"warning": f"Unimplemented function: {fn_name}"}

                await asyncio.sleep(1)
            except Exception as exc:
                print(f"[web-agent] action error {fn_name}: {exc}")
                result_data = {"error": str(exc)}

            if requires_acknowledgement:
                result_data["safety_acknowledgement"] = True
            results.append((call_id, fn_name, result_data))

        return results

    async def _function_responses_from_results(self, results):
        screenshot_bytes = await self.page.screenshot(type="png")
        current_url = self.page.url

        function_responses = []
        for call_id, name, result in results:
            response_data = {"url": current_url}
            response_data.update(result or {})

            # SDK compatibility: attach screenshot parts when supported; otherwise send plain response.
            if hasattr(types, "FunctionResponsePart") and hasattr(types, "FunctionResponseBlob"):
                function_responses.append(
                    types.FunctionResponse(
                        name=name,
                        id=call_id,
                        response=response_data,
                        parts=[
                            types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type="image/png",
                                    data=screenshot_bytes,
                                )
                            )
                        ],
                    )
                )
            else:
                function_responses.append(
                    types.FunctionResponse(name=name, id=call_id, response=response_data)
                )
        return function_responses, screenshot_bytes

    async def run_task(
        self,
        prompt: str,
        update_callback: Callable[[str | None, str], Awaitable[None]] | None = None,
        max_turns: int = 20,
    ) -> str:
        from playwright.async_api import async_playwright

        final_response = "Web agent finished without a final summary."
        print(f"[web-agent] start prompt={prompt}")

        async with async_playwright() as p:
            self.browser = await p.chromium.launch(headless=self.headless)
            self.context = await self.browser.new_context(
                viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            self.page = await self.context.new_page()
            await self.page.goto("https://www.google.com")

            environment_enum = getattr(getattr(types, "Environment", None), "ENVIRONMENT_BROWSER", None)
            computer_use_kwargs = {"environment": environment_enum} if environment_enum is not None else {}
            config = types.GenerateContentConfig(
                tools=[types.Tool(computer_use=types.ComputerUse(**computer_use_kwargs))],
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            )

            initial_screenshot = await self.page.screenshot(type="png")
            await self._maybe_update(update_callback, initial_screenshot, "Web agent initialized")

            chat_history = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part.from_bytes(data=initial_screenshot, mime_type="image/png"),
                    ],
                )
            ]

            for turn in range(max_turns):
                print(f"[web-agent] turn={turn + 1}")
                response = await self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=chat_history,
                    config=config,
                )

                candidates = getattr(response, "candidates", None) or []
                if not candidates:
                    print("[web-agent] model returned no candidates")
                    break

                model_content = getattr(candidates[0], "content", None)
                if not model_content:
                    print("[web-agent] candidate missing content")
                    break

                chat_history.append(model_content)

                function_calls = []
                for part in getattr(model_content, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        final_response = part_text
                        if getattr(part, "thought", False):
                            print(f"[web-agent][thought] {part_text}")
                        else:
                            print(f"[web-agent][agent] {part_text}")
                    function_call = getattr(part, "function_call", None)
                    if function_call:
                        function_calls.append(function_call)

                if not function_calls:
                    print("[web-agent] complete (no function calls)")
                    await self._maybe_update(update_callback, None, "Web agent finished")
                    break

                results = await self.execute_function_calls(function_calls)
                function_responses, screenshot_bytes = await self._function_responses_from_results(results)
                action_names = ", ".join(name for _, name, _ in results) or "no actions"
                await self._maybe_update(update_callback, screenshot_bytes, f"Executed: {action_names}")
                chat_history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(function_response=fr) for fr in function_responses],
                    )
                )

            await self.browser.close()
            self.browser = None
            self.context = None
            self.page = None
            print("[web-agent] browser closed")
            return final_response
