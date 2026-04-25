import logging

from google import genai
from google.genai import types

from agent.prompts import SYSTEM_PROMPT
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class QAAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_iterations: int = 5,
    ) -> None:
        self._registry = registry
        self._model = model
        self._max_iterations = max_iterations
        self._client = genai.Client(api_key=api_key)
        logger.info(
            "QAAgent initialized with model='%s' max_iter=%d", model, max_iterations
        )

    def answer(self, question: str) -> str:
        """
        Responde uma pergunta usando o loop de function calling.

        Fluxo:
        1. Constrói a mensagem inicial com a pergunta.
        2. Envia ao Gemini com os schemas das tools.
        3. Se há function_calls → executa cada uma → adiciona resultados ao histórico.
        4. Reenvia ao Gemini com os resultados.
        5. Repete até resposta final ou max_iterations.
        """
        logger.info("QAAgent.answer: question='%s'", question)

        tool_declarations = self._registry.get_gemini_schemas()
        tools = [types.Tool(function_declarations=tool_declarations)]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
        )

        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part(text=question)],
            )
        ]

        for iteration in range(1, self._max_iterations + 1):
            logger.debug("Agent iteration %d/%d", iteration, self._max_iterations)

            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            candidate = response.candidates[0]
            assistant_parts: list[types.Part] = list(candidate.content.parts)

            function_calls = [p for p in assistant_parts if p.function_call is not None]
            text_parts = [p for p in assistant_parts if p.text]

            contents.append(candidate.content)

            if not function_calls:
                final_text = "\n".join(p.text for p in text_parts if p.text).strip()
                logger.info(
                    "Agent produced final answer after %d iteration(s).", iteration
                )
                return final_text or "Não foi possível gerar uma resposta."

            tool_result_parts: list[types.Part] = []

            for fc_part in function_calls:
                fc = fc_part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                logger.info(
                    "Agent calling tool '%s' with args: %s", tool_name, tool_args
                )

                tool_result = self._registry.dispatch(tool_name, tool_args)
                result_str = tool_result.to_agent_str()

                logger.debug(
                    "Tool '%s' result (truncated): %s", tool_name, result_str[:200]
                )

                tool_result_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name,
                            response={"result": result_str},
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=tool_result_parts))

        logger.warning(
            "Agent reached max_iterations (%d) without final answer.",
            self._max_iterations,
        )
        return "Limite de iterações atingido. Tente reformular a pergunta."
