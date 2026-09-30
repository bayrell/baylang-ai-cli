"""BayLang AI — библиотека для построения AI-агентов поверх OpenRouter.

Модуль содержит базовые абстракции библиотеки:

* :class:`TextMessage` — сообщение диалога в формате OpenAI-совместимого API;
* :class:`Context` — диалоговый контекст с ограничением длины;
* :class:`Provider` / :class:`OpenRouterProvider` — асинхронные провайдеры LLM;
* :class:`Tool` / :class:`ToolRegistry` — подсистема инструментов (function);
* :class:`Agent` — агент с циклом ReAct и fallback-цепочкой провайдеров.

Все сетевые операции выполняются асинхронно через ``httpx``; блокирующие
вызовы в асинхронном контексте не используются.
"""

from __future__ import annotations

import abc
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "BayLangError",
    "ConfigurationError",
    "ProviderError",
    "ToolError",
    "TextMessage",
    "Context",
    "ProviderResponse",
    "Provider",
    "OpenRouterProvider",
    "Tool",
    "ToolRegistry",
    "Agent",
]


# ---------------------------------------------------------------------------
# Иерархия исключений
# ---------------------------------------------------------------------------


class BayLangError(Exception):
    """Базовое исключение библиотеки BayLang AI."""


class ConfigurationError(BayLangError):
    """Ошибка конфигурации: отсутствуют или некорректны обязательные параметры."""


class ProviderError(BayLangError):
    """Ошибка взаимодействия с LLM-провайдером.

    Attributes:
        status_code: HTTP-код ответа или ``None`` при сетевом сбое.
        body: Фрагмент тела ответа для диагностики.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ToolError(BayLangError):
    """Ошибка выполнения инструмента агента.

    Attributes:
        tool_name: Имя инструмента, при выполнении которого произошла ошибка.
    """

    def __init__(self, message: str, tool_name: str = "") -> None:
        super().__init__(message)
        self.tool_name = tool_name


# ---------------------------------------------------------------------------
# Сообщения и контекст
# ---------------------------------------------------------------------------


class TextMessage:
    """Текстовое сообщение диалога в формате OpenAI-совместимого API.

    Attributes:
        role: Роль отправителя (``user``, ``assistant``, ``system`` или ``tool``).
        content: Текст сообщения.
        tool_id: Идентификатор вызова инструмента (для роли ``tool``).
        tools: Список вызовов инструментов (для сообщений ассистента).
    """

    ROLE_USER = "user"
    ROLE_AI = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_TOOL = "tool"

    VALID_ROLES = frozenset({ROLE_USER, ROLE_AI, ROLE_SYSTEM, ROLE_TOOL})

    def __init__(
        self,
        role: str,
        content: str = "",
        *,
        tool_id: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"Недопустимая роль сообщения: {role!r}. Допустимы: {sorted(self.VALID_ROLES)}"
            )
        if not isinstance(content, str):
            raise ValueError("Содержимое сообщения должно быть строкой")
        if tools is None and not content.strip():
            raise ValueError("Содержимое сообщения должно быть непустой строкой")
        self.role = role
        self.content = content
        self.tool_id = tool_id
        self.tools = tools

    def get_data(self) -> dict[str, Any]:
        """Сериализовать сообщение в формате провайдера.

        Returns:
            Словарь вида ``{"role": ..., "content": ...}``; для сообщений
            с tool добавляются поля ``tools`` и ``tool_id``.
        """
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_id is not None:
            data["tool_id"] = self.tool_id
        if self.tools:
            data["tools"] = self.tools
        return data

    @classmethod
    def user(cls, text: str) -> "TextMessage":
        """Создать сообщение пользователя."""
        return cls(cls.ROLE_USER, text)

    @classmethod
    def system(cls, text: str) -> "TextMessage":
        """Создать системное сообщение."""
        return cls(cls.ROLE_SYSTEM, text)

    @classmethod
    def assistant(cls, text: str) -> "TextMessage":
        """Создать сообщение ассистента."""
        return cls(cls.ROLE_AI, text)

    @classmethod
    def tool_result(cls, content: str, tool_id: str) -> "TextMessage":
        """Создать сообщение с результатом выполнения инструмента."""
        return cls(cls.ROLE_TOOL, content, tool_id=tool_id)

    def __repr__(self) -> str:
        preview = self.content if len(self.content) <= 40 else self.content[:37] + "..."
        return f"TextMessage(role={self.role!r}, content={preview!r})"


class Context:
    """Диалоговый контекст: упорядоченная история сообщений.

    Хранилище приватное; внешний доступ к списку сообщений возможен только
    через методы :meth:`add_message`, :meth:`get_data`, :meth:`trim`,
    :meth:`clear`, :meth:`truncate`, :meth:`last_response`, а также итерацию
    и ``len()``.

    Args:
        max_messages: Максимальное число хранимых сообщений. При превышении
            старые сообщения автоматически отсекаются (:meth:`trim`).
    """

    def __init__(self, max_messages: Optional[int] = None) -> None:
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages должен быть не меньше 1")
        self._items: list[TextMessage] = []
        self._max_messages = max_messages

    def add_message(self, message: TextMessage) -> None:
        """Добавить сообщение в конец контекста.

        Args:
            message: Объект :class:`TextMessage`.

        Raises:
            TypeError: Если передан не ``TextMessage``.
        """
        if not isinstance(message, TextMessage):
            raise TypeError("В контекст можно добавлять только объекты TextMessage")
        self._items.append(message)
        if self._max_messages is not None and len(self._items) > self._max_messages:
            self.trim(self._max_messages)

    def get_data(self) -> list[dict[str, Any]]:
        """Получить список сообщений в формате провайдера.

        Returns:
            Список словарей ``{"role": ..., "content": ...}``, готовых
            к передаче в поле ``"messages"`` запроса к LLM.
        """
        return [item.get_data() for item in self._items]

    def trim(self, max_messages: int) -> None:
        """Отсечь старые сообщения, оставив последние ``max_messages``.

        Args:
            max_messages: Сколько последних сообщений сохранить.

        Raises:
            ValueError: Если ``max_messages`` отрицателен.
        """
        if max_messages < 0:
            raise ValueError("max_messages не может быть отрицательным")
        if len(self._items) > max_messages:
            del self._items[:-max_messages]

    def truncate(self, size: int) -> None:
        """Оставить только первые ``size`` сообщений.

        Используется для отката контекста при ошибках провайдера.

        Args:
            size: Желаемый размер контекста.

        Raises:
            ValueError: Если ``size`` отрицателен.
        """
        if size < 0:
            raise ValueError("size не может быть отрицательным")
        if len(self._items) > size:
            del self._items[size:]

    def clear(self) -> None:
        """Полностью очистить контекст."""
        self._items.clear()

    def last_response(self) -> Optional[TextMessage]:
        """Вернуть последнее сообщение роли ``assistant``.

        Returns:
            Сообщение ассистента или ``None``, если ответов ещё не было.
        """
        for item in reversed(self._items):
            if item.role == TextMessage.ROLE_AI:
                return item
        return None

    def __iter__(self):
        """Итерировать по копии списка сообщений."""
        return iter(list(self._items))

    def __len__(self) -> int:
        """Вернуть количество сообщений в контексте."""
        return len(self._items)

    def __repr__(self) -> str:
        return f"Context(messages={len(self._items)})"


# ---------------------------------------------------------------------------
# Провайдеры
# ---------------------------------------------------------------------------


@dataclass
class ProviderResponse:
    """Нормализованный ответ LLM-провайдера.

    Attributes:
        text: Текст ответа модели.
        model: Имя модели, вернувшей ответ.
        usage: Использование токенов (``prompt_tokens`` / ``completion_tokens``).
        raw: Исходный ответ API для отладки.
        tools: Вызовы инструментов из ответа модели (может быть пустым).
    """

    text: str
    model: str
    usage: Optional[dict[str, int]] = None
    raw: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)


class Provider(abc.ABC):
    """Абстрактный LLM-провайдер.

    Наследник обязан реализовать ``get_url``, ``get_model_name``,
    ``get_api_key``, :meth:`send` и :meth:`send_stream`.

    Args:
        model_name: Имя модели по умолчанию.
        api_key: API-ключ провайдера.
        timeout: Таймаут HTTP-запроса в секундах.
        temperature: Температура генерации.
    """

    def __init__(
        self,
        model_name: str = "",
        api_key: str = "",
        timeout: float = 60.0,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    @abc.abstractmethod
    def get_url(self) -> str:
        """Вернуть полный URL chat-completions эндпоинта провайдера."""

    @abc.abstractmethod
    def get_model_name(self) -> str:
        """Вернуть имя используемой модели."""

    @abc.abstractmethod
    def get_api_key(self) -> str:
        """Вернуть API-ключ провайдера."""

    @abc.abstractmethod
    async def send(
        self, context: Context, tools: Optional[list[dict[str, Any]]] = None
    ) -> ProviderResponse:
        """Отправить контекст в модель и вернуть нормализованный ответ.

        Args:
            context: Диалоговый контекст.
            tools: Схемы инструментов в формате OpenAI Function.

        Returns:
            Объект :class:`ProviderResponse`.

        Raises:
            ProviderError: При сетевом сбое или ошибке HTTP.
        """

    @abc.abstractmethod
    async def send_stream(
        self, context: Context, tools: Optional[list[dict[str, Any]]] = None
    ) -> AsyncIterator[str]:
        """Отправить контекст и отдавать чанки текста по мере генерации.

        Args:
            context: Диалоговый контекст.
            tools: Схемы инструментов в формате OpenAI Function.

        Yields:
            Фрагменты текста ответа модели.
        """


class OpenRouterProvider(Provider):
    """Провайдер для OpenRouter (OpenAI-совместимый chat-completions API).

    Args:
        api_key: Ключ OpenRouter (обязателен).
        model_name: Имя модели; по умолчанию ``openai/gpt-4o-mini``.
        url: URL API; по умолчанию ``https://openrouter.ai/api/v1/chat/completions``.
        timeout: Таймаут HTTP-запроса в секундах.
        temperature: Температура генерации.
        referer: Заголовок ``HTTP-Referer`` для аналитики OpenRouter.
        title: Заголовок ``X-Title`` для аналитики OpenRouter.

    Raises:
        ConfigurationError: Если ``api_key`` пуст.
    """

    DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    DEFAULT_REFERER = "https://github.com/baylang/baylang-ai"
    DEFAULT_TITLE = "BayLang AI"

    def __init__(
        self,
        api_key: str = "",
        model_name: str = DEFAULT_MODEL,
        *,
        url: str = DEFAULT_URL,
        timeout: float = 60.0,
        temperature: float = 0.7,
        referer: str = DEFAULT_REFERER,
        title: str = DEFAULT_TITLE,
    ) -> None:
        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
        )
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError("Для OpenRouterProvider требуется непустой api_key")
        self._url = url or self.DEFAULT_URL
        self.referer = referer
        self.title = title
        self._client: Optional[httpx.AsyncClient] = None

    def get_url(self) -> str:
        """Вернуть URL chat-completions эндпоинта OpenRouter."""
        return self._url

    def get_model_name(self) -> str:
        """Вернуть имя модели OpenRouter."""
        return self.model_name

    def get_api_key(self) -> str:
        """Вернуть API-ключ OpenRouter."""
        return self.api_key

    def _build_headers(self) -> dict[str, str]:
        """Собрать HTTP-заголовки запроса (ключ API не логируется)."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

    def _build_payload(
        self, context: Context, tools: Optional[list[dict[str, Any]]] = None, stream: bool = False
    ) -> dict[str, Any]:
        """Собрать тело запроса в формате OpenAI chat-completions."""
        payload: dict[str, Any] = {
            "model": self.get_model_name(),
            "messages": context.get_data(),
            "temperature": self.temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload

    def _parse_response(self, raw: dict[str, Any]) -> ProviderResponse:
        """Нормализовать сырой ответ API в объект ProviderResponse.

        Raises:
            ProviderError: Если в ответе нет поля ``choices``.
        """
        choices = raw.get("choices")
        if not choices:
            raise ProviderError(
                "Ответ провайдера не содержит поля choices",
                body=json.dumps(raw, ensure_ascii=False)[:500],
            )
        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        tools = message.get("tool_calls")
        usage = raw.get("usage")
        return ProviderResponse(
            text=text,
            model=raw.get("model") or self.get_model_name(),
            usage=usage if isinstance(usage, dict) else None,
            raw=raw,
            tools=tools if isinstance(tools, list) else [],
        )

    def _get_client(self) -> httpx.AsyncClient:
        """Вернуть (или создать) переиспользуемый асинхронный HTTP-клиент."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Выполнить POST-запрос и вернуть разобранный JSON-ответ.

        Raises:
            ProviderError: При таймауте, сетевой ошибке, HTTP 4xx/5xx
                или некорректном теле ответа.
        """
        client = self._get_client()
        try:
            response = await client.post(
                self.get_url(), headers=self._build_headers(), json=payload
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Таймаут запроса к {self.get_url()} (>{self.timeout} с)"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Сетевая ошибка при обращении к {self.get_url()}: {exc}"
            ) from exc

        if response.status_code >= 400:
            body = response.text[:1000]
            raise ProviderError(
                f"Провайдер вернул ошибку HTTP {response.status_code}",
                status_code=response.status_code,
                body=body,
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "Провайдер вернул некорректный JSON", body=response.text[:500]
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(
                "Провайдер вернул ответ неожидаемой структуры", body=str(data)[:500]
            )
        return data

    async def send(
        self, context: Context, tools: Optional[list[dict[str, Any]]] = None
    ) -> ProviderResponse:
        """Отправить контекст в OpenRouter и вернуть нормализованный ответ.

        Args:
            context: Диалоговый контекст.
            tools: Схемы инструментов в формате OpenAI Function.

        Returns:
            Объект :class:`ProviderResponse`.

        Raises:
            ProviderError: При сетевом сбое или ошибке HTTP.
        """
        payload = self._build_payload(context, tools=tools, stream=False)
        raw = await self._post_json(payload)
        return self._parse_response(raw)

    async def send_stream(
        self, context: Context, tools: Optional[list[dict[str, Any]]] = None
    ) -> AsyncIterator[str]:
        """Отправить контекст в OpenRouter и отдавать чанки текста (SSE).

        Разбирает строки вида ``data: {...}`` и завершающий ``data: [DONE]``.

        Args:
            context: Диалоговый контекст.
            tools: Схемы инструментов в формате OpenAI Function.

        Yields:
            Фрагменты текста ответа модели.

        Raises:
            ProviderError: При таймауте, сетевой ошибке или HTTP 4xx/5xx.
        """
        payload = self._build_payload(context, tools=tools, stream=True)
        client = self._get_client()
        try:
            async with client.stream(
                "POST", self.get_url(), headers=self._build_headers(), json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")[:1000]
                    raise ProviderError(
                        f"Провайдер вернул ошибку HTTP {response.status_code}",
                        status_code=response.status_code,
                        body=body,
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Не удалось разобрать SSE-чанк: %s", data[:200])
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Таймаут стримингового запроса к {self.get_url()} (>{self.timeout} с)"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Сетевая ошибка при обращении к {self.get_url()}: {exc}"
            ) from exc

    async def aclose(self) -> None:
        """Закрыть HTTP-клиент и освободить соединения."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def __repr__(self) -> str:
        return f"OpenRouterProvider(model={self.get_model_name()!r}, url={self.get_url()!r})"


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------


class Tool:
    """Инструмент агента: обёртка над функцией с JSON-схемой параметров.

    Обработчик вызывается с именованными аргументами (``handler(**params)``)
    и может быть как синхронной, так и асинхронной функцией.

    Args:
        name: Уникальное имя инструмента (латиницей, без пробелов).
        description: Описание для модели.
        parameters_schema: JSON-схема параметров в формате OpenAI Function.
            По умолчанию — пустая схема ``{"type": "object"}``.
        handler: Реализация инструмента. Если ``None``, :meth:`execute`
            бросит :class:`ToolError`.

    Raises:
        ConfigurationError: Если ``name`` или ``description`` пусты.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: Optional[dict[str, Any]] = None,
        handler: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("Инструмент должен иметь непустое имя (name)")
        if not isinstance(description, str) or not description.strip():
            raise ConfigurationError(f"Инструмент {name!r} должен иметь непустое описание")
        self.name = name.strip()
        self.description = description.strip()
        if parameters_schema is None:
            self.parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}
        elif isinstance(parameters_schema, dict):
            self.parameters_schema = parameters_schema
        else:
            raise ConfigurationError(
                f"parameters_schema инструмента {self.name!r} должен быть словарём"
            )
        self.handler = handler

    def get_schema(self) -> dict[str, Any]:
        """Вернуть схему инструмента в формате OpenAI Function.

        Returns:
            Словарь ``{"type": "function", "function": {...}}``.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    async def execute(self, params: Optional[dict[str, Any]] = None) -> Any:
        """Выполнить инструмент с переданными параметрами.

        Args:
            params: Именованные параметры вызова.

        Returns:
            Результат обработчика: строка или сериализуемый объект.

        Raises:
            ToolError: Если обработчик отсутствует, параметры некорректны
                или обработчик завершился исключением.
        """
        if self.handler is None:
            raise ToolError(f"Инструмент {self.name!r} не имеет обработчика", tool_name=self.name)
        arguments = params if params is not None else {}
        if not isinstance(arguments, dict):
            raise ToolError(
                f"Параметры инструмента {self.name!r} должны быть словарём",
                tool_name=self.name,
            )
        try:
            result = self.handler(**arguments)
            if inspect.isawaitable(result):
                result = await result
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                f"Ошибка выполнения инструмента {self.name!r}: {exc}", tool_name=self.name
            ) from exc
        return result

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


class ToolRegistry:
    """Реестр инструментов агента.

    Хранит инструменты по уникальному имени, позволяет строить схемы
    для запроса к модели и выполнять вызовы по имени.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Зарегистрировать инструмент.

        Args:
            tool: Объект :class:`Tool`.

        Raises:
            TypeError: Если передан не ``Tool``.
            ConfigurationError: Если инструмент с таким именем уже
                зарегистрирован.
        """
        if not isinstance(tool, Tool):
            raise TypeError("В реестр можно добавлять только объекты Tool")
        if tool.name in self._tools:
            raise ConfigurationError(f"Инструмент {tool.name!r} уже зарегистрирован")
        self._tools[tool.name] = tool

    def find(self, name: str) -> Optional[Tool]:
        """Найти инструмент по имени.

        Returns:
            Инструмент или ``None``, если не найден.
        """
        return self._tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Вернуть схемы всех инструментов для передачи в запрос к модели.

        Returns:
            Список схем в формате OpenAI Function.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Выполнить инструмент по имени.

        Args:
            name: Имя инструмента.
            params: Именованные параметры вызова.

        Returns:
            Результат выполнения инструмента.

        Raises:
            ToolError: Если инструмент не найден или завершился ошибкой.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Инструмент {name!r} не найден в реестре", tool_name=name)
        return await tool.execute(params)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools)})"


# ---------------------------------------------------------------------------
# Агент
# ---------------------------------------------------------------------------


def _parse_tool(tool: dict[str, Any], seq: int) -> tuple[str, dict[str, Any], str]:
    """Разобрать один tool из ответа модели.

    Args:
        tool: Словарь вызова вида ``{"id": ..., "function": {...}}``.
        seq: Порядковый номер (используется для генерации идентификатора).

    Returns:
        Кортеж ``(имя, аргументы, id вызова)``.
    """
    function = tool.get("function") or {}
    name = function.get("name") or ""
    raw_args = function.get("arguments") or "{}"
    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            arguments = {}
    elif isinstance(raw_args, dict):
        arguments = raw_args
    else:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    tool_id = tool.get("id") or f"tool_{seq}"
    return name, arguments, tool_id


class Agent:
    """AI-агент: диалог с LLM-моделью с поддержкой инструментов и fallback.

    Агент поддерживает классический цикл ReAct: если модель возвращает
    ``tools``, агент выполняет инструменты из реестра, добавляет
    результаты в контекст и повторяет запрос до получения финального ответа.

    Args:
        provider: Основной LLM-провайдер.
        tools: Реестр инструментов; по умолчанию создаётся пустой.
        max_iters: Максимальное число итераций цикла tool.
        fallback_provider: Резервный провайдер для :meth:`send_with_fallback`.
        system_prompt: Системный промпт, добавляемый в начало контекста.
        max_context_messages: Лимит сообщений контекста.

    Raises:
        ConfigurationError: Если ``max_iters`` меньше 1.
    """

    def __init__(
        self,
        provider: Provider,
        tools: Optional[ToolRegistry] = None,
        max_iters: int = 5,
        fallback_provider: Optional[Provider] = None,
        system_prompt: str = "",
        max_context_messages: int = 100,
    ) -> None:
        if max_iters < 1:
            raise ConfigurationError("max_iters должен быть не меньше 1")
        self.provider = provider
        self.tools = tools if tools is not None else ToolRegistry()
        self.max_iters = max_iters
        self.fallback_provider = fallback_provider
        self.system_prompt = system_prompt
        self.context = Context(max_messages=max_context_messages)
        self.last_response: Optional[ProviderResponse] = None

    def _ensure_system_prompt(self, ctx: Context) -> None:
        """Добавить системный промпт в начало контекста (если задан)."""
        if not self.system_prompt:
            return
        data = ctx.get_data()
        if data and data[0]["role"] == TextMessage.ROLE_SYSTEM:
            return
        ctx.add_message(TextMessage.system(self.system_prompt))

    async def _send_with(self, provider: Provider, ctx: Context) -> str:
        """Основной цикл ReAct с указанным провайдером.

        Args:
            provider: Провайдер, через который выполняется запрос.
            ctx: Диалоговый контекст.

        Returns:
            Текст финального ответа модели.

        Raises:
            ProviderError: Если провайдер завершился ошибкой или модель
                не вернула финальный ответ за ``max_iters`` итераций.
        """
        schemas = self.tools.get_schemas() if len(self.tools) > 0 else None
        snapshot = len(ctx)

        for _ in range(self.max_iters):
            try:
                response = await provider.send(ctx, tools=schemas)
            except ProviderError:
                ctx.truncate(snapshot)
                raise
            self.last_response = response

            if response.tools:
                ctx.add_message(
                    TextMessage(TextMessage.ROLE_AI, response.text, tools=response.tools)
                )
                for tools in response.tools:
                    name, arguments, tool_id = _parse_tool(tool, len(ctx))
                    try:
                        result = await self.tools.execute(name, arguments)
                        result_text = (
                            result
                            if isinstance(result, str)
                            else json.dumps(result, ensure_ascii=False, default=str)
                        )
                    except ToolError as exc:
                        result_text = json.dumps({"error": str(exc)}, ensure_ascii=False)
                        logger.warning("Инструмент %r вернул ошибку: %s", name, exc)
                    ctx.add_message(TextMessage.tool_result(result_text, tool_id=tool_id))
                continue

            text = response.text
            if not text.strip():
                ctx.truncate(snapshot)
                raise ProviderError("Модель вернула пустой текст ответа")
            ctx.add_message(TextMessage.assistant(text))
            return text

        ctx.truncate(snapshot)
        raise ProviderError(f"Модель не вернула финальный ответ за {self.max_iters} итераций")

    async def send(self, context: Optional[Context] = None) -> str:
        """Отправить запрос через основной провайдер.

        Args:
            context: Диалоговый контекст; по умолчанию — внутренний контекст
                агента.

        Returns:
            Текст финального ответа модели.

        Raises:
            ProviderError: При ошибке провайдера или исчерпании итераций.
        """
        ctx = context if context is not None else self.context
        self._ensure_system_prompt(ctx)
        return await self._send_with(self.provider, ctx)

    async def send_with_fallback(self, context: Optional[Context] = None) -> str:
        """Отправить запрос через основной провайдер с переходом на резервный.

        При ``ProviderError`` основного провайдера контекст откатывается
        к состоянию до запроса, и запрос повторяется через резервный
        провайдер (если задан). Если резервный тоже завершился ошибкой,
        бросается :class:`ProviderError` с описанием обоих сбоев.

        Args:
            context: Диалоговый контекст; по умолчанию — внутренний контекст
                агента.

        Returns:
            Текст финального ответа модели.

        Raises:
            ProviderError: Если оба провайдера завершились ошибкой.
        """
        ctx = context if context is not None else self.context
        self._ensure_system_prompt(ctx)
        snapshot = len(ctx)
        try:
            return await self._send_with(self.provider, ctx)
        except ProviderError as primary_exc:
            if self.fallback_provider is None:
                raise
            logger.warning(
                "Основной провайдер завершился ошибкой: %s; пробуем резервный", primary_exc
            )
            ctx.truncate(snapshot)
            try:
                return await self._send_with(self.fallback_provider, ctx)
            except ProviderError as fallback_exc:
                raise ProviderError(
                    f"Оба провайдера завершились ошибкой. "
                    f"Основной: {primary_exc}. Резервный: {fallback_exc}"
                ) from fallback_exc

    def reset(self) -> None:
        """Очистить контекст и внутреннее состояние для новой сессии."""
        self.context.clear()
        self.last_response = None

    def __repr__(self) -> str:
        return (
            f"Agent(provider={self.provider!r}, tools={len(self.tools)}, "
            f"max_iters={self.max_iters})"
        )
