"""BayLang AI — библиотека для построения AI-агентов поверх OpenRouter.

Модуль содержит базовые абстракции библиотеки:

* :class:`TextMessage`, :class:`ToolResultMessage`, :class:`ToolsMessage` —
  сообщения диалога в формате OpenAI-совместимого API (текст модели,
  вызовы инструментов и их результаты);
* :class:`Context` — диалоговый контекст с ограничением длины и JSON-сериализацией;
* :class:`Provider` / :class:`OpenRouterProvider` — асинхронные провайдеры LLM;
* :class:`Tool` / :class:`ToolRegistry` — подсистема инструментов (function);
* :class:`Agent` — агент с циклом ReAct и fallback-цепочкой провайдеров.

Основной метод :meth:`Agent.send_with` — асинхронный генератор: он отдаёт
каждое сообщение (текст модели, вызовы инструментов, результаты их работы)
по мере появления, поэтому интерфейс может выводить их пользователю
без задержки. Метод :meth:`Agent.send` собирает генератор и возвращает
финальный текст ответа.

Все сетевые операции выполняются асинхронно через ``httpx``; блокирующие
вызовы в асинхронном контексте не используются.

Провайдер отправляет запросы с подсказкой кеширования промпта:
к последнему сообщению добавляется блок ``cache_control: {"type": "ephemeral"}``,
который включает prompt caching у провайдеров, поддерживающих его.
"""

from __future__ import annotations

import asyncio
import abc
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Union

import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "BayLangError",
    "ConfigurationError",
    "ProviderError",
    "ToolError",
    "TextMessage",
    "ToolResultMessage",
    "ToolsMessage",
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

    Базовый класс сообщений контекста. Наследники :class:`ToolResultMessage`
    и :class:`ToolsMessage` расширяют его для tool-цепочек и JSON-сериализации.

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

    #: Тип сообщения для JSON-сериализации (диспетчеризация при восстановлении).
    MESSAGE_TYPE = "text"

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

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать сообщение в словарь для сохранения в JSON.

        В отличие от :meth:`get_data`, результат содержит поле ``type``
        с диспетчером класса, по которому :meth:`from_dict` восстановит
        исходный тип сообщения.

        Returns:
            Словарь вида ``{"type": "text", "role": ..., "content": ...}``.
        """
        data = self.get_data()
        data["type"] = self.MESSAGE_TYPE
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextMessage":
        """Восстановить сообщение из словаря (:meth:`to_dict`).

        Диспетчеризация выполняется по полю ``type``; для словарей без
        этого поля тип определяется по роли и наличию поля ``tools``.

        Args:
            data: Словарь сообщения.

        Returns:
            Объект :class:`TextMessage`, :class:`ToolResultMessage`
            или :class:`ToolsMessage`.

        Raises:
            ValueError: Если ``data`` не словарь или роль сообщения
                некорректна; для ``tool_result`` — если отсутствует
                непустой ``tool_id``.
        """
        if not isinstance(data, dict):
            raise ValueError("Сообщение должно быть словарём")
        message_type = data.get("type") or cls.infer_type(data)
        if message_type == ToolResultMessage.MESSAGE_TYPE:
            return ToolResultMessage.from_dict(data)
        if message_type == ToolsMessage.MESSAGE_TYPE:
            return ToolsMessage.from_dict(data)
        role = data.get("role") or TextMessage.ROLE_USER
        content = data.get("content") or ""
        return TextMessage(role, content, tool_id=data.get("tool_id"), tools=data.get("tools"))

    @staticmethod
    def infer_type(data: dict[str, Any]) -> str:
        """Определить тип сообщения по роли и наличию ``tools`` (legacy JSON)."""
        if data.get("role") == TextMessage.ROLE_TOOL:
            return ToolResultMessage.MESSAGE_TYPE
        if data.get("tools"):
            return ToolsMessage.MESSAGE_TYPE
        return TextMessage.MESSAGE_TYPE

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
    def tool_result(cls, content: str, tool_id: str) -> "ToolResultMessage":
        """Создать сообщение с результатом выполнения инструмента.

        Возвращает :class:`ToolResultMessage` (обратная совместимость сохранена).
        """
        return ToolResultMessage(content, tool_id=tool_id)

    def __repr__(self) -> str:
        preview = self.content if len(self.content) <= 40 else self.content[:37] + "..."
        return f"TextMessage(role={self.role!r}, content={preview!r})"


class ToolResultMessage(TextMessage):
    """Сообщение с результатом выполнения инструмента (роль ``tool``).

    Содержит идентификатор вызова ``tool_id``, по которому результат
    связывается с соответствующим сообщением :class:`ToolsMessage`.
    Поддерживает JSON-сериализацию через :meth:`to_dict` / :meth:`from_dict`.

    Args:
        content: Текст результата выполнения инструмента.
        tool_id: Идентификатор вызова инструмента (непустая строка).

    Raises:
        ValueError: Если ``tool_id`` пуст.
    """

    MESSAGE_TYPE = "tool_result"

    def __init__(self, content: str, tool_id: str) -> None:
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError("ToolResultMessage требует непустой tool_id")
        super().__init__(TextMessage.ROLE_TOOL, content, tool_id=tool_id)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать сообщение в словарь для сохранения в JSON.

        Returns:
            Словарь ``{"type": "tool_result", "role": "tool",
            "content": ..., "tool_id": ...}``.
        """
        return {
            "type": self.MESSAGE_TYPE,
            "role": self.role,
            "content": self.content,
            "tool_id": self.tool_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultMessage":
        """Восстановить сообщение результата из словаря.

        Поддерживает и поле ``tool_id``, и стандартное для OpenAI
        поле ``tool_call_id``.

        Args:
            data: Словарь сообщения.

        Returns:
            Объект :class:`ToolResultMessage`.

        Raises:
            ValueError: Если ``data`` не словарь или отсутствует
                непустой идентификатор вызова.
        """
        if not isinstance(data, dict):
            raise ValueError("Сообщение должно быть словарём")
        tool_id = data.get("tool_id") or data.get("tool_call_id") or ""
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError("ToolResultMessage требует непустой tool_id")
        return cls(data.get("content") or "", tool_id=tool_id)

    def __repr__(self) -> str:
        preview = self.content if len(self.content) <= 40 else self.content[:37] + "..."
        return f"ToolResultMessage(tool_id={self.tool_id!r}, content={preview!r})"


class ToolsMessage(TextMessage):
    """Сообщение ассистента со списком вызовов инструментов.

    Соответствует ответу модели, содержащему поле ``tool_calls``
    (внутри библиотеки — атрибут ``tools``). Позволяет сохранять
    и восстанавливать контекст с tool-цепочками в JSON.

    Attributes:
        content: Текст ответа модели (может быть пустым, если модель
            вернула только вызовы инструментов).
        tools: Список вызовов инструментов в формате провайдера.

    Args:
        content: Текст ответа модели.
        tools: Список вызовов инструментов (допустим пустой список,
            но не ``None``).

    Raises:
        ValueError: Если ``tools`` не является списком.
    """

    MESSAGE_TYPE = "tools"

    def __init__(
        self,
        content: str = "",
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if tools is None:
            raise ValueError("ToolsMessage требует список tools (допустим пустой)")
        if not isinstance(tools, list):
            raise ValueError("ToolsMessage.tools должен быть списком")
        super().__init__(TextMessage.ROLE_AI, content, tools=tools)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать сообщение в словарь для сохранения в JSON.

        Поле ``tools`` сохраняется всегда (даже если список пуст),
        чтобы восстановление было точным.

        Returns:
            Словарь ``{"type": "tools", "role": "assistant",
            "content": ..., "tools": [...]}``.
        """
        return {
            "type": self.MESSAGE_TYPE,
            "role": self.role,
            "content": self.content,
            "tools": list(self.tools or []),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolsMessage":
        """Восстановить сообщение с tool-вызовами из словаря.

        Args:
            data: Словарь сообщения.

        Returns:
            Объект :class:`ToolsMessage`.

        Raises:
            ValueError: Если ``data`` не словарь или поле ``tools``
                не является списком.
        """
        if not isinstance(data, dict):
            raise ValueError("Сообщение должно быть словарём")
        tools = data.get("tools")
        if tools is None:
            tools = []
        if not isinstance(tools, list):
            raise ValueError("ToolsMessage.tools должен быть списком")
        return cls(data.get("content") or "", tools=tools)

    def __repr__(self) -> str:
        count = len(self.tools or [])
        preview = self.content if len(self.content) <= 40 else self.content[:37] + "..."
        return f"ToolsMessage(tools={count}, content={preview!r})"


class Context:
    """Диалоговый контекст: упорядоченная история сообщений.

    Хранилище приватное; внешний доступ к списку сообщений возможен только
    через методы :meth:`add_message`, :meth:`get_data`, :meth:`to_dict`,
    :meth:`to_json`, :meth:`trim`, :meth:`clear`, :meth:`truncate`,
    :meth:`last_response`, :meth:`last_messages`, а также итерацию и ``len()``.

    Контекст можно сохранять в JSON (:meth:`to_json`) и восстанавливать
    из него (:meth:`from_json`) — tool-цепочки сохраняются полностью.

    Args:
        max_messages: Максимальное число хранимых сообщений. При превышении
            старые сообщения автоматически отсекаются (:meth:`trim`).
    """

    def __init__(self, max_messages: Optional[int] = None) -> None:
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages должен быть не меньше 1")
        self.items: list[TextMessage] = []
        self.max_messages = max_messages

    def add_message(self, message: TextMessage) -> None:
        """Добавить сообщение в конец контекста.

        Args:
            message: Объект :class:`TextMessage` (или его наследника).

        Raises:
            TypeError: Если передан не ``TextMessage``.
        """
        if not isinstance(message, TextMessage):
            raise TypeError("В контекст можно добавлять только объекты TextMessage")
        self.items.append(message)
        if self.max_messages is not None and len(self.items) > self.max_messages:
            self.trim(self.max_messages)

    def get_data(self) -> list[dict[str, Any]]:
        """Получить список сообщений в формате провайдера.

        Returns:
            Список словарей ``{"role": ..., "content": ...}``, готовых
            к передаче в поле ``"messages"`` запроса к LLM.
        """
        return [item.get_data() for item in self.items]

    def to_dict(self) -> list[dict[str, Any]]:
        """Сериализовать контекст в список словарей (для сохранения в JSON).

        Returns:
            Список словарей с полем ``type`` у каждого сообщения.
        """
        return [item.to_dict() for item in self.items]

    def to_json(self, indent: Optional[int] = None) -> str:
        """Сериализовать контекст в JSON-строку.

        Args:
            indent: Отступ для читаемого JSON; ``None`` — компактная строка.

        Returns:
            JSON-строка, обратно читаемая через :meth:`from_json`.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(
        cls,
        data: list[dict[str, Any]],
        max_messages: Optional[int] = None,
    ) -> "Context":
        """Восстановить контекст из списка словарей (:meth:`to_dict`).

        Args:
            data: Список словарей сообщений.
            max_messages: Лимит сообщений нового контекста.

        Returns:
            Новый объект :class:`Context`.

        Raises:
            ValueError: Если ``data`` не список или содержит не-словари.
        """
        if not isinstance(data, list):
            raise ValueError("Контекст должен быть списком сообщений")
        ctx = cls(max_messages=max_messages)
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Каждое сообщение контекста должно быть словарём")
            ctx.add_message(TextMessage.from_dict(item))
        return ctx

    @classmethod
    def from_json(
        cls,
        data: Union[str, bytes],
        max_messages: Optional[int] = None,
    ) -> "Context":
        """Восстановить контекст из JSON-строки (:meth:`to_json`).

        Args:
            data: JSON-строка (или байты) со списком сообщений.
            max_messages: Лимит сообщений нового контекста.

        Returns:
            Новый объект :class:`Context`.

        Raises:
            ValueError: Если строка не является корректным JSON-списком
                сообщений.
        """
        try:
            items = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError("Некорректный JSON контекста") from exc
        return cls.from_dict(items, max_messages=max_messages)

    def trim(self, max_messages: int) -> None:
        """Отсечь старые сообщения, оставшиеся последние ``max_messages``.

        Args:
            max_messages: Сколько последних сообщений сохранить.

        Raises:
            ValueError: Если ``max_messages`` отрицателен.
        """
        if max_messages < 0:
            raise ValueError("max_messages не может быть отрицательным")
        if len(self.items) > max_messages:
            del self.items[:-max_messages]

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
        if len(self.items) > size:
            del self.items[size:]

    def clear(self) -> None:
        """Полностью очистить контекст."""
        self.items.clear()

    def last_response(self) -> Optional[TextMessage]:
        """Вернуть последнее сообщение роли ``assistant``.

        Returns:
            Сообщение ассистента или ``None``, если ответов ещё не было.
        """
        for item in reversed(self.items):
            if item.role == TextMessage.ROLE_AI:
                return item
        return None

    def last_messages(self, count: int) -> list[TextMessage]:
        """Вернуть последние ``count`` сообщений контекста.

        Используется для показа пользователю последних сообщений
        при загрузке истории диалога.

        Args:
            count: Сколько последних сообщений вернуть.

        Returns:
            Список сообщений (от старых к новым); пустой список,
            если ``count <= 0`` или контекст пуст.
        """
        if count <= 0 or not self.items:
            return []
        return list(self.items[-count:])

    def __iter__(self):
        """Итерировать по копии списка сообщений."""
        return iter(list(self.items))

    def __len__(self) -> int:
        """Вернуть количество сообщений в контексте."""
        return len(self.items)

    def __repr__(self) -> str:
        return f"Context(messages={len(self.items)})"


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
        self.fallback_iters = 100
        self.delay = 5
        self.client: Optional[httpx.AsyncClient] = None

    @abc.abstractmethod
    def get_url(self) -> str:
        """Вернуть полный URL chat-completions эндпоинта провайдера."""

    @abc.abstractmethod
    def get_model_name(self) -> str:
        """Вернуть имя используемой модели."""

    @abc.abstractmethod
    def get_api_key(self) -> str:
        """Вернуть API-ключ провайдера."""

    def parse_response(self, raw: dict[str, Any]) -> ProviderResponse:
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

    def get_client(self) -> httpx.AsyncClient:
        """Вернуть (или создать) переиспользуемый асинхронный HTTP-клиент."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self.client

    async def post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Выполнить POST-запрос и вернуть разобранный JSON-ответ.

        Raises:
            ProviderError: При таймауте, сетевой ошибке, HTTP 4xx/5xx
                или некорректном теле ответа.
        """
        client = self.get_client()
        try:
            response = await client.post(
                self.get_url(), headers=self.build_headers(), json=payload
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
        count = 0
        while count < self.fallback_iters:
            count += 1
            try:
                payload = self.build_payload(context, tools=tools, stream=False)
                raw = await self.post_json(payload)
                return self.parse_response(raw)
            except ProviderError:
                await asyncio.sleep(self.delay)
                continue

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
        payload = self.build_payload(context, tools=tools, stream=True)
        client = self.get_client()
        try:
            async with client.stream(
                "POST", self.get_url(), headers=self.build_headers(), json=payload
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
        if self.client is not None and not self.client.is_closed:
            await self.client.aclose()
        self.client = None


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
    DEFAULT_MODEL = "openrouter/auto"
    DEFAULT_REFERER = "https://baylang.com/"
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
        self.url = url or self.DEFAULT_URL
        self.referer = referer
        self.title = title
        self.client: Optional[httpx.AsyncClient] = None

    def get_url(self) -> str:
        """Вернуть URL chat-completions эндпоинта OpenRouter."""
        return self.url

    def get_model_name(self) -> str:
        """Вернуть имя модели OpenRouter."""
        return self.model_name

    def get_api_key(self) -> str:
        """Вернуть API-ключ OpenRouter."""
        return self.api_key

    def build_headers(self) -> dict[str, str]:
        """Собрать HTTP-заголовки запроса (ключ API не логируется)."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

    def build_payload(
        self,
        context: Context,
        tools: Optional[list[dict[str, Any]]] = None,
        stream: bool = False,
        cache_control: bool = True,
    ) -> dict[str, Any]:
        """Собрать тело запроса в формате OpenAI chat-completions.

        Если ``cache_control`` включён (по умолчанию), к последнему сообщению
        добавляется блок ``cache_control: {"type": "ephemeral"}``. Такая
        подсказка включает prompt caching у провайдеров, поддерживающих его
        (например, Anthropic через OpenRouter): кешируется префикс диалога,
        что ускоряет повторные запросы и снижает стоимость токенов.
        """
        messages = context.get_data()
        for message in messages:
            message["cache_control"] = {"type": "ephemeral"}
        payload: dict[str, Any] = {
            "cache_control": {"type": "ephemeral"},
            "model": self.get_model_name(),
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload

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
                f"parameters_schema инструмента {name!r} должен быть словарём"
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
        self.tools: dict[str, Tool] = {}

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
        if tool.name in self.tools:
            raise ConfigurationError(f"Инструмент {tool.name!r} уже зарегистрирован")
        self.tools[tool.name] = tool

    def find(self, name: str) -> Optional[Tool]:
        """Найти инструмент по имени.

        Returns:
            Инструмент или ``None``, если не найден.
        """
        return self.tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Вернуть схемы всех инструментов для передачи в запрос к модели.

        Returns:
            Список схем в формате OpenAI Function.
        """
        return [tool.get_schema() for tool in self.tools.values()]

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
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"Инструмент {name!r} не найден в реестре", tool_name=name)
        return await tool.execute(params)

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: object) -> bool:
        return name in self.tools

    def __iter__(self):
        return iter(self.tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self.tools)})"


# ---------------------------------------------------------------------------
# Агент
# ---------------------------------------------------------------------------


def parse_tool(tool: dict[str, Any], seq: int) -> tuple[str, dict[str, Any], str]:
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
        max_iters: int = 100,
        max_context_messages: Optional[int] = None,
    ) -> None:
        if max_iters < 1:
            raise ConfigurationError("max_iters должен быть не меньше 1")
        self.provider = provider
        self.tools = tools if tools is not None else ToolRegistry()
        self.max_iters = max_iters
        self.context = Context(max_messages=max_context_messages)
        self.last_response: Optional[ProviderResponse] = None
        self.delay = 5

    async def send_with(self) -> AsyncIterator[TextMessage]:
        """Основной цикл ReAct: отдаёт сообщения по мере их появления.

        Асинхронный генератор: каждое сообщение (текст модели, вызовы
        инструментов, результаты их выполнения) добавляется в контекст
        и сразу отдаётся вызывающему коду — например, интерактивному
        режиму, который печатает его пользователю.

        Yields:
            Сообщения диалога:

            * :class:`ToolsMessage` — модель вызывает инструменты;
            * :class:`ToolResultMessage` — результат выполнения инструмента;
            * :class:`TextMessage` — финальный текстовый ответ модели.

        Raises:
            ProviderError: Если провайдер завершился ошибкой или модель
                не вернула финальный ответ за ``max_iters`` итераций.
        """
        schemas = self.tools.get_schemas() if len(self.tools) > 0 else None
        snapshot = len(self.context)

        for _ in range(self.max_iters):
            try:
                response = await self.provider.send(self.context, tools=schemas)
            except ProviderError:
                self.context.truncate(snapshot)
                raise
            self.last_response = response

            text = response.text.strip()

            if response.tools:
                # Ответ с вызовами инструментов: показываем их пользователю,
                # выполняем каждый инструмент и отдаём результат.
                tools_message = ToolsMessage(text, response.tools)
                self.context.add_message(tools_message)
                yield tools_message

                for tool in response.tools:
                    name, arguments, tool_id = parse_tool(tool, len(self.context))
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
                    result_message = ToolResultMessage(result_text, tool_id=tool_id)
                    self.context.add_message(result_message)
                    yield result_message

                await asyncio.sleep(self.delay)
                continue

            if text:
                message = TextMessage.assistant(text)
                self.context.add_message(message)
                yield message
            return

        self.context.truncate(snapshot)
        raise ProviderError(f"Модель не вернула финальный ответ за {self.max_iters} итераций")

    async def send(self) -> str:
        """Отправить запрос и вернуть финальный текст ответа модели.

        Обёртка над :meth:`send_with`: собирает сообщения генератора
        и возвращает текст последнего текстового ответа ассистента.

        Returns:
            Текст финального ответа модели (пустая строка, если модель
            вернула пустой ответ).

        Raises:
            ProviderError: При ошибке провайдера или исчерпании итераций.
        """
        text = ""
        async for message in self.send_with():
            if message.MESSAGE_TYPE == TextMessage.MESSAGE_TYPE:
                text = message.content
        return text

    async def disconnect(self):
        """Закрыть соединения провайдера."""
        if self.provider:
            await self.provider.aclose()

    def reset(self) -> None:
        """Очистить контекст и внутреннее состояние для новой сессии."""
        self.context.clear()
        self.last_response = None

    def __repr__(self) -> str:
        return (
            f"Agent(provider={self.provider!r}, tools={len(self.tools)}, "
            f"max_iters={self.max_iters})"
        )