from __future__ import annotations

import os
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from botocore.tokens import (  # type: ignore[import-untyped]
    DeferredRefreshableToken,
    FrozenAuthToken,
)

from .llm_configurations import LLM, ModelInfo


class AmazonBedrock_LLM(LLM):
    def __init__(
        self,
        name: str,
        model: ModelInfo,
        verbose: bool = False,
    ):
        super().__init__(name, model)
        self.model = model
        self.verbose = verbose
        self.chat_history: list[dict[str, Any]] = []
        self.system_prompt = ""
        self.client: Any | None = None

        bedrock_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if not bedrock_token:
            print("NO AWS_BEARER_TOKEN_BEDROCK provided — running in mock mode")
            return

        region = os.getenv("AWS_DEFAULT_REGION")
        if not region:
            raise ValueError(
                "No AWS_DEFAULT_REGION, set it, exampe us-east-2 tends to work well"
            )

        from datetime import datetime, timedelta, timezone

        def _refresh_bearer_token() -> FrozenAuthToken:
            # Provide a long-lived, timezone-aware expiration so
            # botocore's DeferredRefreshableToken._is_expired() can
            # safely compute remaining time without None errors.
            expiration = datetime.now(tz=timezone.utc) + timedelta(days=365)
            return FrozenAuthToken(token=bedrock_token, expiration=expiration)

        auth_token = DeferredRefreshableToken(
            method="bedrock-bearer-token",
            refresh_using=_refresh_bearer_token,
        )

        print("BEDROCK TOKEN PROVIDED")
        botocore_session: Any = getattr(boto3.Session(), "_session")
        botocore_session._auth_token = auth_token
        self.client = botocore_session.create_client(
            service_name="bedrock-runtime",
            region_name=region,
            config=Config(signature_version="bearer"),
        )

    def _get_response(self, prompt: str) -> str:
        if self.client is None:
            return "Mock Reply"

        bedrock_messages: list[dict[str, Any]] = [
            {"role": msg["role"], "content": [{"text": msg["content"]}]}
            for msg in self.chat_history
            if isinstance(msg.get("role"), str) and isinstance(msg.get("content"), str)
        ]

        bedrock_messages.append({"role": "user", "content": [{"text": prompt}]})

        response: dict[str, Any] = self.client.converse(
            modelId=self.model.model_id,
            messages=bedrock_messages,
            system=[{"text": self.system_prompt}],
        )

        reply = response["output"]["message"]["content"][0]["text"]
        self.chat_history.append({"role": "assistant", "content": reply})
        return reply
