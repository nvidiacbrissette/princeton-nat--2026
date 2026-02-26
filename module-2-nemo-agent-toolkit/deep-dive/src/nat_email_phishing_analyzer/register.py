# SPDX-FileCopyrightText: Copyright (c) 2024-2025,
# NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Any

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace

from .prompt import phishing_prompt
from .prompt import sensitive_info_prompt
from .prompt import intent_classifier_prompt
from .utils import smart_parse
# Ensure evaluator is registered when package is imported
from . import evaluator_register  # noqa: F401

logger = logging.getLogger(__name__)


class EmailPhishingAnalyzerConfig(
    FunctionBaseConfig, OptimizableMixin, name="email_phishing_analyzer"
):
    _type: str = "email_phishing_analyzer"
    llm: LLMRef = OptimizableField(
        description="The LLM to use for email phishing analysis.",
        default="llama_3_405",
        space=SearchSpace(values=["llama_3_405", "llama_3_70"]),
    )
    prompt: str = OptimizableField(
        description=(
            "The prompt template for analyzing email phishing. Use {body} "
            "to insert the email text."
        ),
        default=phishing_prompt,
        space=SearchSpace(
            is_prompt=True,
            prompt_purpose=(
                "Allow an LLM to look at an email body and determine if it is "
                "a phishing attempt."
            ),
        ),
    )


@register_function(
    config_type=EmailPhishingAnalyzerConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def email_phishing_analyzer(
    config: EmailPhishingAnalyzerConfig, builder: Builder
) -> Any:
    """Register the email phishing analysis tool."""

    async def _analyze_email_phishing(text: str) -> str:
        """
        Analyze an email body for signs of phishing using an LLM.

        Args:
            text: The email body text to analyze

        Returns:
            String containing analysis results in a human-readable format
        """
        # Get LLM from builder
        llm = await builder.get_llm(
            llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

        try:
            # Get response from LLM
            response = await llm.apredict(config.prompt.replace("{body}", text))
        except Exception as e:
            logger.error("Error during LLM prediction: %s", e)
            return f"Error: LLM prediction failed {e}"

        try:
            # Parse response using smart_parse
            analysis = smart_parse(response)

            # Handle missing or malformed fields with defaults
            result = {
                "is_likely_phishing": analysis.get("is_likely_phishing", False),
                "explanation": analysis.get(
                    "explanation", "No detailed explanation provided"
                ),
            }

            # Return as JSON string
            return json.dumps(result)
        except json.JSONDecodeError:
            return "Error: Could not parse LLM response as JSON"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _analyze_email_phishing,
        description=(
            "This tool analyzes email content to detect signs of phishing attempts. "
            "It evaluates factors like urgency, generic greetings, grammar mistakes, "
            "unusual requests, and emotional manipulation."
        ),
    )


class SensitiveInfoDetectorConfig(FunctionBaseConfig, OptimizableMixin, name="sensitive_info_detector"):
    _type: str = "sensitive_info_detector"
    llm: LLMRef = OptimizableField(
        description="The LLM to use for sensitive info detection.",
        default="llama_3_405",
        space=SearchSpace(values=["llama_3_405", "llama_3_70"]),
    )
    prompt: str = OptimizableField(
        description=(
            "Prompt template for detecting sensitive info requests. Use {body} to insert the email text."
        ),
        default=sensitive_info_prompt,
        space=SearchSpace(
            is_prompt=True,
            prompt_purpose="Detect whether email asks for sensitive info.",
        ),
    )


@register_function(
    config_type=SensitiveInfoDetectorConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def sensitive_info_detector(
    config: SensitiveInfoDetectorConfig, builder: Builder
) -> Any:
    """Register a tool that detects sensitive info requests in the email."""

    async def _detect(text: str) -> str:
        llm = await builder.get_llm(
            llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

        try:
            response = await llm.apredict(config.prompt.replace("{body}", text))
        except Exception as e:
            logger.error("Error during LLM prediction (sensitive info): %s", e)
            return json.dumps(
                {
                    "requests_sensitive_info": False,
                    "items": [],
                    "explanation": f"LLM error: {e}",
                }
            )

        parsed = smart_parse(response)
        result = {
            "requests_sensitive_info": bool(
                parsed.get("requests_sensitive_info", False)
            ),
            "items": parsed.get("items", []),
            "explanation": parsed.get("explanation", ""),
        }
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _detect,
        description=(
            "Detects whether the email requests sensitive information like passwords, one-time codes, "
            "account numbers, or personal identifiers. Returns a JSON with fields {requests_sensitive_info, "
            "items, explanation}."
        ),
    )


class IntentClassifierConfig(
    FunctionBaseConfig, OptimizableMixin, name="intent_classifier"
):
    _type: str = "intent_classifier"
    llm: LLMRef = OptimizableField(
        description="The LLM to use for intent classification.",
        default="llama_3_405",
        space=SearchSpace(values=["llama_3_405", "llama_3_70"]),
    )
    prompt: str = OptimizableField(
        description=(
            "Prompt template for classifying attacker intent. Use {body} to insert the email text."
        ),
        default=intent_classifier_prompt,
        space=SearchSpace(
            is_prompt=True,
            prompt_purpose="Classify likely attacker intent from email body.",
        ),
    )


@register_function(
    config_type=IntentClassifierConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def intent_classifier(
    config: IntentClassifierConfig, builder: Builder
) -> Any:
    """Register a tool that classifies the likely attacker intent."""

    async def _classify(text: str) -> str:
        llm = await builder.get_llm(
            llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )
        try:
            response = await llm.apredict(config.prompt.replace("{body}", text))
        except Exception as e:
            logger.error("Error during LLM prediction (intent): %s", e)
            return json.dumps(
                {"intent": "other", "confidence": 0.0, "rationale": f"LLM error: {e}"}
            )

        parsed = smart_parse(response)
        intent_val = str(parsed.get("intent", "other")).lower().strip()
        if intent_val not in {
            "credential theft",
            "financial fraud",
            "malware delivery",
            "account recovery scam",
            "other",
        }:
            intent_val = "other"
        conf = parsed.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        result = {
            "intent": intent_val,
            "confidence": max(0.0, min(1.0, conf)),
            "rationale": parsed.get("rationale", ""),
        }
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _classify,
        description=(
            "Classifies the likely attacker objective (for example, credential theft, financial fraud, "
            "malware delivery, account recovery scam, other). Returns JSON {intent, confidence, rationale}."
        ),
    )


class LinkAnalyzerConfig(
    FunctionBaseConfig, OptimizableMixin, name="link_and_domain_analyzer"
):
    _type: str = "link_and_domain_analyzer"
    suspicious_tlds: list[str] = OptimizableField(
        description="List of suspicious TLDs considered risky.",
        default=["zip", "mov", "country", "click", "xyz", "top", "lol", "mom"],
        space=SearchSpace(values=None),
    )
    min_suspicious_score: float = OptimizableField(
        description="Threshold for marking links as suspicious.",
        default=0.5,
        space=SearchSpace(low=0.2, high=0.9, step=0.1),
    )


@register_function(config_type=LinkAnalyzerConfig)
async def link_and_domain_analyzer(
    config: LinkAnalyzerConfig, builder: Builder
) -> Any:
    """Register a pure-Python tool that scores links/domains in the email body."""

    import re
    from urllib.parse import urlparse

    url_pattern = re.compile(r"https?://[^\s)]+", re.IGNORECASE)

    def _score_url(u: str) -> tuple[float, dict]:
        score = 0.0
        reasons: list[str] = []

        parsed = urlparse(u)
        host = parsed.netloc.lower()

        if host.startswith("xn--"):
            score += 0.4
            reasons.append("punycode domain")

        if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", host.split(":")[0]):
            score += 0.4
            reasons.append("raw IP host")

        tld = host.split(".")[-1]
        if tld in set(x.lower() for x in config.suspicious_tlds):
            score += 0.3
            reasons.append(f"suspicious TLD .{tld}")

        path = parsed.path or ""
        if any(
            tok in path.lower()
            for tok in ["login", "verify", "update", "secure", "account", "password"]
        ):
            score += 0.2
            reasons.append("auth-seeming path")

        score = min(1.0, score)
        return score, {"url": u, "reasons": reasons}

    async def _analyze(text: str) -> str:
        # Silence linter for unused builder
        _ = builder
        links = url_pattern.findall(text or "")
        scored = []
        suspicious = []
        for u in links:
            s, meta = _score_url(u)
            record = {"url": meta["url"], "score": s, "reasons": meta["reasons"]}
            scored.append(record)
            if s >= float(config.min_suspicious_score):
                suspicious.append(record)

        result = {
            "total_links": len(links),
            "suspicious_links": suspicious,
            "all_links": scored,
        }
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _analyze,
        description=(
            "Extracts URLs from the email and flags suspicious ones using heuristics (punycode, raw IPs, "
            "suspicious TLDs, auth-like paths). Returns JSON {total_links, suspicious_links, all_links}."
        ),
    )


class PhishingAggregatorConfig(FunctionBaseConfig, OptimizableMixin, name="phishing_risk_aggregator"):
    _type: str = "phishing_risk_aggregator"
    explanation_llm: LLMRef = OptimizableField(
        description="LLM used to draft human-readable explanation.",
        default="llama_3_70",
        space=SearchSpace(values=["llama_3_405", "llama_3_70"]),
    )
    weight_sensitive_info: float = OptimizableField(
        description="Weight for sensitive info signal.",
        default=0.5,
        space=SearchSpace(low=0.1, high=1.0, step=0.1),
    )
    weight_suspicious_links: float = OptimizableField(
        description="Weight for link analysis signal.",
        default=0.3,
        space=SearchSpace(low=0.1, high=1.0, step=0.1),
    )
    weight_intent: float = OptimizableField(
        description="Weight for intent classification signal.",
        default=0.2,
        space=SearchSpace(low=0.1, high=1.0, step=0.1),
    )
    decision_threshold: float = OptimizableField(
        description="Threshold above which email is phishing.",
        default=0.5,
        space=SearchSpace(low=0.3, high=0.8, step=0.05),
    )


@register_function(
    config_type=PhishingAggregatorConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def phishing_risk_aggregator(config: PhishingAggregatorConfig, builder: Builder) -> Any:
    """Register a tool that aggregates signals into a single phishing decision."""

    async def _aggregate(payload: str) -> str:
        """
        Accepts either raw email body as a string, or a JSON object containing:
        {
          "body": str,
          "sensitive_info": {...},
          "intent": {...},
          "links": {...}
        }
        Returns JSON {is_likely_phishing, risk_score, explanation, factors}
        """
        try:
            obj = smart_parse(payload)
        except Exception:
            obj = {"body": payload}

        body = obj.get("body", payload)

        # Extract signals if present
        sens = obj.get("sensitive_info", {})
        intent = obj.get("intent", {})
        links = obj.get("links", {})

        # Compute numeric features with safe defaults
        sens_flag = 1.0 if bool(sens.get("requests_sensitive_info", False)) else 0.0
        intent_score = (
            float(intent.get("confidence", 0.0))
            if str(intent.get("intent", "")).lower()
            in {"credential theft", "financial fraud", "malware delivery", "account recovery scam"}
            else 0.0
        )
        suspicious_links = float(len(links.get("suspicious_links", [])))
        link_score = (
            min(1.0, suspicious_links / max(1.0, float(links.get("total_links", 1.0))))
            if links
            else 0.0
        )

        # Weighted risk score
        risk = (
            config.weight_sensitive_info * sens_flag
            + config.weight_suspicious_links * link_score
            + config.weight_intent * intent_score
        )
        risk = max(0.0, min(1.0, risk))
        decision = bool(risk >= float(config.decision_threshold))

        # Build explanation with LLM for readability
        llm = await builder.get_llm(
            llm_name=config.explanation_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )
        summary_prompt = (
            "Summarize the phishing assessment in 2-4 sentences given the following signals and email body. "
            "Be concise and actionable.\n\n"
            f"Signals JSON: {json.dumps({'sensitive_info': sens, 'intent': intent, 'links': links})}\n\n"
            f"Email body:\n{body}"
        )
        try:
            expl = await llm.apredict(summary_prompt)
        except Exception as e:
            logger.warning("Explanation LLM error: %s", e)
            expl = (
                "Aggregated signals indicate the likelihood above the decision threshold."
            )

        result = {
            "is_likely_phishing": decision,
            "risk_score": round(risk, 3),
            "factors": {
                "sensitive_info": sens,
                "intent": intent,
                "links": links,
            },
            "explanation": expl,
        }
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _aggregate,
        description=(
            "Aggregates signals from other tools into a final phishing decision. Input should be a JSON with keys "
            "{body, sensitive_info, intent, links}. Returns JSON {is_likely_phishing, risk_score, explanation, factors}."
        ),
    )
