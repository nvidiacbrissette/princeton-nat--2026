# SPDX-FileCopyrightText: Copyright (c) 2025,
# NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem

logger = logging.getLogger(__name__)


class PhishingAccuracyEvaluatorConfig(
    EvaluatorBaseConfig, name="phishing_accuracy"
):
    """
    Compute binary accuracy for phishing detection.

    Expects dataset rows to include a `label` with values like `phish` or
    `benign`. A workflow output is considered phishing if either:
      - it is a JSON containing key `is_likely_phishing` true, or
      - the output string includes the substring `likely a phishing`
        (fallback heuristic).
    """
    metric_name: str = "accuracy"


@register_evaluator(config_type=PhishingAccuracyEvaluatorConfig)
async def register_phishing_accuracy_evaluator(
    config: PhishingAccuracyEvaluatorConfig, _builder: EvalBuilder
):
    class PhishingAccuracy(BaseEvaluator):
        def __init__(self):
            super().__init__(
                max_concurrency=8,
                tqdm_desc="Evaluating phishing accuracy",
            )

        async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
            label = str(item.full_dataset_entry.get("label", "")).strip().lower()
            expected_is_phish = label in {
                "phish", "phishing", "spam", "malicious"
            }

            output = item.output_obj
            is_phish_pred = False
            try:
                # Try to parse JSON
                import json
                parsed = None
                if isinstance(output, str):
                    try:
                        parsed = json.loads(output)
                    except Exception:
                        # Attempt to find embedded JSON
                        import re
                        m = re.search(r"{.*}", output, re.DOTALL)
                        if m:
                            try:
                                parsed = json.loads(m.group(0))
                            except Exception:
                                parsed = None
                if isinstance(output, dict):
                    parsed = output

                if parsed is not None and isinstance(parsed, dict):
                    is_phish_pred = bool(
                        parsed.get("is_likely_phishing", False)
                    )
                else:
                    # Fallback string heuristic
                    is_phish_pred = (
                        isinstance(output, str)
                        and ("likely a phishing" in output.lower())
                    )
            except Exception as e:
                logger.debug("Evaluator parsing error for %s: %s", item.id, e)
                is_phish_pred = False

            score = 1.0 if (is_phish_pred == expected_is_phish) else 0.0
            reasoning = {
                "expected_label": label,
                "predicted_is_phish": is_phish_pred,
            }
            return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)

    evaluator = PhishingAccuracy()
    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description="Phishing Accuracy Evaluator",
    )


