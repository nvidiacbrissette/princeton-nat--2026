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

phishing_prompt = """

Examine the following email content and determine if it exhibits signs of
malicious intent. Look for any suspicious signals that may indicate phishing,
such as requests for personal information or a suspicious tone.

Email content:
{body}

Return your findings as a JSON object with these fields:

- is_likely_phishing: (boolean) true if phishing is suspected
- explanation: (string) detailed explanation of your reasoning

"""


# Additional prompts to power richer analysis tools

sensitive_info_prompt = """
You are a security analyst. Review the email body below and determine whether
it requests any sensitive information from the recipient. Consider passwords,
one-time passcodes (OTP), account numbers, routing numbers, Social Security
numbers, credit card numbers, security questions/answers, recovery codes, and
personal identifying information.

Email content:
{body}

Return a JSON object with the following fields:
- requests_sensitive_info: (boolean) true if any sensitive info is requested
- items: (array of strings) list of sensitive items requested
  (for example, "password", "routing number")
- explanation: (string) brief description of the evidence
"""


intent_classifier_prompt = """
Classify the likely attacker objective in the following email. Choose exactly
one intent from this set:
[
  "credential theft",
  "financial fraud",
  "malware delivery",
  "account recovery scam",
  "other"
]

Email content:
{body}

Return a JSON object with the following fields:
- intent: (string) one of the allowed intent values above
- confidence: (number in [0,1]) confidence score
- rationale: (string) short explanation
"""
