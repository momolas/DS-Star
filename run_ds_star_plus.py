# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from agents.ds_star_plus import deep_data_research
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--llm', default='gemini-3.5-flash', type=str)
parser.add_argument('--debug_round', default=5, type=int)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--max_workers', default=10, type=int)
parser.add_argument('--max_round', default=5, type=int)
parser.add_argument('--report_refine_num', default=2, type=int)
P = parser.parse_args()

with open(f"tasks/{P.task}/question.txt", "r") as f:
    question = f.read()
report = deep_data_research(P, question)

print(report)
