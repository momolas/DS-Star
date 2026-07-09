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
# See the License for the specific language governingå permissions and
# limitations under the License.

from agents.ds_star import run_agent
from agents.analyzer import agent_analyze_parallel
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--llm', default='gemini-3.5-flash', type=str)
parser.add_argument('--debug_round', default=5, type=int)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--max_workers', default=10, type=int)
parser.add_argument('--max_round', default=20, type=int)
P = parser.parse_args()

with open(f"tasks/{P.task}/question.txt", "r") as f:
    question = f.read()
with open(f"tasks/{P.task}/guidelines.txt", "r") as f:
    guidelines = f.read()
filenames = os.listdir(f"tasks/{P.task}/data")
agent_analyze_parallel(P, filenames)
contexts = []
for filename in filenames:
    with open(f"tasks/{P.task}/analytics/{filename}.txt", "r") as f:
        contexts.append(f.read())
answer = run_agent(P, question, guidelines, filenames, contexts)

print(answer)