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

import os
from tools.llm import complete
from tools.utils import run_python_code
from agents.analyzer import agent_analyze_parallel
from agents.prompts import initial_plan
from agents.prompts import initial_implementation
from agents.prompts import verify
from agents.prompts import next_plan
from agents.prompts import router
from agents.prompts import implement
from agents.prompts import match_guideline
from agents.prompts import bug_summarization
from agents.prompts import debug_with_context


def debug_code(P, filename, code, bug, filenames, contexts):
    prompt_for_bug_summarization = bug_summarization(P, filename, bug)
    bug = complete(P, prompt_for_bug_summarization)
    prompt_for_debugging = debug_with_context(P, code, bug, filenames, contexts)
    debugged_code = complete(P, prompt_for_debugging)
    result = run_python_code(P, debugged_code, f"{filename}.py")
    debugged_code = debugged_code.replace("```python", "").replace("```", "")
    return debugged_code, result


def coder(P, prompt, filename, filenames, contexts):
    while True:
        code = complete(P, prompt)
        code = code.replace("```python", "").replace("```", "")
        result = run_python_code(P, code, f"{filename}.py")
        if result.returncode != 0:
            debug_round = 0
            while debug_round < P.debug_round and result.returncode != 0:
                code, result = debug_code(P, filename, code, result.stderr, filenames, contexts)
                debug_round += 1
        if result.returncode == 0:
            break
    return result.stdout, code


def run_agent(P, question, guidelines, filenames, contexts, task_id=0):
    with open(f"tasks/{P.task}/logs{task_id}.txt", "a") as f:
        f.write("="*100+"\n")
        f.write("Question\n")
        f.write("="*100+"\n")
        f.write(question+"\n")
    prompt_for_initial_plan = initial_plan(P, question, filenames, contexts)
    plan = complete(P, prompt_for_initial_plan)
    with open(f"tasks/{P.task}/logs{task_id}.txt", "a") as f:
        f.write("="*100+"\n")
        f.write("Plans\n")
        f.write("="*100+"\n")
        f.write("1."+plan+"\n")
    prompt_for_initial_implementation = initial_implementation(P, plan, filenames, contexts)
    result, code = coder(P, prompt_for_initial_implementation, "intermediate", filenames, contexts)
    with open(f"tasks/{P.task}/logs{task_id}.txt", "a") as f:
        f.write("="*100+"\n")
        f.write("Result\n")
        f.write("="*100+"\n")
        f.write(result+"\n")
    plans = [plan]
    prompt_for_verify = verify(P, plans, code, result, question)
    sufficient_status = complete(P, prompt_for_verify)
    with open(f"tasks/{P.task}/logs{task_id}.txt", "a") as f:
        f.write("="*100+"\n")
        f.write("Validation\n")
        f.write("="*100+"\n")
        f.write(sufficient_status+"\n")
    refinement_num = 0
    while 'No' in sufficient_status:
        prompt_for_routing = router(P, question, filenames, contexts, plans, result)
        while True:
            routing_result = complete(P, prompt_for_routing)
            if routing_result == 'Add Step' or routing_result in [f"Step{i+1}" for i in range(len(plans))]:
                break
        if 'Add Step' not in routing_result:
            step_idx = routing_result.replace("Step", "")
            step_idx = step_idx.replace(" ", "")
            plans = plans[:int(step_idx)-1]
        prompt_for_plan = next_plan(P, question, filenames, contexts, plans, result)
        plan = complete(P, prompt_for_plan)
        plans.append(plan)
        with open(f"tasks/{P.task}/logs{task_id}.txt", "a") as f:
            f.write("="*100+"\n")
            f.write("Plans\n")
            f.write("="*100+"\n")
            for i in range(len(plans)):
                f.write(f"{i+1}.{plans[i]}\n")
        prompt_for_implementation = implement(P, plans, filenames, contexts, code)
        result, code = coder(P, prompt_for_implementation, "intermediate", filenames, contexts)
        with open(f"tasks/{P.task}/logs{task_id}.txt", "a") as f:
            f.write("="*100+"\n")
            f.write("Result\n")
            f.write("="*100+"\n")
            f.write(result+"\n")
        prompt_for_verify = verify(P, plans, code, result, question)
        sufficient_status = complete(P, prompt_for_verify)
        with open(f"tasks/{P.task}/logs{task_id}.txt", "a") as f:
            f.write("="*100+"\n")
            f.write("Validation\n")
            f.write("="*100+"\n")
            f.write(sufficient_status+"\n")
        refinement_num += 1
        if refinement_num > P.max_round-1:
            break
    prompt_for_match_guideline = match_guideline(P, code, result, question, guidelines)
    answer = complete(P, prompt_for_match_guideline)
    with open(f"tasks/{P.task}/logs{task_id}.txt", "a") as f:
        f.write("="*100+"\n")
        f.write("Answer\n")
        f.write("="*100+"\n")
        f.write(answer+"\n")
        f.write("="*100+"\n")
        f.write("Code\n")
        f.write("="*100+"\n")
        f.write(code.replace("```python", "").replace("```", "")+"\n")
    return answer

