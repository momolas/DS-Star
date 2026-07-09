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
from agents.prompts import code_for_analyze
from agents.prompts import bug_summarization
from agents.prompts import debug_for_analyze
from concurrent.futures import ThreadPoolExecutor


def debugging_analyze(P, code, bug, filename):
    prompt_for_bug_summarization = bug_summarization(P, filename, bug)
    bug = complete(P, prompt_for_bug_summarization)
    prompt_for_debugging = debug_for_analyze(P, code, bug)
    debugged_code = complete(P, prompt_for_debugging)
    result = run_python_code(P, debugged_code, f"analyze_{filename}.py")
    debugged_code = debugged_code.replace("```python", "").replace("```", "")
    return debugged_code, result


def agent_analyze(P, filename):
    prompt = code_for_analyze(P, filename)
    while True:
        code = complete(P, prompt)
        result = run_python_code(P, code, f"analyze_{filename}.py")
        if result.returncode != 0:
            debug_round = 0
            while debug_round < P.debug_round and result.returncode != 0:
                code, result = debugging_analyze(P, code, result.stderr, filename)
                debug_round += 1
        if result.returncode == 0:
            break
    os.makedirs(f"tasks/{P.task}/analytics", exist_ok=True)
    with open(f"tasks/{P.task}/analytics/{filename}.txt", "w") as f:
        f.write(result.stdout)
    print(f"Analyzed {filename} successfully!")
        
        
def agent_analyze_parallel(P, filenames):
    with ThreadPoolExecutor(max_workers=P.max_workers) as executor:
        futures = [executor.submit(agent_analyze, P, filename) for filename in filenames]
        for future in futures:
            future.result()
    print("Analyzer Agent worked successfully!")
