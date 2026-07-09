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

import ast
import subprocess
import os


def parse_output(json_str):
    start_idx, end_idx = json_str.find("["), json_str.rfind("]")+1
    try:
        output = ast.literal_eval(json_str[start_idx:end_idx])
    except:
        output = False
    return output


def run_python_code(P, response, py_filepath):
    code_text = response.replace("```python", "").replace("```", "")
    output_filepath = os.path.join(f"tasks/{P.task}", py_filepath)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(code_text)
    try:
        result = subprocess.run(
            ["python", py_filepath],
            cwd=f"tasks/{P.task}",
            capture_output=True,
            text=True)
    except:
        result = "Failed"
    if result == "Failed":
        return run_python_code(P, response, py_filepath)
    else:
        return result
