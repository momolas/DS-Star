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
from tools.utils import parse_output
from agents.analyzer import agent_analyze_parallel
from agents.prompts import question_generation
from agents.prompts import generate_report
from agents.prompts import question_generation_for_refine
from agents.prompts import refine_report
from agents.ds_star import run_agent
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


def deep_data_research(P, question):
    filenames = os.listdir(f"tasks/{P.task}/data")
    agent_analyze_parallel(P, filenames)
    contexts = []
    for filename in filenames:
        with open(f"tasks/{P.task}/analytics/{filename}.txt", "r") as f:
            contexts.append(f.read())
    prompt = question_generation(P, question, filenames, contexts)
    while True:
        subquestions = complete(P, prompt)
        subquestions = parse_output(subquestions)
        if subquestions is not False:
            break
    subquestions = [subquestions[i]['question'] for i in range(len(subquestions))]
    subanswers = [None] * len(subquestions)
    with ThreadPoolExecutor(max_workers=P.max_workers) as executor:
        guidelines = "Answer with sufficient explanation."
        futures = {
            executor.submit(run_agent, P, subquestions[i], guidelines, filenames, contexts, i): i
            for i in range(len(subquestions))
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            subanswers[index] = result
    prompt = generate_report(P, question, subquestions, subanswers)
    report = complete(P, prompt)
    with open(f"tasks/{P.task}/report0.md", "a") as f:
        f.write(report)
    report_logs = [report]
    num_previous_sub_questions = len(subquestions)
    previous_sub_questions = subquestions
    previous_sub_answers = subanswers
    for refine_num in range(P.report_refine_num):
        prompt = question_generation_for_refine(P, question, filenames, contexts, report_logs[-1], previous_sub_questions, previous_sub_answers)
        while True:
            subquestions = complete(P, prompt)
            subquestions = parse_output(subquestions)
            if subquestions is not False:
                break
        subquestions = [subquestions[i]['question'] for i in range(len(subquestions))]
        subanswers = [None] * len(subquestions)
        with ThreadPoolExecutor(max_workers=P.max_workers) as executor:
            guidelines = "Answer with sufficient explanation."
            futures = {
                executor.submit(run_agent, P, subquestions[i], guidelines, filenames, contexts, i+num_previous_sub_questions): i
                for i in range(len(subquestions))
            }
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                subanswers[index] = result
        prompt = refine_report(P, question, subquestions, subanswers, report_logs[-1], num_previous_sub_questions)
        report = complete(P, prompt)
        report_logs.append(report)
        with open(f"tasks/{P.task}/report{refine_num+1}.md", "a") as f:
            f.write(report)
        num_previous_sub_questions += len(subquestions)
        previous_sub_questions += subquestions
        previous_sub_answers += subanswers
    return report_logs[-1]
