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

def code_for_analyze(P, filename):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"Generate a Python code that loads and describes the content of {filename}.\n"
    prompt += f"\n"
    prompt += f"# Requirement\n"
    prompt += f"- The file can both unstructured or structured data.\n"
    prompt += f"- If there are too many structured data, print out just few examples. Do not print out all raw data.\n"
    prompt += f"- Print out essential informations. For example, print out all the column names.\n"
    prompt += f"- The Python code should print out the content of {filename}.\n"
    prompt += f"- The code should be a single-file Python program that is self-contained and can be exectued as-is.\n"
    prompt += f"- Your response should only contain a single code block.\n"
    prompt += f"- Do not include dummy contents since we will debug if error occurs.\n"
    prompt += f"- Do not use try: and except: to prevent error. I will debug it later."
    return prompt


def bug_summarization(P, filename, bug):
    prompt = ""
    prompt += f"# Error report\n"
    prompt += f"{bug}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Remove all unnecessary parts of the above error report.\n"
    prompt += f"- We are now running {filename}.py. Do not remove where the error occurred."
    return prompt


def debug_for_analyze(P, code, bug):
    prompt = f"# Code with an error:\n"
    prompt += f"```python\n{code}\n```\n"
    prompt += f"\n"
    prompt += f"# Error:\n"
    prompt += f"{bug}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Please revise the code to fix the error.\n"
    prompt += f"- Provide the improved, self-contained Python script again.\n"
    prompt += f"- There should be no additional headings or text in your response.\n"
    prompt += f"- Do not include dummy contents since we will debug if error occurs.\n"
    prompt += f"- All files and documents are in `./data` directory."
    return prompt


def debug_with_context(P, code, bug, filenames, contexts):
    prompt = f"# Given data: {filenames}\n"
    for i in range(len(filenames)):
        prompt += f"./data/{filenames[i]}\n"
        prompt += f"{contexts[i]}\n"
        prompt += f"\n"
    prompt += f"# Code with an error:\n"
    prompt += f"```python\n{code}\n```\n"
    prompt += f"\n"
    prompt += f"# Error:\n"
    prompt += f"{bug}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Please revise the code to fix the error.\n"
    prompt += f"- Provide the improved, self-contained Python script again.\n"
    prompt += f"- Note that you only have {filenames} available.\n"
    prompt += f"- There should be no additional headings or text in your response.\n"
    prompt += f"- Do not include dummy contents since we will debug if error occurs.\n"
    prompt += f"- All files and documents are in `./data` directory."
    return prompt


def initial_plan(P, question, filenames, contexts):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"In order to answer factoid questions based on the given data, you have to first plan effectively.\n"
    prompt += f"\n"
    prompt += f"# Question\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Given data: {filenames}\n"
    for i in range(len(filenames)):
        prompt += f"./data/{filenames[i]}\n"
        prompt += f"{contexts[i]}\n"
        prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Suggest your very first step to answer the question above.\n"
    prompt += f"- Your first step does not need to be sufficient to answer the question.\n"
    prompt += f"- Just propose a very simple inital step, which can act as a good starting point to answer the question.\n"
    prompt += f"- Your response should only contain an initial step."
    return prompt


def initial_implementation(P, plan, filenames, contexts):
    prompt = f"# Given data: {filenames}\n"
    for i in range(len(filenames)):
        prompt += f"./data/{filenames[i]}\n"
        prompt += f"{contexts[i]}\n"
        prompt += f"\n"
    prompt += f"# Plan\n"
    prompt += f"{plan}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Implement the plan with the given data.\n"
    prompt += f"- Do not print out too much.\n"
    prompt += f"- Your response should be a single markdown Python code (wrapped in ```).\n"
    prompt += f"- There should be no additional headings or text in your response."
    return prompt


def verify(P, plans, code, result, question):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"Your task is to check whether the current plan and its code implementation is enough to answer the question.\n"
    prompt += f"# Plan\n"
    for i in range(len(plans)):
        prompt += f"{i+1}. {plans[i]}\n"
    prompt += f"\n"
    prompt += f"# Code\n"
    prompt += f"```python\n{code}\n```\n"
    prompt += f"\n"
    prompt += f"# Execution result of code\n"
    prompt += f"{result}\n"
    prompt += f"\n"
    prompt += f"# Question\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Verify whether the current plan and its code implementation is enough to answer the question.\n"
    prompt += f"- Your response should only be one of 'Yes' or 'No', without any explanation.\n"
    prompt += f"- If it is enough to answer the question, please answer 'Yes'.\n"
    prompt += f"- Otherwise, please answer 'No'."
    return prompt


def next_plan(P, question, filenames, contexts, plans, result):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"In order to answer factoid questions based on the given data, you have to first plan effectively.\n"
    prompt += f"Your task is to suggest next plan to do to answer the question.\n"
    prompt += f"\n"
    prompt += f"# Question\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Given data: {filenames}\n"
    for i in range(len(filenames)):
        prompt += f"./data/{filenames[i]}\n"
        prompt += f"{contexts[i]}\n"
        prompt += f"\n"
    prompt += f"# Current plans\n"
    for i in range(len(plans)):
        prompt += f"{i+1}. {plans[i]}\n"
    prompt += f"\n"
    prompt += f"# Obtained results from the current plans:\n"
    prompt += f"{result}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Suggest your next step to answer the question above.\n"
    prompt += f"- Your next step does not need to be sufficient to answer the question, but if it requires only final simple last step you may suggest it.\n"
    prompt += f"- Just propose a very simple next step, which can act as a good intermediate point to answer the question.\n"
    prompt += f"- Of course your response can be a plan which could directly answer the question.\n"
    prompt += f"- Your response should only contain an next step without any explanation."
    return prompt


def router(P, question, filenames, contexts, plans, result):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"In order to answer factoid questions based on the given data, you have to first plan effectively.\n"
    prompt += f"Since current plan is insufficient to answer the question, your task is to decide how to refine the plan to answer the question.\n"
    prompt += f"\n"
    prompt += f"# Question\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Given data: {filenames}\n"
    for i in range(len(filenames)):
        prompt += f"./data/{filenames[i]}\n"
        prompt += f"{contexts[i]}\n"
        prompt += f"\n"
    prompt += f"# Current plans\n"
    for i in range(len(plans)):
        prompt += f"Step{i+1}. {plans[i]}\n"
    prompt += f"\n"
    prompt += f"# Obtained results from the current plans:\n"
    prompt += f"{result}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- If you think one of the steps of current plans is wrong, answer among the following options: "
    for i in range(len(plans)-1):
        prompt += f"Step{i+1}, "
    prompt += f"Step{len(plans)}.\n"
    prompt += f"- If you think we should perform new NEXT step, answer as 'Add Step'.\n"
    prompt += f"- Your response should only be 'Step1' - 'Step{len(plans)}' or 'Add Step', without any explanation."
    return prompt


def implement(P, plans, filenames, contexts, code):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"Your task is to implement the next plan with the given data.\n"
    prompt += f"# Given data: {filenames}\n"
    for i in range(len(filenames)):
        prompt += f"./data/{filenames[i]}\n"
        prompt += f"{contexts[i]}\n"
        prompt += f"\n"
    prompt += f"# Base code\n"
    prompt += f"```python\n{code}\n```\n"
    prompt += f"\n"
    prompt += f"# Previous plans\n"
    for i in range(len(plans)-1):
        prompt += f"{i+1}. {plans[i]}\n"
    prompt += f"\n"
    prompt += f"# Current plan to implement\n"
    prompt += f"{plans[-1]}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Implement the current plan with the given data.\n"
    prompt += f"- The implementation should be done based on the base code.\n"
    prompt += f"- The base code is an implementation of the previous plans.\n"
    prompt += f"- Do not print out too much.\n"
    prompt += f"- Your response should be a single markdown Python code (wrapped in ```).\n"
    prompt += f"- There should be no additional headings or text in your response."
    return prompt


def match_guideline(P, code, result, question, guidelines):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"Your task is to generate the answer of the question following the given guideline.\n"
    prompt += f"\n"
    prompt += f"# Reference code\n"
    prompt += f"```python\n{code}\n```\n"
    prompt += f"\n"
    prompt += f"# Execution result of reference code\n"
    prompt += f"{result}\n"
    prompt += f"\n"
    prompt += f"# Question\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Guidelines\n"
    prompt += f"{guidelines}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Output an answer, following the given guidelines.\n"
    return prompt


def question_generation(P, question, filenames, contexts):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"Your task is to write a comprehensive data science report to the given question by using the data listed below.\n"
    prompt += f"In order to do this, you have to first suggest multiple data analysis questions that should be answered to write the report.\n"
    prompt += f"\n"
    prompt += f"# Given data: {filenames}\n"
    for i in range(len(filenames)):
        prompt += f"{filenames[i]}\n"
        prompt += f"{contexts[i]}\n"
        prompt += f"\n"
    prompt += f"# Question\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Suggest multiple factoid data analysis questions that are required to write the report really well.\n"
    prompt += f"- All the questions should be well-answered using the given data.\n"
    prompt += f"- All questions should be answered independently.\n"
    prompt += f"- Expected answer for each question should be concise (e.g., answers should not be a billion rows).\n"
    prompt += f"- Return in valid JSON format:\n"
    prompt +=    "Question = {'question': str}\n"
    prompt += f"Return: list[Question]\n"
    return prompt


def question_generation_for_refine(P, question, filenames, contexts, report, previous_subquestions, previous_subanswers):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"Your task is to complement the given data science report of the given question.\n"
    prompt += f"In order to do this, you have to suggest supplementary multiple data analysis questions that can strengthen to the report.\n"
    prompt += f"\n"
    prompt += f"# Given data: {filenames}\n"
    for i in range(len(filenames)):
        prompt += f"{filenames[i]}\n"
        prompt += f"{contexts[i]}\n"
        prompt += f"\n"
    prompt += f"# Given data science report:\n"
    prompt += f"{report}\n"
    prompt += f"\n"
    prompt += f"# Question\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Used data analysis questions to write the above report:\n"
    for i in range(len(previous_subquestions)):
        prompt += f"## Sub-Question {i}:\n"
        prompt += f"{previous_subquestions[i]}\n"
        prompt += f"## Sub-Answer {i}:\n"
        prompt += f"{previous_subanswers[i]}\n"
        prompt += f"\n"
    prompt += f"\n"
    prompt += f"# Your task\n"
    prompt += f"- Suggest multiple factoid data analysis questions that are required to complement the report.\n"
    prompt += f"- All questions should contain new information that is not included in the report.\n"
    prompt += f"- All the questions should be well-answered using the given data.\n"
    prompt += f"- All questions should be answered independently.\n"
    prompt += f"- All questions are different from the used data analysis questions.\n"
    prompt += f"- Expected answer for each question should be concise (e.g., answers should not be a billion rows).\n"
    prompt += f"- Return in valid JSON format:\n"
    prompt +=    "Question = {'question': str}\n"
    prompt += f"Return: list[Question]\n"
    return prompt


def generate_report(P, question, subquestions, subanswers):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"Your task is to write a **comprehensive data science report** to the given question by using some relevant informations listed below.\n"
    prompt += f"\n"
    prompt += f"# Relevant informations:\n"
    for i in range(len(subquestions)):
        prompt += f"## Sub-Question {i}:\n"
        prompt += f"{subquestions[i]}\n"
        prompt += f"## Sub-Answer {i}:\n"
        prompt += f"{subanswers[i]}\n"
        prompt += f"\n"
    prompt += f"# Question that you have to write a comprehensive data science report:\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Your task:\n"
    prompt += f"- The report should be grounded to the given relevant informations.\n"
    prompt += f"- For the citation, use the Sub-Question number as a citation number (e.g., cite with [0] for the Sub-Question 0).\n"
    prompt += f"- All the {len(subquestions)} Sub-Questions must be used.\n"
    prompt += f"- The data science report should be relevant to given question, should be comprehensive, and should be insightful.\n"
    prompt += f"- The data science report should have nice structure, good readability, and should be professionl.\n"
    prompt += f"- Write a very comprehensive data science report to the given above question."
    return prompt


def refine_report(P, question, subquestions, subanswers, report, num_previous_sub_questions):
    prompt = f"You are an expert data analysist.\n"
    prompt += f"Your task is to complement the given data science report of the given question by using the some relevant informations listed below.\n"
    prompt += f"\n"
    prompt += f"# Additional relevant informations:\n"
    for i in range(len(subquestions)):
        prompt += f"## Sub-Question {i+num_previous_sub_questions}:\n"
        prompt += f"{subquestions[i]}\n"
        prompt += f"## Sub-Answer {i+num_previous_sub_questions}:\n"
        prompt += f"{subanswers[i]}\n"
        prompt += f"\n"
    prompt += f"# Given data science report:\n"
    prompt += f"{report}"
    prompt += f"\n"
    prompt += f"# Question that you have to write a comprehensive data science report:\n"
    prompt += f"{question}\n"
    prompt += f"\n"
    prompt += f"# Your task:\n"
    prompt += f"- Do not modify the given report a lot. Just try to add new information. Try to use all the additional relevant informations.\n"
    prompt += f"- The report should be grounded to the given relevant informations.\n"
    prompt += f"- For the citation, use the Sub-Question number as a citation number (e.g., cite with [13] for the Sub-Question 13).\n"
    prompt += f"- The data science report should be relevant to given question, should be comprehensive, and should be insightful.\n"
    prompt += f"- The data science report should have nice structure, good readability, and should be professionl.\n"
    prompt += f"- Complement the give data science report to the given above question."
    return prompt
