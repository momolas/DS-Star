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
import time
import requests
from google import genai
from openai import OpenAI
from google.genai.types import GenerateContentConfig
from concurrent.futures import TimeoutError
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()


def complete(P, prompt, temperature=1.0):
    if 'gpt' in P.llm:
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        waiting_time = 0.5
        response = None
        while response is None:
            try:
                response = client.responses.create(
                    model=P.llm,
                    input=prompt
                )
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                time.sleep(waiting_time)
                if waiting_time < 5:
                    waiting_time += 0.5
        result = response.output_text
        if isinstance(result, str):
            return result
        else:
            return complete(P, prompt, temperature=temperature)   
            
    if 'gemini' in P.llm:
        waiting_time = 0.5
        response = None
        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=65535,
            candidate_count=1
        )
        while response is None:
            try:
                if os.getenv("VERTEX_AI_PROJECT") is not None:
                    client = genai.Client(vertexai=True, project=os.getenv("VERTEX_AI_PROJECT"), location=os.getenv("VERTEX_AI_LOCATION"))
                else:
                    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        client.models.generate_content,
                        model=P.llm,
                        contents=prompt,
                        config=config
                    )
                    try:
                        response = future.result(timeout=4*60)
                    except TimeoutError:
                        print("TimeoutError: future.result() timed out")
                        response = None
                if response is not None and not response.candidates:
                    response = None
                elif response is not None and response.candidates is not None and not response.candidates[0].content:
                    response = None
                elif response is not None and response.candidates is not None and not response.candidates[0].content.parts:
                    response = None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                time.sleep(waiting_time)
                if waiting_time < 5:
                    waiting_time += 0.5     
        result = response.candidates[0].content.parts[0].text
        output_token = response.usage_metadata.candidates_token_count
        if isinstance(output_token, int):
            return result
        else:
            return complete(P, prompt, temperature=temperature)
