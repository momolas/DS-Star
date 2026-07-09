# <div align="center">DS-STAR: Data Science Agent for Solving Diverse Tasks across Heterogeneous Formats and Open-Ended Queries</div>
<div align="center">Jaehyun Nam<sup>1</sup>, Jinsung Yoon<sup>1</sup>, Jiefeng Chen<sup>1</sup>, Raj Sinha<sup>1</sup>, Jinwoo Shin<sup>2</sup>, and Tomas Pfister<sup>1</sup></div>
<div align="center"><sup>1</sup>Google Cloud AI Research, <sup>2</sup>KAIST</div>
<br><br>

DS-STAR is a state-of-the-art data science agent whose versatility is shown by its ability to automate a range of tasks, from statistical analysis and data wrangling to visualization and deep data research, across various data types.

<div align="center">
  <img src="assets/overview.pdf" alt="DS-STAR Overview" width="90%"/>
</div>

## Key Features

- **Multi-Agent Pipeline**: Orchestrates specialized agents (Data File Analyzer, Planner, Coder, Verifier, Report Writer, etc.) for end-to-end data science automation.
- **Diverse Queries**: Solves Machine Learning, Data Analysis, Data Wrangling, Data Manipulation, Statistical Analysis, Visualization, Deep Data Research, etc.
- **Diverse Datasets**: Handles various heterogeneous data types (CSV, JSON, Markdown, Text, PDF, XLSX, etc.).

## Installation

1.  **Environment Setup**: The project relies on a Conda environment named `ds_star`. You can set up a similar environment with Python 3.11 and install the required dependencies.
    ```bash
    conda create -n ds_star python=3.11
    conda activate ds_star
    pip install -r requirements.txt
    ```

2.  **Environment Variables**: Create a `.env` file in the root of the project and add your API keys:
    ```env
    # Add keys if you have them, otherwise leave them empty
    # You need to specify related keys if you want to use the model
    # For example, if you want to use Gemini models, you need to specify either VERTEX_AI_PROJECT and VERTEX_AI_LOCATION, or GEMINI_API_KEY
    # If you want to use OpenAI models, you need to specify OPENAI_API_KEY
    
    OPENAI_API_KEY=your_openai_api_key
    VERTEX_AI_PROJECT=your_vertex_ai_project
    VERTEX_AI_LOCATION=your_vertex_ai_location
    GEMINI_API_KEY=your_gemini_api_key
    ```

## Usage

You can automate the data science task using the provided Python script.

### DS-STAR for well-defined queries
```bash
python run_ds_star.py \
  --task '<your_task>'
```

### DS-STAR+ for deep data research
```bash
python run_ds_star_plus.py \
  --task '<your_task>'
```

### Key Arguments
*   `--llm`: LLM for agents (e.g., gemini-3.5-flash, gpt-5).
*   `--degug_round`: Maximum round for debugging code scripts.
*   `--task`: (Required) Directory name of the data science task.
*   `--max_workers`: Maximum number of workers for parallel execution.
*   `--max_round`: Maximum number of rounds for DS-STAR.
*   `--report_refine_num`: Number of refinement rounds for DS-STAR+.

## Project Structure

*   `agents/`: Core logic for agents and pipeline execution.
*   `tools/`: Utility functions, e.g., LLM calls.
*   `tasks/<task>/question.txt`: (Required) Data science query in natural language.
*   `tasks/<task>/guidelines.txt`: (Optional) Formatting guideline for the data science query.
*   `tasks/<task>/data/`: Required data files for data science query.

## Citation

If you find this repo or our paper helpful, please cite it as follows:

```bibtex
@article{nam2026ds,
  title={DS-STAR: Data Science Agent for Solving Diverse Tasks across Heterogeneous Formats and Open-Ended Queries},
  author={Nam, Jaehyun and Yoon, Jinsung and Chen, Jiefeng and Sinha, Raj and Shin, Jinwoo and Pfister, Tomas},
  journal={arXiv preprint arXiv:2509.21825},
  year={2026}
}
```

# Disclaimer

This is not an officially supported Google product.