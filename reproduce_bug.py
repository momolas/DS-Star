
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies before importing dsstar
sys.modules['google.generativeai'] = MagicMock()
sys.modules['openai'] = MagicMock()

# Now import dsstar
import dsstar
from dsstar import DS_STAR_Agent, DSConfig

def test_infinite_loop():
    print("Setting up test...")
    
    # Config with auto_debug enabled
    config = DSConfig(
        run_id="test_run",
        auto_debug=True,
        api_key="dummy",
        model_name="gemini-1.5-flash",
        runs_dir="test_runs"
    )
    
    agent = DS_STAR_Agent(config)
    
    # Mock the controller to avoid file system ops and logging
    agent.controller.execute_step = MagicMock(side_effect=lambda step_name, step_func, **kwargs: step_func())
    agent.controller.logger = MagicMock()
    
    # Mock _call_model to always return some code
    agent._call_model = MagicMock(return_value="```python\nprint('fixed')\n```")
    
    # Mock _execute_code to ALWAYS fail
    # We use a counter to break the loop manually if it goes too long, proving the infinite loop
    execution_count = 0
    
    def mock_execute(code, data_files=None):
        nonlocal execution_count
        execution_count += 1
        print(f"Execution attempt {execution_count}")
        if execution_count > 10:
            raise Exception("Infinite loop detected! Stopping test.")
        return "", "Some error occurred"

    agent._execute_code = MagicMock(side_effect=mock_execute)
    
    # Mock other methods to bypass them
    agent.plan_next_step = MagicMock(return_value="Plan step 1")
    agent.generate_code = MagicMock(return_value="```python\nprint('initial')\n```")
    agent.analyze_data = MagicMock(return_value={"result": "summary"})
    
    # Run pipeline (mocking data files)
    try:
        print("Starting pipeline...")
        # We only need to trigger the phase 2 loop
        # But run_pipeline is complex. Let's just call the logic inside phase 2 if possible.
        # Or just run_pipeline and let it hit the loop.
        agent.run_pipeline("query", ["data.csv"])
    except Exception as e:
        print(f"\nCaught expected exception: {e}")
        if "Infinite loop detected" in str(e):
            print("SUCCESS: Infinite loop bug reproduced.")
        else:
            print(f"FAILURE: Unexpected exception: {e}")

if __name__ == "__main__":
    test_infinite_loop()
