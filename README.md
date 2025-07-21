# ADK Artifacts Memory Bank Example

This project demonstrates how to create persistent artifacts using the ADK and save important information into long-term memory using Memory Bank.

## Setup and Usage

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Create a `.env` file:**
    In the `chat_agent` folder, create a file named `.env` with the following content:
    ```
    GOOGLE_GENAI_USE_VERTEXAI=TRUE
    GOOGLE_CLOUD_PROJECT=your_project_id # Replace with your Google Cloud Project ID
    GOOGLE_CLOUD_LOCATION=us-central1          # Or your specific location
    STAGING_BUCKET=gs://your_staging_bucket # Replace with the GCS bucket name for staging files to build the Agent Engine instance
    ```

3.  **Deploy the agent to Agent Engine:**
    ```bash
    python chat_agent/agent.py
    ```
    This will deploy your agents to Agent Engine and display the new agent engine ID.

4.  **Configure `main.py`:**
    Open `main.py` and update the "Change these variables" section with your information, including the agent engine ID from the previous step.

5.  **Launch the UI:**
    ```bash
    python main.py
    ```
