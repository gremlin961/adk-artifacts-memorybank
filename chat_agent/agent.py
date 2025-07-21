# Copyright 2024 Google, LLC. This software is provided as-is,
# without warranty or representation for any use or purpose. Your
# use of it is subject to your agreement with Google.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example Agent Workflow using Google's ADK
# 
# This notebook provides an example of building an agentic workflow with Google's new ADK. 
# For more information please visit  https://google.github.io/adk-docs/



# Vertex AI Modules
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part 

# Vertex Agent Modules
from google.adk.agents import Agent # Base class for creating agents
# Removed unused: from google.adk.runners import Runner
# Removed unused: from google.adk.sessions import InMemorySessionService
# Removed unused: from google.adk.artifacts import InMemoryArtifactService, GcsArtifactService
from google.adk.tools.agent_tool import AgentTool # Wrapper to use one agent as a tool for another
from google.adk.tools import google_search # Removed unused: load_artifacts
from google.adk.tools import ToolContext
from google.adk.tools import load_artifacts, load_memory

# Imports for Agent Engine
from vertexai import agent_engines



# Vertex GenAI Modules (Alternative/Legacy way to interact with Gemini, used here for types)
from google.genai import types as types # Used for structuring messages (Content, Part)

# Google Cloud AI Platform Modules
from google.cloud import storage # Client library for Google Cloud Storage (GCS)


# Other Python Modules
import os # For interacting with the operating system (paths, environment variables)
import warnings # For suppressing warnings
import logging # For controlling logging output
import mimetypes

from dotenv import load_dotenv


# --- Configuration ---
load_dotenv()
project_id = os.environ.get('GOOGLE_CLOUD_PROJECT') # Your GCP Project ID
location = os.environ.get('GOOGLE_CLOUD_LOCATION') # Vertex AI RAG location (can be global for certain setups)
#region = os.environ.get('GOOGLE_CLOUD_REGION') # Your GCP region for Vertex AI resources and GCS bucket

# Configuration for Agent Engine
GOOGLE_CLOUD_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
GOOGLE_CLOUD_LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
STAGING_BUCKET = os.environ["STAGING_BUCKET"]



# Ignore all warnings
#warnings.filterwarnings("ignore")
# Set logging level to ERROR to suppress informational messages
#logging.basicConfig(level=logging.ERROR)

# --- Environment Setup ---
# Set environment variables required by some Google Cloud libraries
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1" # Instructs the google.genai library to use Vertex AI backend
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = location

# --- Agent Tool Definitions ---
# @title Define Tools for creating a ticket, adding notes to a ticket, add a file to the session, and getting the GCS URI

async def add_artifact_from_gcs(uri: str, tool_context: ToolContext) -> str: # Modified signature as per user code
    """
    Adds a specific file from Google Cloud Storage (GCS) to the current session state for agent processing.

    This function takes a GCS URI for a file and wraps it in a `types.Content` object.
    This object is then typically used to make the file's content accessible to the
    agent for tasks like summarization, question answering, or data extraction
    related specifically to that file within the ongoing conversation or session.
    The MIME type is assumed to be inferred by the underlying system or defaults.

    Use this function *after* you have identified a specific GCS URI (e.g., using
    `get_gcs_uri` or similar) that you need the agent to analyze or reference directly.

    Args:
        uri: str - The complete Google Cloud Storage URI of the file to add.
                 Must be in the format "gs://bucket_name/path/to/file.pdf".
                 Example: "gs://my-doc-bucket/reports/q1_report.pdf"

    Returns:
         types.Content - A structured Content object representing the referenced file.
                       This object has `role='user'` and contains a `types.Part`
                       that holds the reference to the provided GCS URI.
                       This Content object can be passed to the agent in subsequent calls.
    """
    
    # Determine the bucket name and blob names
    path_part = uri[len("gs://"):]
    # Split only on the first '/' to separate bucket from the rest
    bucket_name, blob_name = path_part.split('/', 1)

    # Initialize GCS client
    storage_client = storage.Client()

    # Get the bucket and blob objects
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the file content as bytes
    file_bytes = blob.download_as_bytes()

    # Determine the mime type of the file based on the file extension
    # Split the URI by the last '/'
    parts = uri.rsplit('/', 1)
    filename = f'user:{parts[-1]}' # The 'user:' prefix changes the scope of the artifact to the user instead of the session

    # Detect the MIME type of the file
    mime_type, encoding = mimetypes.guess_type(filename)
    #mime_type = "application/pdf"
    #print(f"Detected MIME type: {mime_type}")

    if mime_type == "application/json":
        mime_type = "text/plain"

    # This part attempts to create a Content object referencing the GCS URI.
    file_artifact = types.Part(
        inline_data=types.Blob(
            data=file_bytes,
            mime_type=mime_type
        )
    )
    
    version = await tool_context.save_artifact(
        filename=filename, artifact=file_artifact
    )

    print(version)


async def list_artifacts(tool_context: ToolContext) -> str:
    """Tool to list available artifacts for the user."""
    try:
        available_files = await tool_context.list_artifacts()
        if not available_files:
            return "You have no saved artifacts."
        else:
            # Format the list for the user/LLM
            file_list_str = "\n".join([f"- {fname}" for fname in available_files])
            return f"Here are your available Python artifacts:\n{file_list_str}"
    except ValueError as e:
        print(f"Error listing Python artifacts: {e}. Is ArtifactService configured?")
        return "Error: Could not list Python artifacts."
    except Exception as e:
        print(f"An unexpected error occurred during Python artifact list: {e}")
        return "Error: An unexpected error occurred while listing Python artifacts."
    



# --- Agents ---

# -- Search Agent ---
# This agent's role is to perform a Google search for grounding
search_agent = None
search_agent = Agent(
    model="gemini-2.5-flash", # A robuts and responsive model for performing simple actions
    name="search_agent",
    instruction=
    """
        You are a research expert for your company. You will be provided with a request to perform a Google search for something and you will return your findings.
        
        You will use the `google_search` tool to perform a Google search and respond with the results.
        
        An example workflow proceed with your research.
        
        An example workflow would be:
        1: You will be provided with a topic or question to research
        2: Use the `google_search` tool to perform a Google search for the provided question or topic.
        3: Return the response to the calling agent
        
    """,
    description="Performs searches related to a provided question or topic.",
    tools=[
        google_search,
    ],
)


# --- Reasoning Agent ---
# This agent's role is to generate a detailed response to a users question
reasoning_agent = None
reasoning_agent = Agent(
    model="gemini-2.5-pro", # Advanced model for complex tasks and reasoning
    name="reasoning_agent",
    instruction=
    """
        You are a research expert for your company. You will be provided with a request to research something and you will return your findings.
        
        You have access to the following tools
        2: load_artifacts: use this to load artifiacts such as files and images
        3: list_artifacts: use this tool to list any artifiacts you have access to
        
               
        An example workflow would be:
        1: You will be provided with a topic or question to research.
        2: use the list_artifacts tool to see which artifacts you have access to
        2: Use your tools to research the topic or load artifacts.
        3: Return the response to the user
        
    """,
    description="Performs reasearch related to a provided question or topic.",
    tools=[
        #AgentTool(agent=search_agent), # Make the search_agent available as a tool
        list_artifacts,
        load_artifacts,
    ],
)



# --- Root Agent Definition ---
# @title Define the Root Agent with Sub-Agents

# Initialize root agent variables
root_agent = None
runner_root = None # Initialize runner variable (although runner is created later)

    # Define the root agent (coordinator)
search_agent_team = Agent(
    name="search_support_agent",    # Name for the root agent
    #model="gemini-2.5-flash", # Model for the root agent (orchestration)
    model="gemini-2.0-flash-exp", # Model that supports Audio input and output 
    description="The main coordinator agent. Handles user requests and delegates tasks to specialist sub-agents and tools.", # Description (useful if this agent were itself a sub-agent)
    instruction=                  # The core instructions defining the workflow
    """
        You are the lead support coordinator agent. Your goal is to understand the customer's question or topic, and then delegate to the appropriate agent or tool.

        You have access to specialized tools and sub-agents:
        1. AgentTool `reasoning_agent`: Provide the user's question or topic. This agent will research the topic or question and provide a detailed response. The `reasoning_agent`'s response will be streamed directly to the user.
        
      

        Your workflow:
        1. Start by greeting the user.
        2. Ask the user for a GCS bucket and object name.
        3. Use the add_artifact_from_gcs tool to add the file as an artifact to the session.
        4. Once the user provides you with the information, respond with "Okay, I'll start researching that for you. Please wait a moment.".
        5. Pass to the `reasoning_agent` and provide the user's research request. 

       
    """,
    tools=[
        AgentTool(agent=reasoning_agent), # Make the reasoning_agent available as a tool
        add_artifact_from_gcs,
        list_artifacts,
        load_artifacts,
        load_memory,
        
    ],
    sub_agents=[
        #reasoning_agent, # Add the reasoning_agent as a sub-agent
    ],

)

# Assign the created agent to the root_agent variable for clarity in the next step
root_agent = search_agent_team



if __name__ == '__main__':
    vertexai.init(
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        staging_bucket=STAGING_BUCKET,
    )
    
    remote_app = agent_engines.create(
    agent_engine=root_agent,
    display_name='artifact_agent',
    description='Agent used to test artifacts and Memory Bank',
    requirements=[
        "google-cloud-aiplatform[agent_engines,adk]",
        "vertexai",
        "google-cloud-storage",
        "pydantic",
        "python-dotenv",
        "google-genai",   
        ]
    )

    print(remote_app.resource_name)
