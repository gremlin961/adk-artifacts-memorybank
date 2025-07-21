import os
import json
import asyncio
import logging

from pathlib import Path
from typing import Dict # For type hinting active_websockets

from google.genai.types import (
    Part,
    Content,
)

from google.adk.runners import Runner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.sessions import VertexAiSessionService

from google.adk.artifacts import GcsArtifactService

from google.adk.memory import VertexAiMemoryBankService

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from starlette.websockets import WebSocketState

# Imports from the status-messenger package
import status_messenger
from status_messenger.messenger import current_websocket_session_id_var # Import ContextVar directly
from status_messenger import setup_status_messenger_async, stream_status_updates # Other package imports

# --- Agent Imports for the Google ADK ---
from chat_agent.agent import root_agent # Using the agent from chat_agent
# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#---- Change these variables to match your project -----------
PROJECT_ID = "YOUR_PROJECT_ID" # Your GCP Project ID
LOCATION = "us-central1" # Your GCP Location (keep it as us-central1 for now)
USER_ID = "user_123" # The user ID you want to use for your Agent session
gcs_artifact_bucket = "YOUR_ARTIFACT_BUCKET" # The GCS bucket name to store artifacts
agent_engine_id = "2246055964941746176" # Your Agent Engine ID, created by running "python chat_agent/agent.py"
#-------------------------------------------------------------

APP_NAME = agent_engine_id
session_service = VertexAiSessionService(project=PROJECT_ID, location=LOCATION)
artifact_service = GcsArtifactService(bucket_name=gcs_artifact_bucket)
memory_service = VertexAiMemoryBankService(
    project=PROJECT_ID,
    location=LOCATION,
    agent_engine_id=agent_engine_id
)

# Global store for active websockets, mapping session_id to WebSocket object
active_websockets: Dict[str, WebSocket] = {}

async def start_agent_session(session_id: str, user_id: str): # Changed to async def
    """Starts an ADK agent session."""
    logger.info(f"[{session_id}] Attempting to start agent session for user {user_id}.")
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        #session_id=session_id,
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
        artifact_service=artifact_service,
        memory_service=memory_service,
    )
    run_config = RunConfig(response_modalities=["TEXT"])
    live_request_queue = LiveRequestQueue()
    live_events = runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )
    logger.info(f"[{session_id}] Agent session started with session.id: {session.id}. Live events queue created.")
    return live_events, live_request_queue, runner, session.id

async def agent_to_client_messaging(websocket: WebSocket, live_events, session_id: str):
    """Handles messages from ADK agent to the WebSocket client."""
    async for event in live_events:
        message_to_send = None
        server_log_detail = None
        if event.turn_complete:
            server_log_detail = "Agent turn complete."
            message_to_send = {"type": "agent_turn_complete", "turn_complete": True}
        elif event.interrupted:
            server_log_detail = "Agent turn interrupted."
            message_to_send = {"type": "agent_interrupted", "interrupted": True}
        else:
            part: Part = (event.content and event.content.parts and event.content.parts[0])
            if part and part.text:
                text = part.text
                message_to_send = {"type": "agent_message", "message": text}

        if server_log_detail:
            logger.info(f"[{session_id}] AGENT->CLIENT_TASK: {server_log_detail}")
            # Removed call to send_server_log_to_client

        if message_to_send:
            await websocket.send_text(json.dumps(message_to_send))
    logger.info(f"[{session_id}] Live events stream from agent finished.")
    logger.info(f"[{session_id}] Agent-to-client messaging task finished.")

async def client_to_agent_messaging(websocket: WebSocket, live_request_queue: LiveRequestQueue, session_id: str):
    """Handles messages from WebSocket client to the ADK agent."""
    try:
        while True:
            text = await websocket.receive_text()
            logger.info(f"[{session_id}] CLIENT->AGENT_TASK: Received text: '{text}'")
            content = Content(role="user", parts=[Part.from_text(text=text)])
            live_request_queue.send_content(content=content)
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected by client.")
    except asyncio.CancelledError:
        logger.info(f"[{session_id}] Client-to-agent messaging task cancelled.")
    finally:
        if live_request_queue: # Ensure queue is closed if it exists
            live_request_queue.close()
        logger.info(f"[{session_id}] Client-to-agent messaging task finished.")


# Status Messenger Functions
async def broadcast_status_to_client(websocket: WebSocket, status_text: str, session_id: str):
    if websocket.client_state == WebSocketState.CONNECTED:
        payload = {"type": "status", "data": status_text}
        await websocket.send_text(json.dumps(payload))
        logger.info(f"[{session_id}] SENT_STATUS_TO_CLIENT: {status_text}")
    else:
        logger.warn(f"[{session_id}] WebSocket not connected, cannot send status: {status_text}")


async def status_message_broadcaster():
    logger.info("Status message broadcaster starting.")
    async for ws_session_id, message in status_messenger.stream_status_updates():
        if ws_session_id is None: # Should ideally not happen if ContextVar is always set
            logger.warn(f"Status message with no session ID: {message}. Not broadcasting.")
            continue
        
        ws = active_websockets.get(ws_session_id)
        if ws:
            try:
                await broadcast_status_to_client(ws, message, ws_session_id)
            except Exception as e:
                logger.error(f"[{ws_session_id}] Error sending status via broadcaster: {e}", exc_info=True)
        else:
            # This can happen if the client disconnects but a message was already in queue.
            logger.warn(f"[{ws_session_id}] No active WebSocket for status: {message}")

app = FastAPI(title=APP_NAME, version="0.1.0")

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    # Initialize the status messenger system
    setup_status_messenger_async(loop)
    # Start the background task that broadcasts messages from the queue
    asyncio.create_task(status_message_broadcaster(), name="status_message_broadcaster_task")
    logger.info("Status message broadcaster task scheduled.")

origins = ["*",]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Static files mounted from {STATIC_DIR}")
else:
    logger.error(f"Static directory or index.html not found at {STATIC_DIR}. Frontend may not load.")

@app.get("/")
async def root_path():
    index_html_path = STATIC_DIR / "index.html"
    if index_html_path.is_file():
        return FileResponse(index_html_path)
    logger.error(f"index.html not found at {index_html_path}")
    return {"error": "index.html not found"}, 404

@app.websocket("/ws/{session_id_from_path}")
async def websocket_endpoint(websocket: WebSocket, session_id_from_path: str):
    session_id = session_id_from_path
    user_id = USER_ID  # Define user_id
    await websocket.accept()
    active_websockets[session_id] = websocket # Register connection
    logger.info(f"[{session_id}] WebSocket connected and registered for user {user_id}.")
    
    context_token = None
    live_events = None
    live_request_queue = None
    runner = None  # Initialize runner to None
    agent_to_client_task = None
    client_to_agent_task = None

    try:
        # CRITICAL: Set the session ID in the context for this async task chain.
        context_token = current_websocket_session_id_var.set(session_id)
        logger.info(f"[{session_id}] Initializing agent session backend.")
        live_events, live_request_queue, runner, adk_session_id = await start_agent_session(session_id, user_id)

        agent_to_client_task = asyncio.create_task(
            agent_to_client_messaging(websocket, live_events, session_id),
            name=f"agent_to_client_{session_id}"
        )
        client_to_agent_task = asyncio.create_task(
            client_to_agent_messaging(websocket, live_request_queue, session_id),
            name=f"client_to_agent_{session_id}"
        )
        
        # Wait for both tasks to complete.
        # asyncio.gather will propagate the first exception raised by any of the tasks.
        # If client_to_agent_task ends due to WebSocketDisconnect, gather will raise it (or complete if no exception).
        # If agent_to_client_task ends (e.g. agent turn done), gather will wait for client_to_agent_task.
        try:
            await asyncio.gather(agent_to_client_task, client_to_agent_task)
        except WebSocketDisconnect:
            logger.info(f"[{session_id}] A task ended due to WebSocketDisconnect (expected on client close).")
        # Other exceptions will propagate and be caught by the outer try/except.
        # Log completion of tasks if gather finishes without raising an unhandled exception from them.
        if agent_to_client_task.done() and not agent_to_client_task.exception():
            logger.info(f"[{session_id}] Task {agent_to_client_task.get_name()} completed.")
        if client_to_agent_task.done() and not client_to_agent_task.exception():
             logger.info(f"[{session_id}] Task {client_to_agent_task.get_name()} completed.")

    except WebSocketDisconnect: # Catch WebSocketDisconnect specifically if it propagates from gather
        logger.info(f"[{session_id}] WebSocket disconnected by client (caught in outer try-block).")
    except Exception as e: # General exception catch for the whole endpoint
        logger.error(f"[{session_id}] Unhandled error in websocket_endpoint's try-block: {e}", exc_info=True)
    finally:
        logger.info(f"[{session_id}] Client disconnecting / cleaning up tasks...")
        if context_token: # CRITICAL: Reset the context variable.
            current_websocket_session_id_var.reset(context_token)
        active_websockets.pop(session_id, None) # Unregister connection
        logger.info(f"[{session_id}] WebSocket session cleaned up.")

        # Simpler task cancellation
        all_tasks = []
        if agent_to_client_task: all_tasks.append(agent_to_client_task)
        if client_to_agent_task: all_tasks.append(client_to_agent_task)
        if 'pending' in locals() and pending: all_tasks.extend(list(pending))
        
        for task in all_tasks:
            if task and not task.done():
                logger.info(f"[{session_id}] Cancelling task: {task.get_name()}")
                task.cancel()
                # No need to await cancellation for simplicity in this example

        if runner:
            logger.info(f"[{session_id}] Generating and saving session memories for adk_session_id: {adk_session_id}")
            completed_session = await runner.session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=adk_session_id)
            await memory_service.add_session_to_memory(completed_session)
            logger.info(f"[{session_id}] Closing ADK live_request_queue.")
            live_request_queue.close() # Simplified: no try-except
        
        if websocket.client_state == WebSocketState.CONNECTED:
            logger.info(f"[{session_id}] Server explicitly closing WebSocket.")
            await websocket.close(code=1000) # Simplified: no try-except
        logger.info(f"[{session_id}] Client cleanup finished.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Uvicorn server on http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, app_dir=str(Path(__file__).parent))
