"""
WebSocket streaming for live experiment visualization.

Streams MuJoCo simulation state from workers to browser clients.
"""

import asyncio
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import pubsub_v1

router = APIRouter()

# Active WebSocket connections per experiment
# {experiment_id: {websocket1, websocket2, ...}}
active_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections for experiment streaming."""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, experiment_id: str, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        if experiment_id not in self.connections:
            self.connections[experiment_id] = set()
        self.connections[experiment_id].add(websocket)

    def disconnect(self, experiment_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if experiment_id in self.connections:
            self.connections[experiment_id].discard(websocket)
            if not self.connections[experiment_id]:
                del self.connections[experiment_id]

    async def broadcast(self, experiment_id: str, message: dict):
        """Send message to all connections for an experiment."""
        if experiment_id not in self.connections:
            return

        dead_connections = set()
        for websocket in self.connections[experiment_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.add(websocket)

        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(experiment_id, ws)

    def get_viewer_count(self, experiment_id: str) -> int:
        """Get number of active viewers for an experiment."""
        return len(self.connections.get(experiment_id, set()))


manager = ConnectionManager()


@router.websocket("/stream/{experiment_id}")
async def websocket_stream(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for live experiment streaming.

    Clients connect to receive real-time simulation state updates.
    State is received from workers via Pub/Sub and forwarded to clients.
    """
    # TODO: Verify user has access to this experiment
    # token = websocket.query_params.get("token")
    # user = verify_token(token)

    await manager.connect(experiment_id, websocket)

    try:
        # Send initial connection info
        await websocket.send_json(
            {
                "type": "connected",
                "experiment_id": experiment_id,
                "viewers": manager.get_viewer_count(experiment_id),
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (camera control, etc.)
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=30.0
                )

                # Handle client commands
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "camera_control":
                    # Forward camera control to worker (via Pub/Sub)
                    # This allows synchronized camera for all viewers
                    pass

            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        manager.disconnect(experiment_id, websocket)
    except Exception as e:
        manager.disconnect(experiment_id, websocket)
        raise


async def publish_state_to_clients(experiment_id: str, state: dict):
    """
    Publish simulation state to all connected clients.

    Called by Pub/Sub subscription handler when new state arrives from worker.
    """
    message = {
        "type": "state",
        "experiment_id": experiment_id,
        "state": state,
    }
    await manager.broadcast(experiment_id, message)


async def publish_event_to_clients(experiment_id: str, event_type: str, data: dict):
    """
    Publish experiment event to all connected clients.

    Events: started, completed, failed, attempt_started, goal_reached, etc.
    """
    message = {
        "type": "event",
        "event_type": event_type,
        "experiment_id": experiment_id,
        "data": data,
    }
    await manager.broadcast(experiment_id, message)


# Pub/Sub subscriber for receiving state from workers
# This runs in a background task when the server starts

_subscriber_task = None


async def start_pubsub_subscriber():
    """
    Start background task to receive state updates from workers.

    Workers publish to 'experiment-state' topic, we subscribe and forward to WebSocket clients.
    """
    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        "g1-alignment", "experiment-state-web"
    )

    def callback(message):
        try:
            data = json.loads(message.data.decode("utf-8"))
            experiment_id = data.get("experiment_id")

            if data.get("type") == "state":
                # Use asyncio to call async broadcast
                asyncio.create_task(
                    publish_state_to_clients(experiment_id, data.get("state", {}))
                )
            elif data.get("type") == "event":
                asyncio.create_task(
                    publish_event_to_clients(
                        experiment_id, data.get("event_type", "unknown"), data
                    )
                )

            message.ack()
        except Exception as e:
            print(f"Error processing message: {e}")
            message.nack()

    # Start subscriber in background
    future = subscriber.subscribe(subscription_path, callback)

    try:
        future.result()
    except Exception as e:
        print(f"Subscriber error: {e}")
        future.cancel()
