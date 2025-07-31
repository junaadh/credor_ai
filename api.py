import asyncio
import io
import os
import logging
from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID
import face_recognition
import numpy as np
import cv2
import aiohttp
from aiohttp import web
import asyncpg
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image
import uuid
from pathlib import Path


# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai-worker")

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
API_HOST = os.getenv("API_HOST", "http://localhost:8080")
MODEL_PATH = os.getenv("MODEL_PATH", "./model.h5")
AI_ID = os.getenv("AI_ID")
AI_PASSWORD = os.getenv("AI_PASSWORD")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

if DEBUG_MODE:
    DEBUG_DIR = Path("debug")
    DEBUG_DIR.mkdir(exist_ok=True)

auth_token: str | None = None


# ─── Model Output Structure ────────────────────────────────────────────────────
class PredictionResult(TypedDict):
    confidence: float
    job_id: str
    detected_at: str
    label: str
    serial_id: str
    media_url: str
    post_url: str


# ─── Load Model ────────────────────────────────────────────────────────────────
logger.info("Loading model from %s …", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
logger.info("Model loaded successfully.")


# ─── Retry Helper ──────────────────────────────────────────────────────────────
async def retry_async(fn, retries=3, delay=1, backoff=2):
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning("Retrying (%d/%d) after error: %s", attempt + 1, retries, e)
            await asyncio.sleep(delay)
            delay *= backoff


# ─── Inference ─────────────────────────────────────────────────────────────────
async def run_inference(image_bytes: bytes) -> tuple[str, float] | None:
    try:
        logger.debug("Running inference …")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
        tensor = tf.convert_to_tensor(
            [tf.keras.preprocessing.image.img_to_array(img) / 255.0]
        )
        preds = model.predict(tensor, verbose=0)[0]
        confidence = float(preds.max())
        label = str(preds.argmax())
        logger.debug("Inference result: label=%s confidence=%.4f", label, confidence)
        return label, confidence
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        return None


# ─── Encode Queue Worker ──────────────────────────────────────────────────────────
async def encode_worker(
    queue: asyncio.Queue, session: aiohttp.ClientSession, pool: asyncpg.Pool
):
    while True:
        user_id = await queue.get()
        try:
            logger.info("Encoding worker processing user_id=%s", user_id)
            await process_encoding(session, pool, user_id)
        except Exception as e:
            logger.exception("Encoding job failed for user_id=%s: %s", user_id, e)
        finally:
            queue.task_done()


# ─── Job Queue Worker ──────────────────────────────────────────────────────────
async def job_worker(
    queue: asyncio.Queue, session: aiohttp.ClientSession, pool: asyncpg.Pool
):
    while True:
        job_id, user_id = await queue.get()
        try:
            logger.info("Worker picked up job %s %s", job_id, user_id)
            await process_job(session, pool, job_id, user_id)
        except Exception as e:
            logger.exception("Job %s failed during processing: %s", job_id, e)
        finally:
            queue.task_done()


# ─── Process Encoding ───────────────────────────────────────────────────────────────
async def process_encoding(
    session: aiohttp.ClientSession, pool: asyncpg.Pool, user_id: UUID
):
    async def download():
        async with session.get(
            f"{API_HOST}/api/admin/bucket?user_id={user_id}",
            headers={"Authorization": f"Bearer {auth_token}"},
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch image: HTTP {resp.status}")
            return await resp.read()

    image_bytes = await retry_async(download)

    # Load and detect face
    image_np = np.frombuffer(image_bytes, np.uint8)
    bgr_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        raise Exception("No face detected")

    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    if not encodings:
        raise Exception("Failed to encode face")

    encoding = encodings[0]
    encoding_bytes = encoding.astype(np.float32).tobytes()

    # Debug: Show image with face box
    if DEBUG_MODE:
        for top, right, bottom, left in face_locations:
            cv2.rectangle(bgr_image, (left, top), (right, bottom), (0, 255, 0), 2)
        debug_filename = DEBUG_DIR / f"{uuid.uuid4()}_match.jpg"
        cv2.imwrite(str(debug_filename), bgr_image)
        print(f"[DEBUG] Saved: {debug_filename}")
    # Store encoding in DB
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE encodings SET encoding = $1 WHERE user_id = $2",
            encoding_bytes,
            user_id,
        )
    logger.info("Stored encoding for user_id=%s", user_id)


# ─── Process Job ───────────────────────────────────────────────────────────────
async def process_job(
    session: aiohttp.ClientSession, pool: asyncpg.Pool, job_id: UUID, user_id: UUID
) -> None:
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, image_url, post_uri FROM post_images WHERE job_id = $1",
                job_id,
            )
            enc_row = await conn.fetchrow(
                "SELECT encoding FROM encodings WHERE user_id = $1", user_id
            )
            user_encoding = (
                np.frombuffer(enc_row["encoding"], dtype=np.float32)
                if enc_row and enc_row["encoding"]
                else None
            )
        logger.info("Fetched %d images for job %s", len(rows), job_id)
    except Exception as e:
        logger.exception("DB fetch failed for job %s: %s", job_id, e)
        return

    for row in rows:
        serial_id = row["id"]
        media_url = row["image_url"]
        post_url = row["post_uri"]

        try:

            async def download():
                headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
                async with session.get(media_url, headers=headers) as resp:
                    if resp.status != 200:
                        raise Exception(f"HTTP {resp.status}")
                    if not resp.headers.get("Content-Type", "").startswith("image/"):
                        raise Exception(f"Not an image: {media_url}")
                    return await resp.read()

            media_bytes = await retry_async(download)

            face_match = True
            if user_encoding is not None:
                image_np = np.frombuffer(media_bytes, np.uint8)
                bgr_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_image)
                encodings = face_recognition.face_encodings(rgb_image, face_locations)

                if not encodings:
                    logger.warning("No face found in image: %s", media_url)
                    continue  # Skip this image

                distances = face_recognition.face_distance(encodings, user_encoding)
                best_distance = distances.min()
                threshold = 0.6  # lower = stricter match
                face_match = best_distance <= threshold

                if DEBUG_MODE:
                    for top, right, bottom, left in face_locations:
                        color = (0, 255, 0) if face_match else (0, 0, 255)
                        cv2.rectangle(bgr_image, (left, top), (right, bottom), color, 2)

                    debug_filename = (
                        DEBUG_DIR
                        / f"{uuid.uuid4()}_{'match' if face_match else 'no_match'}.jpg"
                    )
                    cv2.imwrite(str(debug_filename), bgr_image)
                    print(f"[DEBUG] Saved: {debug_filename}")

                if not face_match:
                    payload: PredictionResult = {
                        "confidence": -1.0,
                        "job_id": str(job_id),
                        "user_id": str(user_id),
                        "media_url": media_url,
                        "detected_at": str(datetime.now(timezone.utc)),
                        "label": "not_match",
                        "serial_id": serial_id,
                        "post_url": post_url,
                    }
                    await retry_async(
                        lambda: session.post(
                            f"{API_HOST}/api/ai/job",
                            headers={
                                "Authorization": f"Bearer {auth_token}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )
                    )
                    logger.info(
                        "Posted non-match result: serial_id=%s label=not_match",
                        serial_id,
                    )
                    continue  # skip inference if not matched

            inference = await run_inference(media_bytes)
            if inference is None:
                continue

            label, confidence = inference
            payload: PredictionResult = {
                "confidence": confidence,
                "job_id": str(job_id),
                "user_id": str(user_id),
                "media_url": media_url,
                "detected_at": str(datetime.now(timezone.utc)),
                "label": "real" if (label == 1) else "fake",
                "serial_id": serial_id,
                "post_url": post_url,
            }

            await asyncio.sleep(5)  # throttle
            await retry_async(
                lambda: session.post(
                    f"{API_HOST}/api/ai/job",
                    headers={
                        "Authorization": f"Bearer {auth_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            )
            logger.info(
                "Posted result: serial_id=%s label=%s confidence=%.4f",
                serial_id,
                label,
                confidence,
            )

        except Exception as e:
            logger.exception("Failed serial_id=%s url=%s: %s", serial_id, media_url, e)

    try:
        await retry_async(
            lambda: session.get(
                f"{API_HOST}/api/ai/job/{job_id}",
                headers={"Authorization": f"Bearer {auth_token}"},
            )
        )
        logger.info("Completed job %s notification.", job_id)
    except Exception as e:
        logger.exception("Final callback for job %s failed: %s", job_id, e)


# ─── API Endpoint Handler ──────────────────────────────────────────────────────
async def handle_encode(request: web.Request) -> web.Response:
    try:
        user_id = request.query.get("user_id")
        if not user_id:
            return web.Response(status=400, text="Missing user_id")
        user_uuid = UUID(user_id)
        logger.info("Received /encode request for user_id=%s", user_uuid)
        await request.app["encode_queue"].put(user_uuid)
        return web.Response(status=202, text="Encode job accepted")
    except Exception as e:
        logger.warning("Invalid /encode request: %s", e)
        return web.Response(status=400, text="Invalid Payload")


# ─── API Endpoint Handler ──────────────────────────────────────────────────────
async def handle_predict(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        print(f"handle_predict: {data}")
        job_id = UUID(data["job_id"])
        user_id = UUID(data["user_id"])
        logger.info("Received /predict request for job_id=%s", job_id)
        await request.app["queue"].put((job_id, user_id))
        return web.Response(status=202, text="Job accepted")
    except Exception as e:
        logger.warning("Invalid /predict payload: %s", e)
        return web.Response(status=400, text="Invalid Payload")


# ─── API Endpoint Handler ──────────────────────────────────────────────────────
async def authenticate(session: aiohttp.ClientSession) -> str:
    logger.info("Authenticating with server…")
    payload = {
        "email": AI_ID,
        "password": AI_PASSWORD,
    }
    async with session.post(
        f"{API_HOST}/api/admin/login",
        headers={"Content-Type": "application/json"},
        json=payload,
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Authentication failed (HTTP {resp.status})")
        data = await resp.json()
        token = data.get("access_token")
        if not token:
            raise Exception("No token in authentication response")
        logger.info("Authentication successful.")
        return token


# ─── Main Entrypoint ───────────────────────────────────────────────────────────
async def main():
    logger.info("Initializing AI Worker service…")
    pool = await asyncpg.create_pool(dsn=DB_URL)
    session = aiohttp.ClientSession()
    queue: asyncio.Queue = asyncio.Queue()
    encode_queue: asyncio.Queue = asyncio.Queue()

    app = web.Application()
    app["pool"] = pool
    app["session"] = session
    app["queue"] = queue
    app["encode_queue"] = encode_queue
    app.add_routes(
        [web.post("/predict", handle_predict), web.get("/encode", handle_encode)]
    )

    global auth_token
    auth_token = await authenticate(session)

    # Start worker tasks
    for i in range(4):  # Number of parallel workers
        asyncio.create_task(job_worker(queue, session, pool))
        logger.info("Started worker #%d", i + 1)

    for i in range(2):
        asyncio.create_task(encode_worker(encode_queue, session, pool))
        logger.info("Started encode worker #%d", i + 1)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 9000)
    await site.start()
    logger.info("✅ AI Worker is live at http://0.0.0.0:9000/predict")

    while True:
        await asyncio.sleep(3600)


# ─── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
