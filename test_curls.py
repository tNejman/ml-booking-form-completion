import json
import random
import time

import requests

# --- Configuration ---
API_URL = "http://localhost:8080"
NUM_REQUESTS = 50

SAMPLE_DESCRIPTIONS = [
    "Cozy apartment in the city center with 2 bedrooms and wifi.",
    "Spacious villa with a pool and sea view, 4 bedrooms, large kitchen.",
    "Small studio near the metro, perfect for students, 1 bed.",
    "Luxury penthouse with 3 baths and a private gym and AC.",
    "Rustic cottage in the woods, fireplace included, 2 beds.",
    "Modern loft downtown, high ceilings and large windows.",
    "Beachfront bungalow with direct access to the sand, 3 bedrooms.",
    "Suburban house with a large garden and garage, sleeps 6.",
    "Historic townhouse in the old quarter, newly renovated.",
    "Shared room in a friendly hostel, cheap and clean.",
]


def generate_simulated_correction(prediction_data):
    """
    Simulates a human correcting the AI's mistakes.
    Updated to SKIP amenities.
    """
    # Create a copy so we don't mess up the original data
    feedback = prediction_data.copy()

    # --- 1. Remove Amenities (Requested Update) --

    # --- 2. Simulate Correction: Room Type ---
    # 30% chance the user changes the room type
    if random.random() < 0.30:
        feedback["room_type"] = random.choice(
            ["Entire home/apt", "Private room", "Shared room"]
        )

    # --- 3. Simulate Correction: Bedrooms ---
    # 40% chance the user adjusts the number of bedrooms
    if random.random() < 0.40:
        current = feedback.get("bedrooms") or 1.0
        feedback["bedrooms"] = max(0.0, current + random.choice([-1.0, 0.0, 1.0]))

    # Remove internal metadata
    if "model_version" in feedback:
        del feedback["model_version"]

    return feedback


def main():
    print(f"--- Starting Data Generation ({NUM_REQUESTS} requests) ---")
    print(f"Target API: {API_URL}\n")

    headers = {"Content-Type": "application/json"}

    try:
        requests.get(f"{API_URL}/health")
    except requests.exceptions.ConnectionError:
        print(f"CRITICAL ERROR: Cannot connect to {API_URL}. Is the server running?")
        return

    for i in range(NUM_REQUESTS):
        description = random.choice(SAMPLE_DESCRIPTIONS)

        # Step 1: Request Prediction
        try:
            resp = requests.post(
                f"{API_URL}/predict/ab_test",
                headers=headers,
                json={"description": description},
            )

            # If server crashes (500), print error and skip
            if resp.status_code == 500:
                print(
                    f"[{i + 1}] Server Error 500. Did you apply the JSON fix to main.py?"
                )
                continue

            resp.raise_for_status()
            pred_data = resp.json()
            pred_id = pred_data["prediction_id"]

            print(
                f"[{i + 1}/{NUM_REQUESTS}] Prediction ID: {pred_id} (Amenities skipped)"
            )

            # Step 2: Generate Feedback (Without Amenities)
            feedback_payload = generate_simulated_correction(pred_data)
            feedback_payload["prediction_id"] = pred_id

            # Step 3: Send Feedback
            fb_resp = requests.post(
                f"{API_URL}/feedback", headers=headers, json=feedback_payload
            )
            fb_resp.raise_for_status()

        except Exception as e:
            print(f"      -> Error: {e}")
            time.sleep(1)

    print("\n--- Generation Complete ---")


if __name__ == "__main__":
    main()
