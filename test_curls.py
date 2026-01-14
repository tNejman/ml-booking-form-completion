import json
import random
import time

import requests

API_URL = "http://localhost:8080"
NUM_REQUESTS = 100

TEST_CASES = [
    {
        "description": "Cozy apartment in the city center with 2 bedrooms and wifi.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "bathrooms_text": "1 bath",
            "bedrooms": 2.0,
            "beds": 2.0,
            "accommodates": 4.0,
            "amenities": ["Wifi", "Kitchen", "Heating"],
        },
    },
    {
        "description": "Small studio near the metro, perfect for students, 1 bed.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "bathrooms_text": "1 bath",
            "bedrooms": 0.0,
            "beds": 1.0,
            "accommodates": 2.0,
            "amenities": ["Wifi", "Heating"],
        },
    },
    {
        "description": "Luxury penthouse with 3 baths and a private gym and AC.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "bathrooms_text": "3 baths",
            "bedrooms": 3.0,
            "beds": 3.0,
            "accommodates": 6.0,
            "amenities": ["Air conditioning", "Gym", "Wifi", "Elevator"],
        },
    },
    {
        "description": "Modern loft downtown, high ceilings and large windows.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Loft",
            "bathrooms_text": "1.5 baths",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 2.0,
            "amenities": ["Wifi", "Kitchen", "Elevator"],
        },
    },
    {
        "description": "Business ready suite near convention center, fast internet and workspace.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "bathrooms_text": "1 bath",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 2.0,
            "amenities": ["Wifi", "Dedicated workspace", "Iron", "Hair dryer"],
        },
    },
    {
        "description": "Spacious villa with a pool and sea view, 4 bedrooms, large kitchen.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Villa",
            "bathrooms_text": "3 baths",
            "bedrooms": 4.0,
            "beds": 6.0,
            "accommodates": 10.0,
            "amenities": ["Pool", "Kitchen", "Air conditioning", "Wifi"],
        },
    },
    {
        "description": "Rustic cottage in the woods, fireplace included, 2 beds.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "House",
            "bathrooms_text": "1 bath",
            "bedrooms": 1.0,
            "beds": 2.0,
            "accommodates": 4.0,
            "amenities": ["Indoor fireplace", "Heating", "Free parking on premises"],
        },
    },
    {
        "description": "Large family house with garden, garage, and BBQ grill. Sleeps 8.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "House",
            "bathrooms_text": "2.5 baths",
            "bedrooms": 4.0,
            "beds": 5.0,
            "accommodates": 8.0,
            "amenities": ["Kitchen", "Free parking on premises", "BBQ grill", "Garden"],
        },
    },
    {
        "description": "Ski chalet with hot tub and sauna, right on the slopes.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Chalet",
            "bathrooms_text": "2 baths",
            "bedrooms": 3.0,
            "beds": 4.0,
            "accommodates": 6.0,
            "amenities": ["Hot tub", "Sauna", "Ski-in/ski-out", "Heating"],
        },
    },
    {
        "description": "Cozy houseboat moored in the marina, gentle waves.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Boat",
            "bathrooms_text": "1 bath",
            "bedrooms": 2.0,
            "beds": 2.0,
            "accommodates": 4.0,
            "amenities": ["Waterfront", "Kitchen", "Wifi"],
        },
    },
    {
        "description": "Vintage Airstream camper with outdoor seating and fire pit.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Camper/RV",
            "bathrooms_text": "1 bath",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 2.0,
            "amenities": [
                "Free parking on premises",
                "Fire pit",
                "Outdoor dining area",
            ],
        },
    },
    {
        "description": "Tiny house eco-stay, solar powered, compost toilet.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Tiny house",
            "bathrooms_text": "1 bath",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 2.0,
            "amenities": ["Kitchen", "Heating", "Essentials"],
        },
    },
    {
        "description": "Historic castle room with stone walls and antique furniture.",
        "truth": {
            "room_type": "Private room",
            "property_type": "Castle",
            "bathrooms_text": "1 private bath",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 2.0,
            "amenities": ["Heating", "Essentials", "Shampoo"],
        },
    },
    {
        "description": "Farm stay with animals, fresh eggs daily, and tractor rides.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Farm stay",
            "bathrooms_text": "1 bath",
            "bedrooms": 2.0,
            "beds": 3.0,
            "accommodates": 5.0,
            "amenities": ["Kitchen", "Free parking on premises", "Animals"],
        },
    },
    {
        "description": "Yurt in the mountains, stargazing roof, off-grid experience.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Yurt",
            "bathrooms_text": "Outdoor shower",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 2.0,
            "amenities": ["Heating", "Mountain view", "Essentials"],
        },
    },
    {
        "description": "Shared room in a friendly hostel, cheap and clean.",
        "truth": {
            "room_type": "Shared room",
            "property_type": "Hostel",
            "bathrooms_text": "Shared half-bath",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 1.0,
            "amenities": ["Wifi", "Breakfast"],
        },
    },
    {
        "description": "Private room in a shared apartment, owner has a cat.",
        "truth": {
            "room_type": "Private room",
            "property_type": "Apartment",
            "bathrooms_text": "1 shared bath",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 1.0,
            "amenities": ["Wifi", "Kitchen", "Washer"],
        },
    },
    {
        "description": "Sofa bed in living room for crash pad, very basic.",
        "truth": {
            "room_type": "Shared room",
            "property_type": "Apartment",
            "bathrooms_text": "1 shared bath",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 1.0,
            "amenities": ["Wifi", "Essentials"],
        },
    },
    {
        "description": "Event space only, no sleeping, large hall.",
        "truth": {
            "room_type": "Entire home/apt",
            "property_type": "Other",
            "bathrooms_text": "2 baths",
            "bedrooms": 0.0,
            "beds": 0.0,
            "accommodates": 0.0,
            "amenities": ["Wifi", "Chairs", "Tables"],
        },
    },
    {
        "description": "Luxury tent glamping with electricity and real bed.",
        "truth": {
            "room_type": "Private room",
            "property_type": "Tent",
            "bathrooms_text": "Shared bath",
            "bedrooms": 1.0,
            "beds": 1.0,
            "accommodates": 2.0,
            "amenities": ["Electricity", "Heating", "Wifi"],
        },
    },
]


def main():
    print(f"--- Starting A/B Test Simulation ({NUM_REQUESTS} requests) ---")
    print(f"Target API: {API_URL}\n")

    headers = {"Content-Type": "application/json"}

    try:
        requests.get(f"{API_URL}/health")
    except requests.exceptions.ConnectionError:
        print(f"CRITICAL ERROR: Cannot connect to {API_URL}. Is the server running?")
        return

    for i in range(NUM_REQUESTS):
        case = random.choice(TEST_CASES)
        description = case["description"]
        ground_truth = case["truth"]

        try:
            print(f"[{i + 1}/{NUM_REQUESTS}] Sending: '{description[:40]}...'")

            resp = requests.post(
                f"{API_URL}/app/predict/ab_test",
                headers=headers,
                json={"description": description},
            )

            if resp.status_code != 200:
                print(f"      -> API Error {resp.status_code}: {resp.text}")
                continue

            pred_data = resp.json()
            pred_id = pred_data["prediction_id"]
            model_used = pred_data.get("model_version", "unknown")

            print(f"      -> Got Prediction ID: {pred_id} (Model: {model_used})")

            feedback_payload = ground_truth.copy()
            feedback_payload["prediction_id"] = pred_id

            fb_resp = requests.post(
                f"{API_URL}/app/feedback", headers=headers, json=feedback_payload
            )
            fb_resp.raise_for_status()
            print("      -> Feedback sent successfully (Ground Truth)")

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.5)

    print("\n--- Simulation Complete ---")


if __name__ == "__main__":
    main()
