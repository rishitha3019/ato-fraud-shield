import argparse, hashlib, os, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng = np.random.default_rng(42)
random.seed(42)

CITIES = {
    "New York": (40.7128, -74.0060), "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298), "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740), "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936), "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970), "Cincinnati": (39.1031, -84.5120),
    "Miami": (25.7617, -80.1918), "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903), "Atlanta": (33.7490, -84.3880),
    "Boston": (42.3601, -71.0589),
}
CITY_NAMES = list(CITIES.keys())
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
OS_LIST = ["Windows 10", "Windows 11", "macOS 13", "macOS 14", "Ubuntu 22.04", "iOS 17", "Android 14"]
SCREEN_RESOLUTIONS = ["1920x1080", "1440x900", "1366x768", "2560x1440", "1280x800", "375x812", "390x844"]
MALICIOUS_IP_PREFIXES = ["185.220.", "192.42.", "198.98.", "45.142.", "91.108."]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def device_fingerprint(browser, os, resolution):
    return hashlib.md5(f"{browser}|{os}|{resolution}".encode()).hexdigest()[:16]

def jitter_coords(lat, lon, radius_km=50):
    offset = radius_km / 111
    return round(lat + rng.uniform(-offset, offset), 4), round(lon + rng.uniform(-offset, offset), 4)

def random_ip(malicious=False):
    if malicious:
        return random.choice(MALICIOUS_IP_PREFIXES) + f"{rng.integers(1,254)}.{rng.integers(1,254)}"
    return fake.ipv4_public()

def build_user_profiles(n_users):
    profiles = {}
    for uid in range(1, n_users + 1):
        city = random.choice(CITY_NAMES)
        browser, os_, res = random.choice(BROWSERS), random.choice(OS_LIST), random.choice(SCREEN_RESOLUTIONS)
        profiles[uid] = {
            "user_id": uid, "home_city": city,
            "home_lat": CITIES[city][0], "home_lon": CITIES[city][1],
            "browser": browser, "os": os_, "resolution": res,
            "device_fp": device_fingerprint(browser, os_, res),
            "login_hour_mean": rng.integers(7, 22),
            "login_hour_std": rng.uniform(1.5, 3.5),
            "session_duration_mean": rng.uniform(120, 900),
            "actions_per_min_mean": rng.uniform(1.5, 8.0),
            "account_age_days": rng.integers(30, 2000),
        }
    return profiles

def generate_legit_event(user, event_time, prev):
    if rng.random() < 0.04:
        br, os_, res = random.choice(BROWSERS), random.choice(OS_LIST), random.choice(SCREEN_RESOLUTIONS)
        fp, new_dev = device_fingerprint(br, os_, res), True
    else:
        br, os_, res, fp, new_dev = user["browser"], user["os"], user["resolution"], user["device_fp"], False
    lat, lon = jitter_coords(user["home_lat"], user["home_lon"], 40)
    hour = int(np.clip(rng.normal(user["login_hour_mean"], user["login_hour_std"]), 0, 23))
    hrs = (event_time - prev["event_time"]).total_seconds()/3600 if prev else 168.0
    dist = haversine_km(prev["lat"], prev["lon"], lat, lon) if prev else 0.0
    vel = dist / max(hrs, 0.01) if prev else 0.0
    fa1h = int(rng.poisson(0.1))
    return {
        "event_id": fake.uuid4(), "user_id": user["user_id"], "event_time": event_time,
        "hour_of_day": hour, "day_of_week": event_time.weekday(),
        "city": user["home_city"], "lat": lat, "lon": lon, "ip_address": random_ip(False),
        "hours_since_last_login": round(hrs, 2), "km_from_last_login": round(dist, 2), "velocity_kmh": round(vel, 2),
        "browser": br, "os": os_, "screen_resolution": res, "device_fingerprint": fp, "is_new_device": int(new_dev),
        "failed_attempts_1h": fa1h, "failed_attempts_6h": fa1h + int(rng.poisson(0.2)),
        "failed_attempts_24h": fa1h + int(rng.poisson(0.4)),
        "session_duration_sec": round(abs(rng.normal(user["session_duration_mean"], 60)), 1),
        "actions_per_minute": round(abs(rng.normal(user["actions_per_min_mean"], 0.5)), 2),
        "time_to_first_action_sec": round(rng.uniform(2, 30), 2),
        "account_age_days": user["account_age_days"],
        "hour_deviation_from_mean": abs(hour - user["login_hour_mean"]),
        "is_fraud": 0, "fraud_type": "none",
    }

def generate_credential_stuffing(user, event_time, prev):
    ev = generate_legit_event(user, event_time, prev)
    fa1h = int(rng.integers(5, 30))
    ev.update({"event_id": fake.uuid4(), "ip_address": random_ip(True),
        "failed_attempts_1h": fa1h, "failed_attempts_6h": fa1h + int(rng.integers(10, 50)),
        "failed_attempts_24h": fa1h + int(rng.integers(20, 80)),
        "is_new_device": 1, "device_fingerprint": fake.md5()[:16],
        "hour_of_day": int(rng.integers(1, 5)),
        "session_duration_sec": round(rng.uniform(10, 90), 1),
        "time_to_first_action_sec": round(rng.uniform(0.5, 3.0), 2),
        "actions_per_minute": round(rng.uniform(15, 40), 2),
        "is_fraud": 1, "fraud_type": "credential_stuffing"})
    ev["hour_deviation_from_mean"] = abs(ev["hour_of_day"] - user["login_hour_mean"])
    return ev

def generate_impossible_travel(user, event_time, prev):
    ev = generate_legit_event(user, event_time, prev)
    far = [c for c in CITY_NAMES if haversine_km(user["home_lat"], user["home_lon"], CITIES[c][0], CITIES[c][1]) > 1000] or CITY_NAMES
    city = random.choice(far)
    alat, alon = jitter_coords(CITIES[city][0], CITIES[city][1], 20)
    ev["city"], ev["lat"], ev["lon"] = city, alat, alon
    if prev:
        dist = haversine_km(prev["lat"], prev["lon"], alat, alon)
        gap = max((event_time - prev["event_time"]).total_seconds()/3600, 0.25)
        ev["km_from_last_login"] = round(dist, 2)
        ev["velocity_kmh"] = round(dist/gap, 2)
        ev["hours_since_last_login"] = round(gap, 2)
    ev.update({"event_id": fake.uuid4(), "ip_address": random_ip(rng.random() > 0.4),
        "is_new_device": int(rng.random() > 0.5), "is_fraud": 1, "fraud_type": "impossible_travel"})
    if ev["is_new_device"]:
        ev["device_fingerprint"] = fake.md5()[:16]
    return ev

def generate_ato_session(user, event_time, prev):
    ev = generate_legit_event(user, event_time, prev)
    fa1h = int(rng.integers(0, 3))
    ev.update({"event_id": fake.uuid4(), "device_fingerprint": fake.md5()[:16], "is_new_device": 1,
        "actions_per_minute": round(rng.uniform(20, 60), 2),
        "time_to_first_action_sec": round(rng.uniform(0.3, 2.0), 2),
        "session_duration_sec": round(rng.uniform(30, 180), 1),
        "failed_attempts_1h": fa1h, "failed_attempts_6h": fa1h + int(rng.integers(0, 5)),
        "is_fraud": 1, "fraud_type": "ato_session_abuse"})
    return ev

def generate_slow_burn(user, event_time, prev):
    ev = generate_legit_event(user, event_time, prev)
    hour = int(np.clip(user["login_hour_mean"] + rng.uniform(4, 8) * rng.choice([-1, 1]), 0, 23))
    ev.update({"event_id": fake.uuid4(), "device_fingerprint": fake.md5()[:16], "is_new_device": 1,
        "failed_attempts_1h": 0, "failed_attempts_6h": int(rng.integers(0, 2)),
        "actions_per_minute": round(user["actions_per_min_mean"] * rng.uniform(1.3, 2.5), 2),
        "time_to_first_action_sec": round(rng.uniform(1.0, 5.0), 2),
        "hour_of_day": hour, "hour_deviation_from_mean": abs(hour - user["login_hour_mean"]),
        "is_fraud": 1, "fraud_type": "slow_burn_ato"})
    return ev

FRAUD_GENS = [generate_credential_stuffing, generate_impossible_travel, generate_ato_session, generate_slow_burn]

def simulate(n_users, n_events, fraud_rate):
    print(f"Building {n_users} user profiles...")
    profiles = build_user_profiles(n_users)
    last_event = {}
    uids = rng.choice(list(profiles.keys()), size=n_events, replace=True)
    base = datetime(2024, 10, 1)
    times = sorted([base + timedelta(seconds=int(rng.integers(0, 90*24*3600))) for _ in range(n_events)])
    fraud_idx = set(rng.choice(n_events, size=int(n_events*fraud_rate), replace=False).tolist())
    records = []
    print(f"Simulating {n_events:,} events ({int(n_events*fraud_rate):,} fraud)...")
    for i, (uid, t) in enumerate(zip(uids, times)):
        u, prev = profiles[int(uid)], last_event.get(int(uid))
        ev = random.choice(FRAUD_GENS)(u, t, prev) if i in fraud_idx else generate_legit_event(u, t, prev)
        ev["event_time"] = t
        records.append(ev)
        last_event[int(uid)] = ev
        if (i+1) % 10000 == 0:
            print(f"  ... {i+1:,} / {n_events:,}")
    df = pd.DataFrame(records).sort_values("event_time").reset_index(drop=True)
    df["velocity_kmh_log"] = np.log1p(df["velocity_kmh"])
    df["failed_attempts_ratio"] = df["failed_attempts_1h"] / (df["failed_attempts_24h"] + 1)
    df["impossible_velocity_flag"] = (df["velocity_kmh"] > 900).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = (df["hour_of_day"] < 6).astype(int)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-users", type=int, default=1000)
    p.add_argument("--n-events", type=int, default=50000)
    p.add_argument("--fraud-rate", type=float, default=0.03)
    p.add_argument("--output-dir", type=str, default="data/raw")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "login_events.csv")
    df = simulate(args.n_users, args.n_events, args.fraud_rate)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print(f"Shape: {df.shape}")
    print(df["is_fraud"].value_counts())
    print(df[df["is_fraud"]==1]["fraud_type"].value_counts())

if __name__ == "__main__":
    main()
