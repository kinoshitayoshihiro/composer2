import yaml
with open("data/processed_chordmap_with_emotion.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
print("Sections:", list(data.get("sections", {}).keys()))