import base64

with open("ai.mp3", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

print(encoded)