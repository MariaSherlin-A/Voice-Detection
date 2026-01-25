import base64

with open("audio.mp3", "rb") as f:
    print(base64.b64encode(f.read()).decode())
