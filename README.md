# Introduction
This project showcases a facial recognition system utilizing the DeepFace library coupled with YOLOv8 for real-time capability. It includes a passive and active anti-spoofing detection mechanism, making it more robust to spoof attempts.

# Usage
To get started, you need to populate the database with some images. Below is the structure of the project, including the database (`db`):

```bash
data/
├── db
│   ├── db.pkl
│   ├── person1
│   │   ├── 2023-12-14-072531.jpg
│   │   └── 2024-06-19-204427.jpg
│   ├── person2
│   │   ├── 2023-12-14-072531.jpg
│   │   └── 2024-06-19-204427.jpg
│   └── person3
│       ├── 2023-12-14-072531.jpg
│       └── 2024-06-19-204427.jpg
└── weights
    ├── 2.7_80x80_MiniFASNetV2.pth
    ├── 4_0_0_80x80_MiniFASNetV1SE.pth
    ├── facenet512_weights.h5
    └── yolov8n-face.pt
```

No need to download the model weights separately; just populate the db with images of individuals. We adhere to the LFW (Labeled Faces in the Wild) format for image organization.

We use Docker to simplify the setup. Just run the following command to get started:
```bash
docker compose up
```

Afterwards, access the graphical user interface (GUI) via your web browser at http://localhost:8090.
