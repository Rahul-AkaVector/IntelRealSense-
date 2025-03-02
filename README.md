# Intel RealSense Project

## Overview

Intel RealSense is a Python-based project designed for detecting body angles and mimicking movements to robots. It utilizes AI and computer vision techniques to track hand posture, calculate elbow angles, and generate movement commands.

## Features

- Real-time body angle detection
- Hand posture tracking
- AI-powered movement analysis
- Robot motion integration

## Usage

🔴 **Note:** This project requires an Intel RealSense camera to function properly. It has been tested on the Intel RealSense D435 camera, but compatibility with other models may vary. Please check your camera configuration before use.

## Installation

```bash
# Clone the repository
git clone https://github.com/Rahul-AkaVector/IntelRealSense-.git
cd IntelRealSense

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python elbow_angle3.py
```

## Contributing

Feel free to fork the repo and submit pull requests!

## License

This project is licensed under the MIT License.
