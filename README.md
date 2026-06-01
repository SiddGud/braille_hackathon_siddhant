# BrailleVision
**AI-powered Braille recognition for an inclusive world.**

BrailleVision is a real-time computer vision application that detects physical Braille text via a webcam or image upload, mathematically repairs OCR anomalies, and translates the text into spoken audio using on-device TTS. 

It is designed with **dual-accessibility**: Sighted users can navigate the sleek interactive dashboard, while visually impaired users can seamlessly operate the entire system using voice commands.

## ⚠️ CRITICAL: How to Capture Braille Images

Because computer vision relies heavily on shadow mapping to detect physical bumps, **you must capture the photos correctly** for the YOLOv8 model to work. 

If your model is returning garbage text, it is because your lighting is wrong! Follow these two golden rules (similar to the reference images shared in our WhatsApp group):

### 1. Use the BACK of the paper
This specific YOLOv8 model was trained to recognize the mirrored crater patterns from the **back** of punched Braille paper. 
- Do **NOT** scan the raised bumps on the front of the paper. 
- Flip the paper over so the dots appear as indented holes/craters.

### 2. Shine a flashlight from the Bottom-Left
Braille AI cannot see dots; it sees the *shadows* cast by the dots.
- Do **NOT** use overhead lighting (like a ceiling lamp directly above the paper). Overhead lighting casts shadows at the bottom of the dots, which blinds the model.
- Lay the paper flat on a table. 
- Hold a phone flashlight at a low angle from the **bottom-left corner**, shining diagonally across the page. This casts the shadows on the top-right inner walls of the craters, which is exactly the lighting environment the model was trained on!

## Tech Stack
- **AI/ML:** YOLOv8 Nano, OpenCV, CLAHE Preprocessing
- **Backend:** Python, FastAPI, WebSockets
- **Frontend:** HTML5, Vanilla JavaScript, WebRTC (Live Camera)
- **Audio:** Web Speech API TTS

## How It Works
1. **Camera Capture:** High-resolution 1280x720 live feed captured via WebRTC, streamed as base64 frames over WebSocket at 2 FPS.
2. **Preprocessing:** OpenCV CLAHE enhances contrast and adaptive thresholding isolates Braille dots.
3. **AI Detection:** YOLOv8 nano model detects and classifies individual Braille cells with sub-100ms inference time on CPU.
4. **Heuristic Repair:** The backend runs a mathematical fuzzy-matching algorithm and sequence heuristic to automatically repair catastrophic OCR failures (like misinterpreting 'w' as 'r').
5. **Speech Output:** Decoded characters are assembled into words and converted to natural speech.

---
*Built with ❤️ for Empowering Independence*
