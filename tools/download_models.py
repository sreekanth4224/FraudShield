from modules.detectors.face_classifier import FaceDeepfakeClassifier
from modules.detectors.voice_classifier import VoiceDeepfakeClassifier
from modules.detectors.custom_heads import CustomFaceDetector, CustomVoiceDetector
from modules.detectors.scene_classifier import SceneDeepfakeClassifier

def main():
    for cls in (FaceDeepfakeClassifier, VoiceDeepfakeClassifier, SceneDeepfakeClassifier,
                CustomFaceDetector, CustomVoiceDetector):
        clf = cls.get()
        print(f"{cls.__name__:<26} {cls.status}" + (f" — {clf.name}" if clf else ""))

if __name__ == "__main__":
    main()
