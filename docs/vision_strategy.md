# Vision Strategy

Only one tortoise is expected in the enclosure, so multi-object tracking is not required.

## Baseline

The current implementation uses OpenCV MOG2 background subtraction to find motion. This keeps the project runnable with mock images and provides a replaceable detector boundary.

## Classifier upgrade

The next accuracy step is a tortoise classifier, for example YOLOv8n or YOLO11n. The classifier should replace the detector implementation while retaining the `Detection` contract.

## Tracking upgrade

CSRT is the preferred tracker for one object. DeepSORT or ByteTrack can be evaluated if the system later supports multiple animals.

## Shell identification

Optional ORB or AKAZE shell-pattern matching can be added if more than one tortoise needs to be distinguished.