# Imports
import cv2
import torch
from PIL import Image
from torchvision import transforms

from Input import train_dataset
from Model import ConvNet
from face_detector import FaceDetector

prototype_path = 'models/deploy.prototxt.txt'
face_detection_model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
model_path = 'detector_state.pth'
font = cv2.FONT_HERSHEY_DUPLEX


def load_model(file_path):
    """function to load saved state of model"""
    trained_model = ConvNet()
    trained_model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    trained_model.eval()

    return trained_model


# Load saved model
model = load_model(model_path)
model.class_to_idx = train_dataset.class_to_idx


def process_image(image):
    """apply normalization to image for the pytorch model"""
    try:
        img = Image.fromarray(image)
        transform = transforms.Compose([transforms.Resize((100, 100)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4437, 0.3848, 0.3613], [0.2972, 0.2702, 0.2581])
                                        ])
        image = transform(img)
        return image
    except IOError:
        pass


def inference(images):
    """method to obtain predictions on passed images"""
    with torch.no_grad():
        classification = []
        index_to_class = {value: key for key, value in model.class_to_idx.items()}

        # forward pass
        image = process_image(images)
        output = model(image[None])
        label = output.numpy().argmax()
        classification.append(index_to_class[label])

    return classification[-1]


def classification(frame, faces):
    for (start_x, start_y, width, height) in faces:
        # clamp coordinates that are outside of the image
        start_x, start_y = max(start_x, 0), max(start_y, 0)
        # obtain face coordinates
        face_img = frame[start_y:start_y + height, start_x:start_x + width]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # make prediction
        prediction = inference(face_img)
        # create bounding box on face and add text label
        label = prediction
        if label == 'with_mask':
            colour = (0, 255, 0)
        elif label == 'without_mask':
            colour = (0, 0, 255)

        cv2.rectangle(frame, (start_x, start_y), (start_x + width, start_y + height), colour, 1)
        cv2.putText(frame, label, (start_x, start_y - 10), font, 0.5, colour, 2)

    return frame


def image_prediction(image):
    """function that detects human faces in a given image and makes prediction on it"""
    # load the model for face detection
    face_detector = FaceDetector(prototype_path, face_detection_model_path)
    # read input image
    image = cv2.imread(image)
    # detect faces in input image
    faces = face_detector.detect(image)
    # pass detected face for classification
    classification(image, faces)
    # display the output image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_prediction():
    face_detector = FaceDetector(prototype_path, face_detection_model_path)
    video_capture = cv2.VideoCapture(0)

    while True:
        # capture frame by frame
        ret, frame = video_capture.read()
        faces = face_detector.detect(frame)
        classification(frame, faces)

        # display the output image
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        video_capture.release()
        cv2.destroyAllWindows()
