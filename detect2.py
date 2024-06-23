import os
import cv2
from ultralytics import YOLO

def main():
    # Get IP webcam URL from the user
    ip_webcam_url = input("Enter the IP webcam URL (e.g., http://<IP_ADDRESS>:<PORT>/video): ")

    # Construct the model path in the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, 'best.pt')

    # Verify the model path
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load YOLO model
    model = YOLO(model_path)

    # Open a connection to the IP webcam
    cap = cv2.VideoCapture(ip_webcam_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    else:
        print("Successfully opened video stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Perform YOLO detection on the frame
        results = model(frame, save=False)  # save=False to not save images

        # Plot the results on the frame
        res_plotted = results[0].plot()

        # Display the frame with detections
        cv2.imshow('YOLO Detection', res_plotted)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
