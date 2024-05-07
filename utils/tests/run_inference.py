import os
import time
import threading
import traceback

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import queue

framerate  = 20
pframerate  = 20
model_color = {}

def draw_boxes(frame, bboxes, labels, scores, bcolor):
    print(f"Drawing boxes: {bboxes}, {labels}, {scores}, {bcolor}")
    try:
        for bbox, label, score in zip(bboxes[0], labels[0], scores[0]):
            if score > 0.01:
                label = int(label[0])
                #print(f"Label: {label}, Score: {score}  Bbox: {bbox} Bcolor: {bcolor}")
                y_min, x_min, y_max, x_max = bbox
                height, width = frame.shape[:2]
                x_min = int(x_min * width)
                y_min = int(y_min * height)
                x_max = int(x_max * width)
                y_max = int(y_max * height)
                thickness = 2
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bcolor, thickness)  
                cv2.putText(frame, f"{label}: {score}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcolor, thickness)
                cv2.putText(frame, f"Original FPS: {framerate}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, f"Playback FPS: {pframerate}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error drawing boxes: {e}")


def get_model_color(model_name):
    global model_color
    for model in os.listdir("./models"):  
        if model == model_name:
            col = (0, 0, 0)     
            if model_name not in model_color:
                model_color[model_name] = get_random_color()
                col = model_color[model_name]
            elif model_name in model_color:
                col = model_color[model_name]
            return col
    

def get_random_color():
    color_choices = {
        0: (255, 0, 0),      # Red
        1: (0, 255, 0),      # Green
        2: (255, 255, 0),    # Yellow
        3: (0, 0, 255),      # Blue
        4: (255, 165, 0),    # Orange
        5: (128, 0, 128),    # Purple
        6: (0, 255, 255),    # Cyan
        7: (255, 192, 203),  # Pink
        8: (0, 128, 0),      # Dark Green
        9: (255, 255, 255),  # White
        10: (0, 0, 0)        # Black
    }
    return color_choices[np.random.randint(0, 4)]


def wait_for_triton():
    while True:
        try:
            triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=True)
            if triton_client.is_server_ready():
                return triton_client
            else:
                print("waiting for triton server to be ready")
                time.sleep(15)
        except Exception as e:
            print("channel creation failed: " + str(e))
            time.sleep(15)


def run_tests():
    print("Running tests")
    triton_client = wait_for_triton()
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")
    for video in os.listdir(f"{current_directory}/utils/tests/samplevideos"):
        print(f"Video: {video}")
        cap = cv2.VideoCapture(f"{current_directory}/utils/tests/samplevideos/{video}")
        target_size = (768, 512)
        stored_results = {'bboxes': None, 'labels': None, 'scores': None}
        lock = threading.Lock()
        processing_queue = queue.Queue()
        playback_queue = queue.Queue()
        def inference_thread_function(triton_client, model_name, bcolor, input_data, stored_results, lock):
            try:
                inputs = []
                outputs = []
                batch_size = 1
                input_data_batch = np.expand_dims(input_data.astype(np.uint8), axis=0)
                input_data_batch = np.repeat(input_data_batch, batch_size, axis=0)  
                inputs.append(grpcclient.InferInput("image", (batch_size,) + input_data.shape, "UINT8"))
                inputs[0].set_data_from_numpy(input_data_batch)  
                outputs.append(grpcclient.InferRequestedOutput("predicted_bboxes"))
                outputs.append(grpcclient.InferRequestedOutput("predicted_labels"))
                outputs.append(grpcclient.InferRequestedOutput("predicted_scores"))
                results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
                with lock:
                    stored_results['bboxes'] = results.as_numpy("predicted_bboxes").astype(np.float32)
                    stored_results['labels'] = results.as_numpy("predicted_labels").astype(np.int32)
                    stored_results['scores'] = results.as_numpy("predicted_scores").astype(np.float32) 
                    stored_results['bcolor'] = bcolor
            except InferenceServerException as e:
                print("inference failed: ")

        def read_frames():
            global framerate
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                framerate = int(cap.get(cv2.CAP_PROP_FPS))
                print(f"Frame rate: {framerate}")
                if not ret:
                    break
                frame_count += 1
                processing_queue.put(frame)
            cap.release()
            processing_queue.put(None)
        
        def inference_and_draw():
            while True:
                frame = processing_queue.get()
                if frame is None:
                    break
                input_data = cv2.resize(frame, target_size)
                for modeldir in os.listdir("./models"):
                    model_name = modeldir
                    bcolor = get_model_color(model_name)
                    inference_thread_instance = threading.Thread(target=inference_thread_function, args=(triton_client, model_name, bcolor, input_data, stored_results, lock))
                    inference_thread_instance.start()
                    inference_thread_instance.join()
                    with lock:
                        if stored_results['bboxes'] is not None:
                            frame = draw_boxes(frame, stored_results['bboxes'], stored_results['labels'], stored_results['scores'], stored_results['bcolor'])
                playback_queue.put(frame)
                processing_queue.task_done()
        

        def playback():
            global pframerate
            buffer_time = 30 
            start_time = time.time()
            while True:
                if processing_queue.empty() and time.time() - start_time > buffer_time:
                    break
                if playback_queue.empty():
                    time.sleep(0.1)
                    continue
                frame = playback_queue.get()
                if frame is None:
                    break
                cv2.imshow("Playback", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_delay = 1 / int(framerate)
                time.sleep(frame_delay)
                pframerate = int(1 / frame_delay)
                playback_queue.task_done()

        read_thread = threading.Thread(target=read_frames)
        inference_thread = threading.Thread(target=inference_and_draw)
        playback_thread = threading.Thread(target=playback)
        read_thread.start()
        inference_thread.start()
        time.sleep(6)
        playback_thread.start()
        read_thread.join()
        inference_thread.join()
        playback_thread.join()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    run_tests()
