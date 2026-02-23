# Traffic-MOT-Threat-Classification

This is a personal project I made to learn more about how to track different objects with computer vision, and apply classifications to them based on their activities.
Anyone is free to test and use this project. This was created with the assistive help of AI.

Structure:<br>
The model used for this project is the YOLOv8 small model, and I wanted to recognise 4 classes from the coco dataset: [car=2, motorcycle=3, bus=5, truck=7]
In addition to creating a basic working model, I've incoporated a tilling method where the tile size is 1080 so that larger videos are split into smaller chunks, and object recognition is performed on all of the tiles per frame
in order to increase the confidence scores of smaller objects. After testing, the best confidence value was found to be 0.3. Afterwards, a "threat" classification system was implemented into 3 categories for wether a car was moving too fast, loiting for too long, or no threat.

The speed was calculated by how many pixels a detected car has moved, and it it crossed a threshold, it will classify it as "WARN". Likewise with the loiting, every car has its own timer to detect how long it has remained in a certain area. Once it has loitered for more than 5 seconds, it will classify it as a "WARN". This project uses ByteTrack to track all information of objects it recognises. If a tracked object is not seen in the frame for more than 2 seconds, it will automatically delete its information from the bytetrack in order to save space.

Inference Speed:
The project script was run on a rtx 4060 GPU with 2 different video sizes. The first video was of size [2160x3840] and has an average inference time of 3.8 fps. The second video was tested on [1080x1920] and has an average inference time of 16.4 fps.

Overall, this has been an insightful and fullfilling project to depend my understanding on computer vision and what its capabilities are.

<p align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/2b305d5a-3594-4572-97f8-5cee054aa1f6" /> <img width="500" alt="image" src="https://github.com/user-attachments/assets/1af29c65-9203-49ce-a019-4a79e320dc7b" />
</p>

