
# Object Detection YOLOv8 ✨

Leveraging the power of YOLOv8 I have developed Cars counting system including bounding boxes around cars and having unique IDs using Tracker (sort) and counting when passing through a specific line in a frame.


## Result 🎞

![Result](https://github.com/bilalansar3/Cars-Count-YOLOv8/blob/main/images/carCount.gif)


### Working 🛠
I designed a mask for the original frame and use **BITWISE_AND**  to make a **masked frame** and pass this masked frame to model [YOLOv8](https://github.com/ultralytics/ultralytics) that detects the cars, trucks, buses, motorcycle only. Then the output frame shows Bounding Box and Counter Line to count vehicles passing through this line. 

[SORT](https://github.com/abewley/sort/tree/master) tracker is used to count the unique Ids of all cars.

![Working](https://github.com/bilalansar3/Cars-Count-YOLOv8/blob/main/images/bitwise%20and.gif)

## Run This Project 💻


**Install Libraries (Dependencies)**

```bash
  pip install -r requirements.txt
```
put video named ***input.mp4*** then run app.py
```bash
  py app.py
```
after complete compilation you will have ***OutPut.mp4*** file that is the result of model
    
## 🔗 Get In Touch

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bilalahmad3/)
[![github](https://img.shields.io/badge/github-333333?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bilalansar3)
