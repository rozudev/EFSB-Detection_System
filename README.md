# EFSB-Detection_System
This is my fist big project for AI in Agrculture, I originated this idea for my 10th Grade Research for AI in Agrculture research development, a eggplant and shoot borer (EFSB) symptoms Real time Detection System for eggplant farms. In this research project I combined OpenCV and YOLO, I used two models of YOLO (yolo11 and yolo26) for experimenting, testing and comparison of overall accuracy. The Eggplant Fruit and Shoot Borer (EFSB), Leucinodes orbonalis, is the most destructive pest of eggplants in South and Southeast Asia, capable of causing 70-80% yield loss, and up to 100% in severe cases.

The goal of this project is to help and support eggplant farmers on dealing with destructive pest and creat a good impact in the agricultural sector. This project contributes to sustainable farming by enabling early detection of EFSB infestation which can help reduce unnecesssary pescticide usage.

This project that I made is still a functional software prototype, I manualy collected 305 pictures of healthy eggplant and EFSB infected eggplant at a small eggplant farm and then I annotated the costum dataset of 305 eggplant images for classification using roboflow, after that I used kaggle to get the weight file (best.pt) and convert it into torch.ONNX format, lastly I programmed it using Python (3.11).

![image alt](https://github.com/rozudev/EFSB-Detection_System/blob/e5c945339f58c3c384ed7fb449bea39b5b076d9c/research-documentation_pictures/7x2QnYNg.jpg)


For the final product, I will be using Raspberry Pi 5, Raspberry Pi camera module 3 and design systems. We as a group will aim to collect 1,000+ pictures of both healthy and infected eggplant for more efficient and accurate detection of the AI model. We will be saving and contribute fairly on the expeness for the materials needed (Raspberry Pi 5, Raspeberry Pi Camera module 3 and many more)

Date: 24/04/2026

-----------------------------------------------------------------------------------------------------------------------------------------

As the Project Creator and Technical Lead of my research group, I initiated and designed the project concept, developed the AI Prototype and system architecture, while collaborating with my research group on the project expenses.

![image alt](https://github.com/rozudev/EFSB-Detection_System/blob/e5c945339f58c3c384ed7fb449bea39b5b076d9c/research-documentation_pictures/Bnh8qL0v.jpg)

Here is the systems that I draw for the early design of our project:

![image alt](https://github.com/rozudev/EFSB-Detection_System/blob/e5c945339f58c3c384ed7fb449bea39b5b076d9c/research-documentation_pictures/JjcYb4Xk.jpg)

![image alt](https://github.com/rozudev/EFSB-Detection_System/blob/e5c945339f58c3c384ed7fb449bea39b5b076d9c/research-documentation_pictures/PeRBHBPi.jpg)

![image alt](https://github.com/rozudev/EFSB-Detection_System/blob/e5c945339f58c3c384ed7fb449bea39b5b076d9c/research-documentation_pictures/r5JVte_C.jpg)

![image alt](https://github.com/rozudev/EFSB-Detection_System/blob/e5c945339f58c3c384ed7fb449bea39b5b076d9c/research-documentation_pictures/uAza_y8_.jpg)

Here is also the link of my video on my YT channel where I explain about the project and the systems that I created.
https://www.youtube.com/watch?v=GRup5Qj9Rrk&t=3s

Date: 16/05/2026

--------------------------------------------------------------------------------------------------------------------------------------------

I added a Warning System on my functionl software prototype using Arduino UNO and components (red LED, green LED and Buzzer), I used pyfirmata2 to allow arduino to communicate with my python code.

![image alt](https://github.com/rozudev/EFSB-Detection_System/blob/e5c945339f58c3c384ed7fb449bea39b5b076d9c/research-documentation_pictures/bbihnQ2u.jpg)

The red LED will turn on and the buzzer will make a noise if theres a infected efsb eggplant detected, it will only turn green if the eggplant is healthy and there is no infected eggplant detected.

Date: 11/06/2026




