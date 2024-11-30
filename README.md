 A comprehensive study on the effectiveness of four Convolutional Neural Network (CNN) models - ResNet, EfficientNet_b0, MobileNetV3 large, and 
Vision Transformer (ViT) - in classifying images from a Knife Dataset.The primary focus lies in examining how different learning rates influence the training performance 
of these models. Employing a systematic approach, the study evaluates each model's accuracy and efficiency under varying learning rate conditions. Key findings 
indicate significant variances in model performance, with certain models displaying superior adaptability to learning rate modifications. These results provide insightful 
revelations about the optimal deployment of CNN models for image classification tasks, particularly in applications requiring the discrimination of knife images. The study's 
findings offer valuable contributions to the field of image classification, highlighting critical considerations in the selection and tuning of CNN models for specific tasks. The experiment was performed on Google Colab with Tesla T4 GPU. 

# Data Augmentation

When training pretrained models for classifying knife images from 192 classes, a strategic sequence of image augmentations was applied to the training data, aiming to enhance the model's capability to generalize and perform effectively on varied data. This process is crucial in computer vision tasks, particularly when dealing with many classes.

The images were first resized to specific dimensions, ensuring uniformity in the size of the input images for consistent processing by the model. Color jittering was then applied, adjusting the brightness of the images by a factor of 0.2, while keeping contrast, saturation, and hue constant. This step introduced variations in lighting conditions, helping the model to recognize knives in different illumination.

To introduce rotational variance, the images were randomly rotated between 0 and 180 degrees. This augmentation is essential for the model to learn to recognize knives regardless of their orientation in the image. Complementing this, random vertical and horizontal flips were applied with a probability of 0.5. These flips added symmetry-related variations to the dataset, enabling the model to identify inverted versions of knives and further enhancing its robustness against orientation changes.

After these spatial and color transformations, the images were converted into tensors, scaling the pixel values to a range suitable for neural network processing. Finally, the images were normalized, a crucial step involving adjusting pixel values based on dataset-specific mean and standard deviation. This normalization is key to stabilizing the learning process and is tailored to the requirements of the pretrained models used.

Overall, these augmentations collectively prepared the dataset to simulate a wide range of real-world scenarios. Such preparation is pivotal in developing a robust classifier capable of accurately identifying various knife classes under different conditions, ultimately boosting the model's accuracy and generalization capabilities on unseen data.
