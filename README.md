 A comprehensive study on the effectiveness of four Convolutional Neural Network (CNN) models - ResNet, EfficientNet_b0, MobileNetV3 large, and 
Vision Transformer (ViT) - in classifying images from a Knife Dataset.The primary focus lies in examining how different learning rates influence the training performance 
of these models. Employing a systematic approach, the study evaluates each model's accuracy and efficiency under varying learning rate conditions. Key findings 
indicate significant variances in model performance, with certain models displaying superior adaptability to learning rate modifications. These results provide insightful 
revelations about the optimal deployment of CNN models for image classification tasks, particularly in applications requiring the discrimination of knife images. The study's 
findings offer valuable contributions to the field of image classification, highlighting critical considerations in the selection and tuning of CNN models for specific tasks. The experiment was performed on Google Colab with Tesla T4 GPU. 

## Knife Dataset

In the development of machine learning models for knife classification, a comprehensive dataset plays a pivotal role in ensuring accuracy and robustness. The dataset employed in this project is meticulously partitioned into three subsets: training, testing, and validation, each serving a distinct purpose in the model's learning and evaluation process.

The training set, comprising 9,624 images, forms the foundation of the model's learning phase. This extensive collection of images allows the model to learn and recognize diverse patterns and characteristics of different knives. The richness and variety of this dataset are crucial for the model to develop a deep understanding and thereby improve its predictive accuracy.

For model evaluation, a testing set consisting of 350 images is utilized. This set is pivotal in assessing the model's performance on new, unseen data. It provides insights into the model's generalization capabilities and its effectiveness in real-world scenarios. The testing phase is crucial for identifying any potential biases or shortcomings in the model post-training. Additionally, the validation set, containing 400 images, is integral to the model tuning process. It aids in fine-tuning the model parameters and ensures that the model does not overfit the training data. The validation set acts as a benchmark during the training process, allowing for continuous monitoring and adjustments to achieve optimal performance. Crucially, the dataset encompasses 192 distinct classes of knives as shown in Figure 1. presenting a wide array of knife types for the model to learn from. This diversity is key to developing a nuanced and sophisticated classification model. The methodology adopted for this project is grounded in supervised learning. In this approach, the model is trained on labeled data, where each image is tagged with a specific class. This enables the model to learn the association between the features of the images and their corresponding labels. Supervised learning is particularly suited for classification tasks, as it allows for precise categorization based on learned patterns.

![image](https://github.com/user-attachments/assets/fcf64f7d-9bd0-4d65-9572-4b6d2f050b99)

In summary, the structured and comprehensive nature of
the dataset, along with the use of supervised learning
techniques, forms the backbone of this knife classification
project. The careful division of the dataset into training,
testing, and validation sets, combined with the variety
offered by 192 knife classes, ensures a robust framework
for the development and evaluation of the machine learning


## Data Augmentation

When training pretrained models for classifying knife images from 192 classes, a strategic sequence of image augmentations was applied to the training data, aiming to enhance the model's capability to generalize and perform effectively on varied data. This process is crucial in computer vision tasks, particularly when dealing with many classes.

The images were first resized to specific dimensions, ensuring uniformity in the size of the input images for consistent processing by the model. Color jittering was then applied, adjusting the brightness of the images by a factor of 0.2, while keeping contrast, saturation, and hue constant. This step introduced variations in lighting conditions, helping the model to recognize knives in different illumination.

To introduce rotational variance, the images were randomly rotated between 0 and 180 degrees. This augmentation is essential for the model to learn to recognize knives regardless of their orientation in the image. Complementing this, random vertical and horizontal flips were applied with a probability of 0.5. These flips added symmetry-related variations to the dataset, enabling the model to identify inverted versions of knives and further enhancing its robustness against orientation changes.

After these spatial and color transformations, the images were converted into tensors, scaling the pixel values to a range suitable for neural network processing. Finally, the images were normalized, a crucial step involving adjusting pixel values based on dataset-specific mean and standard deviation. This normalization is key to stabilizing the learning process and is tailored to the requirements of the pretrained models used.

Overall, these augmentations collectively prepared the dataset to simulate a wide range of real-world scenarios. Such preparation is pivotal in developing a robust classifier capable of accurately identifying various knife classes under different conditions, ultimately boosting the model's accuracy and generalization capabilities on unseen data.
