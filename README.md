# EEG-Signal-Classification
CS554 - Machine Learning Class Project

Purpose of the project is analyzing the performance change of feeding image data of EEG signals to CNN instead of time series data.
## Converting Time Series Data to Image
Collected three different time series data must be transformed
into 2D images because CNNs take an image as input
to the Neural Network. That is why, Gramian Angular Field
and Markov Transition Field are chosen methods to images
obtained from time series data. Gramian Angular field
represents temporal correlation between each time point. First,
scale the time series data between −1 and 1 using Min-
Max scaler. Then, convert the scaled time series into polar
coordinates. Finally, after taking the inner product of the polar
encoded values, time series data are gotten like image form.
Moreover, Markov Transition Field follows similar steps.

## Results
After converting time series data to image, comparisons are made according to Markov Transition Field conversion since it gave the best results amongst the datasets.

![results](https://github.com/sinanutkuulu/EEG-Signal-Classification/assets/92628109/39400097-6f6f-484b-a536-d4032e33418a)

Loss Plot of One of the Dataset on CNN: 

![Loss_Curve](https://github.com/sinanutkuulu/EEG-Signal-Classification/assets/92628109/ee5b5fea-e6c8-41a2-ae60-249eee975edf)

