# REAL VS FAKE NEWS DETECTION

### Introduction
The term "fake news" has evolved to mean different things to different people. We are
defining "fake news" broadly to mean news stories that are untrue, meaning they lack verified
information, sources, or quotes. These articles may occasionally contain propaganda that is
intended to deceive the reader or be produced as "clickbait" to generate income (the writer profits
on the number of people who click on the story). Fake news stories have been more prevalent on
social media in recent years, in part because to how simple and quick it is to spread anything
there, with the potential to lead to violence, suicides, and other tragedies.
In this project I would work with a dataset from Kaggle containing more than 40 000
news articles categorized as true or false. The purpose of this project is to determine if there is a
good possibility to decide if a news article is fake or real, based on the content of the article. For
that purpose, several data models will be trained and tested, to determine which, one is better. I
will further discuss preliminary testing results with test/training data.

### Related Work
The dataset used for this project contains of two files: Fake.csv and True.csv. A cleaning
of the data had to be performed in order to prepare the data for training and validation of models.
Once the data was ready for processing, I had to analyze the data in order to understand it and
determine the best approach for selecting a model. After a research and analysis of the current
data, I determined that the best model to use would be Recurrent neural network (RNN).
Recurrent neural networks are a class of neural networks that is powerful for modeling sequence
data such as time series or natural language. Schematically, a RNN layer uses a for loop to iterate
over the timesteps of a sequence, while maintaining an internal state that encodes information
about the timesteps it has seen so far. Once I have my model trained and validated, I have to
compare it to the results of other models, to determine which one performs best in test
environment.

### Proposed Method
To be able to determine which model would be the best fit for the task, my plan is to
prepare the data and train several models. After analyzing the results of the models accuracy, I
would validate the scores and decide which one would be the best to use. Proposed models to test
are: Logistic Regression, Decision Tree Classification, Random Forest Classifier, Recurrent
neural networks. Models would be compared on the base of accuracy and precision. Here higher
number the better.


### Data
For this task I choose to download data from Kaggle. The data set contains two files. Real.csv
and Fake.csv.
![image](https://user-images.githubusercontent.com/5628399/207475925-93a89aaf-3b87-4195-ab71-140b6469c641.png)

In order to prepare the data, I created categorical field called ‘class’ with values 1 for true and 0 for false. Additionally, because this data was scrapped from various websites, data cleaning is required. 
![image](https://user-images.githubusercontent.com/5628399/207475962-e2530159-579a-4817-9d3f-0c5632183d03.png)


The final step of data preparation is vectorization. Vectorization is process of mapping our natural language into numerical values which is used to make predictions using different machine learning algorithms. Let's make two variables that represent independent and depend on variables in numerical calculations. Linear equation represented by y=mx+c is used during internal calculation. I decided to use TfidfVectorizer for this project.

  ## Data Visualization
  When we visualize the data, we can easily see that the Fake news data is more diverse. There are only 2 subjects in the real data set and there are 6 subjects in the fake data set. 

![image](https://user-images.githubusercontent.com/5628399/207476057-6d294264-a5b7-4b4d-a53b-684ee5e60dff.png)

![image](https://user-images.githubusercontent.com/5628399/207476070-1e0d3ba2-be36-4e6a-95dd-224e1c6d156f.png)


Furthermore, when we do a word count, we can see that there are many fake news articles with less than 300-400 words, in contrast to the real news, where most of the articles have more than 500 words. 

![image](https://user-images.githubusercontent.com/5628399/207476093-898d5997-cf18-4c13-8cea-00df0d411986.png)


![image](https://user-images.githubusercontent.com/5628399/207476106-e2a9efd2-793c-4b56-bb9c-4b4361277efb.png)


Additional to the word count diagrams, we can include wordcloud to be able to see which are the most used words in both fake and real news. This might be related to the fact that most people nowadays are more likely to read the entire article if it is less than 500 words. In contrast, not many people would invest the time to read a 1500 or even 2000 words article.

![image](https://user-images.githubusercontent.com/5628399/207476137-879d8cc9-eced-43b6-a401-9d17fac80071.png)

![image](https://user-images.githubusercontent.com/5628399/207476145-0a4e5ef9-fc63-49ee-96bb-ed6d0c07e843.png)


As we can see there a many similarities. There are a lot of words that are used in both real and fake news. However, when we look closer, we see that “video” is the most used word in the fake news title. In contrast video is not even in the real news wordcloud. This is done in order for the article title to be appealing. If there is a video of it, it must be true, right?


### Experiments

#### Logistic Regression 

From the beginning, this project must decide if some news is fake or real. The result from this task was always going to be binary. For that reason, I choose to test how well a Logistic Regression will work. Another advantage of this model is that it is relatively simple and easy to implement, which makes it ideal first choice.

![image](https://user-images.githubusercontent.com/5628399/207476209-81be7115-f191-45a0-9342-1912afcef57a.png)


#### Decision Tree Classifier
Classification trees are a subset of decision trees that rely on a judgment being made based on a "Yes" or "No" (Real or Fake) response to a question. Therefore, a classification tree is a type of tree that decides if news is True or False by posing several connected questions and using the replies to arrive at a workable answer. These kinds of trees are typically created using a technique known as binary recursive partitioning. The data is divided into distinct modules, or partitions, using the binary recursive partitioning technique, and then these partitions are further spliced into each branch of the decision tree classifier.

![image](https://user-images.githubusercontent.com/5628399/207476237-9a99ddc2-a7c9-4263-bffa-4d76da6bda96.png)


#### Random Forest Classifier

Overfitting is a concern for decision trees, but not for random forests. The reason for this is because random forest classifiers employ random subsets to address this issue. Random forests are slower than decision trees. Multiple decision trees are used in random forests, which requires a lot of processing power and adds to the processing time. Random forests are more difficult to comprehend than decision trees, and while it is simple to convert the former to the rules, it is more challenging to do so with the latter.

  ![image](https://user-images.githubusercontent.com/5628399/207476275-04169dd0-4f28-485b-bb1d-f024563d3568.png)


#### Recurrent neural networks
A recurrent neural network is a neural network that is specialized for processing a sequence of data x(t)= x(1), . . . , x(τ) with the time step index t ranging from 1 to τ. For tasks that involve sequential inputs, such as speech and language, it is often better to use RNNs. In an NLP problem, if you want to predict the next word in a sentence it is important to know the words before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Another way to think about RNNs is that they have a “memory” which captures information about what has been calculated so far.

![image](https://user-images.githubusercontent.com/5628399/207476301-355fadf5-5a70-4c51-95a0-c8d1b11c160b.png)
![image](https://user-images.githubusercontent.com/5628399/207476309-a5728f3e-c25b-4de7-901c-8c74a94c3645.png)


### Results and Discussion


|   Model      | Accuracy on testing set  | Precision on testing set  |  Recall on testing set |
| ------------- | ------------- | ------------- | ------------- |
| Random Forest Classifier | 0.907976827 | 0.905455386 | 0.903784033 |
| Decision Tree Classifier | 0.890597148  | 0.861072584 | 0.907208962 |
| Logistic Regression | 0.601270053 | 0.548081368 | 0.593491865 |
| Recurrent Neural Networks | 0.989193405 | 0.984281091 | 0.993235363 |


 From the table above we can clearly see that the best model is RRN with LSTM layer. More data would be required to be able to fine tune the model but at the current state, we have one very accurate model for predicting real vs fake news.
 
 ## Accuracy and lost
 From the figures below we can say that the model gets stable training loss after around 5 epochs and that accuracy hits peak after around 2 epochs. Because of this, there is no reason to run the model with more than 5 epochs.
 
 ![image](https://user-images.githubusercontent.com/5628399/207477268-1f9ee578-11cc-4747-aaff-b85efa948134.png)  ![image](https://user-images.githubusercontent.com/5628399/207477272-83e33a7e-01d9-4e56-a573-13e330cd672e.png)


#### Confusion Matrix
The confusion matrix below gives us an idea of how accurate the model is with this current test data set. There are very few false-positive and false-negative results.

![image](https://user-images.githubusercontent.com/5628399/207477314-0c827340-c0f6-419c-90ec-3bb892ded53f.png)


## Conclusion

Recurrent neural networks (RNNs) are a type of neural network in which the results of one step are fed into the current step as input. Traditional neural networks have inputs and outputs that are independent of one another, but there is a need to remember the previous words in situations where it is necessary to anticipate the next word in a sentence. As a result, RNN was developed, which utilized a Hidden Layer to resolve this problem. The Hidden state, which retains some information about a sequence, is the primary and most significant characteristic of RNNs. This neural network works perfectly for my project as it is showing the most promising results. Compared to the Logistic Regression, Decision Tree Classifier and Random Forest Classifier models, RNN with LSTM layer is performing extremely well. This is because of the hidden layer RNN has, so that each iteration has information about the previous one, and by doing so, improving the accuracy of the prediction.

## References

Encyclopedia Britannica, Encyclopedia Britannica, Inc., https://www.britannica.com/dictionary/news. 

“Fake News.” FAKE NEWS | Definition in the Cambridge English Dictionary, https://dictionary.cambridge.org/us/dictionary/english/fake-news. 

Storyblocks, https://www.storyblocks.com/video/stock/animation-text-news-breaking-and-news-intro-graphic-with-lines-and-world-map-in-studio-abstract-background-elegant-and-luxury-dynamic-style-for-news-template-hwnt-ut7vkelrtgka. Accessed 8 Dec. 2022. 

Trandafoiu, Ruxandra. Wither Fake News: COVID-19 and Its Impact on Journalism, https://blogs.edgehill.ac.uk/isr/wither-fake-news-covid-19-and-its-impact-on-journalism/. Accessed 8 Dec. 2022. 

Bisaillon, Clément. “Fake and Real News Dataset.” Kaggle, 26 Mar. 2020, https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset.

“Recurrent Neural Networks (RNN) with Keras:   TensorFlow Core.” TensorFlow, https://www.tensorflow.org/guide/keras/rnn.

“Sklearn.ensemble.RandomForestClassifier.” Scikit-Learn, scikit-learn.org/stable/modules/generated
/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier. Accessed 19 Nov. 2022.

“Sklearn.tree.DecisionTreeClassifier.” Scikit-Learn, scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision+tree+classification. Accessed 18 Nov. 2022.

 “Logistic Regression 3-Class Classifier.” Scikit-Learn, scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html?highlight=logistic+regression. Accessed 19 Nov. 2022.

“Fake and Real News Dataset.” Www.kaggle.com, www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset.

“Introduction to Recurrent Neural Network - GeeksforGeeks.” GeeksforGeeks, 3 Oct. 2018, www.geeksforgeeks.org/introduction-to-recurrent-neural-network/.


