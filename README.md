
Customer Segmentation 
------------

Electric vehicles are emerging faster and it is important to know the presence of different types of customers in the industry.
Identification of potential customer preferences will assist vehicle companys to decide what type of vehicles they should make in order to acheive the best profit.

**Project Purpose**
This study aims to identify the potential customer segments focusing on their socio-economic background, environmental or locational cues and specifications of electric vehicle to identify the polarity of their preference towards electric vehicle. 

   
![ScreenShot](https://github.com/wjj1019/Customer-Segmentation---Case-Study/blob/main/Data/Customer-segmentation.png)

Data (Harvard Dataverse)
------------

The dataset consists of 52 Features which corresponds to the answers to the survery questions regarding an indiviudals situation relevant to vehicle purchasing options (Electric or Gasoline). 

The survey was designed and implemented in SurveyMonkey, an online survey platform, and respondents were recruited and paid through Amazon Mechanical Turk (MTurk). Respondents were recruited from among car owners who have completed at least 100 tasks on MTurk with a minimum 95% acceptance rate. (2019-07-09)

[Descriptions to Each Features](https://github.com/wjj1019/Customer-Segmentation---Case-Study/blob/main/Data/Feature%20Explanation.xlsx)
   
Statistical and Machine Learning Model Used
------------
* K-Mode Clustering [Algorithm Explained](https://github.com/wjj1019/Customer-Segmentation---Case-Study/blob/main/K-Mode%20Selection%20(Customer%20Distribution)/Algorithm%20Explanation%20Doc.pdf)
* Multinomial Naive Bayes 
* Hypothesis testing - Chi-square, Kruskal
* Box Cox Transform

Methodology
------------
1. Feature Engineering
- Each features with different classes were in mix of numerical and categorical (string), we have unified all the features into numerical representation
- Label Encoding and pd.cut to seperate large classes (Eg. age -Continuous -> Age Groups)into multipl classes

2. Cluster Distribution
- There were total of 52 Features and each with different classes (ranging from 2 - 5)
- Clustering was done to extract the different characteristics between potential groups of customers.
- Since the number of features were too large to be sepearated by several clustering combinations, we have divided the features into 4 Sections
(Each Section represents a specific category - Social, Economic, Behavior, Vehicle Specification Preference)

3. Feature Selection
- Among each Feature Sections, using Chi-Square method, only the ones with high importance to Buy Choice (Dependent Variable) were selected.

4. Machine Learning Model
- K-Mode for distributing each sample into optimal cluster
- Naive Bayes (Multinomial) to identify the probability of a customer buying an electric car given all the features
(Total of 27 Different Cluster combinations computed - Each having different preference towards buy choice)

