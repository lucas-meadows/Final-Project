UFC Sports Betting Classifier

This project aims to use pre-fight market data and static fighter characteristics to restrospectively classify whether large odds movements occured. We imported data Kaggle which contained information about individual fighters in the UFC, and reduced this dataset to static features, such that there was no risk of data leakage with features (such as fighter records) that needed to be updated. We also took snapshots across major betting markets to produce betting odds data. Using such datasets, we produced logistic regression models to analyse this question, produced promising results with potential for application amongst sports bettors.

In this github repository, data_pipeline.py is responsible for pulling data from the-odds-api.com, an API which gives access to betting market data for various sports. It is also responsible for parsing this data, taking these snapshots and producing a comprehensible csv file. Likewise, it imports the data set from kaggle, ready for use.

Feature_Engineering.py is responsible for accessing the data through our pipeline, and processing the market data such that we get the features we desire. It makes sure to remove any features, that could potentially cause data leakage and affect our results. It then returns the market data with its engineered features so that it can be used elsehwere.

Classification.py calls upon both datasets, merges them and then prepares them for the various models we produced. This includes the Random Forest Classifier we used preliminarily to check our concern of models being overly complex for our smaller dataset. IT also has our Logistic Regression, Cross-Validated Logistic Regression, and Calibration Curve to check ur models performances. These models can all be trained on the data downloaded from the first two files, data_pipeline.py and Feature_Engineering.py

