RECRUITMENT TASK REPORT
PART1 (word similarity scores)
LIBRARIES USED:
•	NumPy (Np)
•	spaCy
•	Pandas (pd)
•	Gensim
•	SciPy stats
Steps followed:
1.	Read the content of the file located at '/content/SimLex-999.txt'.
2.	Display the content of the file.
3.	Attempt to read the file content as a CSV file using pd.read_csv, 
4.	Manually create a Pandas DataFrame (df) with the provided data.
5.	Download the "word2vec-google-news-300" model using gensim.downloader.
6.	Use the downloaded Word2Vec model to calculate the similarity between two words ('absorb' and 'withdraw').
7.	Load the 'en_core_web_lg' spaCy model.
8.	Tokenize and preprocess the sentences from the DataFrame.
9.	Train a Word2Vec model on the tokenized sentences.
10.	Calculate the similarity score for each pair of words in the DataFrame.
11.	Scale the similarity scores and add them to the DataFrame.
12.	Use scipy.stats.spearmanr to calculate the Spearman correlation coefficient between the 'SimLex999' column and the scaled similarity scores.
13.	Print the Spearman correlation coefficient.

PART2(phrase and sentence similarity)
Phrase similarity task :
Libraries used:
•	datasets
•	pandas
•	spacy
•	numpy
•	scikit-learn
•	en-core-web-lg (spaCy model)


Steps followed for execution:
1.	Loaded the "PiC/phrase_similarity" dataset using the datasets library.
2.	Preprocessed the text data, including tokenization, lemmatization, and lowercase conversion.
3.	Split the dataset into training and testing sets using train_test_split from scikit-learn.
4.	Utilized spaCy to obtain word embeddings for each phrase.
5.	Calculated the cosine similarity between the embeddings to determine the similarity score.
6.	Defined a threshold and classified the pairs as similar or not based on the similarity score.
7.	Evaluated the model using accuracy, precision, recall, and F1-score.
Conclusion:
The model achieved an accuracy of approximately 49.69%, precision of 49.78%, recall of 70.25%, and an F1-score of 58.27%. These metrics indicate the performance of the model on the given text similarity task. Depending on the specific requirements of your application, you may need to adjust the threshold or explore different models to improve performance.
It's important to note that the success of the model depends on the quality and representativeness of the training data, as well as the chosen similarity metric and threshold. Further experimentation and tuning may be needed to optimize the model for your specific use case.

Sentence similarity task:
Libraries Used:
•	datasets
•	pandas
•	pyarrow
•	spacy
•	scipy.spatial
•	scikit-learn

Steps Followed for execution:
1.	Data Loading: Gathered datasets using datasets and pandas, plus a Parquet file with pyarrow. 
2.	Text Preprocessing: Prepared text for analysis using spaCy's language model. 
3.	Sentence Embedding: Represented sentences as numerical vectors with spaCy's word vectors. 
4.	Similarity Calculation: Measured similarity between sentence pairs using cosine similarity. 
5.	Model Evaluation: Assessed performance with accuracy, precision, recall, and F1-score. 
6.	Conclusion: The model effectively identifies similar sentences, but precision could be improved. Potential for enhancement through optimization or advanced techniques.
