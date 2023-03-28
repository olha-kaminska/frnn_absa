## Fuzzy Rough Nearest Neighbour Methods for Aspect-Based Sentiment Analysis
Code for the paper written by [Olha Kaminska](https://scholar.google.com/citations?hl=en&user=yRgJkEwAAAAJ), [Chris Cornelis](https://scholar.google.com/citations?hl=en&user=ln46HlkAAAAJ), and [Veronique Hoste](https://scholar.google.com/citations?hl=en&user=WxOsW3IAAAAJ) and published at the [Electronics journal](https://www.mdpi.com/journal/electronics), particularly, the special issue [AI for Text Understanding](https://www.mdpi.com/journal/electronics/special_issues/AI_recognition) (Volume 12, 2023).

In this work we considered fuzzy-rough-based nearest-neighbours approaches for the Aspect-Based Sentiment Analysis (ABSA) task.

The data for these paper were provided by [SentEmo project](http://www.sentemo.org/), where we considered FMCG (Fast Moving Consumer Goods) dataset of English products reviews.

### Repository Overview ###
- The **code** directory contains .py files with different functions:
  - *preprocessing.py* - functions for data uploading and preperation;
  - *embeddings.py* - functions for tweets embeddings with different methods;
  - *fuzzy_eval.py* - functions for fuzzy-rough-based approach and cross-validation evaluation;
  - *pipeline.py* - function to perform pipeline-based solutions.
- The **data** directory contains *README_data_download.md* file with instruction on uploading necessary dataset files.
- The **model** directory contains *README_model_download.md* file with instruction on uploading necessary models that should be saved in the *model* folder.
- The file **Example.ipynb** provides an overview of all function and their usage on the example of Anger dataset. It is built as a pipeline described in the paper with corresponded results. 
- The file **requirements.txt** contains the list of all necessary packages and versions used with the Python 3.7.4 environment.

### MDPI link ###
https://www.mdpi.com/2154078

### DOI ###
https://doi.org/10.3390/electronics12051088 

### Abstract ###
*Fine-grained sentiment analysis, known as Aspect-Based Sentiment Analysis (ABSA), establishes the polarity of a section of text concerning a particular aspect. Aspect, sentiment, and emotion categorisation are the three steps that make up the configuration of ABSA, which we looked into for the dataset of English reviews. In this work, due to the fuzzy nature of textual data, we investigated machine learning methods based on fuzzy rough sets, which we believe are more interpretable than complex state-of-the-art models. The novelty of this paper is the use of a pipeline that incorporates all three mentioned steps and applies Fuzzy-Rough Nearest Neighbour classification techniques with their extension based on ordered weighted average operators (FRNN-OWA), combined with text embeddings based on transformers. After some improvements in the pipeline’s stages, such as using two separate models for emotion detection, we obtain the correct results for the majority of test instances (up to 81.4%) for all three classification tasks. We consider three different options for the pipeline. In two of them, all three classification tasks are performed consecutively, reducing data at each step to retain only correct predictions, while the third option performs each step independently. This solution allows us to examine the prediction results after each step and spot certain patterns. We used it for an error analysis that enables us, for each test instance, to identify the neighbouring training samples and demonstrate that our methods can extract useful patterns from the data. Finally, we compare our results with another paper that performed the same ABSA classification for the Dutch version of the dataset and conclude that our results are in line with theirs or even slightly better.*

### BibTeX citation: ###
>@Article{electronics12051088, AUTHOR = {Kaminska, Olha and Cornelis, Chris and Hoste, Veronique}, TITLE = {Fuzzy Rough Nearest Neighbour Methods for Aspect-Based Sentiment Analysis}, JOURNAL = {Electronics}, VOLUME = {12}, YEAR = {2023}, NUMBER = {5}, ARTICLE-NUMBER = {1088}, URL = {https://www.mdpi.com/2079-9292/12/5/1088}, ISSN = {2079-9292}, DOI = {10.3390/electronics12051088}}
