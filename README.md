# Sentiment-Analysis-Tool-for-Social-Media-Investigation

A machine learning and NLP-based system for automated sentiment analysis of social media data, designed to streamline investigative processes. This tool leverages real-time data scraping, OCR-based text extraction, machine learning classification, and interactive dashboards to provide actionable insights into public sentiment, trends, and anomalies.

---

## Project Overview

With billions of posts shared across social media daily, investigators, researchers, and businesses face a major challenge in extracting meaningful insights efficiently. Manual analysis is slow, error-prone, and cannot keep up with the scale of online data.

This project introduces a **web-based sentiment analysis tool** that automatically collects, preprocesses, and analyzes social media content — including **text embedded in images** — to determine sentiment polarity (positive or negative). The system integrates:

* **Natural Language Processing (NLP)** for text preprocessing and feature extraction
* **Machine Learning algorithms (Naive Bayes, TF-IDF)** for sentiment classification
* **OCR (Tesseract + OpenCV)** for extracting text from memes, screenshots, and image-based posts
* **API integration and automation** for live data collection
* **Interactive dashboards and automated reports** for clear visualization of results

This solution enables investigators to detect patterns, anomalies, and suspicious activities in real time, supporting faster and more accurate decision-making.

---

## Problem Statement

Social media platforms generate vast volumes of unstructured, noisy data - much of it in the form of images containing text. Traditional sentiment analysis systems fail to process image-based text effectively and often lack investigative focus.

Our system addresses these gaps by:

* Extracting text from images using OCR
* Preprocessing and classifying sentiment automatically
* Presenting results in a clear, investigator-friendly format

This allows users to quickly identify harmful, hateful, or trending content and focus on high-priority cases.

---

## Objectives

1. **Sentiment Model Development** - Build and train a machine learning model using labeled social media datasets for positive/negative classification.
2. **OCR-based Text Extraction** - Use Tesseract OCR integrated with OpenCV to extract text from images.
3. **Sentiment Classification** - Clean and process text, then categorize it based on sentiment polarity.
4. **System Integration** - Create a unified pipeline that connects text extraction, sentiment analysis, and visualization, making it usable for real-world investigations.

---

## Methodology

1. **Data Acquisition** - Collect social media posts from platforms like Twitter, Facebook, and Instagram. Use Sentiment140 dataset from Kaggle for initial training.

2. **Data Preprocessing** -

   * Lowercasing text
   * Removing special characters, URLs, hashtags, and stopwords
   * Handling class imbalance by sampling equal positive/negative examples
   * TF-IDF vectorization for feature representation

3. **Model Development** -

   * Train a **Multinomial Naive Bayes** classifier on preprocessed data
   * Evaluate using **accuracy, precision, recall, F1-score, and confusion matrix**

4. **OCR Integration** -

   * Use OpenCV to read images and convert to grayscale
   * Extract text using PyTesseract
   * Preprocess and classify extracted text

5. **Result Visualization** -

   * Generate classification reports
   * Present insights via web-based dashboard

---

## System Architecture

**Core Components:**

* **Frontend:** Web interface for uploading images/text and viewing results
* **Backend:** Python Flask/Django API handling preprocessing, ML inference, and reporting
* **Database:** Stores past investigations, results, and user interactions
* **ML Model:** Naive Bayes classifier with TF-IDF feature extraction
* **OCR Engine:** Tesseract + OpenCV

The system supports modular integration, allowing additional ML models or APIs to be added without redesigning the architecture.

---

## Results

* **Accuracy Achieved:** 77.2% on test data
* **Precision:** 0.80 (positive sentiment)
* **Recall:** 0.82 (negative sentiment)
* **F1-Score:** \~0.77 (balanced performance)

Example outputs demonstrate successful sentiment detection from both plain text and OCR-extracted content, including real social media screenshots.

---

## Tech Stack

* **Languages:** Python, JavaScript
* **Libraries & Frameworks:**

  * Scikit-learn (Naive Bayes, evaluation metrics)
  * OpenCV + PyTesseract (OCR processing)
  * Pandas, NumPy (data preprocessing)
  * Matplotlib / Seaborn (visualization)
* **APIs:** Twitter API, Facebook Graph API (for live data scraping)
* **Database:** SQL / NoSQL storage (optional for persistence)

---

## Future Scope

* Integration with **deep learning models** (BERT, RoBERTa) for higher accuracy
* Support for **multimodal sentiment analysis** (text + emojis + image context)
* **Real-time dashboards** with live alerts for suspicious activities
* **Multilingual sentiment detection** for global applicability
* Automated **trend correlation** with real-world events for crisis management

---

## Contact

For questions, collaborations, or contributions:
**Ira Padole**
📧 [irapadole2004@gmail.com](mailto:irapadole2004@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/ira-padole-3487062b4) • [Portfolio](https://irapadole.com)

---
