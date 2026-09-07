# AI/Data Science Projects

This folder contains my main data science/AI projects, organized by topic.
- [Korean](./README_KOR.md)

## 1. Constructing text-embedded hybrid recommendation models for korean dining resturants   

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/rating_recsys
- Goal: Constructing recommnedation model using crawled user review data from restaurant recommendation website
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat&logo=selenium&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
- Main notebook/script: `diningcode_analysis.ipynb`
- Highlights:
  - Constructing real-time datasets using web-crawling techniques with Selenium
  - Storing & retrieving crawled data into local MySQL sever using SQL queries
  - Text embedding user reviews and integrating into traditional recommendation models to achive accurate rating predictions. 
  - 

---

## 2. News Simplification for Youth via Fine-Tuning LLMs (Large-Language Models) 

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/NLP_Newspaper_CUAI
- Goal: Translating hard news articles for young students and teenagers.
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black) ![LLM Fine-Tuning](https://img.shields.io/badge/LLM%20Fine--Tuning-412991?style=flat&logo=openai&logoColor=white)
- Main notebook/script: `Final_NLP_Newspaper.ipynb`
- Highlights:
  - Constructing parallel corpus using LLM(GPT-4o) based data augmentation
  - Adapting and fine-tuning [Gemma 3-1B](https://huggingface.co/google/gemma-3-1b-it) for the given TST(Text-Style-Transfer) task 
  - Validating model's ability to maintain 100% of its factual accuracy in **75%+** of the original data, also improving readability for younger audiences.

---

## 3. Data visualization and composing business report for global supermarket retail data with Seaborn

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/Global_Supermarket_Analysis
- Goal: Conducting exploratory data analysis (EDA) and loss analysis to develop strategies to improve loss status.
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white)
- Main notebook/script: `supermarket_analysis.ipynb`, `Global_supermarket_Analysis.pdf`
- Highlights:
  - Comprehensive EDA on global sales and logistics data
  - Loss analysis on specific features that is crucial in economic industry
  - Composing data-driven business strategy report

---
