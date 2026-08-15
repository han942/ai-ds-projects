# AI/Data Science Projects

This folder contains my main data science/AI projects, organized by topic.
- [Korean](./README_KOR.md)
---

## 1. Constructing text-embedded hybrid recommendation models for korean dining resturants   

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/rating_recsys
- Goal: Construction of recommnedation model using crawled user review data from restaurant recommendation website
- Tech stack: Python, mySQL, PyTorch, Pandas, Selenium, Scikit-learn
- Main notebook/script: `diningcode_analysis.ipynb`
- Highlights:
  - Constructing real-time datasets using web-crawling techniques with Selenium
  - Storing & retrieving crawled data into local MySQL sever using SQL queries
  - Text embedding user reviews and integrating into traditional recommendation models to achive accurate rating predictions. 
  - 

---

## 2. News Simplification for Youth via Fine-Tuning LLMs (Large-Language Models) 

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/NLP_Newspaper_CUAI
- Goal: Simplify hard news articles for young students and teenagers.
- Tech stack: Python, LLM Fine-Tuning, PyTorch
- Main notebook/script: `Final_NLP_Newspaper.ipynb`
- Highlights:
  - Constructing parallel corpus using LLM(GPT-4o) based data augmentation
  - Adapting and fine-tuning **[`Gemma 3-1B model`]**(https://huggingface.co/google/gemma-3-1b-it) for the given TST(Text-Style-Transfer) task 
  - Validating model's ability to maintain 100% of its factual accuracy in **75%+** of the original data, also improving readability for younger audiences.

---

## 3. Data visualization and composing business report for global supermarket retail data with Seaborn

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/Global_Supermarket_Analysis
- Goal: Exploratory data analysis and visualization on global supermarket sales data
- Tech stack: Python, Pandas, Matplotlib/Seaborn
- Main notebook/script: `supermarket_analysis.ipynb`, `Global_supermarket_Analysis.pdf`
- Highlights:
  - Comprehensive EDA on global sales and logistics data
  - Loss analysis on specific features that is crucial in economic industry
  - Composing data-driven business strategy report

---
