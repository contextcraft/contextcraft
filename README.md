![Banner](Mnamer.png)
# Replication package for paper : "Automated Generation of Context-Rich Prompts for LLM-based Method Name Suggestion"

# ContextCraft:
ContextCraft is an automated algorithm to generating context-rich prompts for LLMs that generate the expected method names according to the prompts. For a given query (functional description), it retrieves a few best examples whose functional descriptions have
the greatest similarity with the query. From the examples, it identifies tokens that are likely to appear in the final method name as well as their likely positions, picks up pivot words that are semantically related to tokens in the according method names, and specifies
the evaluation results of the LLM on the selected examples. All such outputs (tokens with probabilities and position information, pivot words accompanied by associated name tokens and similarity scores, and evaluation results) together with the query and the selected examples are then filled in a predefined prompt template, resulting in a context-rich prompt..
# Dataset:
Following dataset is used to evalatute the approach
- [English Dataset:](https://github.com/contextcraft/contextcraft/tree/main/Dataset) Method Names with English Functional Descriptions (Dataset of Baseline).

# Approach 
![ContextCraft](ContextCraftOverviewFinal.PNG)
# Best Example Extraction:
This module retrieves related examples from a dataset based on a given functional description (𝑓𝑑) to aid in method name generation. Similarity between the query (𝑓𝑑) and corpus examples (𝑒) is measured by comparing their functional descriptions (𝐷(𝑒)):
![Best Example Eq1](eq1.PNG)

Similarity is calculated using cosine similarity on vectorized descriptions, utilizing BERT embeddings for state-of-the-art performance:
![Best Example Eq2](eq2.PNG)

The number of examples is limited to ten to comply with prompt length constraints in large language models.[Source Code](https://github.com/contextcraft/contextcraft/blob/main/Source_Code/Embedding_Vector_DataBase.ipynb)
#  [ Probabilistic Token Positioning:]() 
## Overview

Predict token positions in method names based on their appearance in functional descriptions. [Source Code](https://github.com/contextcraft/contextcraft/blob/main/Source_Code/PTPandPWI.ipynb)

## Concept
Compute the likelihood of tokens from descriptions appearing in various positions within method names (prefix, infix, suffix).

## Process

### Decomposition
- Split functional descriptions by whitespace and punctuation.
- Split method names by camelCase and underscore conventions.

### Probability Calculation
- **Prefix Probability (𝑃prefix)**: Likelihood of token 𝑡 appearing as the first token in method names.
  ```math
  𝑃prefix(𝑡) = \frac{\text{Occurrences of 𝑡 as prefixes of names}}{\text{Occurrences of 𝑡 in descriptions}}
### Algorithm
![PTP Algorithm](PTP_algorithm.PNG)
# [Pivot Word Identification](https://github.com/propaki/Automethod/tree/main/SFT-Training-Corpus):
## Description
Pivot Word Identification (PWI) is a process designed to enhance the generation of method names from functional descriptions. While directly copying tokens from functional descriptions to method names is one approach, PWI identifies semantically similar words (pivot words) that can inform method name generation even if they aren't directly copied. These pivot words are added to the prompt to improve the relevance and accuracy of the generated method names.

## Steps in Pivot Word Identification

### 1. Initialization

- **Extract Tokens**: Extract tokens from both the functional description and the method name.
- **Setup**: Initialize an empty set to store identified pivot words.

### 2. Iterate Through Method Name Tokens

- **Compare Tokens**: For each token in the method name, identify tokens in the functional description that are semantically similar.
- **Compute Similarity**: Use cosine similarity of BERT embeddings to measure semantic similarity between tokens.
- **Identify Pivot Words**: If a token from the functional description has a similarity score above a predefined threshold and is the highest compared to others, mark it as a pivot word.

### 3. Record Pivot Words

- **Collect Data**: Record each identified pivot word along with the associated method name token and their similarity score.

### 4. Remove Duplicates

- **Ensure Uniqueness**: Ensure that the pivot words list contains unique entries by retaining only the instance with the greatest similarity score for each word.

### 5. Return Results

- **Output**: Return the final list of pivot words, including their corresponding method name tokens and similarity scores.

## Outcome

The PWI process results in a list of pivot words from the functional description that, while not directly part of the method name, are semantically related and useful for generating the method name. This technique enhances the precision and contextual relevance of method name generation in automated systems.

## Requirements

- **Natural Language Processing Tools**: Tokenizers and vector similarity measures.
- **BERT Embeddings**: To compute semantic similarity between tokens.

## References

- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## Example Usage

Below is a conceptual outline for using PWI in your project. This is not a full code implementation but illustrates the high-level steps:

1. **Prepare Functional Descriptions and Method Names**: Extract tokens from your functional descriptions and corresponding method names.
2. **Compute Similarities**: Use BERT embeddings to compute cosine similarity between tokens.
3. **Identify and Record Pivot Words**: Determine pivot words based on similarity scores and store them.
4. **Remove Duplicates**: Ensure pivot words list is unique by removing duplicates.
5. **Utilize Pivot Words**: Use the identified pivot words to enhance your method name generation process.

For a more detailed implementation, refer to the main code in the repository.
![PWI Algorithm](PWI.PNG)
# [LLM-based Feedback Mechanism:](https://github.com/propaki/Automethod/tree/main/Source-Code/RNN-Attn-Copy.ipynb)
We meticulously reproduced and implemented the baseline model in "[Source-Code](https://github.com/propaki/Automethod/tree/main/Source-Code)", which is a [RNN-Attn-Copy](https://github.com/propaki/Automethod/tree/main/Source-Code/RNN-Attn-Copy.ipynb) equipped with both attention and copying mechanisms. This advanced architecture was chosen as our benchmark for assessing the performance of alternative models due to its proven prowess in sequence-to-sequence translation tasks and its exceptional ability to grasp contextual nuances within sequences.
![PWI Algorithm](PromptTemplate_LMS.PNG)
# mNamer Approach: Semantic-Driven Preprocessing and Fine-Tuning:
The mNamer methodology is structured around two core components:
Semantic-Driven Preprocessing and Fine-Tuning, designed to optimize the process of generating accurate Java method names from functional descriptions.
## 1) Semantic-Driven Preprocessing:
This phase begins with the selection of the best examples of functional descriptions and method names from our dataset. Utilizing the Python script **[SelectBestExamples.ipynb](https://github.com/propaki/Automethod/blob/main/Source-Code/SelectBestExamples.ipynb)** found in the Source-Code folder, we identify the top 1,800 examples. These are then divided into four subsets as follows:
![SubDatasets](SubDataset.PNG)

- 300 samples for **[Fine_Tuning.csv](https://github.com/propaki/Automethod/blob/main/English_Best_Examples/Fine_Tuning.csv)**
- 500 for **[Best_Shots.csv](https://github.com/propaki/Automethod/blob/main/English_Best_Examples/Best_Shots.csv)**
- 500 for **[Prompt_Evaluation.csv](https://github.com/propaki/Automethod/blob/main/English_Best_Examples/Prompt_Evaluation.csv)**
- 500 for **[ReinforcementLearning.csv](https://github.com/propaki/Automethod/blob/main/English_Best_Examples/ReinforcementLearning.csv)**

The 500 Best-Shots are converted into vectors using the BERT model through **[EmbeddingBestExamples.ipynb](https://github.com/propaki/Automethod/blob/main/Source-Code/EmbeddingBestExamples.ipynb)**, to calculate semantic similarity. These vectors are stored in the **[English_Embedded](https://github.com/propaki/Automethod/tree/main/English_Best_Examples/English_Embedded)** folder.
![SubDatasets](PromptEngineering.PNG)
Using **[Prompt_Evaluation.csv](https://github.com/propaki/Automethod/blob/main/English_Best_Examples/Prompt_Evaluation.csv)**, we assess the prompt corpus in the **[OptiPrompt](https://github.com/propaki/Automethod/tree/main/OptiPrompts)** folder. To select 30 relevant shots for a given functional description, **[SelectBestShots.ipynb](https://github.com/propaki/Automethod/blob/main/Source-Code/SelectBestShots.ipynb)** compares semantic similarity with the vectors in the **[English-Embedded](https://github.com/propaki/Automethod/tree/main/English_Best_Examples/English_Embedded)** folder, extracting the top 30 relevant examples. This process ensures the creation of an optimal prompt containing best shots that are semantically relevant to the input functional description.
## 2) Fine-Tuning
The fine-tuning stage is divided into two key parts: Supervised Fine Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).

### 2A) Supervised Fine Tuning (SFT): 
Using **English-SFT-Training-Corpus.JSONL**, which contains 300 **[Fine_Tuning.csv](https://github.com/propaki/Automethod/blob/main/English_Best_Examples/Fine_Tuning.csv)** samples in a conversational style, we fine-tune the LLMs (e.g., [GPT-3.5-turbo](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates)) with **[Fine-Tuning(SFT+RLHF).ipynb](https://github.com/propaki/Automethod/blob/main/Source-Code/Fine-Tuning(SFT+RLHF).ipynb)**.
### 2 B) Customized RLHF:
We evaluate the semantic similarity between generated method names and the actual method names provided in **[ReinforcementLearning.csv](https://github.com/propaki/Automethod/blob/main/English_Best_Examples/ReinforcementLearning.csv)**, corresponding to the given functional descriptions. Feedback is then given to the SFT-tuned LLM through prompts to further refine its output.
Through this meticulous approach, mNamer aims to enhance the LLM's ability to generate method names that are not only accurate but also semantically aligned with the developers' intentions, thereby improving code readability and maintainability.
![SubDatasets](RLHF.PNG)
# mNamer

This snippet gives a clear, step-by-step guide for users to replicate the study, ensuring they understand how to set up their environment correctly. Make sure to include any additional specific instructions or prerequisites needed directly in your README or linked documentation to assist users further.
git clone https://github.com/propaki/Automethod.git

fine-tuned ready to chat ChatGPT extention availabe at [Mnamer](https://chat.openai.com/g/g-T58v7ELEM-mnamer)
The source code is the centerpiece of this repository, showcasing the application of BERT-based semantic model for both Semantic Driven Preprocessing and BERT-based RLHF in Postprocessing for LLMs to improve its method naming capabilities. This model represents a significant advancement in the field of automated method naming.

Getting Started
To get started with mNamer:
Install the required dependencies: pip install -r requirements.txt
Follow the instructions in the usage_instructions.md file for detailed steps on how to train and evaluate the models using the provided datasets and prompts.
Contribution
Contributions to LLM-MethodNamer are welcome. If you have suggestions for improvement or want to report bugs, please open an issue or submit a pull request.

