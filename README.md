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
### Algorithm 1
![PTP Algorithm](PTP_algorithm.PNG)
# [Pivot Word Identification](https://github.com/propaki/Automethod/tree/main/SFT-Training-Corpus):
## Overview
Pivot Word Identification (PWI) is a process designed to enhance the generation of method names from functional descriptions. While directly copying tokens from functional descriptions to method names is one approach, PWI identifies semantically similar words (pivot words) that can inform method name generation even if they aren't directly copied. These pivot words are added to the prompt to improve the relevance and accuracy of the generated method names.[Source Code](https://github.com/contextcraft/contextcraft/blob/main/Source_Code/PTPandPWI.ipynb)

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


## Example Usage

Below is a conceptual outline for using PWI in your project. This is not a full code implementation but illustrates the high-level steps:

1. **Prepare Functional Descriptions and Method Names**: Extract tokens from your functional descriptions and corresponding method names.
2. **Compute Similarities**: Use BERT embeddings to compute cosine similarity between tokens.
3. **Identify and Record Pivot Words**: Determine pivot words based on similarity scores and store them.
4. **Remove Duplicates**: Ensure pivot words list is unique by removing duplicates.
5. **Utilize Pivot Words**: Use the identified pivot words to enhance your method name generation process.

## Algorithm 2
![PWI Algorithm](PWI.PNG)
# [LLM-based Feedback Mechanism:]()
## Overview

The LLM-based Feedback Mechanism (LFM) evaluates the effectiveness of a Language Learning Model (LLM) in generating method names from functional descriptions. Unlike token-based processes, LFM provides quantitative feedback by comparing the generated method names with the ground truth names from examples. This feedback mechanism measures the accuracy of the LLM's output and highlights discrepancies to improve model performance.[Source Code](https://github.com/contextcraft/contextcraft/blob/main/Source_Code/LLM_FeedBack.ipynb)

## Steps in LLM-based Feedback Mechanism

### 1. Generate Method Name

- **Feed Prompt to LLM**: Input the functional description from each example into the LLM.
- **Generate Method Name**: Request the LLM to generate a method name based on the provided description.

### 2. Compare Generated Name with Ground Truth

- **Compute Edit Distance**: Compare the method name generated by the LLM with the actual method name in the example using character-based edit distance.
- **Assess Similarity**: Evaluate how closely the generated name matches the ground truth based on the edit distance score.

### 3. Generate Feedback Message

- **Specify Differences**: Create a feedback message (`msg`) detailing how the generated method name differs from the actual method name.
- **Highlight Discrepancies**: The message should include specific information about insertions, deletions, or substitutions needed to align the generated name with the ground truth.

## Outcome

The LFM process provides quantitative feedback on the LLM's method name generation. By comparing generated names with ground truth names and highlighting differences, it helps in identifying areas where the LLM's performance can be improved.

## Requirements

- **Language Learning Model (LLM)**: A model capable of generating method names from functional descriptions.
- **Edit Distance Calculation**: Tools or algorithms to compute the edit distance between two strings.

## References

- **Edit Distance**: [Edit Distance - Wikipedia](https://en.wikipedia.org/wiki/Edit_distance)

## Example Usage

Below is a conceptual outline for using LFM in your project:

1. **Prepare Data**: Collect functional descriptions and corresponding ground truth method names from your dataset.
2. **Generate Names**: Use the LLM to generate method names based on the functional descriptions.
3. **Compare Names**: Calculate the edit distance between each generated method name and the ground truth method name.
4. **Provide Feedback**: Create feedback messages that specify the differences and suggest corrections.
# Template-based Prompt Generation
With the outputs from the preceding sections, we have collected all the information to create a context-rich prompt [Source Code](https://github.com/contextcraft/contextcraft/blob/main/Source_Code/ContextCraft.ipynb). The prompt is composed of five parts:
1. **Best Examples**: Retrieved from the corpus
2. **Probabilistic Token Positioning (PTP)**: Tokens identified for potential copying
3. **Pivot Words**: Useful words identified from the functional descriptions
4. **LLM-based Feedback**: Quantitative feedback on LLM-generated method names
5. **Query**: Composed of predefined text (an order to the LLM) and the functional description.
### Example Prompt
Here’s how an example prompt is structured:
![PWI Algorithm](PromptTemplate_LMS.PNG)
# Results 

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

