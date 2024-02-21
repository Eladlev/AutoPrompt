
# How AutoPrompt works

This document outlines the optimization process flows of AutoPrompt. The framework is designed with modularity and adaptability in mind, allowing for easy extension of the prompt calibration process from classification tasks to generative tasks. 


##   Classification Pipeline Overview 

The classification pipeline executes a calibration process involving the following steps:

1. **User Input:**
   - The user provides an initial prompt and task description to kickstart the calibration process.

2. **Challenging Examples:**
   - A set of challenging examples is proposed to the user to enhance the model's performance.

3. **Annotation:**
   - The provided examples are annotated, utilizing either a human-in-the-loop approach or leveraging Language Model (LLM) capabilities.

4. **Prediction:**
   - The annotated samples are evaluated using the current prompt to assess model performance.

5. **Prompt Analysis:**
   - The pipeline analyzes the prompt scores and identifies instances of large errors.

6. **Prompt Refinement:**
   - A new prompt is suggested based on the evaluation results, aiming to improve model accuracy.

7. **Iteration:**
   - Steps 2-6 are iteratively repeated until convergence, refining the prompt and enhancing the model's performance throughout the process. 


## Generation Pipeline Overview 

The generation pipeline shares a common structure with the classification flow but introduces a modification step for generation prompts. The process unfolds as follows:

1. **User Input:**
   - The user provides an initial prompt and task description for the generation process.

2. **Prompt Modification (LLM):**
   - The initial prompt is transformed into a classification-compatible input using a Language Model (LLM), creating an intermediary task for boolean classification or ranking.

3. **Annotation (Classification):**
   - Challenging examples are annotated for boolean classification or ranking based on the modified prompts. This step is analogous to the classification flow.

4. **Ranker Calibration (LLM):**
   - Utilizing the annotated examples, a ranking prompt (implemented as an LLM estimator) is fitted.

5. **Calibration (Generation):**
   - The original generation prompt is calibrated using the ranking LLM estimator (now used for evaluation), resulting in enhanced prompt formulations for generation tasks.
   


The modular architecture of the pipeline demonstrates the flexibility of the core calibration process and effectiveness for both classification and generation tasks. The additional step in the generation flow seamlessly integrates with the overall iterative prompt calibration approach.




