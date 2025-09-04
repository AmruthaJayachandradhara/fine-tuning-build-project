# End-to-End Fine-Tuning Project Report

## Synthetic Data Generation

### Approaches Tried
- **Initial GPT-based Generation**: Started with basic prompts to generate job title variations
- **Prompt Engineering**: Developed comprehensive prompt template with 5 distinct variation categories:
  1. Alternative Keywords (semantic similarity with different terms)
  2. Seniority & Experience Level (Junior, Senior, Lead, Principal)
  3. Technical Skills & Specializations (specific technologies, certifications)
  4. Job Context & Descriptors (location, urgency, team context)
  5. Formatting Variations (punctuation, structure diversity)

### What Worked
- **Structured Prompt Template**: Created detailed instructions in `synthetic_data/data/initial_prompt.txt` that explicitly guides the model to generate semantically similar but linguistically diverse job titles
- **Category-based Variation**: Breaking down variations into 5 specific types ensured comprehensive coverage of real-world job posting patterns
- **2 variations per title**: Balanced approach between dataset size and generation quality

### What Didn't Work Initially
- **Generic prompts**: Early attempts with simple "generate variations" prompts produced low-quality, repetitive data
- **Insufficient diversity**: Initial synthetic data lacked the linguistic variety found in real job postings

### Final Approach
- Used comprehensive prompt template with examples and specific instructions
- Generated `jittered_titles.csv` with semantically related but lexically diverse job title pairs
- Focused on creating training data that teaches the model to understand semantic similarity across different phrasings

### Future Improvements
- **Larger scale generation**: Could expand to more variations per title
- **Domain-specific prompts**: Create specialized prompts for different industries
- **Quality filtering**: Implement automated quality assessment for generated pairs

---

## Model Training

### Base Model Selection
- **Chosen Model**: `sentence-transformers/all-mpnet-base-v2`
  - Strong baseline performance on semantic similarity tasks
  - Good balance of model size and capability
  - Well-suited for sentence embedding tasks

### Training Architecture & Techniques

#### LoRA (Low-Rank Adaptation) Implementation
- **Why LoRA**: Enables parameter-efficient fine-tuning, reducing memory requirements and training time
- **LoRA Configuration**:
  - **Target Modules**: `["query", "key", "value", "dense"]` - focused on attention mechanisms
  - **Bias Setting**: `"none"` - no bias adaptation for efficiency
  - **Task Type**: `"FEATURE_EXTRACTION"` - optimized for embedding generation

#### Hyperparameter Optimization with Optuna
- **Automated Tuning**: Implemented comprehensive hyperparameter search using Optuna
- **Optimized Parameters**:
  1. **LoRA Rank (r)**: Searched [8, 16, 32, 64] → Optimal: **16**
  2. **LoRA Alpha**: Searched [8, 16, 32] → Optimal: **16** 
  3. **LoRA Dropout**: Searched [0.1, 0.3] → Optimal: **0.226**
  4. **Learning Rate**: Searched [1e-6, 1e-4] → Optimal: **1.09e-05**
  5. **Batch Size**: Searched [16, 32, 64] → Optimal: **32**
  6. **Triplet Margin**: Searched [0.1, 0.5] → Optimal: **0.123**
  7. **Warmup Steps**: Searched [100, 500] → Optimal: **359**

#### Training Results
- **Final Training Loss**: **0.159**
- **Best Validation Loss**: **0.1864**
- **Training Steps**: **3,525**
- **Parameter Efficiency**: Only **1.35%** of parameters trainable (1,499,136 out of 110,985,600)
- **Training Duration**: Optimized for efficiency with LoRA

### Loss Function & Training Strategy
- **Triplet Loss**: Used for learning embeddings where similar job titles are closer than dissimilar ones
- **Multiple Negative Ranking**: Enhanced training with multiple negative samples per positive pair
- **Warmup Strategy**: Gradual learning rate increase for stable training

### What Worked
- **LoRA Fine-tuning**: Dramatically reduced training time and memory usage while maintaining performance
- **Optuna Optimization**: Automated hyperparameter search found optimal configuration
- **Triplet Loss**: Effective for learning semantic similarity in job title embeddings

### What Didn't Work Initially
- **Full model fine-tuning**: Too resource-intensive and slower convergence
- **Manual hyperparameter tuning**: Time-consuming and suboptimal results
- **Basic training arguments**: Initial default settings led to poor convergence

### Future Improvements
- **Larger datasets**: Scale up synthetic data generation for more robust training
- **Multi-task learning**: Combine job title similarity with other related tasks
- **Advanced loss functions**: Experiment with contrastive learning or SimCLR approaches

---

## Evaluation & Results

### Model Performance Metrics

#### Quantitative Results
- **Training Loss Reduction**: From initial high loss to **0.159** final training loss
- **Validation Performance**: Achieved **0.1864** validation loss
- **Parameter Efficiency**: **98.65%** parameter reduction compared to full fine-tuning
- **Convergence**: Stable training with optimal hyperparameters found via Optuna

#### Hyperparameter Importance Analysis
Based on Optuna optimization:
1. **Learning Rate**: 0.33 importance - most critical for convergence
2. **LoRA Alpha**: 0.30 importance - crucial for adaptation strength
3. **Batch Size**: 0.20 importance - affects training stability
4. **LoRA Rank**: 0.10 importance - balances capacity vs efficiency
5. **Other parameters**: Lower but meaningful contributions

### Streamlit Application Evaluation

#### Deployed Application Features
**Link**: http://localhost:8502

#### Core Functionality
- **Job Search Engine**: Compare default vs fine-tuned model results side-by-side
- **Similarity Scoring**: Configurable similarity thresholds and result limits
- **Real-time Inference**: Interactive search with immediate results

#### Advanced Features Added
1. **📊 Model Analytics Dashboard**
   - Performance metrics comparison (mean similarity, standard deviation)
   - Interactive similarity distribution histograms using Plotly
   - Real-time statistics showing model behavior differences

2. **🎯 Advanced Search Options**
   - Multi-query batch processing with CSV export
   - Query comparison (side-by-side analysis)
   - Search history tracking and replay functionality
   - Configurable search parameters

3. **📈 Interactive Visualizations**
   - PCA-based 2D embedding space visualization
   - Interactive scatter plots showing embedding clusters
   - Explained variance analysis for dimensionality reduction
   - Model comparison visualizations

4. **⚙️ Settings & Configuration**
   - Performance tuning controls
   - Data export capabilities (analytics reports, search results)
   - Session management and history clearing
   - Customizable display options

5. **🔍 Enhanced Job Search**
   - Color-coded similarity scores (high/medium/low)
   - "Find similar jobs" functionality for result exploration
   - Result overlap analysis between models
   - Threshold-based filtering

### Comparative Analysis
The Streamlit app demonstrates clear improvements from fine-tuning:
- **Better Semantic Understanding**: Fine-tuned model shows improved clustering of semantically similar job titles
- **Reduced Noise**: More consistent similarity scores for related positions
- **Enhanced Relevance**: Top results more aligned with user intent

---

## Summary

This project successfully implemented an end-to-end fine-tuning pipeline for job title similarity using LoRA-enhanced sentence transformers. **Key achievements include**:

- **Efficient Training**: LoRA reduced trainable parameters by 98.65% while maintaining performance
- **Automated Optimization**: Optuna-based hyperparameter tuning achieved optimal configuration automatically
- **Production-Ready Application**: Comprehensive Streamlit app with advanced analytics and visualization features
- **Semantic Improvement**: Fine-tuned model demonstrates better understanding of job title relationships compared to baseline

**Key Learnings**: Parameter-efficient fine-tuning with LoRA provides an excellent balance between performance and resource requirements. Structured prompt engineering for synthetic data generation significantly impacts model quality. Automated hyperparameter optimization is crucial for achieving optimal results without manual trial-and-error.

**Next Steps**: Scale synthetic data generation, experiment with multi-task learning approaches, and explore advanced contrastive learning methods. The current framework provides a solid foundation for expanding to larger datasets and more complex semantic understanding tasks.

---

## Technical Specifications

### Environment & Dependencies
- **Base Model**: `sentence-transformers/all-mpnet-base-v2`
- **Fine-tuning Framework**: HuggingFace Transformers + PEFT (LoRA)
- **Optimization**: Optuna for automated hyperparameter search
- **Visualization**: Streamlit + Plotly + scikit-learn
- **Hardware**: Optimized for CPU/GPU with parameter-efficient training

### Repository Structure
```
fine-tuning-build-project/
├── synthetic_data/          # Data generation pipeline
├── fine_tuning/            # Training notebooks and models
├── streamlit_app/          # Production application
└── requirements.txt        # Dependencies
```

### Model Artifacts
- **Training Checkpoints**: `fine_tuning/data/trained_models/`
- **LoRA Adapters**: Saved separately for efficient deployment
- **Embeddings**: Pre-computed for fast inference in Streamlit app
