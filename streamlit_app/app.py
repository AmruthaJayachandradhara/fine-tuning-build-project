import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
from datetime import datetime

# Set up page configuration and CSS.
st.set_page_config(page_title="Job Posting Search Engine", layout="centered")
st.markdown(
    """
    <style>
    .block-container {
        padding: 2rem 2rem !important;
        max-width: 1200px;
    }
    .section-spacing {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .header-container > div {
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper: detect device.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

st.title('🔍 Job Posting Search Engine')
device = get_device()

# Sidebar Navigation
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🔍 Job Search", "📊 Model Analytics", "🎯 Advanced Search", "📈 Visualizations", "⚙️ Settings"]
)

# Model Selection in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.radio(
    "Select primary model:",
    ["Fine-tuned Model", "Default Model", "Compare Both"],
    index=0
)

# Search Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Search Settings")
max_results = st.sidebar.slider("Max results to show:", 5, 20, 10)
min_similarity = st.sidebar.slider("Minimum similarity threshold:", 0.0, 1.0, 0.3, 0.05)

# Initialize session state variables.
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "app_state" not in st.session_state:
    st.session_state.app_state = "search"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "fine_tuned"
# Instead of using "user_input" to preserve the search query, we use "saved_search".
if "saved_search" not in st.session_state:
    st.session_state.saved_search = ""
if "app_state" not in st.session_state:
    # "search": initial search input form,
    # "results": search results are available,
    # "similar_jobs": a job has been selected to view similar jobs.
    st.session_state.app_state = "search"

# ----- Functions for loading resources -----
@st.cache_resource
def load_fine_tuned_embeddings():
    embeddings = np.load(os.path.join('data', 'fine_tuned_embeddings.npy'))
    return embeddings

@st.cache_resource
def load_default_embeddings():
    embeddings = np.load(os.path.join('data', 'default_embeddings.npy'))
    return embeddings

@st.cache_resource
def load_job_postings():
    job_postings_df = pd.read_parquet(os.path.join('data', 'job_postings.parquet'))
    job_postings_df['posting'] = job_postings_df['job_posting_title'] + ' @ ' + job_postings_df['company']
    return job_postings_df['posting'].to_list()

@st.cache_resource
def load_fine_tuned_model():
    fine_tuned_model_path = os.path.join('data', 'fine_tuned_model')
    model = SentenceTransformer(fine_tuned_model_path, device=device)
    return model

@st.cache_resource
def load_default_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    return model

# ----- Load Resources -----
# For demonstration, limit to the first 5000 job postings.
fine_tuned_embeddings = torch.tensor(load_fine_tuned_embeddings()[:5000], device=device)
default_embeddings = torch.tensor(load_default_embeddings()[:5000], device=device)
job_postings = load_job_postings()[:5000]
fine_tuned_model = load_fine_tuned_model()
default_model = load_default_model()

# ----- New Feature Functions -----
def create_embedding_visualization():
    """Create 2D visualization of embeddings using PCA"""
    st.subheader("📊 Embedding Space Visualization")
    
    # Select embeddings to visualize
    embed_choice = st.selectbox("Choose embeddings to visualize:", 
                               ["Fine-tuned", "Default", "Both"])
    
    if embed_choice == "Fine-tuned":
        embeddings_np = fine_tuned_embeddings.cpu().numpy()
        titles = ["Fine-tuned"] * len(embeddings_np)
    elif embed_choice == "Default":
        embeddings_np = default_embeddings.cpu().numpy()
        titles = ["Default"] * len(embeddings_np)
    else:  # Both
        embeddings_np = np.vstack([
            fine_tuned_embeddings.cpu().numpy(),
            default_embeddings.cpu().numpy()
        ])
        titles = ["Fine-tuned"] * len(fine_tuned_embeddings) + ["Default"] * len(default_embeddings)
    
    # Use PCA for dimensionality reduction
    sample_size = min(1000, len(embeddings_np))  # Limit for performance
    indices = np.random.choice(len(embeddings_np), sample_size, replace=False)
    
    with st.spinner("Computing PCA..."):
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_np[indices])
    
    # Create interactive plot
    df_plot = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'model': [titles[i] for i in indices],
        'job': [job_postings[i % len(job_postings)] for i in indices]
    })
    
    fig = px.scatter(df_plot, x='x', y='y', color='model', 
                     hover_data=['job'],
                     title=f"PCA Visualization of Job Embeddings ({sample_size} samples)")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show explained variance
    st.info(f"📈 Explained variance: PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}")

def show_model_analytics():
    """Display model performance analytics"""
    st.subheader("📊 Model Performance Analytics")
    
    # Compute similarity statistics
    with torch.inference_mode():
        # Compute pairwise similarities for both models
        ft_similarities = torch.mm(fine_tuned_embeddings, fine_tuned_embeddings.T)
        default_similarities = torch.mm(default_embeddings, default_embeddings.T)
        
        # Remove diagonal (self-similarities)
        ft_similarities.fill_diagonal_(0)
        default_similarities.fill_diagonal_(0)
        
        # Get statistics
        ft_mean = ft_similarities.mean().item()
        ft_std = ft_similarities.std().item()
        default_mean = default_similarities.mean().item()
        default_std = default_similarities.std().item()
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Fine-tuned Mean Similarity", f"{ft_mean:.3f}")
    with col2:
        st.metric("📊 Fine-tuned Std Dev", f"{ft_std:.3f}")
    with col3:
        st.metric("🔄 Default Mean Similarity", f"{default_mean:.3f}")
    with col4:
        st.metric("📈 Default Std Dev", f"{default_std:.3f}")
    
    # Similarity distribution comparison
    st.subheader("📈 Similarity Score Distributions")
    
    # Sample similarities for histogram
    sample_size = 10000
    ft_sample = ft_similarities.flatten()[:sample_size].cpu().numpy()
    default_sample = default_similarities.flatten()[:sample_size].cpu().numpy()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ft_sample, name="Fine-tuned", opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=default_sample, name="Default", opacity=0.7, nbinsx=50))
    
    fig.update_layout(
        title="Distribution of Pairwise Similarity Scores",
        xaxis_title="Similarity Score",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def advanced_search_interface():
    """Advanced search interface with multiple queries and filters"""
    st.subheader("🎯 Advanced Search Options")
    
    # Multi-query search
    st.markdown("### 🔍 Multi-Query Search")
    search_mode = st.radio("Search mode:", ["Single Query", "Batch Queries", "Query Comparison"])
    
    if search_mode == "Single Query":
        query = st.text_input("Enter job title:", key="advanced_single")
        if query:
            perform_search([query], max_results, min_similarity)
            
    elif search_mode == "Batch Queries":
        queries_text = st.text_area("Enter multiple job titles (one per line):", 
                                   height=100, key="advanced_batch")
        if queries_text:
            queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
            if queries:
                perform_batch_search(queries, max_results, min_similarity)
                
    elif search_mode == "Query Comparison":
        col1, col2 = st.columns(2)
        with col1:
            query1 = st.text_input("Query 1:", key="comp_query1")
        with col2:
            query2 = st.text_input("Query 2:", key="comp_query2")
        
        if query1 and query2:
            compare_queries(query1, query2, max_results)
    
    # Search history
    st.markdown("### 📚 Search History")
    if st.session_state.search_history:
        for i, (timestamp, query) in enumerate(st.session_state.search_history[-10:]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{timestamp}: {query}")
            with col2:
                if st.button("🔄", key=f"rerun_{i}"):
                    perform_search([query], max_results, min_similarity)
    else:
        st.info("No search history yet.")

def perform_search(queries, max_results, min_similarity):
    """Perform search with given parameters"""
    for query in queries:
        # Add to search history
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.search_history.append((timestamp, query))
        
        st.markdown(f"### Results for: *{query}*")
        
        with torch.inference_mode():
            # Get embeddings for query
            default_query_embedding = default_model.encode([query], normalize_embeddings=True, convert_to_tensor=True)[0]
            finetuned_query_embedding = fine_tuned_model.encode([query], normalize_embeddings=True, convert_to_tensor=True)[0]
            
            # Compute similarities
            default_sim = torch.inner(default_query_embedding, default_embeddings)
            finetuned_sim = torch.inner(finetuned_query_embedding, fine_tuned_embeddings)
            
            # Filter by threshold and get top results
            default_filtered = default_sim[default_sim >= min_similarity]
            finetuned_filtered = finetuned_sim[finetuned_sim >= min_similarity]
            
            if len(default_filtered) == 0 and len(finetuned_filtered) == 0:
                st.warning(f"No results found above similarity threshold {min_similarity:.2f}")
                continue
            
            # Get top indices
            top_default = torch.argsort(default_sim, descending=True)[:max_results]
            top_finetuned = torch.argsort(finetuned_sim, descending=True)[:max_results]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔄 Default Model**")
                for i, idx in enumerate(top_default):
                    if default_sim[idx] >= min_similarity:
                        st.write(f"{i+1}. {job_postings[idx.item()]}")
                        st.write(f"   Score: {default_sim[idx]:.3f}")
            
            with col2:
                st.markdown("**🎯 Fine-tuned Model**")
                for i, idx in enumerate(top_finetuned):
                    if finetuned_sim[idx] >= min_similarity:
                        st.write(f"{i+1}. {job_postings[idx.item()]}")
                        st.write(f"   Score: {finetuned_sim[idx]:.3f}")
        
        st.markdown("---")

def perform_batch_search(queries, max_results, min_similarity):
    """Perform batch search"""
    st.info(f"🔍 Processing {len(queries)} queries...")
    
    results_data = []
    
    for query in queries:
        with torch.inference_mode():
            default_query_embedding = default_model.encode([query], normalize_embeddings=True, convert_to_tensor=True)[0]
            finetuned_query_embedding = fine_tuned_model.encode([query], normalize_embeddings=True, convert_to_tensor=True)[0]
            
            default_sim = torch.inner(default_query_embedding, default_embeddings)
            finetuned_sim = torch.inner(finetuned_query_embedding, fine_tuned_embeddings)
            
            # Get best matches
            best_default_idx = torch.argmax(default_sim)
            best_finetuned_idx = torch.argmax(finetuned_sim)
            
            results_data.append({
                'Query': query,
                'Default Best Match': job_postings[best_default_idx.item()],
                'Default Score': default_sim[best_default_idx].item(),
                'Fine-tuned Best Match': job_postings[best_finetuned_idx.item()],
                'Fine-tuned Score': finetuned_sim[best_finetuned_idx].item()
            })
    
    # Display results table
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Add download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"batch_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def compare_queries(query1, query2, max_results):
    """Compare two queries side by side"""
    st.markdown(f"### Comparing: *{query1}* vs *{query2}*")
    
    with torch.inference_mode():
        # Get embeddings for both queries
        q1_default = default_model.encode([query1], normalize_embeddings=True, convert_to_tensor=True)[0]
        q1_finetuned = fine_tuned_model.encode([query1], normalize_embeddings=True, convert_to_tensor=True)[0]
        q2_default = default_model.encode([query2], normalize_embeddings=True, convert_to_tensor=True)[0]
        q2_finetuned = fine_tuned_model.encode([query2], normalize_embeddings=True, convert_to_tensor=True)[0]
        
        # Compute query-to-query similarity
        default_q_sim = torch.inner(q1_default, q2_default).item()
        finetuned_q_sim = torch.inner(q1_finetuned, q2_finetuned).item()
        
        # Display query similarities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔄 Default Model Query Similarity", f"{default_q_sim:.3f}")
        with col2:
            st.metric("🎯 Fine-tuned Model Query Similarity", f"{finetuned_q_sim:.3f}")
        
        # Get top results for each query
        q1_default_sim = torch.inner(q1_default, default_embeddings)
        q1_finetuned_sim = torch.inner(q1_finetuned, fine_tuned_embeddings)
        q2_default_sim = torch.inner(q2_default, default_embeddings)
        q2_finetuned_sim = torch.inner(q2_finetuned, fine_tuned_embeddings)
        
        # Show top results comparison
        st.markdown("### Top Results Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Results for: {query1}**")
            st.markdown("*Default Model:*")
            top_indices = torch.argsort(q1_default_sim, descending=True)[:5]
            for i, idx in enumerate(top_indices):
                st.write(f"{i+1}. {job_postings[idx.item()]} ({q1_default_sim[idx]:.3f})")
            
            st.markdown("*Fine-tuned Model:*")
            top_indices = torch.argsort(q1_finetuned_sim, descending=True)[:5]
            for i, idx in enumerate(top_indices):
                st.write(f"{i+1}. {job_postings[idx.item()]} ({q1_finetuned_sim[idx]:.3f})")
        
        with col2:
            st.markdown(f"**Results for: {query2}**")
            st.markdown("*Default Model:*")
            top_indices = torch.argsort(q2_default_sim, descending=True)[:5]
            for i, idx in enumerate(top_indices):
                st.write(f"{i+1}. {job_postings[idx.item()]} ({q2_default_sim[idx]:.3f})")
            
            st.markdown("*Fine-tuned Model:*")
            top_indices = torch.argsort(q2_finetuned_sim, descending=True)[:5]
            for i, idx in enumerate(top_indices):
                st.write(f"{i+1}. {job_postings[idx.item()]} ({q2_finetuned_sim[idx]:.3f})")

# ----- Main App Logic Based on Page Selection -----

if page == "📊 Model Analytics":
    show_model_analytics()
    
elif page == "🎯 Advanced Search":
    advanced_search_interface()
    
elif page == "📈 Visualizations":
    create_embedding_visualization()
    
elif page == "⚙️ Settings":
    st.subheader("⚙️ Application Settings")
    
    st.markdown("### 🔧 Performance Settings")
    cache_size = st.slider("Cache size (number of embeddings):", 1000, 10000, 5000)
    
    st.markdown("### 📊 Display Settings")
    show_scores = st.checkbox("Show similarity scores", value=True)
    show_indices = st.checkbox("Show result indices", value=False)
    
    st.markdown("### 💾 Data Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Search History"):
            st.session_state.search_history = []
            st.success("Search history cleared!")
    
    with col2:
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session reset!")
            st.rerun()
    
    st.markdown("### 📥 Export Options")
    if st.button("📊 Export Analytics Report"):
        # Create a simple analytics report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_embeddings': len(fine_tuned_embeddings),
            'search_history_count': len(st.session_state.search_history),
            'device': str(device)
        }
        
        st.json(report_data)
        
        report_json = pd.DataFrame([report_data]).to_json(orient='records')
        st.download_button(
            "💾 Download Report",
            report_json,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

elif page == "🔍 Job Search":
    # Original job search functionality
    st.subheader("🔍 Job Title Search Engine")
    st.markdown("Compare results between the default and fine-tuned models:")
    
    # Search input
    user_query = st.text_input("🔍 Enter a job title to search for similar positions:")
    
    if user_query:
        # Add to search history
        timestamp = datetime.now().strftime("%H:%M")
        if (timestamp, user_query) not in st.session_state.search_history:
            st.session_state.search_history.append((timestamp, user_query))
        
        with torch.inference_mode():
            # Embed the user query with both models
            default_query_embedding = default_model.encode([user_query], normalize_embeddings=True, convert_to_tensor=True)[0]
            finetuned_query_embedding = fine_tuned_model.encode([user_query], normalize_embeddings=True, convert_to_tensor=True)[0]
            
            # Compute similarities
            default_similarities = torch.inner(default_query_embedding, default_embeddings)
            finetuned_similarities = torch.inner(finetuned_query_embedding, fine_tuned_embeddings)
            
            # Filter by minimum similarity threshold
            default_mask = default_similarities >= min_similarity
            finetuned_mask = finetuned_similarities >= min_similarity
            
            # Check if any results meet the threshold
            if not default_mask.any() and not finetuned_mask.any():
                st.warning(f"⚠️ No results found with similarity ≥ {min_similarity:.2f}. Try lowering the threshold in the sidebar.")
            else:
                # Get top results
                top_default_indices = torch.argsort(default_similarities, descending=True)[:max_results]
                top_finetuned_indices = torch.argsort(finetuned_similarities, descending=True)[:max_results]
                
                # Filter by threshold
                top_default_indices = top_default_indices[default_similarities[top_default_indices] >= min_similarity]
                top_finetuned_indices = top_finetuned_indices[finetuned_similarities[top_finetuned_indices] >= min_similarity]
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔄 Default Model Results")
                    if len(top_default_indices) > 0:
                        for i, idx in enumerate(top_default_indices):
                            similarity = default_similarities[idx].item()
                            job_title = job_postings[idx.item()]
                            
                            # Use different styling based on similarity score
                            if similarity >= 0.8:
                                st.success(f"**{i+1}.** {job_title}")
                            elif similarity >= 0.6:
                                st.info(f"**{i+1}.** {job_title}")
                            else:
                                st.write(f"**{i+1}.** {job_title}")
                            
                            st.caption(f"Similarity: {similarity:.3f}")
                            
                            if st.button(f"🔍 Find similar jobs", key=f"default_similar_{idx}"):
                                st.session_state.selected_job = job_title
                                st.session_state.app_state = "similar_jobs"
                                st.rerun()
                    else:
                        st.info("No results above similarity threshold")
                
                with col2:
                    st.markdown("### 🎯 Fine-tuned Model Results")
                    if len(top_finetuned_indices) > 0:
                        for i, idx in enumerate(top_finetuned_indices):
                            similarity = finetuned_similarities[idx].item()
                            job_title = job_postings[idx.item()]
                            
                            # Use different styling based on similarity score
                            if similarity >= 0.8:
                                st.success(f"**{i+1}.** {job_title}")
                            elif similarity >= 0.6:
                                st.info(f"**{i+1}.** {job_title}")
                            else:
                                st.write(f"**{i+1}.** {job_title}")
                            
                            st.caption(f"Similarity: {similarity:.3f}")
                            
                            if st.button(f"🔍 Find similar jobs", key=f"finetuned_similar_{idx}"):
                                st.session_state.selected_job = job_title
                                st.session_state.app_state = "similar_jobs"
                                st.rerun()
                    else:
                        st.info("No results above similarity threshold")
                
                # Show summary statistics
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔄 Default Results", len(top_default_indices))
                with col2:
                    st.metric("🎯 Fine-tuned Results", len(top_finetuned_indices))
                with col3:
                    overlap = len(set(top_default_indices.tolist()) & set(top_finetuned_indices.tolist()))
                    st.metric("🤝 Overlapping Results", overlap)

# Handle similar jobs view
if st.session_state.app_state == "similar_jobs" and st.session_state.selected_job is not None:
    st.markdown(f"### 🔍 Jobs Similar to: *{st.session_state.selected_job}*")
    
    if st.button("🔙 Back to Search"):
        st.session_state.app_state = "search"
        st.session_state.selected_job = None
        st.rerun()
    
    with torch.inference_mode():
        # Find the embedding for the selected job
        selected_job_idx = None
        for i, job in enumerate(job_postings):
            if job == st.session_state.selected_job:
                selected_job_idx = i
                break
        
        if selected_job_idx is not None:
            # Get embeddings for the selected job
            selected_default_embedding = default_embeddings[selected_job_idx]
            selected_finetuned_embedding = fine_tuned_embeddings[selected_job_idx]
            
            # Compute similarities to all other jobs
            default_similarities = torch.inner(selected_default_embedding, default_embeddings)
            finetuned_similarities = torch.inner(selected_finetuned_embedding, fine_tuned_embeddings)
            
            # Remove self-similarity
            default_similarities[selected_job_idx] = -1
            finetuned_similarities[selected_job_idx] = -1
            
            # Get top similar jobs
            top_default_indices = torch.argsort(default_similarities, descending=True)[:max_results]
            top_finetuned_indices = torch.argsort(finetuned_similarities, descending=True)[:max_results]
            
            # Filter by threshold
            top_default_indices = top_default_indices[default_similarities[top_default_indices] >= min_similarity]
            top_finetuned_indices = top_finetuned_indices[finetuned_similarities[top_finetuned_indices] >= min_similarity]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔄 Default Model - Similar Jobs")
                for i, idx in enumerate(top_default_indices):
                    similarity = default_similarities[idx].item()
                    job_title = job_postings[idx.item()]
                    st.write(f"**{i+1}.** {job_title}")
                    st.caption(f"Similarity: {similarity:.3f}")
            
            with col2:
                st.markdown("#### 🎯 Fine-tuned Model - Similar Jobs")
                for i, idx in enumerate(top_finetuned_indices):
                    similarity = finetuned_similarities[idx].item()
                    job_title = job_postings[idx.item()]
                    st.write(f"**{i+1}.** {job_title}")
                    st.caption(f"Similarity: {similarity:.3f}")
        else:
            st.error("Selected job not found in dataset.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    🚀 Powered by Fine-tuned Sentence Transformers | 
    📊 Built with Streamlit | 
    🎯 Enhanced with LoRA Fine-tuning
</div>
""", unsafe_allow_html=True)

# =============================================================================
# State Machine:
#
# app_state: "search" -> enter query, "results" -> display search results,
# "similar_jobs" -> display similar job postings.
#
# Transitions:
# - When user types a query and submits, set app_state = "results"
# - When user clicks a "Show most similar jobs" button,
#       set st.session_state.selected_job and app_state = "similar_jobs"
# - When user clicks "Back to search",
#       clear selected_job, set app_state = "results" (or "search" if you prefer)
# =============================================================================

if st.session_state.app_state == "similar_jobs" and st.session_state.selected_job is not None:
    # Similar-jobs view.
    selected_index = st.session_state.selected_job
    st.header("Similar Jobs for:")
    st.write(f"**{job_postings[selected_index]}**")
    st.markdown("<hr>", unsafe_allow_html=True)
    # Compute similar jobs for both models.
    with torch.inference_mode():
        # Default model similar jobs.
        default_embedding = default_embeddings[selected_index]
        default_sim = torch.inner(default_embedding, default_embeddings)
        default_sim[selected_index] = -1  # Exclude the job itself.
        default_top_indices = torch.argsort(default_sim, descending=True)[:5]
        # Fine-tuned model similar jobs.
        finetuned_embedding = fine_tuned_embeddings[selected_index]
        finetuned_sim = torch.inner(finetuned_embedding, fine_tuned_embeddings)
        finetuned_sim[selected_index] = -1
        finetuned_top_indices = torch.argsort(finetuned_sim, descending=True)[:5]
    st.markdown(
        """
        <div class="section-spacing">
            <h3 style="margin-bottom:1rem;">Similar Jobs (Default vs. Fine-Tuned)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Display headers.
    col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
    with col_rank:
        st.write("")  # placeholder for rank header.
    with col_default:
        st.markdown("<div class='header-container'><div>Default Model</div></div>", unsafe_allow_html=True)
    with col_finetuned:
        st.markdown("<div class='header-container'><div>Fine-Tuned Model</div></div>", unsafe_allow_html=True)
    # Show similar jobs result rows.
    for i in range(5):
        col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
        with col_rank:
            st.markdown(f"<h4>{i+1}.</h4>", unsafe_allow_html=True)
        with col_default:
            idx = default_top_indices[i].item()
            st.write(f"**{job_postings[idx]}**")
            st.write(f"Score: {default_sim[idx]:.4f}")
        with col_finetuned:
            idx = finetuned_top_indices[i].item()
            st.write(f"**{job_postings[idx]}**")
            st.write(f"Score: {finetuned_sim[idx]:.4f}")
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Back to search", key="clear_selection"):
        st.session_state.selected_job = None
        # Transition back to "results" without wiping the search query.
        st.session_state.app_state = "results"
        st.rerun()
else:
    # Either in the initial search mode or in search results mode.
    # Use the saved_search value as the default in the text input.
    user_input = st.text_input(
        "Enter a job title:",
        value=st.session_state.get("saved_search", ""),
        key="user_input"
    )
    if user_input:
        # Save the query separately so that it persists even if the widget is re-rendered.
        st.session_state.saved_search = user_input
        st.session_state.app_state = "results"
        with torch.inference_mode():
            default_query_embedding = default_model.encode(
                [user_input],
                normalize_embeddings=True,
                convert_to_tensor=True,
            )[0]
            finetuned_query_embedding = fine_tuned_model.encode(
                [user_input],
                normalize_embeddings=True,
                convert_to_tensor=True,
            )[0]
            default_sim = torch.inner(default_query_embedding, default_embeddings)
            finetuned_sim = torch.inner(finetuned_query_embedding, fine_tuned_embeddings)
            top10_default = torch.argsort(default_sim, descending=True)[:10]
            top10_finetuned = torch.argsort(finetuned_sim, descending=True)[:10]
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-spacing">
                <h3 style="margin-bottom:1rem;">Top Matches (Default vs. Fine-Tuned)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Column headers above search results.
        col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
        with col_rank:
            st.write("")  # empty header for rank.
        with col_default:
            st.markdown("<div class='header-container'><div>Default Model</div></div>", unsafe_allow_html=True)
        with col_finetuned:
            st.markdown("<div class='header-container'><div>Fine-Tuned Model</div></div>", unsafe_allow_html=True)
        # Build search results rows.
        for i in range(len(top10_default)):
            col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
            with col_rank:
                st.markdown(f"<h4>{i+1}.</h4>", unsafe_allow_html=True)
            with col_default:
                job_index = top10_default[i].item()
                st.write(f"**{job_postings[job_index]}**")
                st.write(f"Score: {default_sim[job_index]:.4f}")
                if st.button("Show most similar jobs", key=f"default_{job_index}"):
                    st.session_state.selected_job = job_index
                    st.session_state.app_state = "similar_jobs"
                    st.rerun()
            with col_finetuned:
                job_index = top10_finetuned[i].item()
                st.write(f"**{job_postings[job_index]}**")
                st.write(f"Score: {finetuned_sim[job_index]:.4f}")
                if st.button("Show most similar jobs", key=f"finetuned_{job_index}"):
                    st.session_state.selected_job = job_index
                    st.session_state.app_state = "similar_jobs"
                    st.rerun()
    else:
        st.info("Please enter a job title to start searching.")