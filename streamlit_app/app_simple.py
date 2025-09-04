import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from datetime import datetime

# Set up page configuration
st.set_page_config(page_title="Job Posting Search Engine", layout="centered")

st.title('🔍 Job Posting Search Engine')

# Simple device detection
@st.cache_resource
def get_device():
    return torch.device("cpu")  # Force CPU for deployment stability

device = get_device()

# Load sample data - no external files needed
@st.cache_resource
def load_sample_data():
    sample_jobs = [
        "Software Engineer @ Google",
        "Data Scientist @ Apple", 
        "Product Manager @ Microsoft",
        "ML Engineer @ Amazon",
        "Frontend Developer @ Meta",
        "Backend Developer @ Netflix",
        "DevOps Engineer @ Tesla",
        "AI Researcher @ OpenAI",
        "UX Designer @ Adobe",
        "Marketing Analyst @ Spotify",
        "Full Stack Developer @ Airbnb",
        "Cloud Architect @ AWS",
        "Security Engineer @ Cloudflare",
        "Mobile Developer @ Uber",
        "Database Administrator @ Oracle",
        "Business Analyst @ Salesforce",
        "QA Engineer @ Microsoft",
        "Product Designer @ Figma",
        "Technical Writer @ GitHub",
        "Sales Engineer @ Stripe"
    ] * 50  # Create 1000 sample jobs
    return sample_jobs

# Load default model only
@st.cache_resource
def load_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    return model

# Generate embeddings
@st.cache_resource
def generate_embeddings():
    jobs = load_sample_data()
    model = load_model()
    st.info("🔄 Generating embeddings... (this happens once)")
    embeddings = model.encode(jobs, normalize_embeddings=True, show_progress_bar=True)
    return embeddings

# Load resources
job_postings = load_sample_data()
model = load_model()
embeddings = torch.tensor(generate_embeddings(), device=device)

# Sidebar
st.sidebar.title("🎛️ Settings")
max_results = st.sidebar.slider("Max results:", 5, 20, 10)
min_similarity = st.sidebar.slider("Similarity threshold:", 0.0, 1.0, 0.3, 0.05)

# Main search interface
st.subheader("🔍 Search for Similar Jobs")
user_query = st.text_input("Enter a job title to search:")

if user_query:
    with torch.inference_mode():
        # Encode user query
        query_embedding = model.encode([user_query], normalize_embeddings=True, convert_to_tensor=True)[0]
        
        # Compute similarities
        similarities = torch.inner(query_embedding, embeddings)
        
        # Get top results
        top_indices = torch.argsort(similarities, descending=True)[:max_results]
        
        # Filter by threshold
        top_indices = top_indices[similarities[top_indices] >= min_similarity]
        
        if len(top_indices) > 0:
            st.markdown("### 🎯 Search Results")
            for i, idx in enumerate(top_indices):
                similarity = similarities[idx].item()
                job_title = job_postings[idx.item()]
                
                # Color-coded results
                if similarity >= 0.8:
                    st.success(f"**{i+1}.** {job_title}")
                elif similarity >= 0.6:
                    st.info(f"**{i+1}.** {job_title}")
                else:
                    st.write(f"**{i+1}.** {job_title}")
                
                st.caption(f"Similarity: {similarity:.3f}")
        else:
            st.warning(f"No results found above similarity threshold {min_similarity:.2f}")

# Simple analytics
if st.sidebar.button("📊 Show Analytics"):
    st.subheader("📊 Dataset Analytics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Jobs", len(job_postings))
    with col2:
        st.metric("Embedding Dim", embeddings.shape[1])
    with col3:
        st.metric("Device", str(device))
    
    # Simple visualization
    st.subheader("📈 Embedding Visualization")
    with st.spinner("Computing PCA..."):
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings.cpu().numpy()[:500])  # Sample for performance
    
    df_plot = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'job': job_postings[:500]
    })
    
    fig = px.scatter(df_plot, x='x', y='y', hover_data=['job'], 
                     title="PCA Visualization of Job Embeddings")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    🚀 Powered by Sentence Transformers | 📊 Built with Streamlit
</div>
""", unsafe_allow_html=True)
