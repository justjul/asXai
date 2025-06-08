import os
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import pandas as pd


img_asxai_logo = 'img/asXai_logo.png'
img_arXiv_monthly = 'img/arxiv_submissions_monthly.png'
img_scispace_logo = 'img/sciSpace_logo.png'
img_scite_logo = 'img/scite_logo.png'
img_paperdoc_logo = 'img/paperDoc_logo.jpg'
img_asxai_login = 'img/Login.png'

img_pipeline = 'img/Pipeline_overview.svg'
img_DB_manage = 'img/DB_management.svg'
img_reranker_train_flow = 'img/Reranker_training.svg'

metadata_sample = "data/metadata_sample_2025.csv"
textdata_sample = "data/textdata_sample_2025.csv"

img_chat_service = 'img/chat_service.svg'
img_reranker_model = 'img/Reranker_model.svg'
img_chunk_embedding = 'img/Chunk_embedding.svg'
img_triplet_loss = 'img/Triplet_loss.svg'

img_mlops_stack = 'img/CICD.png'


@st.cache_data
def load_metadata():
    return pd.read_csv(metadata_sample, index_col=0)


@st.cache_data
def load_textdata():
    return pd.read_csv(textdata_sample, index_col=0)


def get_image_base64(img_path):
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


img_login_base64 = get_image_base64(img_asxai_login)

# css custom pour les typo / les bocs etc
custom_css = """
<style>
    /* Styles pour spécifier la taille du texte */
    body {
        font-size: 16px; /* Taille de la police pour tout le texte */
        font-family: 'Roboto', sans-serif; /* Utiliser Roboto pour tout le texte */
        background-color: #eee;
    }
    h1 {
        font-size: 40px; /* Taille de la police pour les titres de niveau 1 */
        font-family: 'Roboto', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
    h2 {
        font-size: 28px; /* Taille de la police pour les titres de niveau 2 */
        font-family: 'Roboto Light', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
    p {
        font-size: 16px; /* Taille de la police pour les paragraphes */
        font-family: 'Roboto Light', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
    
    /* Styles pour les images */
    img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    }

    .img-scale-small {
        width: 200px; /* Définir la largeur de l'image */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    .img-scale-medium {
        width: 400px; /* Définir la largeur de l'image */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    .img-scale-large {
        width: 600px; /* Définir la largeur de l'image */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Styles pour les blocs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:20px;
    }

    .container {
    display: flex;
    justify-content: center;
    align-items: center;
    }
    
    .expander {
    display: flex;
    justify-content: center;
    align-items: center;
    }
    
    .expander-content {
        font-size: 10px; /* Taille de police pour le contenu de l'expander */
    }

    .stTabs [data-baseweb="tab-list"] {
		gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
		height: 70px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 10px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #bf0203;
        color: white;
	}

    .stTabs-content {
        font-size: 10px;
    }
    
    .streamlit-tabs {
        font-size: 20px;
    }
    
    div.st-emotion-cache-16txtl3 {
        padding: 2rem 2rem;
    }

    .block-container {
        padding-top: 1rem;
    }
    
    
   
</style>
"""
st.set_page_config(layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

# DEBUT CODE STREAMLIT************************************************************************************************************

# SOMMAIRE
st.sidebar.image(img_asxai_logo, use_container_width=True)
pages = ["Project overview", "Pipeline overview", "Dataset overview", 'Core components',
         "Performances", "Conclusions"]
page = st.sidebar.radio("### Layout", pages)
st.sidebar.subheader("Authors")
st.sidebar.markdown("""
                    - [Julien Fournier](link)
                    - [Honorat Nembe](link)
                    """)


# page 0############################################################################################################################################
if page == pages[0]:
    st.title("A new RAG for science and for all")
    tab1, tab2, tab3 = st.tabs(["Context", "Aims", "Results"])

    with tab1:
        st.header("Retrival augmented generation for Science")

        col1, col2, col3 = st.columns([2, 0.2, 2])
        with col1:

            st.markdown(
                """
        Retrieval-Augmented Generation (RAG) pipelines enable large language models to ground their responses in actual documents, 
    reducing hallucinations and improving factual accuracy.

    In the scientific domain, this is particularly critical—researchers must stay up to date with the latest discoveries 
    while relying on accurate, verifiable outputs.
    """)

            st.markdown("""
    However, most existing RAG-based tools for scientific search and summarization are **commercial**:
        - Behind paywalls
        - Rate-limited
        - Often not transparent or open-source
            """)

            st.markdown(
                """
            **The volume of new scientific literature is massive and growing:**

            - 📈 **arXiv**: ~24,000 new articles per month  
            - 🧬 **bioRxiv**: ~6,000 new articles per month  
            - 🏛️ **Semantic Scholar**: >1 million articles indexed yearly
    """)

            st.markdown("Some popular commercial tools include:")

            col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
            with col1_1:
                st.image(img_scispace_logo, width=100, caption="SciSpace")
            with col1_2:
                st.image(img_scite_logo, width=100, caption="Scite")
            with col1_3:
                st.image(img_paperdoc_logo, width=100, caption="PaperDoc")

        with col3:
            st.subheader("📊 arXiv Monthly Submissions")
            st.markdown(
                """
                <iframe src="https://tableau.cornell.edu/t/PublicContent/views/arXivSubmissions/LineGraphByArchive?:embed=y&:toolbar=bottom"
                        width="780"
                        height="500"
                        frameborder="0">
                </iframe>
                """,
                unsafe_allow_html=True
            )

    with tab2:
        st.markdown("""
        #### Democratize access to scientific knowledge with a Retrieval-Augmented Generation (RAG) pipeline that is:   
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🧩 Open Source")
            st.markdown("""
            - Fully transparent codebase and architecture  
            - Built with modular components (FastAPI, Qdrant, Ollama, Streamlit)  
            - Enables contributions, auditing, and reproducibility  
            - Encourages adoption in research and education
            """)

        with col2:
            st.subheader("💸 Free Access")
            st.markdown("""
            - No paywalls or usage limits  
            - Hosted or local deployment possible  
            - Accessible to individual researchers, educators, and labs  
            - Low-resource models supported (e.g., Ollama)
            """)

        with col3:
            st.subheader("🔬 Built for Scientists")
            st.markdown("""
            - Inline **citations** and reference tracking  
            - Capable of **technical Q&A** with proper grounding  
            - Interact via **chat interface** enriched by retrieved articles  
            - Future: search by metadata, field filtering, PDF preview
            """)

        st.markdown("---")
        st.success(
            "asXai bridges the gap between cutting-edge AI and accessible, trustworthy scientific discovery.")

    with tab3:
        st.header("📎 Demo of the deployed **asXai RAG pipeline**")

        st.markdown("""
        Submit queries, and receive responses grounded in actual research articles.
        """)

        st.markdown(f"""
            <a href="https://goose-beloved-kit.ngrok-free.app/" target="_blank">
                <img src="data:image/png;base64,{img_login_base64}" width="1400" alt="Open asXai Demo">
            </a>
            """, unsafe_allow_html=True)

        st.info(
            "Note: It may take a few seconds to load the first time depending on server activity.")

# page 1############################################################################################################################################
if page == pages[1]:
    st.title("Pipeline overview")
    tab1, tab2 = st.tabs(
        ["Online Service Architecture", "Offline Service Architecture"])

    with tab1:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.header("")
            st.markdown("""


            """)
            st.markdown("""
            - The pipeline consists of **modular services** communicating via **Kafka** and using a shared local file system.
            - Authentication is managed via **Firebase**, while chat and search flows are powered by **FastAPI** workers.
            - Prometheus, Grafana, and MLflow handle monitoring and observability.
            - Ngrok + Nginx provide secure tunneling and routing for the frontend.
                        
            """)
        with col2:
            st.image(img_pipeline, caption="System-Level Service Architecture",
                     use_container_width=True)

    with tab2:
        col1, col2, col3, col4 = st.columns([0.2, 1, 1, 0.2])

        with col2:
            st.header("Database management")
            st.markdown("""
            - Scientific articles are sourced from **Semantic Scholar** and **arXiv**.
            - PDFs are downloaded via **Selenium**, parsed with **pdfminer**, chunked, and embedded with `e5-large-instruct`.
            - Embeddings are saved in **Qdrant** collection for efficient retrieval.
            """)

        with col3:
            st.header("Reranker model training")
            st.markdown("""
            - **Qdrant** provides training data.
            - A transformer reranker is trained using **contrastive loss** to align citation relevance.
            - Models are logged and tracked via **MLflow**.
            """)

        col1, col2, col3, col4 = st.columns([0.2, 1, 0.51, 0.69])
        with col2:
            st.image(img_DB_manage, caption="Database management",
                     use_container_width=True)
        with col3:
            st.image(img_reranker_train_flow, caption="Reranker training flow",
                     use_container_width=True)

# page 2############################################################################################################################################
if page == pages[2]:
    st.title("Dataset")

    st.markdown("""
    Data used to power the **asXai** RAG pipeline come from Semantics Scholar and arXiv (for now).
    """)

    # Load data
    metadata_df = load_metadata()
    textdata_df = load_textdata()

    st.subheader("🔖 Metadata Preview")
    st.markdown("""
    **metadata** : Titles, authors, publication year, source (e.g. arXiv, Semantic Scholar), citation count, and unique paper ID.
    This metadata is used for filtering, payload enrichment, and user-facing context.
    """)
    st.dataframe(metadata_df, use_container_width=True)

    st.subheader("📄 Text Data Preview")
    st.markdown("""
    **textdata**: holds text extracted from the article PDFs: Each row contains a chunk of ~170 tokens embedded from a window 
    of 512 tokens for context.
    These are used during retrieval and alignment with user queries.
    """)
    st.dataframe(textdata_df, use_container_width=True)


# page 3############################################################################################################################################
if page == pages[3]:
    st.title("Core Components")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Chunk Embedding", "Model Training", "Chat Service", "MLOps stack"])

    # Tab 1: Chunk Embedding
    with tab1:
        col1, col2 = st.columns([0.6, 1.2])

        with col1:
            st.subheader("Embedding Scientific Texts into Semantic Vectors")
            st.markdown("""
            - Articles are split into overlapping windows of **512 tokens**.
            - A central chunk of **~170 tokens** is extracted and used for embedding.
            - Embeddings are computed using a pre-trained model (`e5-large-instruct`) and stored in **Qdrant** with article metadata.
            - This strategy balances **granularity** and **context**, ensuring effective semantic search across long documents.
            """)

        with col2:
            st.image(img_chunk_embedding, use_container_width=True)

    # Tab 2: Triplet Loss Model Training
    with tab2:
        col1, col2, col3 = st.columns([0.8, 1, 0.3])
        with col1:
            st.subheader("Reranker Training with Triplet Loss")
            st.markdown("""
            - A transformer-based reranker is trained to score and rank retrieved chunks.
            - Training uses **triplet loss** with:
                - Anchor: user query embedding  
                - Positive: cited article chunk  
                - Negative: non-cited chunk  
            - Optimizes the embedding space to **prioritize relevance** and **citation alignment**.
            """)
        with col2:
            st.image(img_reranker_model, use_container_width=True,
                     caption="8-layer reranker transformer")

        col1, col2, col3 = st.columns([0.8, 0.7, 0.6])
        with col2:
            st.image(img_triplet_loss, use_container_width=True)

    # Tab 3: Chat Service
    with tab3:
        col1, col2 = st.columns([0.6, 1.2])

        with col1:
            st.subheader("Interactive Chat Grounded in Literature")
            st.markdown("""
            - Users interact via a real-time **chat interface**.
            - Incoming queries are parsed with an LLM (e.g., `gemma3:12b` via **Ollama**).
            - Retrieved chunks are added to the context and passed to the LLM for generation.
            - The response includes **article citations** and feeds a **Citation Score**, used to retrain the reranker.

            🔁 All responses are streamed back using **Server-Sent Events (SSE)** for a smooth user experience.
            """)

        with col2:
            st.image(img_chat_service, use_container_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(""" <Initial instruction>: \n        
            If you can answer the user's question accurately based on your knowledge 
            and previous messages in the conversation: answer it direclty and 
            if you used previous messages, cite your source by including (id) 
            where id are the ARTICLE IDs from the database but DO NOT include 
            external references. \n
            If your answer needs to be cross-checked for accuracy:
            Step 1: Define at least three questions that covers different aspects 
            of the user's query.
            Step 2: Expand each of these questions by rephrasing them in a more 
            descriptive and complete sentence,
            suitable for retrieving relevant scientific documents. 
            Make complete sentences not just key words.
            Step 3: At the end, return the questions defined at step 2, as: \n
              'SEARCHING: question1' 'SEARCHING: question2' etc.

            """)

        with col2:
            st.markdown("""<Refinement instruction>:\n
            Provide an accurate answer to the user's question based on the 
            provided articles and on your knowledge.\n
            Structure your response in multiple paragraph describing one 
            element of the response.
            Cite your sources by including where appropriate (ARTICLE_ID) where 
            ARTICLE_ID are the hashed ids provided in the context\n
            for each article.\n

            """)

    with tab4:
        st.subheader("DevOps, MLOps & Monitoring")

        st.markdown("""
            This system follows modern **CI/CD** and **MLOps** practices for reliability, reproducibility, and scalability.
        """)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            st.markdown("""
            #### 🔄 Continuous Integration / Continuous Deployment (CI/CD)
            - GitHub handles code versioning and team collaboration.
            - Docker + Docker Compose define infrastructure as code for all services.
            - Local and remote environments mirror each other for reproducibility.
            """)
        with col2:
            st.markdown("""
            #### 🧪 MLOps with MLflow
            - ML models (e.g., reranker) are logged, versioned, and evaluated in **MLflow**.
            - Best-performing models are promoted to "production" and automatically used by workers.
            """)
        with col3:
            st.markdown("""
            #### 📊 Monitoring
            - **Prometheus** tracks API call volume, latency, and system resource usage.
            - **Grafana** dashboards provide real-time observability into system performance.
            - Alerts can be configured for system failures or overload conditions.
            """)
        with col4:
            st.markdown("""
            #### 🐳 Containerized Ecosystem
            - Every service (LLM, frontend, database, Kafka, reranker, etc.) is deployed in its own container.
            - Docker Compose ensures they communicate correctly and restart on failure.
            """)

        st.markdown("""
        > ✅ This setup enables fast iteration, safe deployment, and continuous learning from usage data.
        """)

        col1, col2, col3 = st.columns([0.3, 1, 0.3])
        with col2:
            st.image(img_mlops_stack, use_container_width=True)


# page 4############################################################################################################################################
if page == pages[4]:
    st.title("Performance Overview")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("""
        ### ⚙️ System Metrics (from Prometheus)

        - 🔄 Average search latency: **1.2s**
        - 📥 Chat inference latency: **~2.8s** per query
        - 🧠 Ollama inference time: **~1.5s**
        - ✅ Uptime stability: **100%** (Docker auto-restart on failure)
        """)
    with col2:
        st.markdown("""
        ### 📊 Model Performance (MLflow)

        - Latest Reranker: `v2`  
        - Accuracy: **81.2%**
        - Triplet loss: **0.0018**
        - Promoted to production: **2025-06-06**
        """)

    with col3:
        st.markdown("""
        ### ✅ Observed Improvements

        - With reranking, relevant citations appear in top 3 results **+37%** more often.
        - Latency with reranking pipeline remains < 3.5s end-to-end.
        """)

    st.header("")
    st.header("Future improvements")

    st.markdown("""
    The current pipeline is functional and modular, but there are several areas planned for improvement:

    #### 🧠 Model & Retrieval Enhancements
    - Fine-tune the reranker on more diverse scientific domains.
    - Experiment with hybrid retrieval (dense + sparse retrieval fusion).
    - Add full-text filtering by field of study, author, or publication venue.

    #### ⚙️ System Optimization
    - Reduce cold-start latency by caching embedding and search results.
    - Add batching capabilities for query processing to improve throughput.
    - Enable parallel reranker inference for multi-chunk ranking.

    #### 🧪 Evaluation & Feedback
    - Build internal evaluation set with human-labeled relevance judgments.
    - Add user feedback mechanism to improve reranking in-the-loop.

    #### 🌐 Deployment & Scalability
    - Migrate to Kubernetes for better orchestration and autoscaling.
    - Replace Ngrok with Cloudflare Tunnel or custom domain.
    - Add SSO and role-based access for enterprise use.

    #### 🧾 UI/UX Improvements
    - Add citation context viewer (see where in the paper a chunk was retrieved).
    - Improve PDF ingestion preview (show extracted sections before indexing).
    - Display citation scores and confidence alongside chat answers.

    > ✅ These improvements aim to make asXai faster, more accurate, and more valuable to scientific users.
    """)


# page 5############################################################################################################################################
if page == pages[5]:
    st.title("Conclusion")

    st.markdown("""
        Over the past weeks, **asXai** has grown into a modular and functional **Retrieval-Augmented Generation (RAG)** pipeline 
        built for scientific literature.
    """)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("""
        ### 🔍 What We've Built
        - A working chat assistant grounded in real, up-to-date scientific publications.
        - A fully containerized system using Docker, Kafka, Ollama, Qdrant, and FastAPI.
        - Custom training of a reranker using triplet loss, monitored through MLflow.
        - Prometheus + Grafana monitoring with SSE-based live response streaming.
        - A React-based frontend integrated with Firebase Auth.
        """)
    with col2:
        st.markdown("""
        ### 🧩 What Makes It Unique
        - **Open-source**, **free**, and built with scientists in mind.
        - Transparent and adaptable for research, education, or production use.
        - Citation-aware responses and technical fluency using lightweight LLMs.
        """)
    with col3:
        st.markdown("""
        ### 🔭 What’s Next
        - Continue improving model quality, speed, and user experience.
        - Enable user feedback loops for adaptive reranking.
        - Deploy a public-facing, persistent version for the research community.
        """)

    st.divider()

    # Final demo section
    st.subheader("🚀 Try the Demo")
    st.markdown("""
    Click below to open the live **asXai demo** in a new tab:

    <a href="https://goose-beloved-kit.ngrok-free.app/?ngrok-skip-browser-warning=true" target="_blank">
        <img src="img/asXai_logo.png" alt="asXai Demo" width="250"/>
    </a>
    """, unsafe_allow_html=True)

    st.caption(
        "⚠️ Note: The demo may take a few seconds to load. Make sure the backend services are running.")
