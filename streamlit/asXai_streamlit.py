import os
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
img_asxai_logo = BASE_DIR / "img" / "asXai_logo.png"
img_arXiv_monthly = BASE_DIR / 'img'/'arxiv_submissions_monthly.png'
img_scispace_logo = BASE_DIR / 'img'/'sciSpace_logo.png'
img_scite_logo = BASE_DIR / 'img'/'scite_logo.png'
img_paperdoc_logo = BASE_DIR / 'img' / 'paperDoc_logo.jpg'
img_asxai_login = BASE_DIR / 'img' / 'Login.png'

img_pipeline = BASE_DIR / 'img'/'Pipeline_overview.svg'
img_DB_manage = BASE_DIR / 'img'/'DB_management.svg'
img_reranker_train_flow = BASE_DIR / 'img/Reranker_training.svg'

metadata_sample = BASE_DIR / "data"/"metadata_sample_2025.csv"
textdata_sample = BASE_DIR / "data"/"textdata_sample_2025.csv"

img_chat_service = BASE_DIR / 'img'/'chat_service.svg'
img_reranker_model = BASE_DIR / 'img'/'Reranker_model.svg'
img_chunk_embedding = BASE_DIR / 'img'/'Chunk_embedding.svg'
img_triplet_loss = BASE_DIR / 'img'/'Triplet_loss.svg'

img_mlops_stack = BASE_DIR / 'img'/'CICD.png'


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
img_logo_base64 = get_image_base64(img_asxai_logo)


def bilingual(en, fr):
    return en if language == 'En' else fr


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
# st.sidebar.image(str(img_asxai_logo), use_container_width=True)
language = st.sidebar.radio(
    "Language", ["En", "Fr"], horizontal=True, label_visibility="collapsed")

st.sidebar.markdown(f"""
    <a href = "https://goose-beloved-kit.ngrok-free.app/?ngrok-skip-browser-warning=true" target = "_blank" >
        <img src="data:image/png;base64,{img_logo_base64}" width="400" alt="Open asXai Demo">
    </a >
    """, unsafe_allow_html=True)

pages = ["Project overview", "Pipeline overview", "Dataset overview", 'Core components',
         "Performances", "Perspectives", "Conclusions"]
page = st.sidebar.radio("### Layout", pages)
st.sidebar.subheader("Authors")

JF_linkedin_url = "https://www.linkedin.com/in/julien-fournier-63530537"
HN_linkedin_url = "https://www.linkedin.com/in/honorat-nembe-502451242/"
st.sidebar.markdown(f"""
                    - [Julien Fournier]({JF_linkedin_url})
                    - [Honorat Nembe]({HN_linkedin_url})
                    """)


# page 0############################################################################################################################################
if page == pages[0]:
    st.title("A new RAG for science and for all")
    tab1, tab2, tab3 = st.tabs(["Context", "Aims", "Results"])

    with tab1:
        col1, col2, col3 = st.columns([2, 0.2, 2])
        with col1:
            st.header("Retrieval augmented generation on scientific litterature")

            st.markdown(bilingual("""
        Retrieval-Augmented Generation (RAG) pipelines enable large language models to ground their responses in actual documents,
    reducing hallucinations and improving factual accuracy.

    In the scientific domain, this is particularly critical—researchers must stay up to date with the latest discoveries
    while relying on accurate, verifiable outputs.
    """, """
        Les pipelines de type **RAG** permettent aux LLMs d’appuyer leurs réponses sur des documents réels, 
    réduisant ainsi les hallucinations et améliorant la précision des informations.

    Dans le domaine scientifique, c’est particulièrement crucial : les chercheurs doivent rester à jour sur les dernières découvertes  
    tout en s’appuyant sur des réponses fiables et vérifiables.
    """))

            st.markdown(bilingual("""
    However, most existing RAG-based tools for scientific search and summarization are **commercial**:
        - Behind paywalls
        - Rate-limited
        - Often not transparent or open-source
    """, """
        Cependant, la plupart des outils RAG existants pour la synthèse scientifique sont **commerciaux** :
        - Derrière des paywalls  
        - Limités en nombre de requêtes  
        - Souvent ni transparents ni open source
    """))

            st.markdown(bilingual("""
        **The volume of new scientific literature is massive and growing:**

        - 📈 **arXiv**: ~24,000 new articles per month
        - 🧬 **bioRxiv**: ~6,000 new articles per month
        - 🏛️ **Semantic Scholar**: >1 million articles indexed yearly
    """, """
        **Le volume de nouvelles publications scientifiques est considerable et en constante augmentation :**

        - 📈 **arXiv** : environ 24 000 nouveaux articles par mois  
        - 🧬 **bioRxiv** : environ 6 000 nouveaux articles par mois  
        - 🏛️ **Semantic Scholar** : plus d'un million d'articles indexés chaque année
    """))

            st.markdown(bilingual(
                "Some popular commercial tools include:",
                "Exemple de solutions existantes:"
            ))

            col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
            with col1_1:
                st.image(str(img_scispace_logo), width=100, caption="SciSpace")
            with col1_2:
                st.image(str(img_scite_logo), width=100, caption="Scite")
            with col1_3:
                st.image(str(img_paperdoc_logo), width=100, caption="PaperDoc")

        with col3:
            st.header("")
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
            - Transparent codebase and architecture for backend and frontend
            - Built with popular modules (FastAPI, Qdrant, Ollama, React)
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
            - Future: suggests new unexplored research questions
            """)

        st.markdown("---")

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
        col1, col2, col3, col4 = st.columns([0.3, 1.5, 1.5, 0.3])

        with col2:
            st.header("Backend")
            st.markdown(bilingual("""
            - Three main backend services: **chat, search, retrieval**
            - Communications via **Kafka** and/or using a **shared local file system**.
            - LLMs inferences are served by **Ollama**
            - Semantic retrieval is performed by **Qdrant**
            - A **reranking model** trained on citations and logged in **MLflow**
            """, """
            - Trois services principaux : **chat, search, retrieval**
            - Communication via **Kafka** ou par système de fichiers local partagé.
            - Inférences LLM servies par **Ollama**
            - Recherche sémantique réalisée par **Qdrant**
            - Un **modèle de reranking** est entraîné sur les citations et loggé dans **MLflow**
            """))

        with col3:
            st.header("Frontend and Monitoring")
            st.markdown(bilingual("""
            - React UIX with SSE-based response streaming
            - **Ngrok** + **Nginx** for secure tunneling and local routing.
            - Authentication managed via **Firebase**.
            - **Prometheus**, **Grafana** for monitoring and observability.
            """, """
            - Interface React avec streaming des réponses via SSE
            - **Ngrok** + **Nginx** pour le tunneling sécurisé et le routage local.
            - Authentification gérée via **Firebase**.
            - **Prometheus**, **Grafana** pour le monitoring.
            """))

        col1, col2, col3 = st.columns([0.3, 3, 0.3])
        with col2:
            st.image(str(img_pipeline), caption="System-Level Service Architecture",
                     use_container_width=True)

    with tab2:
        col1, col2, col3, col4 = st.columns([0.2, 1, 1, 0.2])

        with col2:
            st.header("Database management")
            st.markdown(bilingual("""
            - Scientific articles are sourced from **Semantic Scholar** and **arXiv**.
            - PDFs are downloaded via **Selenium**, parsed with **pdfminer**, chunked, and embedded with **e5-large-instruct**.
            - Embeddings are saved in **Qdrant** collection.
            """, """
            - Les articles scientifiques proviennent de **Semantic Scholar** et **arXiv**.
            - Les PDFs sont téléchargés via **Selenium**, parsés avec **pdfminer**, découpés puis vectorisés avec **e5-large-instruct**.
            - Les embeddings sont enregistrés dans une collection **Qdrant**.
            """))

        with col3:
            st.header("Reranker model training")
            st.markdown(bilingual("""
            - **Qdrant** provides training data.
            - The model is trained using a **contrastive loss** to align citation relevance.
            - Models are logged and tracked via **MLflow**.
            """, """
            - **Qdrant** fournit les données d'entraînement.
            - Le modèle est entraîné via une **loss contrastive** pour refléter la pertinence des citations.
            - Les modèles sont loggés et suivis dans **MLflow**.
            """))

        col1, col2, col3, col4 = st.columns([0.2, 1, 0.51, 0.69])
        with col2:
            st.image(str(img_DB_manage), caption="Database management",
                     use_container_width=True)
        with col3:
            st.image(str(img_reranker_train_flow), caption="Reranker training flow",
                     use_container_width=True)

# page 2############################################################################################################################################
if page == pages[2]:
    st.title("Dataset")

    st.markdown(bilingual("""
    Data used to power the **asXai** RAG pipeline come from Semantics Scholar and arXiv (for now).
    """, """
    Les données utilisées dans le pipeline **asXai** proviennent de Semantic Scholar et arXiv (pour l’instant).
    """))

    # Load data
    metadata_df = load_metadata()
    textdata_df = load_textdata()

    metadata_df = metadata_df[['paperId', 'authorName', 'publicationDate',
                               'citationCount', 'fieldsOfStudy', 'openAccessPdf', 'referenceIds']]
    textdata_df = textdata_df[['paperId', 'title',
                               'abstract', 'main_text', 'pdf_status', 'full_text']]

    st.subheader("🔖 Metadata Preview")
    st.markdown(bilingual(f"""
    **metadata** : Titles, authors, publication year, source (e.g. arXiv, Semantic Scholar), citation count, and unique paper ID.
    This metadata is used for filtering, payload enrichment, and user-facing context.

    Fields: **{[col for col in metadata_df.columns]}**
    """, f"""
    **metadata** : Titres, auteurs, année de publication, source (par ex. arXiv, Semantic Scholar), nombre de citations et ID unique de l’article.
    Ces métadonnées servent au filtrage, à l’enrichissement des payloads et au contexte utilisateur.

    Champs : **{[col for col in metadata_df.columns]}**
    """))
    st.dataframe(metadata_df, use_container_width=True)

    st.subheader("📄 Text Data Preview")
    st.markdown(bilingual(f"""
    **textdata**: holds text extracted from the article PDFs. These are chunked, embedded and stored to Qdrant
    for retrieval.

    Fields: **{[col for col in textdata_df.columns]}**
    """, f"""
    **textdata** : contient le texte extrait des fichiers PDF des articles. Il est découpé en segments, vectorisé et stocké dans Qdrant
    pour être utilisé lors de la récupération d’informations.

    Champs : **{[col for col in textdata_df.columns]}**
    """))
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
            st.markdown(bilingual(
                """
            - Articles are split into overlapping windows of **512 tokens**.
            - A central chunk of **~170 tokens** is extracted and used for embedding.
            - Embeddings are computed using a pre-trained model (**e5-large-instruct**) and stored in **Qdrant** with article metadata.
            - This strategy balances **granularity** and **context**, ensuring effective semantic search across long documents.
            """,
                """
            - Les articles sont divisés en fenêtres superposées de **512 tokens**.
            - Un segment central d’environ **170 tokens** est extrait et utilisé pour l'embedding.
            - Les embeddings sont calculés à l’aide d’un modèle pré-entraîné (**e5-large-instruct**) et stockés dans **Qdrant** avec les métadonnées.
            - Cette stratégie équilibre **granularité** et **contexte**, garantissant une recherche sémantique efficace dans les documents longs.
            """))

        with col2:
            st.subheader("")
            st.subheader("")
            st.image(str(img_chunk_embedding), use_container_width=True)

    # Tab 2: Triplet Loss Model Training
    with tab2:
        col1, col2, col3 = st.columns([0.8, 1, 0.3])
        with col1:
            st.subheader("Reranker Training with Triplet Loss")
            st.markdown(bilingual(
                """
                - A **transformer-based reranker** is trained to score and rank retrieved chunks.
                - Training uses a **triplet loss** with:
                    - Anchor: article embeddings
                    - Positive: cited article embeddings
                    - Negative: non-cited article embeddings
                - Optimizes the embedding space to prioritize **citation alignment**.
                """,
                """
                - Un **reranker basé sur un transformeur** est entraîné pour scorer et classer les segments retrouvés.
                - L'entraînement utilise une **perte triplet** avec :
                    - Anchor : embeddings de l'article de départ
                    - Positive : embeddings d’un article cité
                    - Negative : embeddings d’un article non cité
                - L’espace d’embedding est optimisé pour favoriser l’**alignement des citations**.
                """))
        with col2:
            st.subheader("")
            st.image(str(img_reranker_model), use_container_width=True,
                     caption="8-layer reranker transformer")

        col1, col2, col3 = st.columns([0.8, 0.7, 0.6])
        with col2:
            st.image(str(img_triplet_loss), use_container_width=True)

    # Tab 3: Chat Service
    with tab3:
        col1, col2 = st.columns([0.6, 1.2])

        with col1:
            st.subheader("Interactive Chat Grounded in Literature")
            st.markdown(bilingual(
                """
                - User's query is processed according to the **initial instruction** and reformulated by the **Chat worker**
                which send it to the **Search API**
                - Search queries are parsed with an LLM (e.g., **gemma3:12b** via **Ollama**).
                - The top-K retrieved chunks are returned to the Chat service and added to the context of the LLM for generation
                according to the **Refinement instruction**.
                - The response includes **article citations** and feeds a **Citation Score**, used to train a lightweight
                rescorer model.
                - All responses are streamed back to the frontend using **Server-Sent Events (SSE)**.
                """,
                """
                - La requête de l’utilisateur est traitée via une **instruction initiale** puis reformulée par le **Chat worker** avant d’être envoyée à l’**API de recherche**.
                - Les requêtes sont analysées avec un LLM (par exemple, **gemma3:12b** via **Ollama**).
                - Les top-K segments retrouvés sont ajoutés au contexte de génération du LLM selon une **instruction de raffinement**.
                - Les réponses incluent les **citations** utilisées et alimentent un **score de citation** pour entraîner un reranker léger.
                - Toutes les réponses sont transmises en direct via **Server-Sent Events (SSE)**.
                """))

        with col2:
            st.subheader("")
            st.image(str(img_chat_service), use_container_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(""" **Initial instruction**: \n
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
            st.markdown("""**Refinement instruction**:\n
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

        st.markdown(bilingual(
            """
            This system follows basic **CI/CD** and **MLOps** practices for production-level pipelines.
            """,
            """
            Ce système suit les bonnes pratiques de base de **CI/CD** et de **MLOps** pour les pipelines de niveau production.
            """))

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            st.markdown(bilingual(
                """
                #### 🔄 Continuous Integration / Continuous Deployment (CI/CD)
                - GitHub handles code versioning.
                - Docker + Docker Compose define infrastructure as code for all services.
                - Local and remote environments mirror each other for reproducibility.
                """,
                """
                #### 🔄 Intégration et déploiement continus (CI/CD)
                - GitHub gère la gestion de version du code.
                - Docker + Docker Compose définissent l’infrastructure comme du code pour tous les services.
                - Les environnements locaux et distants sont identiques pour garantir la reproductibilité.
                """))
        with col2:
            st.markdown(bilingual(
                """
                #### 🧪 MLOps with MLflow
                - Reranker models are logged, versioned, and evaluated in **MLflow**.
                - Best-performing model is promoted to "production" and automatically used by workers.
                """,
                """
                #### 🧪 MLOps avec MLflow
                - Les modèles de reranking sont enregistrés, versionnés et évalués avec **MLflow**.
                - Le meilleur modèle est promu en production et utilisé automatiquement.
                """))
        with col3:
            st.markdown(bilingual(
                """
                #### 📊 Monitoring
                - **Prometheus** tracks API call volume, latency, and system resource usage.
                - **Grafana** dashboards provide real-time observability.
                - Alerts can be configured for system failures or overload conditions.
                """,
                """
                #### 📊 Monitoring
                - **Prometheus** suit les volumes d’appels API, latence et consommation système.
                - Les tableaux de bord **Grafana** fournissent une observabilité en temps réel.
                - Des alertes peuvent être configurées en cas de panne ou de surcharge.
                """))
        with col4:
            st.markdown(bilingual(
                """
                #### 🐳 Containerized Ecosystem
                - Every service (LLM, Qdrant, frontend, Chat service, Kafka, Search service, etc.) is deployed in its own container.
                - Docker Compose ensures they communicate correctly and restart on failure.
                """,
                """
                #### 🐳 Écosystème Conteneurisé
                - Chaque service (LLM, Qdrant, frontend, Chat, Kafka, Search, etc.) est déployé dans son propre conteneur.
                - Docker Compose garantit leur bon fonctionnement et redémarrage en cas d’échec.
                """))

        st.markdown(bilingual(
            """
            > ✅ This setup enables fast iteration, safe deployment, and continuous learning from usage data.
            """,
            """
            > ✅ Cette configuration permet une itération rapide, un déploiement sécurisé, et un apprentissage continu à partir des données d’usage.
            """))

        col1, col2, col3 = st.columns([0.3, 1, 0.3])
        with col2:
            st.image(str(img_mlops_stack), use_container_width=True)


# page 4############################################################################################################################################
if page == pages[4]:
    st.title("Performance Overview")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(bilingual(
            """
            ### ⚙️ System Metrics (from Prometheus)

            - 🔄 Average search latency: **1.2s**
            - 📥 Chat inference latency: **~2.8s** per query
            - 🧠 Ollama inference time: **~1.5s**
            - ✅ Uptime stability: **100%** (Docker auto-restart on failure)
            """,
            """
            ### ⚙️ Indicateurs système (via Prometheus)

            - 🔄 Latence moyenne des recherches : **1.2s**
            - 📥 Latence d'inférence du chat : **~2.8s** par requête
            - 🧠 Temps d'inférence Ollama : **~1.5s**
            - ✅ Stabilité de disponibilité : **100%** (reboot auto via Docker)
            """))
    with col2:
        st.markdown(bilingual(
            """
            ### 📊 Model Performance (MLflow)

            - Latest Reranker: `v2`
            - Accuracy: **81.2%**
            - Triplet loss: **0.0018**
            - Promoted to production: **2025-06-06**
            """,
            """
            ### 📊 Performance du modèle (MLflow)

            - Reranker actuel : `v2`
            - Précision : **81.2%**
            - Perte triplet : **0.0018**
            - Promu en production : **2025-06-06**
            """))

    with col3:
        st.markdown(bilingual(
            """
            ### ✅ Observed Improvements

            - With reranking, relevant citations appear in top 3 results **+37%** more often.
            - Latency with reranking pipeline remains < 3.5s end-to-end.
            """,
            """
            ### ✅ Améliorations observées

            - Grâce au reranking, les citations pertinentes apparaissent dans le top 3 **+37%** plus souvent.
            - La latence totale reste < 3.5s avec reranking.
            """))

# page 5############################################################################################################################################
if page == pages[5]:
    st.title("Future improvements")

    st.markdown(bilingual(
        """
        The current pipeline is functional and modular, but there are several areas planned for improvement:
        """,
        """
        Le pipeline actuel est fonctionnel et modulaire, mais plusieurs pistes d’amélioration sont prévues :
        """))
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(bilingual(
            """
            #### 🧠 Model & Retrieval Enhancements
            - Improve fine-tuning of the notebook specific blender models.
            - Extract citations from pdf to increase training set of the Reranker model.
            """,
            """
            #### 🧠 Améliorations du modèle et de la recherche
            - Mieux ajuster les modèles blender spécifiques aux notebooks.
            - Extraire les citations des PDFs pour enrichir l'entraînement du reranker.
            """))
    with col2:
        st.markdown(bilingual(
            """
            #### ⚙️ System Optimization
            - Implement multiple LLMs agents working in parallel to generate different parts of the response.
            - Go from mid-size to large-scale LLMs
            """,
            """
            #### ⚙️ Optimisation système
            - Mettre en place des agents LLM parallèles pour générer différentes parties des réponses.
            - Passer à des modèles LLM plus grands.
            """))
    with col3:
        st.markdown(bilingual(
            """
            #### 🧪 Evaluation & Feedback
            - Build internal evaluation set with human-labeled relevance judgments.
            - Add user feedback mechanism to improve reranking in-the-loop.
            """,
            """
            #### 🧪 Évaluation & retour utilisateur
            - Créer un jeu d’évaluation avec annotations humaines sur la pertinence.
            - Ajouter un retour utilisateur pour affiner le reranking en continu.
            """))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(bilingual(
            """
            #### 🌐 Deployment & Scalability
            - Migrate to Kubernetes for better orchestration and autoscaling.
            - Replace Ngrok with Cloudflare Tunnel or custom domain.
            """,
            """
            #### 🌐 Déploiement & montée en charge
            - Migrer vers Kubernetes pour une meilleure orchestration et montée en charge.
            - Remplacer Ngrok par Cloudflare Tunnel ou un nom de domaine dédié.
            """))
    with col2:
        st.markdown(bilingual(
            """
            #### 🧾 UI/UX Improvements
            - Display citation scores and confidence alongside chat answers.
            - Add option to export citations to a new notebook or bibtex file
            - Make citation suggestions on texts provided by the user
            """,
            """
            #### 🧾 Améliorations de l’UI/UX
            - Afficher le score de citation et la confiance avec chaque réponse.
            - Permettre d’exporter les citations vers un notebook ou fichier BibTeX.
            - Proposer des suggestions de citations pour des textes fournis.
            """))

    st.markdown(bilingual(
        """
        > ✅ These improvements should make asXai faster, more accurate, and more valuable to scientific users.
        """,
        """
        > ✅ Ces améliorations rendront asXai plus rapide, précis et utile pour les chercheurs.
        """))


# page 6############################################################################################################################################
if page == pages[6]:
    st.title("Conclusion")

    st.markdown(bilingual(
        """
        **asXai** is a modular and functional **Retrieval-Augmented Generation (RAG)** pipeline
        built for scientific literature.
        """,
        """
        **asXai** est un pipeline **RAG (Retrieval-Augmented Generation)** modulaire et fonctionnel,
        conçu pour la littérature scientifique.
        """))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(bilingual(
            """
            ### 🔍 What We've Built
            - A working chat assistant grounded in real, up-to-date scientific publications.
            - Fully containerized system ready to deploy.
            - Custom training of a reranker using triplet loss, monitored through MLflow.
            - A React-based frontend with Firebase Auth and SSE-based live response streaming.
            """,
            """
            ### 🔍 Ce que nous avons construit
            - Un assistant de chat fonctionnel ancré dans des publications scientifiques récentes et réelles.
            - Un système entièrement conteneurisé, prêt à déployer.
            - Entraînement personnalisé du reranker via une perte triplet, suivi avec MLflow.
            - Un frontend React avec authentification Firebase et flux SSE.
            """))
    with col2:
        st.markdown(bilingual(
            """
            ### 🧩 What Makes It Different
            - **Open-source**, **free**, and built for researchers and students.
            - Transparent and adaptable for research, education, or production use.
            - Citation-aware responses and technical fluency using lightweight LLMs.
            """,
            """
            ### 🧩 Ce qui le rend unique
            - **Open-source**, **gratuit**, conçu pour chercheurs et étudiants.
            - Transparent et adaptable pour la recherche, l’enseignement ou la production.
            - Réponses sensibles aux citations et techniquement précises grâce à des LLMs légers.
            """))
    with col3:
        st.markdown(bilingual(
            """
            ### 🔭 What’s Next
            - Continue improving model quality, speed, and user experience.
            - Enable user feedback loops for adaptive reranking.
            - Deploy a public-facing, persistent version for the research community.
            - Improve monitoring.
            """,
            """
            ### 🔭 Prochaines étapes
            - Continuer à améliorer la qualité du modèle, la vitesse et l’expérience utilisateur.
            - Intégrer un retour utilisateur pour affiner le reranking.
            - Déployer une version publique et persistante pour la communauté scientifique.
            - Améliorer la supervision.
            """))

    st.divider()

    # Final demo section
    st.subheader(bilingual("🚀 Try the Demo", "🚀 Lancer la démo"))
    demo_url = "https://goose-beloved-kit.ngrok-free.app/?ngrok-skip-browser-warning=true"
    st.markdown(bilingual(
        f"""Click to open the live [asXai demo]({demo_url}) in a new tab.""",
        f"""Cliquez pour ouvrir la [démo asXai]({demo_url}) dans un nouvel onglet.""",
    ))
