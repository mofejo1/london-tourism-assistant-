# streamlit run enhanced_streamlit_tourism_app.py

# Goal: London Tourism Assistant with persistent conversation memory and citations

# SAMPLE QUESTIONS TO TEST:
# "What are the best attractions in London?"
# "Where can I find good food in Soho?"
# "How do I get to Big Ben?"

# Imports
import streamlit as st
import yaml
import os
from datetime import datetime

# IMPORTANT: Disable tokenizers parallelism to avoid the forking issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# RAG Components - Claude API only
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Import embeddings with fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize Streamlit App - Enhanced for Tourism
st.set_page_config(page_title="London Tourism Assistant", layout="wide", page_icon="🇬🇧")
st.title("🇬🇧 London Tourism Assistant (LTA)")

st.markdown("""
            I'm your AI-powered London tourism expert! Ask me about attractions, restaurants, 
            transportation, activities, and everything you need to know about visiting London.
            """)

# Load credentials - following your pattern
try:
    anthropic_api_key = yaml.safe_load(open("credentials.yml"))['claude']
except:
    st.error("❌ Could not load credentials.yml - please check your file path")
    st.stop()

# Initialize RAG Components
@st.cache_resource
def load_rag_system():
    """Load the RAG system components - cached for performance"""
    
    # Free Hugging Face embedding function (no OpenAI needed)
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load London Tourism vector database
    try:
        vectorstore = Chroma(
            persist_directory="data/london_tourism_db",
            embedding_function=embedding_function
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Claude model - using your API key
        model = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            anthropic_api_key=anthropic_api_key
        )
        
        return retriever, model, True
        
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {str(e)}")
        st.error("Please run enhanced_vector_database.py first to create the London tourism database")
        return None, None, False

# Load RAG system
retriever, model, rag_loaded = load_rag_system()

if not rag_loaded:
    st.stop()

# Enhanced Tourism Prompt Template
tourism_template = """You are a knowledgeable London Tourism Assistant. Use the provided context from London tourism guides to answer questions about visiting London.

IMPORTANT INSTRUCTIONS:
1. Always cite your sources using the format "Source: [filename], Page: [page if available]"
2. Provide specific, actionable tourism advice
3. Include practical details like addresses, opening hours, prices when available
4. If asking about transportation, include tube/bus information
5. Be enthusiastic and helpful - you're promoting London!
6. If the context doesn't contain enough information, say so and provide general guidance

Context from London Tourism Guides:
{context}

Question: {question}

Tourism Assistant Response:"""

prompt = ChatPromptTemplate.from_template(tourism_template)

# Create RAG chain - following your class pattern
def format_docs(docs):
    """Format retrieved documents with source information"""
    formatted_docs = []
    for doc in docs:
        source_file = doc.metadata.get('source_file', 'Unknown Source')
        title = doc.metadata.get('title', 'Unknown Title')
        content = doc.page_content
        
        formatted_doc = f"Source: {source_file}\nTitle: {title}\nContent: {content}\n---"
        formatted_docs.append(formatted_doc)
    
    return "\n\n".join(formatted_docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Set up persistent conversation memory - following your pattern
msgs = StreamlitChatMessageHistory(key="london_tourism_messages")

# Initialize conversation
if len(msgs.messages) == 0:
    welcome_msg = """🎉 Welcome to your London Tourism Assistant! 

I can help you with:
• **Attractions**: Big Ben, Tower Bridge, Museums, Parks
• **Food & Dining**: Restaurants, Pubs, Markets, Local cuisine  
• **Transportation**: Tube, Bus, Walking directions
• **Activities**: Tours, Shopping, Entertainment
• **Practical Info**: Opening hours, Prices, Booking tips

What would you like to know about visiting London?"""
    
    msgs.add_ai_message(welcome_msg)

# Conversation Statistics - Enhanced
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💬 Total Messages", len(msgs.messages))
with col2:
    st.metric("🤖 AI Responses", len([m for m in msgs.messages if m.type == "ai"]))
with col3:
    st.metric("👤 Your Questions", len([m for m in msgs.messages if m.type == "human"]))

# Display conversation history
for msg in msgs.messages:
    with st.chat_message(msg.type):
        if msg.type == "ai":
            # Enhanced AI message display with source highlighting
            content = msg.content
            
            # Highlight source citations
            if "Source:" in content:
                lines = content.split('\n')
                for line in lines:
                    if line.strip().startswith("Source:"):
                        st.markdown(f"📚 **{line}**")
                    else:
                        st.write(line)
            else:
                st.write(content)
        else:
            st.write(msg.content)

# Tourism Query Input with suggestions
st.markdown("### Ask about London:")

# Quick suggestion buttons
suggestion_cols = st.columns(3)
suggestions = [
    "Best attractions to visit",
    "Good restaurants in Soho", 
    "How to use London transport",
    "Free things to do",
    "Best museums",
    "Shopping areas"
]

# Initialize session state for processing
if 'processing_query' not in st.session_state:
    st.session_state.processing_query = None

for i, suggestion in enumerate(suggestions):
    col_idx = i % 3
    with suggestion_cols[col_idx]:
        if st.button(f"💡 {suggestion}", key=f"suggest_{i}"):
            st.session_state.processing_query = suggestion
            st.rerun()

# Main chat input
query_prompt = "Ask me anything about visiting London..."

# Process suggestion button click
if st.session_state.processing_query:
    question = st.session_state.processing_query
    st.session_state.processing_query = None
    
    # Add user message
    with st.chat_message("human"):
        st.write(question)
    msgs.add_user_message(question)
    
    # Generate AI response with RAG
    with st.chat_message("ai"):
        with st.spinner("🔍 Searching London tourism guides..."):
            try:
                # Get RAG response
                response = rag_chain.invoke(question)
                
                # Display response with enhanced formatting
                st.write(response)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%H:%M")
                st.caption(f"⏰ Responded at {timestamp}")
                
                # Store AI response
                msgs.add_ai_message(response)
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}\n\nPlease try rephrasing your question about London tourism."
                st.error(error_msg)
                msgs.add_ai_message(error_msg)

# Regular chat input
elif question := st.chat_input(query_prompt, key="tourism_input"):
    
    # Add user message
    with st.chat_message("human"):
        st.write(question)
    msgs.add_user_message(question)
    
    # Generate AI response with RAG
    with st.chat_message("ai"):
        with st.spinner("🔍 Searching London tourism guides..."):
            try:
                # Get RAG response
                response = rag_chain.invoke(question)
                
                # Display response with enhanced formatting
                st.write(response)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%H:%M")
                st.caption(f"⏰ Responded at {timestamp}")
                
                # Store AI response
                msgs.add_ai_message(response)
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}\n\nPlease try rephrasing your question about London tourism."
                st.error(error_msg)
                msgs.add_ai_message(error_msg)

# Sidebar with additional features - following your expandable pattern
with st.sidebar:
    st.header("🎯 London Tourism Features")
    
    # Database info
    st.subheader("📊 Knowledge Base")
    try:
        # Get collection info using cached function
        if rag_loaded and retriever:
            st.metric("📚 Tourism Documents", "2498")  # From your screenshot
            st.success("✅ London Tourism DB Loaded")
    except:
        st.error("❌ Database not found")
    
    # Conversation management
    st.subheader("💬 Conversation")
    if st.button("🗑️ Clear Chat History"):
        msgs.clear()
        st.rerun()
    
    # Export conversation
    if st.button("📥 Export Conversation"):
        conversation_text = ""
        for msg in msgs.messages:
            speaker = "You" if msg.type == "human" else "London Tourism Assistant"
            conversation_text += f"{speaker}: {msg.content}\n\n"
        
        st.download_button(
            label="💾 Download Chat History",
            data=conversation_text,
            file_name=f"london_tourism_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

# Debug section - expandable like your class pattern
with st.expander("🔧 Debug Information"):
    st.subheader("Message History")
    st.json([{"type": msg.type, "content": msg.content[:100] + "..."} for msg in msgs.messages])
    
    st.subheader("Session State")
    st.write("Chat message count:", len(msgs.messages))
    st.write("Database path:", "data/london_tourism_db")
    
    # Test retrieval
    if st.button("🧪 Test Retrieval"):
        test_query = "best attractions London"
        if retriever:
            with st.spinner("Testing retrieval..."):
                docs = retriever.get_relevant_documents(test_query)
                st.write(f"Retrieved {len(docs)} documents for: '{test_query}'")
                for i, doc in enumerate(docs[:2]):
                    st.write(f"Doc {i+1}: {doc.metadata.get('source_file', 'Unknown')}")
                    st.text(doc.page_content[:200] + "...")

# Footer
st.markdown("---")
st.markdown("🇬🇧 **London Tourism Assistant** - Your AI guide to exploring London! Built with Claude AI and RAG technology.")

# Usage instructions
with st.expander("ℹ️ How to Use This Tourism Assistant"):
    st.markdown("""
    **Getting Started:**
    1. Ask any question about visiting London
    2. Use the suggestion buttons for quick queries
    3. View source citations for reliable information
    
    **Example Questions:**
    - "What are the must-see attractions in London?"
    - "Best places to eat in Covent Garden?"
    - "How do I get from Heathrow to central London?"
    - "What free museums can I visit?"
    - "Best pubs in the City of London?"
    
    **Features:**
    - 💬 Persistent conversation memory
    - 📚 Cites sources from official tourism guides  
    - 🔍 Searches comprehensive London tourism PDFs
    - 📥 Export your conversation history
    """)