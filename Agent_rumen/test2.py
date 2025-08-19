# =============================================================================
# 法律文档智能分析系统 - AI Agent 团队协作平台
# 基于 agno 框架的多智能体法律文档分析工具
# =============================================================================

# 标准库导入
import os                           # 操作系统接口模块
import tempfile                     # 临时文件处理模块
from dotenv import load_dotenv      # 环境变量加载器

# 第三方库导入
import streamlit as st              # Web 应用框架

# agno 框架组件导入
from agno.agent import Agent                           # 智能体核心类
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader  # PDF 知识库和阅读器
from agno.vectordb.qdrant import Qdrant              # Qdrant 向量数据库
from agno.tools.duckduckgo import DuckDuckGoTools     # DuckDuckGo 搜索工具
from agno.models.openai import OpenAIChat             # OpenAI 兼容聊天模型
from agno.embedder.openai import OpenAIEmbedder       # OpenAI 兼容嵌入模型
from agno.document.chunking.document import DocumentChunking  # 文档分块策略

# LangChain 组件导入（备用）
from langchain.chat_models import init_chat_model      # LangChain 聊天模型初始化器
from langchain_community.embeddings import DashScopeEmbeddings  # DashScope 嵌入模型

# =============================================================================
# 全局配置和环境变量
# =============================================================================

# 加载 .env 文件中的环境变量
load_dotenv()

# 千问（DashScope）API 配置
api_key = os.getenv('DASHSCOPE_API_KEY')              # 从环境变量读取 API Key
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 千问 OpenAI 兼容接口
chat_model = "qwen-plus"                               # 使用的聊天模型名称

# =============================================================================
# 全局常量定义
# =============================================================================

COLLECTION_NAME = "legal_documents"  # Qdrant 向量数据库中的集合名称

# =============================================================================
# Streamlit 会话状态初始化函数
# =============================================================================

def init_session_state():
    """初始化 Streamlit 会话状态变量
    
    会话状态用于在用户交互过程中保持数据持久性，
    包括 API 密钥、数据库连接、知识库、智能体团队等关键组件
    """
    # API 配置相关状态
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None        # OpenAI API Key（实际存储 DashScope Key）
    
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None        # Qdrant 向量数据库 API Key
    
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None            # Qdrant 服务器 URL
    
    # 核心组件状态
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None             # Qdrant 向量数据库实例
    
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None            # 法律智能体团队（团队协调者）
    
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None        # PDF 知识库实例
    
    # 文件处理状态追踪
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()      # 已处理文件名集合（防重复处理）




def init_qdrant():
    """Initialize Qdrant client with configured settings."""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        # Create Agno's Qdrant instance which implements VectorDb
        vector_db = Qdrant(
            collection=COLLECTION_NAME,
            url="https://525830b4-3563-4709-90dd-a8fb3062a921.us-west-2-0.aws.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Bbi4iks1zDmPpyFfmNvFub-EHMuagq2AIqQm43xHsR8",
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key="sk-475a692cd53a4a9980b5e0128aa7a57e",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        )
        return vector_db
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None
    
def process_document(uploaded_file, vector_db: Qdrant):
    """
    Process document, create embeddings and store in Qdrant vector database
    
    Args:
        uploaded_file: Streamlit uploaded file object
        vector_db (Qdrant): Initialized Qdrant instance from Agno
    
    Returns:
        PDFKnowledgeBase: Initialized knowledge base with processed documents
    """

    if not st.session_state.openai_api_key:
        
        raise ValueError("OpenAI API key not provided")
    
    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    # Save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue()) # 获取文件的二进制内容,将内容写入临时文件
            temp_file_path = temp_file.name

        st.info("Loading and processing document...")

        # Create a PDFKnowledgeBase with the vector_db
        knowledge_base = PDFKnowledgeBase(
            path=temp_file_path,  # Single string path, not a list
            vector_db=vector_db,
            reader=PDFReader(),
            chunking_strategy=DocumentChunking(
                chunk_size=1000,
                overlap=200
            )
        )
        # PDFKnowledgeBase should auto-load documents when initialized with path
        st.success("Documents loaded successfully!") 
                
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass
            
        return knowledge_base               
    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")
    

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API Configuration")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
            
        qdrant_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else "",
            help="Enter your Qdrant API key"
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key
        
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url if st.session_state.qdrant_url else "",
            help="Enter your Qdrant instance URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url
        
        if all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
            try:
                if not st.session_state.vector_db:
                    # Make sure we're initializing a QdrantClient here
                    st.session_state.vector_db = init_qdrant()
                    if st.session_state.vector_db:
                        st.success("Successfully connected to Qdrant!")
            except Exception as e:
                st.error(f"Failed to connect to Qdrant: {str(e)}")

        st.divider()
        if all([st.session_state.openai_api_key, st.session_state.vector_db]):
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf'])

            if uploaded_file:

                if uploaded_file.name not in st.session_state.processed_files:
                    with st.spinner("Processing document..."):
                        try:
                            knowledge_base = process_document(uploaded_file, st.session_state.vector_db)
                            if knowledge_base:
                                st.session_state.knowledge_base = knowledge_base

                                # Add the file to processed files   
                                # 这行代码是用来防止重复处理同一个文件的。
                                st.session_state.processed_files.add(uploaded_file.name)

                                # 初始化智能体
                                # Initialize agents
                                legal_researcher = Agent(
                                name="Legal Researcher",
                                role="Legal research specialist",
                                model=OpenAIChat(
                                    id = "qwen-plus",
                                    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                    api_key = st.session_state.openai_api_key
                                ),
                                tools=[DuckDuckGoTools()],
                                knowledge=st.session_state.knowledge_base,
                                search_knowledge=True,
                                instructions=[
                                    "Find and cite relevant legal cases and precedents",
                                    "Provide detailed research summaries with sources",
                                    "Reference specific sections from the uploaded document",
                                    "Always search the knowledge base for relevant information"
                                ],
                                show_tool_calls=True,
                                markdown=True
                            )

                            
                                contract_reviewer = Agent(
                                name="Contract Reviewer",
                                role="Contract review specialist",
                                model=OpenAIChat(
                                    id = "qwen-plus",
                                    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                    api_key = st.session_state.openai_api_key
                                ),
                                knowledge=st.session_state.knowledge_base,
                                search_knowledge=True,
                                instructions=[
                                        "Review contracts thoroughly",
                                        "Identify key terms and potential issues",
                                        "Reference specific clauses from the document"
                                    ],
                                markdown=True
                            
                            )

                                legal_strategist = Agent(
                                name="Legal Strategist",
                                role="Legal strategy specialist",
                                model=OpenAIChat(
                                    id = "qwen-plus",
                                    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                    api_key = st.session_state.openai_api_key
                                ),
                                knowledge=st.session_state.knowledge_base,
                                search_knowledge=True,
                                instructions=[
                                        "Develop comprehensive legal strategies",
                                        "Provide actionable recommendations",
                                        "Consider both risks and opportunities"
                                    ],
                                markdown=True
                            )

                                st.session_state.legal_team = Agent(
                                    name="Legal Team",
                                    role="Legal team coordinator",
                                    model=OpenAIChat(
                                        id = "qwen-plus",
                                        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                        api_key = st.session_state.openai_api_key
                                    ),
                                    team=[legal_researcher, contract_reviewer, legal_strategist],
                                knowledge=st.session_state.knowledge_base,
                                search_knowledge=True,
                                instructions=[
                                        "Coordinate analysis between team members",
                                        "Provide comprehensive responses",
                                        "Ensure all recommendations are properly sourced",
                                        "Reference specific parts of the uploaded document",
                                        "Always search the knowledge base before delegating tasks"
                                    ],
                                markdown=True
                            )
                                st.success("✅ Document processed and team initialized!")

                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")

                else:
                    # File already processed, just show a message
                    st.success("✅ Document already processed and team ready!")
            st.divider()
            st.header("Analysis options")
            analysis_type = st.selectbox(
                        "Select Analysis Type",
                        [
                            "Contract Review",
                            "Legal Research",
                            "Risk Assessment",
                            "Compliance Check",
                            "Custom Query"
                        ]
            )
        else:
            st.warning("Please configure all API credentials to proceed")
    # # Main content area
    if not all([st.session_state.openai_api_key, st.session_state.vector_db]):
        st.info("Please configure all API credentials to proceed")
    elif not uploaded_file:
        st.info("👈 Please upload a legal document to begin analysis")
    
    elif st.session_state.legal_team:
        # Create a dictionary for analysis type icons
        analysis_icons = {
            "Contract Review": "📑",
            "Legal Research": "🔍",
            "Risk Assessment": "⚠️",
            "Compliance Check": "✅",
            "Custom Query": "💭"
        }
        # Dynamic header with icon  
        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
        # Dynamic header with icon
        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
  
        analysis_configs = {
            "Contract Review": {
                "query": "Review this contract and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
                "description": "Detailed contract analysis focusing on terms and obligations"
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to this document.",
                "agents": ["Legal Researcher"],
                "description": "Research on relevant legal cases and precedents"
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities in this document.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Compliance Check": {
                "query": "Check this document for regulatory compliance issues.",
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Comprehensive compliance analysis"
            },
            "Custom Query": {
                "query": None,
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Custom analysis using all available agents"
            }
        }
        # Replace the existing user_query section with this:

        if analysis_type == "Custom Query":
            user_query = st.text_area(
                "Enter your specific query:",
                help="Add any specific questions or points you want to analyze"
            )
        else:
            user_query = None  # Set to None for non-custom queries
        

        if st.button("Analyze"):
            if analysis_type == "Custom Query" and not user_query:
                st.warning("Please enter a query")
            else:
                with st.spinner("Analyzing document..."):
                    try:
                        # Ensure OpenAI API key is set
                        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                        
                        # Combine predefined and user queries
                        if analysis_type != "Custom Query":
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            Please search the knowledge base and provide specific references from the document.
                            """
                        else:
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            {user_query}
                            
                            Please search the knowledge base and provide specific references from the document.
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        response = st.session_state.legal_team.run(combined_query)
                        
                        # Display results in tabs
                        tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        
                        with tabs[0]:
                            st.markdown("### Detailed Analysis")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[1]:
                            st.markdown("### Key Points")
                            key_points_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:    
                                {response.content}
                                
                                Please summarize the key points in bullet points.
                                Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[2]:
                            st.markdown("### Recommendations")
                            recommendations_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:
                                {response.content}
                                
                                What are your key recommendations based on the analysis, the best course of action?
                                Provide specific recommendations from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    else:
        st.info("Please upload a legal document to begin analysis")






if __name__ == "__main__":
    main() 
                                    


