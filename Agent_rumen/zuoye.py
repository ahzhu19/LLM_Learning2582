# 数据探索 Agent - 使用 DeepSeek 模型完成 Titanic 数据分析
# 
# 任务要求：
# 1. 数据的summry，即统计数据的均值，方差，最大最小值等。
# 2. 缺失值NULL的填充，用均值
# 3. 对数据画图，例如统计Survived列的分布。
# 4. 利用sklearn作为工具，训练模型并完成模型的预测。
# 注意：以上功能需要都用大模型来完成，不能手写代码实现。

from dotenv import load_dotenv
import os
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()


# 配置 DeepSeek 模型（使用 OpenAI 兼容接口）
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 端点
    temperature=0  # 确保结果稳定
)

# 创建绘图工具
from langchain_core.tools import tool
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import json
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import ChatPromptTemplate
agent = create_python_agent(
    llm=llm, 
    tool=PythonREPLTool(),
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
)

data_location = "Agent_rumen/titanic_cleaned.csv"

df = pd.read_csv(data_location)

# 对数据进行summary

def get_data_summary(agent, data_location):
    template_string = """
    Generate a JSON summary for each column in the dataset located at {location}. \
    The summary should include statistics and information for each column:

    You MUST print the above summary in a json format.
    What is the final printed value? Only tell me the printed part without any extra information, for example: ','.
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)

    location_prompt = prompt_template.format_messages(
        location=data_location)

    response = agent.run(location_prompt)
    data_summary = json.loads(response)
    return data_summary

# NULL值填充

def remove_null(agent, data_location):
    template_string = """
    Load the dataset at location: {location} and handle null values on the following conditions:

    CONDITION 1: For numerical columns: Replace null values with the mean of the respective column.

    CONDITION 2: For string (categorical) columns: Replace null values with random values that are not null from the same column.

    You MUST perform data cleaning in the following order:
    1. Count how many null values are there in the dataset before data cleaning in each column.
    2. Handle null values from each column based on the above conditions
    3. Count how many null values are there in the dataset after data cleaning in each column. 
    4. Save this new data file in the same file from where it was loaded.

    What is the count of values that were null before and after this data cleaning in each column? Return the output in a json format \
    where each entry corresponds to the column name followed by the count of null values before and after data cleaning.
    You MUST substitute all variables in the dataset before replying.
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    location_prompt = prompt_template.format_messages(location=data_location)
    response = agent.run(location_prompt)
    return response


# 对数据画图
def plot_figure(agent, data_location, query):
    template_string = """
    Load the dataset from {location} and create a visualization based on the user's request: {user_query}
    
    You MUST follow these steps:
    1. Load the data from the specified location
    2. Analyze the user's query to understand what type of visualization they want
    3. Create the appropriate plot using matplotlib or seaborn
    4. Customize the plot with proper title, labels, and styling
    5. Save the plot to 'Agent_rumen/generated_plot.png'
    6. Display the plot
    
    Common plot types to consider based on the query:
    - For distribution: histogram, box plot, violin plot
    - For comparison: bar chart, grouped bar chart
    - For correlation: scatter plot, heatmap
    - For time series: line plot
    - For categorical data: count plot, pie chart
    
    Make sure to:
    - Choose appropriate colors and styling
    - Add meaningful titles and axis labels
    - Handle any missing values appropriately
    - Ensure the plot is saved successfully
    
    What visualization did you create and where was it saved?
    """
    
    prompt_template = ChatPromptTemplate.from_template(template_string)
    location_prompt = prompt_template.format_messages(
        location=data_location, 
        user_query=query
    )
    response = agent.run(location_prompt)
    return response


# 模型的训练和预测
def train_model(agent, data_location, query):






