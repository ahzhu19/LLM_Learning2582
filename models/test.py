from modelscope.pipelines import pipeline
from modelscope.preprocessors import TextGenerationPreprocessor

# 加载模型
model_dir = './models/langboat/mengzi-t5-base'

# 手动创建预处理器
preprocessor = TextGenerationPreprocessor(model_dir)

# 创建 pipeline
pipeline_ins = pipeline(
    task='text2text-generation',
    model=model_dir,
    preprocessor=preprocessor,
    device='cpu'  # 或 'cuda:0'
)

# 测试输入
input_text = '中国的首都位于<extra_id_0>。'
result = pipeline_ins(input=input_text)
print(result)
