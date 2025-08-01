
# 4、分词是中文自然语言处理的基础任务之一。给定一段中文文本，设计一个分词算法，将文本分割成有意义的词序列。
# 分词算法需要能够识别出文本中的词语边界，并正确地将它们分开。
# 输入：
# 一个字符串 sentence，表示需要进行分词的中文文本。
# 一个字典列表。
# 输出：
# 一个列表 words，包含文本中所有分词后的结果。

dictionary = {"我","爱","自然","自然语言","语言","处理"}
sentence = "我爱自然语言处理"

class Segmenter(object):
    def __init__(self, dictionary):
        # 把传入的dictionary从set转换为字典
        self.dictionary = {word: 1 for word in dictionary}

    def segment_sentence(self, sentence):
        """
        使用前向最大匹配算法进行中文分词
        """
        result = []
        i = 0
        
        while i < len(sentence):
            # 尝试找到最长的匹配词
            longest_word = ""
            for j in range(i + 1, len(sentence) + 1):
                word = sentence[i:j]
                if word in self.dictionary:
                    longest_word = word
            
            # 如果找到了词典中的词，添加到结果中
            if longest_word:
                result.append(longest_word)
                i += len(longest_word)
            else:
                # 如果没找到，将单个字符作为一个词
                result.append(sentence[i])
                i += 1
                
        return result


# 测试代码
segmenter = Segmenter(dictionary)
result = segmenter.segment_sentence(sentence)
print(f"原句: {sentence}")
print(f"分词结果: {result}")
