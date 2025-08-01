# 2、给定一个包含多个句子的文本数据集，每个句子由若干单词组成。你需要设计一个简单的算法，统计每个单词在文本中出现的频率，
# 并输出出现频率最高的前N个单词，请使用二叉搜索树（BST）来存储单词及其对应的频率。
# 输入：
# 一个包含多个句子的列表 sentences，每个句子由若干单词组成。
# 一个整数 N，表示需要输出出现频率最高的前N个单词。
# 输出：
# 一个包含N个元组的列表 top_n_words，每个元组包含一个单词及其对应的频率，按频率从高到低排序。

def top_n_words(sentences, N):
    word_freq = {}
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:N]
