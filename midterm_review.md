# Language Models

### 1. Task Definition and Applications (任务定义及应用)
- **语言模型 (Language Modeling)** 的目标是**预测下一个单词**或**整句话的概率**。语言模型主要在自然语言处理中用于**语音识别**、**拼写纠正**、**机器翻译**、**文本生成**等应用中【122†source】。
- 例如，在自动完成或输入法中，模型会根据已输入的上下文预测下一个最有可能的单词。

### 2. Probability of the Next Word and Probability of a Sentence (下一个单词和句子的概率)
- **下一个单词的概率**：语言模型用于预测当前上下文中下一个单词的可能性。
- **整句话的概率**：也可以计算一句话的整体概率，例如：“Stocks plunged this morning, despite a cut in interest rates...”【122†source】。
- 使用**链式法则 (Chain Rule)** 计算整个句子的概率，即通过各个单词在之前上下文条件下出现的概率的乘积来表示整个句子的概率。
**计算一句话的整体概率**，指的是使用**语言模型**来估计给定的句子出现的可能性。这在自然语言处理（NLP）中是一个非常重要的概念，用于多种任务，例如**语音识别**、**机器翻译**、**拼写检查**等。下面我会详细解释什么是句子的整体概率，以及如何计算它。

#### 2.1 句子整体概率的意义
在**语言模型**中，我们希望对一句话的流利度进行量化，以便可以在多种可能的句子中选择最合理的那个。**句子的概率**反映了该句子在模型（通常是基于大量语料库训练的模型）中出现的可能性。

例如，在处理以下两个句子时：
1. **“The cat is on the mat.”**
2. **“Mat the cat on is the.”**

我们可以直观地看出，第一个句子更符合英语的语法和用法。**句子的整体概率**就是用来数值化地描述这种“合理性”，使模型能够判断哪个句子更“自然”或更符合训练数据的模式。

#### 2.2 计算句子整体概率的方法
通常情况下，句子的整体概率是通过语言模型来估算的。语言模型的目标是对句子中每个单词的出现概率进行建模，从而计算出整个句子的概率。

假设有一个句子 **W = w_1, w_2, ..., w_n**，那么句子的整体概率可以用**链式法则 (Chain Rule)** 来表示为：
$$[
P(W) = P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_1, w_2) \cdot ... \cdot P(w_n | w_1, w_2, ..., w_{n-1})
$$]

**语言模型**通常会通过简化的方式估计每个单词的条件概率，以避免复杂计算：
- **一元模型 (Unigram)**：假设每个单词的出现是独立的，因此
  $$[
  P(W) = P(w_1) \cdot P(w_2) \cdot ... \cdot P(w_n)
  $$]
- **二元模型 (Bigram)**：假设每个单词只依赖于前一个单词
 $$[
  P(W) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_2) \cdot ... \cdot P(w_n | w_{n-1})
  $$]
- **三元模型 (Trigram)**：假设每个单词只依赖于前两个单词
  $$[
  P(W) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_1, w_2) \cdot ... \cdot P(w_n | w_{n-2}, w_{n-1})
  $$]

通过这种方式，**句子的整体概率**就是通过所有这些单词的条件概率相乘得出的结果。这种概率越高，意味着这个句子在模型中越合理，也就是说在训练语料库中它出现的可能性更大。

#### 2.3 具体例子
让我们举一个例子，假设有句子**“Stocks plunged this morning”**，我们可以用 **Bigram 模型**来估计它的概率：
- **P(“Stocks plunged this morning”)** 可以分解为：
  $$[
  P("Stocks") \cdot P("plunged" | "Stocks") \cdot P("this" | "plunged") \cdot P("morning" | "this")
  $$]
每个条件概率（如 **P("plunged" | "Stocks")**）表示的是，在见到“Stocks”之后，“plunged”出现的可能性。这些概率通常通过对大量文本数据进行统计得到。

#### 2.4 为什么句子的整体概率有用？
- **语音识别**：在语音识别中，模型会根据声学模型识别到的音节选择可能的词序列，而句子概率有助于在多个候选词序列中选择最符合自然语言结构的那一个。
- **机器翻译**：在机器翻译中，语言模型用于生成更自然的翻译。高概率的句子通常是语法正确、语义连贯的句子。
- **文本生成**：在文本自动生成中，语言模型可以帮助生成合理的句子结构，使得生成的文本更贴近人类自然语言的风格。

#### 2.5 使用 n-gram 计算句子概率的局限
- **上下文限制**：例如，Bigram 或 Trigram 模型只能考虑前 1 或 2 个单词的上下文，因此对于更长距离的依赖性，n-gram 模型的效果并不好。
- **数据稀疏性**：由于句子和词的组合非常多，即使是一个相对大的语料库，也无法涵盖所有可能的 n-gram 组合。因此，计算这些组合的概率时可能会出现数据稀疏性问题。这种情况下，通常会使用一些**平滑技术**（例如 **拉普拉斯平滑** 或 **Katz 回退**）来处理未见过的 n-gram。

### 总结
**计算一句话的整体概率**就是用一个语言模型来计算这句话中每个单词依赖于之前单词的概率之积，从而得到整个句子的出现概率。这个概率表示了句子的自然程度和合理性，在语言建模任务中有非常广泛的应用。希望这些解释能帮助您理解“句子整体概率”的概念！如果有其他问题，欢迎继续提问！


### 3. Markov Independence Assumption (马尔可夫独立性假设)
- **马尔可夫独立性假设**用于简化语言模型的复杂性。假设某个单词的出现**只依赖于前面的几个单词**，而不是整个之前的句子。这种假设称为**k-阶马尔可夫假设**。
- 在**n-gram模型**中，通常假设下一个词只依赖于之前的 **(n-1)** 个词。例如，在**二元模型 (Bigram)** 中，假设下一个单词只依赖于前一个单词【122†source】。

### 4. n-gram Language Models (n-元语言模型)
- **n-元语言模型** 是基于马尔可夫假设的一种统计语言模型，用于预测句子中的下一个词。
  - **Unigram**: 每个单词独立出现。
  - **Bigram**: 每个单词只依赖于它前一个单词。
  - **Trigram**: 每个单词依赖于它前两个单词。
- **例子**：
  - 对于句子 “I want to eat Chinese food” 的三元模型 (trigram)，我们可以写出概率公式：
    $$[
    P(i|START, START) \cdot P(want|START, i) \cdot P(to|i, want) \cdot \cdots
    $$]

### 5. Role of the START and END Markers (开始和结束标记的作用)
- **START** 和 **END** 标记用于指定句子的边界，在语言模型中非常重要。
- 这些标记帮助模型在计算句子开头和结尾的概率时更精确。例如：
  - “START I want to eat Chinese food END”，这种形式可以保证模型考虑句子结构的完整性【122†source】。

### 6. Estimating n-gram Probabilities from a Corpus (从语料库中估计 n-元概率)
- **最大似然估计 (Maximum Likelihood Estimate, MLE)**：用来计算从语料库中观察到的 n-元组合的概率。通常用该 n-元的出现次数除以其前 **(n-1)** 元的出现次数来得到【122†source】。
#### **具体的例子**
假设我们有一个语料库，包含以下两句话：
1. "I want to eat Chinese food"
2. "I want to play football"

我们来估计以下 **trigram** 的概率：**P(to | I, want)**，即在单词 "I" 和 "want" 之后，出现 "to" 的概率。

#### **步骤 1：统计频率**
首先，我们要统计三元组 **(I, want, to)** 在语料库中的出现次数，和它的前两个单词的二元组合 **(I, want)** 的出现次数。

1. **C(I, want, to)**：我们在语料库中找到 "I want to" 的三元组合：
   - 在第一句话 **"I want to eat Chinese food"** 中，"I want to" 出现了 1 次。
   - 在第二句话 **"I want to play football"** 中，"I want to" 也出现了 1 次。
   - 因此，**C(I, want, to) = 2**。

2. **C(I, want)**：我们接下来要统计 "I want" 这个二元组合的出现次数：
   - 在第一句话和第二句话中，"I want" 各出现了 1 次，总共 **2 次**。
   - 因此，**C(I, want) = 2**。

#### **步骤 2：计算 Trigram 概率**
根据 **最大似然估计 (MLE)** 的公式，三元组合的概率可以表示为：

$$[
P(to | I, want) = \frac{C(I, want, to)}{C(I, want)}
$$]

其中：
- **C(I, want, to) = 2**，表示 "I want to" 这个三元组合在语料库中出现了 2 次。
- **C(I, want) = 2**，表示 "I want" 这个二元组合在语料库中出现了 2 次。

因此：

$$[
P(to | I, want) = \frac{2}{2} = 1.0
$$]


### 7. Dealing with Unseen Tokens (处理未见过的词元)
- 由于语言的多样性，训练数据中不可能包含所有可能的 n-元组合。这种情况被称为**数据稀疏性**。Data Sparsity
- 为了解决这个问题，未见过的词元通常被替换为 **UNK** (未知) 标记，且使用各种平滑技术分配概率给这些未见过的词元。

### 8. Smoothing and Back-off Techniques (平滑和回退技术)
平滑技术使得模型不会对高频词过度依赖，同时也给了罕见词或未知词一定的机会。这种平衡可以让模型在面对新数据或未在训练集中出现的词时更加稳健。
- **Additive Smoothing (加性平滑)**：比如**拉普拉斯平滑**，通过给所有可能事件添加一个小常数（如1）来确保所有事件都有非零概率。
  ![image](https://github.com/user-attachments/assets/9a81a56e-5332-4694-a9cd-e7613248f632)

- **Discounting (折扣法)**：将一部分概率质量从见过的事件分配给未见过的事件。
  ![image](https://github.com/user-attachments/assets/b44516a0-3cd9-46b3-a63b-c421e9db8dfa)

- **Linear Interpolation (线性插值)**：结合不同长度的 n-元模型，例如结合 unigram、bigram 和 trigram，以填补 n-gram 数据的稀疏性。Use denser distributions of shorter ngrams to “fill in” sparse ngram distributions.注意是shorter
  ![image](https://github.com/user-attachments/assets/5f145d85-c607-48f5-9fd5-3a46b21ad179)

- **Katz’ Backoff (Katz 回退)**：对于未见过的 n-gram，使用低阶的 n-gram（比如从 trigram 回退到 bigram，或从 bigram 回退到 unigram）来估计它的概率。这种方法通常结合**Good-Turing 平滑**来对低阶模型进行平滑【122†source】。

### 9. Perplexity (困惑度)
- **困惑度 (Perplexity)** 是评估语言模型的常用指标，它衡量了模型在预测某些样本时的好坏程度。困惑度越低，说明模型对样本的预测越好。
- **定义**：
  $$[
  PP(W) = 2^{H(W)}
  $$]
  其中 **H(W)** 是样本集的**交叉熵**。困惑度可以理解为模型对于词汇预测的“有效词汇大小”，困惑度越高表示模型不确定性越大，越低表示模型更加确定【122†source】。

### 10. 语言模型在自然语言处理中的应用
- **语音识别**：通过比较句子的概率，选择最可能的候选句子。例如，识别 "recognize speech" 和 "wreck a nice beach" 时，模型可能更倾向于选择更常见的表达 "recognize speech"。
- **机器翻译**：通过预测翻译候选句子的概率，选择最可能的翻译。
- **拼写纠正**：通过选择语法上和上下文上更合理的单词组合来进行拼写纠正，例如“my cat eats fish”相比于“my xat eats fish”的概率更高【122†source】。

### 总结
- **语言模型** 是用于预测文本序列中下一个单词或整句话的概率的模型。它们广泛用于 NLP 中的许多任务。
- **n-元语言模型** 使用**马尔可夫假设**来简化对上下文的建模，使用 **n-元模型** 预测每个词的概率。
- **平滑和回退技术** 用于解决训练数据中未见过的词和 n-gram 的问题，确保所有事件都有适当的概率质量。
- **困惑度** 是衡量语言模型在预测上的表现的一个重要指标，困惑度越低表示模型的预测能力越强。

# Sequence Labeling (POS tagging)

### 1. Task Definition and POS Tagging (任务定义与词性标注)
- **序列标注 (Sequence Labeling)** 的任务是将输入序列中的每个元素分配一个标签。例如，**词性标注 (Part of Speech Tagging, POS Tagging)** 是一种序列标注任务，将每个单词分配一个词性标签（如名词、动词等）。
- **POS Tagging** 的目标是将一组单词（如“the koala put the keys on the table”）映射到一组词性标签，如：
  ```
  the      koala     put      the      keys      on      the      table
  DT       NN        VBD      DT       NNS       IN      DT       NN
  ```

### 2. Hidden Markov Model (隐马尔可夫模型, HMM)
#### HMM 基础
- **HMM** 是一种生成概率模型，用于在序列标注任务中标注序列。
- **观测值 (Observations)**：序列中的单词，即输入的文本。
- **隐藏状态 (Hidden States)**：词性标签的序列，这些是我们希望通过 HMM 找到的。
- **转移概率 (Transition Probabilities)**：从一个词性标签转移到下一个词性标签的概率。
- **发射概率 (Emission Probabilities)**：给定一个词性标签，生成某个单词的概率。

HMM 中，通过这些概率可以描述一个单词序列是如何生成的，其中：
- **状态转移**使用转移概率来描述词性之间的相互关系（如从“名词”到“动词”）。
- **发射概率**则描述词性如何产生某个特定单词（例如，“名词”标签下生成“dog”的概率是多少）。

#### Markov Chain (马尔可夫链)
- **马尔可夫假设**：序列中的当前状态只依赖于前一个状态，而不依赖于更早的状态。
- 这使得计算更简单，因为我们只需要考虑前一个标签。

### 3. Three Tasks on HMMs (HMM 的三大任务)
#### 3.1 Decoding (解码)
- **解码**是指找到给定一组单词时，最可能的词性标签序列。
- 解决解码问题最常用的方法是**维特比算法 (Viterbi Algorithm)**，它是一种动态规划算法，可以高效地找到最可能的路径。
- **Viterbi 算法**基于马尔可夫假设来逐步计算每一步的最优概率，并将最优路径保存在一个表中，用于后续步骤中的选择【132†source】。

#### 3.2 Evaluation (评估)
- **评估**是计算给定词序列的整体概率。这可以看作是计算整个句子的概率。
- **前向算法 (Forward Algorithm)** 用于评估，和 Viterbi 类似，但不同之处在于，它会计算所有可能路径的总和，而不是寻找最优路径。这种评估方法可以用于评估模型对词序列的整体描述能力，而不仅是最优的路径【132†source】。

#### 3.3 Training (训练)
- **训练**指的是从训练数据中估计发射和转移概率，常用的方法是**最大似然估计 (Maximum Likelihood Estimate, MLE)**。
- 在训练过程中，我们从标注好的数据中计算各个标签对之间的转移概率以及各个单词对应标签的发射概率【132†source】。

### 4. Spurious Ambiguity (虚假歧义)
- **虚假歧义**是指多个不同的隐藏序列可能会产生相同的观测序列。例如，给定单词序列 “time flies like an arrow”，可以有不同的词性标注组合，这些组合在某些情况下可能会导致不同的句法理解。

### 5. Extending HMMs to Trigrams (将 HMM 扩展到三元组)
- 在标准的 HMM 中，通常使用二元依赖（bigram），即当前词性标签依赖于前一个标签。
- 通过扩展到三元组（trigram），当前的标签可以依赖于前两个标签，从而提高预测的精度。然而，这样做也会带来计算复杂度的增加，以及稀疏性的问题，需要通过一些平滑技术（如 Katz' 回退）来缓解【132†source】。

### 6. Applying HMMs to Other Sequence Labeling Tasks (将 HMM 应用于其他序列标注任务)
- 除了词性标注，**HMM** 还可以用于其他序列标注任务，比如**命名实体识别 (Named Entity Recognition, NER)**。在 NER 中，任务是找到文本中的实体（如人名、地名、组织名等），并对它们进行标注。
- 在 NER 中，我们通常使用 **B.I.O. 标签**：
  - **B**：实体的开头 (Beginning)。
  - **I**：实体的内部 (Inside)。
  - **O**：不属于任何实体 (Outside)。

这种标签可以更好地标记多词组成的实体，例如“New York City”，B.I.O. 标记将是 **B-Location I-Location I-Location**，表示这是一个由多个词组成的地名【132†source】。

### 总结
- **序列标注 (Sequence Labeling)** 是将序列中的每个元素分配给一个标签的任务。
- **HMM (隐马尔可夫模型)** 是一种用于序列标注的生成模型，它利用观测序列（如单词序列）和隐藏状态（如词性标签序列）之间的概率关系来进行推断。
- **维特比算法** 是用于解码 HMM 的常用方法，可以找到给定单词序列下最可能的标签序列。
- **前向算法** 则用于计算整个观测序列的概率，考虑所有可能的路径。
- **B.I.O. 标签** 被广泛用于命名实体识别等序列标注任务，以标记实体的开始、内部和非实体部分。

# Parsing with Context Free Grammars

### 1. Tree Representation of Constituency Structure and Phrase Labels (树表示及短语标签)
- **树的表示**：在语言中，句子的结构可以用树来表示，其中每个节点代表一个短语（例如名词短语 NP、动词短语 VP）或词性标签。
- **短语标签**：在短语结构树中，标签通常使用“XP”的形式，其中 X 代表短语的主要词类（例如，名词短语的标签为 NP，动词短语的标签为 VP）。

### 2. CFG Definition (上下文无关文法的定义)
- **上下文无关文法 (Context-Free Grammar, CFG)** 包括以下四部分：
  - **终结符 (Terminals)**：构成句子的基本元素，通常是单词。
  - **非终结符 (Nonterminals)**：用于表示短语或句子的类别，例如 NP、VP。
  - **起始符 (Start Symbol)**：通常表示句子的符号，一般用 S 表示。
  - **生成规则 (Productions/Rules)**：定义如何将非终结符展开为终结符或其他非终结符。例如：
    - \( S \to NP \ VP \)
    - \( VP \to V \ NP \)

### 3. Derivations and Language of a CFG (推导与文法语言)
- **推导**：从起始符开始，通过不断应用生成规则，逐步将非终结符展开为终结符。
- **派生树 (Derivation Tree) vs. 派生字符串 (Derived String)**：派生树用于展示如何通过应用规则来生成句子，而派生字符串则是生成的句子的形式。
- **CFG 的语言**：CFG 的语言是指所有可以通过起始符生成的终结符序列的集合。

### 4. Regular Grammars / Languages and Complexity Classes (正规文法/语言与复杂度类别)
- **正规文法 (Regular Grammar)** 是上下文无关文法的一种特殊形式，它只能生成有限制的结构，通常使用有限状态自动机（FSA）来表示。
- **复杂度类别**：
  - **正规语言**：使用有限状态自动机生成。
  - **上下文无关语言**：使用下推自动机（Pushdown Automata）生成。
  - **Chomsky 层级 (Chomsky Hierarchy)**：将语言的复杂度分为多种类别，从简单的正规语言到复杂的递归可枚举语言。

### 5. Center Embeddings and Cross-serial Dependencies (中心嵌套与交叉序列依赖)
- **中心嵌套 (Center Embeddings)** 是语言中的一种嵌套结构，比如句子 “The mouse the cat the dog chased ate died” 表示了一种递归嵌套，这种结构超出了正规语言的范畴【140†source】。
- **交叉序列依赖 (Cross-serial Dependencies)** 是上下文无关语言难以表示的一种现象。例如在荷兰语中的交叉依赖结构，这些结构无法用简单的上下文无关文法来描述。

### 6. Probabilistic Context-Free Grammars (PCFG)
- **PCFG** 是为每条生成规则分配一个概率的上下文无关文法。规则的概率反映了在给定非终结符的情况下，特定规则被应用的可能性。
- **树的概率 vs. 句子的概率**：树的概率是通过规则的概率相乘计算的，句子的概率则是所有可能派生树的概率总和【141†source】。

### 7. CKY Parsing Algorithm and Dynamic Programming (CKY 解析算法与动态规划)
- **CKY 算法**是一种用于解析上下文无关文法的动态规划算法，使用自下而上的方式，从每个单词开始逐步构建更大的结构，直到生成整个句子的解析树【141†source】。
- **数据结构**：CKY 使用一个二维的“解析表 (Parse Table)”来存储不同片段的非终结符覆盖信息。每个单元格表示输入序列中的某一子区间由哪个非终结符覆盖。
- **拆分位置与回溯指针 (Backpointers)**：在 CKY 算法中，拆分位置用于将输入区间分成两个子区间，从而找到最佳的解析路径，而回溯指针则用于重建解析树。

### 8. Recognition vs. Parsing (识别问题与解析问题)
- **识别 (Recognition)**：检查某个输入句子是否属于某种语法描述的语言，即是否可以用 CFG 生成。
- **解析 (Parsing)**：不但需要判断某个句子是否属于语法，还需要找到其符合语法规则的解析树。

### 9. Top-down vs. Bottom-up Parsing (自顶向下 vs. 自底向上解析)
- **自顶向下解析 (Top-down Parsing)**：从起始符开始，试图通过应用规则生成输入句子。这种方法基于语法驱动。
- **自底向上解析 (Bottom-up Parsing)**：从输入句子开始，尝试找到与规则匹配的子树，逐步构建更大的结构直到起始符。CKY 算法是一种典型的自底向上方法。

### 10. Parsing with PCFGs and CKY Extension (用 PCFG 进行解析)
- **PCFG 解析中的 CKY**：我们可以对 CKY 进行扩展，利用概率来找到输入句子的最高概率解析树。这需要对 CKY 算法进行修改，使其能够自底向上计算每个子结构的解析概率，并使用动态规划来确定最优解析【141†source】。

### 小结
- **上下文无关文法 (CFG)** 是自然语言处理中常用的工具，用于建模句子结构。
- **派生树与解析**是展示句子如何通过规则生成的关键工具。
- **CKY 算法** 是一种高效的解析算法，通过动态规划找到最优的解析结构。
- **PCFG** 结合了概率信息，能够用于找到最有可能的解析树。

# Dependency parsing

### 1. Dependency Parsing Basics (依存解析基础)
- **依存解析**的目标是基于单词之间的语法依存关系来构建句子的解析结构。与短语结构不同，依存解析更强调词与词之间的语法关系。
- **依存关系**由**头 (head)** 和**依赖项 (dependent)** 组成，例如，"the girl likes a friendly boy" 中，"likes" 是句子的中心词，而 "girl" 是主语 (subject) 的依赖项【149†source】。
  
### 2. Transition-Based Dependency Parsing (基于转换的依存解析)
- **状态 (Configuration)**：基于转换的依存解析器通过一系列的状态进行句子的解析。每个状态通常由三个部分组成：
  1. **栈 (Stack)**：存放已处理的单词。
  2. **缓冲区 (Buffer)**：存放未处理的单词。
  3. **部分依存树 (Partial Dependency Tree A)**：目前已经构建的依存关系。

#### 2.1 Transitions (转换操作)
- **Shift**：将缓冲区的第一个单词移到栈中。形式为：
  - `(σ, wi | β, A) => (σ | wi, β, A)`
  - 例如，如果当前缓冲区是 `[had, little, effect]`，执行 shift 操作后，"had" 会移动到栈上【149†source】。
  
- **Left-Arc**：将缓冲区中第一个单词作为依赖，构建从栈顶到缓冲区的弧，并将栈顶移出栈。
  - `(σ | wi, wj | β, A) => (σ, wj | β, A ∪ {wj, r, wi})`
  - 例如，如果栈顶是 "news"，缓冲区第一个单词是 "had"，执行 left-arc 后，"news" 成为 "had" 的依赖项【149†source】。

- **Right-Arc**：将栈顶单词作为依赖，构建从栈顶到缓冲区的弧。
  - `(σ | wi, wj | β, A) => (σ, wi | β, A ∪ {wi, r, wj})`
  - 例如，如果栈顶是 "had"，缓冲区第一个单词是 "effect"，执行 right-arc 后，"had" 成为 "effect" 的依赖项。

#### 2.2 Arc-Standard 和 Arc-Eager 系统
- **Arc-Standard** 系统中，依赖弧的构建必须在相关单词都在栈中的时候进行。而在**Arc-Eager** 系统中，可以在单词依然在缓冲区中的时候进行，这样可以更快地生成依存树。
- **Arc-Eager** 增加了一种操作：**Reduce**，用于移除已经完成其所有依赖关系的单词。这样可以更加灵活地管理栈中的内容【149†source】。

#### 2.3 Predicting the Next Transition Using Discriminative Classifiers (用判别分类器预测下一个转换)
- **分类器**（例如感知机、SVM、神经网络）用于预测下一个应执行的转换操作。
- **特征定义**：特征是从当前状态中提取的，例如栈顶单词、缓冲区中第一个单词的词性、词汇等。通常包括：
  1. **地址 (Address)**：例如“栈顶”或“缓冲区第一个”。
  2. **属性 (Attribute)**：如词性、词形、词干、嵌入等【149†source】。

#### 2.4 Training the Parser from a Treebank (从语料库训练解析器)
- 使用标注的依存树（例如 Penn Treebank）来生成**Oracle Transitions**，这些转换序列用于训练模型。通过这些标注的数据，模型可以学习如何在不同的状态下选择正确的转换操作【149†source】。

### 3. Graph-Based Dependency Parsing (基于图的依存解析)
- **总体思路**：将每个单词视为图中的一个节点，开始时构建一个完全连接的图，然后为每条边分配一个分数，最后选择得分最高的边来生成依存树。
- **最大生成树 (Maximum Spanning Tree, MST)**：通过图算法（如 Chu–Liu/Edmonds 算法）来寻找一个最大得分的生成树，确保每个单词有且只有一个父节点，以形成一个无环、连接的依存树【149†source】。
  
### 4. Dependency Tree Properties (依存树的特性)
- **投射性 (Projectivity)**：
  - **投射性依存树**是指在从句子的根节点出发的所有依存边之间不会相互交叉的情况。这对于大部分语言来说是常见的，而对于一些语言（如德语、匈牙利语等），非投射性结构更为频繁。
  - **基于转换的解析方法**通常只能生成投射性结构，而**基于图的解析方法**可以处理非投射性结构【149†source】。

### 5. Example: Parsing a Sentence Using Transition-Based Parsing (使用转换解析法解析句子)
- 考虑句子“Economic news had little effect on financial markets”：
  1. **初始状态**：栈中只有根节点，缓冲区包含所有单词。
  2. **转换序列**：通过 Shift、Left-Arc、Right-Arc 等操作，依次将单词移入栈中并建立依存关系，直到所有单词处理完成。
  3. **终止状态**：栈中只剩根节点，缓冲区为空，依存树构建完成【149†source】。

### 小结
- **依存解析**用于表示句子中单词之间的语法依存关系，通常使用树状结构。
- **基于转换的依存解析**通过状态转换来逐步构建依存树，依赖于分类器预测下一个最优转换。
- **基于图的依存解析**通过图算法寻找最大生成树，可以处理更复杂的依存结构，包括非投射性结构。

这些考点涉及依存解析的基本方法、解析的不同类型（基于转换和基于图）、以及解析器的训练和特征选择。如果您对某些部分有更多的疑问或需要更详细的例子，请告诉我，我会进一步为您解释！


# Machine Learning

根据老师的PDF课件内容【157†source】【158†source】【159†source】，以下是关于**机器学习和神经网络模型**考试考点的详细解释。

### 1. Generative vs. Discriminative Algorithms (生成模型与判别模型)
- **生成模型 (Generative Algorithms)** 假设观察到的数据是由某种隐藏类别标签“生成”的，它会为每个类别建立不同的模型，通过比较每个模型的适应性来进行预测。例如，朴素贝叶斯就是典型的生成模型【157†source】。
- **判别模型 (Discriminative Algorithms)** 则直接学习类别与数据之间的边界，模型学习的是给定数据的情况下类别的条件分布，即 **P(y|x)**，例如逻辑回归和支持向量机（SVM）【157†source】。

### 2. Supervised Learning: Classification vs. Regression (监督学习：分类与回归)
- **分类 (Classification)**：输出是离散类别。例如，预测电子邮件是“垃圾邮件”还是“非垃圾邮件”。
- **回归 (Regression)**：输出是连续数值。例如，预测房屋的价格。
- **训练与测试误差**：训练误差是模型在训练数据上的误差，测试误差是在新的未见过的数据上的误差。**过拟合 (Overfitting)** 是指模型在训练集上表现很好，但在测试集上表现不佳，这是因为模型过于拟合训练数据中的噪声【157†source】。

### 3. Linear Models and Perceptron Learning (线性模型与感知器学习)
- **线性模型**是基于输入特征的加权和预测输出的模型，例如感知器，它使用**激活函数**将线性组合映射为输出。
- **感知器学习算法**：迭代地调整权重来找到一个线性超平面，将数据划分为不同类别。感知器算法通过使用误分类点来调整权重直到收敛（即整个训练数据集上无误分类）【157†source】。

### 4. Activation Function (激活函数)
- **步阶函数 (Step Function)**：输出要么是 0 要么是 1，用于表示激活的二元状态，但它不可导，因此无法用于梯度下降。
- **常用的替代激活函数**包括 **sigmoid** 和 **ReLU**，这些函数使得模型可以更好地进行非线性映射【158†source】。

### 5. Linear Separability and XOR Problem (线性可分与 XOR 问题)
- **线性可分 (Linear Separability)** 是指数据集可以被一个线性超平面完全分隔。感知器只能解决线性可分问题。
- **XOR 问题**是一个经典的线性不可分问题，因为它不能通过一条直线来划分，需要引入**多层感知器 (MLP)** 来解决【158†source】。

### 6. Logistic Regression and Log-Linear Models (逻辑回归与对数线性模型)
- **逻辑回归**是一种线性模型，通过**sigmoid 函数**将输出映射到 (0, 1) 范围，用于二元分类任务，输出可以解释为某个类别的概率。
- **Logit 函数**：给定一个概率 **p**，将其转换为对数几率 **log-odds**，逻辑回归的输出被视为该类别的对数几率【157†source】。

### 7. Maximum Entropy Markov Models (MEMM)
- **MEMM** 结合了马尔可夫链和对数线性模型，用于序列标注问题，例如词性标注。它假设当前状态只依赖于前一个状态，但可以看到整个输入序列。
- **特征函数**用于定义观察到的单词及其相邻的标记的关系，用于模型训练和预测【157†source】。

### 8. Feed-Forward Neural Networks and Multilayer Neural Nets (前馈神经网络与多层神经网络)
- **前馈神经网络**（没有反馈连接）是神经网络中最基本的类型，包括输入层、隐藏层和输出层。
- **多层神经网络 (MLP)** 可以通过多个隐藏层进行计算，能学习非线性关系。理论上，两个隐藏层就足以表示任何函数，尽管需要足够多的神经元【158†source】。
- **激活函数**：包括 **sigmoid**、**ReLU** 和 **tanh**，这些函数用于引入非线性，使网络可以学习复杂的映射关系【158†source】。

### 9. Computation Graphs and Backpropagation (计算图与反向传播)
- **计算图**是一种用于表示神经网络中计算过程的有向无环图，方便追踪数据流和计算导数。
- **反向传播算法**用于训练神经网络，通过链式法则计算每个权重对最终误差的贡献，从输出层向前逐层计算梯度。
- **反向传播的基本思路**：通过计算误差的偏导数来更新网络中的权重，以最小化损失函数【158†source】。

### 10. Softmax and Regularization (Softmax 与正则化)
- **Softmax 函数**通常用于多类分类问题的输出层，将每个类的分数标准化为概率（所有类别的概率之和为 1）【158†source】。
- **正则化 (Regularization)**：为了避免过拟合，通常会引入正则化项（如 **L2 正则化**），来约束权重的大小，防止参数过大导致模型复杂度过高【157†source】。

### 11. Word Embeddings (词嵌入)
- **词嵌入 (Word Embeddings)** 是用于将单词映射到低维向量空间的技术，例如 **Word2Vec**。通过这样的映射，模型可以学习到单词之间的语义关系，比如“king - man + woman ≈ queen”。
- **Skip-Gram 和 CBOW 模型**：Skip-Gram 通过预测上下文词来训练嵌入，而 CBOW 则是通过上下文预测目标词。二者都使用**负采样 (Negative Sampling)** 或**层次 Softmax** 来提高训练效率【159†source】。

希望这些解释能够帮助您理解考试中的相关考点。如果有任何不清楚的地方或者需要进一步的具体例子，请告诉我，我会更详细地为您解释！
