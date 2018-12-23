# Final Exam Topics

### Final Topics

The final exam will take place on Tuesday Dec 18, 7:10pm-9:10pm (120min) in 451 Computer Science (regular classroom). 

Please note that the final exam is cumulative, so any topics from before the midterm are fair game. In particular, pay attention to topics that were _not_ tested on the midterm. 

While only the content in the uploaded lecture notes is expected for the final, textbook references and links to other background material are provided for each of the topics.

(*) indicates that you should be familiar with the basic concept, but details will not be tested on the exam.

**General Linguistics Concepts **(Parts of J&M 2nd ed. Ch 1 but mostly split over different chapters in J&M. Ch.1 not yet available in 3rd ed.)

- Levels of linguistic representation: phonetics/phonology, morphology, syntax, semantics, pragmatics
- Ambiguity, know some examples in syntax and semantics, including PP attachment, noun-noun compounds. 
- Garden-path sentences. 
- Type/Token distinction.
- Know the following terms: sentence, utterance, word form, stem, lemma, lexeme.
- Parts of speech:
  - know the 9 traditional POS and some of the Penn Treebank tags
- Types of Linguistic Theories (Prescriptive, Descriptive, Explanatory)
- Syntax: 
  - Constituency and Recursion. Constituency tests. 
  - Dependency.
  - Grammatical Relations.
  - Subcategorization / Valency (and relationship to semantic roles).
  - Long-distance dependencies.
  - Syntactic heads (connection between dependency and constituency structure).
  - Center embeddings.
  - Dependency syntax: 
    - Head, dependent
    - dependency relations and labels
    - projectivity
  - Agreement.

**Text Processing **(Split over different chapters in J&M. Parts of [J&M 3rd ed. Ch. 6 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/6.pdf))

- Tokenization (word segmentation).
- Sentence splitting. 
- Lemmatization. 
- Know why these are useful and challenging. 

**Probability Background **

- Prior vs. conditional probability.
- Sample space, basic outcomes
- Probability distribution
- Events
- Random variables
- Bayes' rule
- conditional independence
- discriminative vs. generative models
- Noisy channel model. 
- Calculating with probabilities in log space.

**Text Classification (**[J&M 3rd ed. Ch. 6 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/6.pdf)**)**

- Task definition and applications
- Document representation: Set/Bag-of-words, vector space model
- Naive Bayes' and independence assumptions.

**Language Models (**J&M 2nd ed. Ch 4.1-4.8, [J&M 3rd ed. Ch 4 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/4.pdf)**)**

- Task definition and applications

- Probability of the next word vs. probability of a sentence

- Markov independence assumption. 

- n-gram language models. 

- Role of the END marker. 

- Estimating ngram probabilities from a corpus: 

  - Maximum Likelihood Estimates

  - Dealing with Unseen Tokens

  - Smoothing and Back-off: 

    ​

    - Additive Smoothing
    - Discounting
    - Linear Interpolation
    - Katz' Backoff

- Perplexity

**Sequence Labeling (POS tagging) (**J&M 2nd ed Ch 5.1-5.5, [J&M 3rd ed. Ch 10.1-10.4 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/10.pdf)**)**

- Linguistic tests for part of speech.
- Hidden Markov Model: 
  - Observations (sequence of tokens)
  - Hidden states (sequence of part of speech tags)
  - Transition probabilities, emission probabilities
  - Markov chain
  - Three tasks on HMMS: Decoding, Evaluation, Training
    - Decoding: Find the most likely sequence of tags:
      - Viterbi algorithm (dynamic programming, know the algorithm and data structures involved)
    - Evaluation: Find the probability of a sequence of words
      - Spurious ambiguity: multiple hidden sequences lead to the same observation. 
      - Forward algorithm (difference to Viterbi).
    - Training: We only discussed maximum likelihood estimates. There are unsupervised techniques as well.  (*)
  - Extending HMMs to trigrams.
- Applying HMMs to other sequence labeling tasks, for example Named Entity Recognition
  - B.I.O. tags for NER. 

**Parsing with Context Free Grammars (**J&M 2nd ed Ch. 12 and 13.1-13.4** **and 14.1-14.4 and Ch. 16, 
​                                                                  [J&M 3rd ed. Ch. 11.1-11.5 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/11.pdf) and [Ch. 12.1-12.2 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/12.pdf) [Earley not covered in 3rd. ed] and [Ch 13.1-13.4 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/13.pdf), [complexity classes not covered in 3rd ed.] )

- Tree representation of constituency structure. Phrase labels.
- CFG definition: terminals, nonterminals, start symbol, productions
- derivations and language of a CFG.
- derivation trees (vs. derived string). 
- Regular grammars (and know that these are equivalent to finite state machines and regular expressions - you don't need to know FSAs and regular expression in detail for the midterm).  (*)
- Complexity classes. 
  - "Chomsky hierarchy"
  - Center embeddings as an example of a non-regular phenomenon. 
  - Cross-serial dependencies as an example of a non-context-free phenomenon.
- Probabilitistic context free grammars (PCFG): 
  - Maximum likelihood estimates. 
  - Treebanks. 
- Recognition (membership checking) problem vs. parsing
- Top-down vs. bottom-up parsing. 
- CKY parser:
  - bottom-up approach. 
  - Chomsky normal form. 
  - Dynamic programming algorithm. (know the algorithm and required data structure: CKY parse table). Split position. 
  - Backpointers.
  - Parsing with PCFGs (compute tree probabilities or sentence probability)
- Earley parser:
  - Top-down approach. 
  - Does not require CNF. 
  - Parser state definition. Initial state, goal states.
  - Three operations: Scan, Predict, Complete
  - Dynamic programming algorithm. (know the algorithm and required data structure: parse "Chart" organized by end-position).

**Other Grammar Formalisms **

- Unification Grammar  (J&M 2nd. ed Ch. 15.1-15.3, not covered in 3rd ed.) (*)

  - Feature structures (as Attribute Value Matrix or DAG)
  - Reentrancy in feature structures
  - Unification
  - Unification constraints on grammar rules
  - Know how these are used to enforce agreement

- Lexicalized tree adjoining grammars (not in J&M, supplementary material:

   

  Abeillé & Rambow (2000): "Tree Adjoining Grammar: An Overview", in Abeillé & 

  [Rambow, "Tree Adjoining Grammars", CSLI, U Chicago Press](http://www.cs.columbia.edu/~rambow/papers/intro-only.pdf)) (*)

  ​

  - Two types of elementary trees: initial trees and auxiliary trees. 
  - Substitution nodes. 
  - Foot nodes in auxiliary trees. 
  - Adjunction. 
  - Derived tree vs. derivation tree
  - Know that TAG is more expressive than CFG in the complexity hierarchy. 

- Combinatory Categorial Grammar (CCG)

  - Categories with / and \ 
  - Forward and backward and backward application. 
  - Combinators: forward/backward composition, type raising (*)
  - Relationship to lambda calculus. 

- Hyperedge Replacement Grammars (for AMR graphs)  (*)

**Dependency parsing (**Not in J&M 2nd ed., [J&M 3rd ed. Ch 14.1-14.5 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/14.pdf), 
​                                     Supplementary material: [Küber, McDonald, and Nivre (2009): Dependency Parsing, Ch.1 to 4.2 [ebook available through CU library\]](https://clio.columbia.edu/catalog/7851052))

- Grammar based vs. data based. 
- Data based approaches: 
  - Graph algorithms. vs. transition based 
- Transition based dependency parsing: 
  - States (configuration): Stack, Buffer, Partial dependency tree
  - Transitions (Arc-standard system): Shift, Left-Arc, Right-Arc
  - Predicting the next transition using discriminative classifiers. 
    - Feature definition (address + attribute)
  - Training the parser from a treebank: 
    - Oracle transitions from annotated dependency tree. 
  - Difference between arc-standard and arc-eager. 
- Graph based approaches (*) (only need to be familiar with the basic concepts - no details):
  - Edge-factored model. 
  - Compute Maximum Spanning Tree on completely connected graph of all words. 
  - Can be done using the Chu-Liu-Edmonds algorithm (not covered in detail). 

**Machine Learning **(Some textbook references below. Also take a look at Michael Collins' detailed notes on a variety of topics: <http://www.cs.columbia.edu/~mcollins/notes-spring2018.html>)

- Generative vs. discriminative algorithms 

- Supervised learning. Classification vs. regression problems. 

- Loss functions: Least squares error. Classification error. 

- Training vs. testing error. Overfitting and how to prevent it. 

- Linear Models. 

  - activation function. 
  - perceptron learning algorithm (* i will not ask you to do this on the final)
  - linear separability and the XOR problem. 

- Feature functions

- Log-linear / maximum entropy models  (J&M 3rd. ed. ch. 7, J&M 2nd ed. ch. 6.6)

  - Log-likelihood of the model on the training data. 
  - Simple gradient ascent. 
  - Regularization
  - MEMM (Maximum entropy markov models): 
    - POS tagging with MEMMs.

- Feed-forward neural nets  (J&M 3rd. ed. ch. 8, also take a look at

   

  Yoav Goldberg's book "Neural Network Methods for Natural Language Processing" (Links to an external site.)Links to an external site.

   

  [available as PDF if you are on the Columbia network] )

  - Multilayer neural nets.
  - Different activation functions (sigmoid, ReLU, tanh)
  - Softmax activation.  
  - Input representation options: 
    - one-hot representation for features, word embeddings, feature function value. 
  - Output representation options: 
    - Single score. 
    - Probability distribution (softmax) 
  - Backpropagation (*,  know the idea of propagating back partial derivatives of the loss with respect to each weight, but you don't have to understand the details).

**Formal Lexical Semantics** (J&M 3rd ed. ch 17, J&M 2nd ed. ch 19.1-19.3 and 20.1-20.4 )

- Word senses, lexemes, homonymy, polysemy, metonymy, zeugma. 
- WordNet, synsets
- lexical relations (in WordNet)
  - synonym, antonym
  - hypernym, hyponym
  - meronym
- Word-sense disambiguation: 
  - Supervised learning approach and useful features. 
  - Lesk algorithm (J&M 3rd ed. ch. 17.6)
  - Bootstrapping approach (*) (don't have to know the details, J&M 3rd ed. ch.  17.8). 
- Lexical substitution task

**Distributional (Vector-based) Lexical Semantics** (J&M 3rd ed. ch 15 & 16, not in 2nd ed.)

- Distributional hypothesis
- Co-occurence matrix
- Distance/Similarity metrics (euclidean distance, cosine similarity)
- Dimensions (parameters)of Distributional Semantic Models
  - Preprocessing, term definition, context definition, feature weighting, normalization, dimensionality reduction, similarity/distance measure
- Semantic similarity and relatedness (paradigmatic vs. syntagmatic relatedness)
  - Effect of context size on type of relatedness.
- Sparse vs. Dense Vectors
- One-hot representation (of words, word-senses, features)
- Word Embeddings 
- Word2Vec embeddings using a neural network. 
  - Skip-gram model 
  - CBOW (*)

**Semantic Role Labeling **(J&M Ch. 22)

- Frame Semantics 
  - Frame, Frame Elements (specific to each frame)
  - Valence Patterns
  - FrameNet:
    - Frame definitions
    - Lexical Units
    - Example annotations to illustrate valence patterns.
    - Frame-to-frame relations and frame-element relations (*) (you do not need to remember an exhaustive list of these relations). 
  - PropBank: 
    - Differences to FrameNet.
    - Semantic Roles (ARGx)
    - predicate-argument structure.
    - framesets
  - Semantic Role Labeling:
    - Steps of the general syntax-based approach
      - Target identification, semantic role/frame element identification and labeling. 
    - Features for semantic role role/FE identification and labeling. (*) (you do not need to remember an exhaustive list, but have a general sense of which features are important).
      - Selectional restrictions and preferences. 
      - Parse-tree path features

**Semantic Parsing (full-sentence semantic analysis) **(J&M 2nd ed. Ch 17, not in 3rd ed.)

- Goals for meaning representations (unambiguous, canonical form, supports inference, expressiveness).
- First-order logic (aka Predicate Logic):
  - Syntax of FOL: Constants, Functions, Terms, Quantifiers, Variables, Connectives
  - Semantics for FOL: Model-theoretic sematnics. 
  - Event logical (Neo-Davidsonian) representation.
- Semantic analysis with first-order logic: 
  - Principle of compositionality (and examples for non-compositional phenonema)
  - Lambda expressions.
  - Function application. 
  - Higher-order functions.
  - Types. 
- Categorical Grammar and Combinatory Categorial Grammar (CCG)  (*)

**Abstract Meaning Representation **(not in textbook, planned for 3rd edition)

- meaning of vertices (entities) and edges (relations) (* you do not have to remember specific special relations and concepts in the AMR annotation guidelines). 
- reentrancy
- ARG-of edges (inverse relations)
- constants
- relation to event logic (*)
- AMR parsing (*)
  - JAMR approach (*)
  - Hyperedge Replacement Grammar (HRG) approach (*)

**Machine Translation (MT****) **(J&M 2nd ed. Ch 25)

- Challenges for MT, word order, lexical divergence
- Vauquois triangle.
- Faithfulness vs. Fluency

**Statistical MT** (J&M 2nd ed. Ch, also see Michael Collins' notes on IBM M2 here: <http://www.cs.columbia.edu/~mcollins/ibm12.pdf>)

- Parallel Corpora


- Noisy Channel Model for MT
- Word alignments
- IBM Model 2: 
  - Alignment variables and model definition. 
  - EM training for Model 2 
- Phrase-based MT (*)
- MT Evaluation: 
  - BLEU Score 

**Recurrent Neural Nets**

- Recurrent neural nets. 
  - Neural language model (without RNNs)  
  - Basic RNN concept, hidden layer as state representation.
  - Common usage patterns: 
    - Acceptor / Encoder
    - Transducer
      - Transducer as generator 
      - Conditioned transduction (encoder-decoder)
  - Backpropagtion through time (BPTT) (*)
  - LSTMs (*) 
- Neural MT
  - Attention Mechanisms (*)