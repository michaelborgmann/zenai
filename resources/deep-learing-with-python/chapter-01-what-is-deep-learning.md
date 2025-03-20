# Chapter 1: What is deep learning?

Deep learning, a subset of machine learning, has driven major AI advancements, powering chatbots, self-driving cars, and virtual assistants. While often surrounded by media hype, its impact is undeniable.

At its core, deep learning uses artificial neural networks to recognize patterns and learn from data, excelling in tasks like image recognition and natural language processing. Though its potential is vast, understanding its real capabilities versus exaggerated claims is crucial.

This chapter provides essential context on AI, machine learning, and deep learning, exploring its achievements, significance, and future direction.

---

## 1.1 Artificial intelligence, machine learning, and deep learning

Artificial intelligence (AI) is a broad field focused on creating machines that can perform tasks requiring human intelligence. Machine learning (ML) is a subset of AI that enables systems to learn from data without explicit programming. Deep learning (DL), in turn, is a specialized branch of ML that uses neural networks to model complex patterns and make predictions.

Understanding these distinctions is key to grasping AI’s capabilities and evolution.

### Artificial Intelligence

Artificial Intelligence (AI) emerged in the 1950s, with pioneers in computer science questioning whether machines could replicate human thinking. The field officially began in 1956 at a Dartmouth workshop led by John McCarthy, where researchers proposed that machines could simulate human intelligence by using language, solving problems, and improving themselves. Though the workshop didn’t solve these challenges, it set in motion a lasting intellectual revolution.

AI aims to automate tasks typically performed by humans. It encompasses various approaches, including machine learning and deep learning, but also extends to areas not involving learning. Initially, AI didn’t focus on "learning" until the 1980s. Early AI, like chess programs, relied on manually crafted rules and databases (symbolic AI). However, symbolic AI struggled with complex tasks like image classification or speech recognition, which led to the rise of machine learning as a more flexible solution to tackle these challenges.

### Machine Learning

Machine learning (ML) emerged as a significant subfield of AI in the 1990s, driven by advances in hardware and the availability of large datasets. It revolves around systems that "learn" from examples rather than being explicitly programmed. A machine learning system is trained with input data and corresponding outcomes, enabling it to derive rules for automating tasks. For example, when automating photo tagging, the system learns statistical patterns by analyzing labeled images.

Unlike traditional statistical methods, which are suitable for smaller, simpler datasets, machine learning excels with large, complex datasets (e.g., millions of high-resolution images). It’s an engineering-driven field, deeply reliant on empirical results and hardware improvements. ML, especially deep learning, involves less mathematical theory compared to other disciplines like physics or chemistry, emphasizing practical applications and rapid advancements.

### Learning Rules and Representations from Data

Machine learning models learn to transform input data into meaningful outputs based on known examples. This requires three elements: input data (e.g., speech files or images), expected outputs (e.g., transcriptions or image tags), and a way to measure accuracy for feedback, which guides learning.

The core challenge in machine learning, including deep learning, is transforming data into useful representations that lead to correct outputs. A "representation" is a different way of encoding data to make a task easier. For example, representing images in RGB or HSV formats helps simplify tasks like identifying colors or adjusting saturation.

In simpler tasks, like classifying points on a 2D plane, a coordinate change can effectively separate categories (e.g., black and white points). However, for more complex tasks, such as classifying handwritten digits, manually designing data transformations becomes difficult and brittle. Here, machine learning automates this process by searching for useful representations and rules based on feedback signals, making it far more efficient than manual rule-writing.

In essence, machine learning is about automating the search for effective representations and rules to solve diverse tasks, from image tagging to autonomous driving.

### The "Deep" in "Deep Learning"

Deep learning is a subfield of machine learning that focuses on learning data representations through multiple layers, each providing increasingly meaningful features. The term "deep" refers to the number of layers in a model, which can range from tens to hundreds. This approach contrasts with traditional machine learning, where typically only one or two layers of data representations are learned, hence the term "shallow learning."

Deep learning uses neural networks—models with layers stacked on top of each other—to learn these representations. Despite being inspired by concepts from neurobiology, particularly the visual cortex, deep learning is not a model of the brain. The learning mechanisms of deep learning models are distinct and not directly analogous to brain processes. It's best to view deep learning as a mathematical framework for transforming data representations.

In practice, deep learning models transform input data (e.g., images) through successive layers, progressively distilling information that becomes more useful for solving specific tasks, such as recognizing digits in images. While the idea is simple, the power of deep learning lies in scaling these mechanisms, which can seem almost magical in their effectiveness.

### Understanding How Deep Learning Works

Deep learning involves mapping inputs (e.g., images) to targets (e.g., labels) by learning transformations through layers in a neural network. These transformations are defined by parameters called weights, which are learned by exposure to examples. A deep network may contain millions of parameters, making it challenging to find the correct values for all of them, as changing one weight impacts the entire model.

The learning process is driven by the **loss function**, which measures the difference between the network's predictions and the actual targets. This score is used to adjust the weights to reduce the loss. The key mechanism for this adjustment is the **Backpropagation algorithm**, facilitated by an optimizer. Initially, weights are random, leading to poor performance, but with each training iteration, the weights are adjusted to minimize the loss. After many iterations over large datasets, the network becomes trained, with output closely matching the desired targets. This training process is simple in concept, but its large-scale implementation produces powerful results.

### The Age of Generative AI

Generative AI, exemplified by applications like ChatGPT, Gemini, and Claude for text, and Midjourney for images, showcases the power of foundation models. These models generate creative and informative content based on simple prompts, blurring the line between human and machine creativity.

At the core of generative AI are foundation models, which are trained through **self-supervised learning**. They reconstruct content (such as text or images) by learning from vast, unlabeled datasets—sometimes exceeding one petabyte of data. This method removes the need for manual annotations, greatly scaling model training. Some of these models have hundreds of billions of parameters and cost tens of millions to develop.

These models function as broad knowledge repositories and can be applied to diverse tasks without requiring specialized retraining. By querying the vast amounts of information they’ve memorized, these models can solve new problems with minimal input.

Though generative AI became widely recognized in 2022, its roots go back to the 1990s. The first edition of this book, published in 2017, already discussed generative AI techniques and predicted the eventual integration of AI into cultural content creation.

### What Deep Learning Has Achieved So Far

Deep learning has sparked a technological revolution over the last decade. Initially, from 2013 to 2017, it made significant advances in perceptual tasks, followed by rapid progress in natural language processing from 2017 to 2022. From 2022 onwards, the field has been dominated by transformative generative AI applications.

Key breakthroughs include:

* **Generative AI applications** like ChatGPT, Gemini, and GitHub Copilot
* **Photorealistic image generation** and human-level image classification
* **Human-level speech and handwriting transcription**
* **Improved machine translation** and text-to-speech conversion
* **Human-level autonomous driving**, deployed in cities like Phoenix and San Francisco (2024)
* **Superhuman performance** in games like Go, Chess, and Poker
* **Enhanced recommender systems**, such as those used by YouTube, Netflix, and Spotify

Deep learning has also found success in solving problems once thought unsolvable, such as transcribing ancient manuscripts, detecting plant diseases with smartphones, assisting medical professionals with imaging, and predicting natural disasters. With every new achievement, deep learning is increasingly integrated across sectors like science, medicine, energy, transportation, software development, agriculture, and even the arts.

### Beware of the Short-Term Hype

While the recent successes in AI have been impressive, they have sparked excessive hype, especially regarding the future potential of AI in the short term. For instance, after the release of GPT-4 in 2023, some predictions suggested mass unemployment and a massive productivity surge. However, by 2025, these claims have proven to be overstated: unemployment remains low and productivity growth has not seen the dramatic increases that were promised.

Generative AI has undoubtedly made a significant impact, generating billions of dollars in revenue annually, but its influence remains limited compared to the grand predictions. More sensational claims involve the imminent rise of Artificial General Intelligence (AGI) or even "superintelligence" that surpasses human capabilities, causing widespread concern. These fears of AGI were fueled by early predictions from tech elites, notably linked to companies like DeepMind and OpenAI, who once believed AGI was just a few years away.

However, the current AI technologies, including generative models, are still far from achieving AGI. Today’s AI is best described as **cognitive automation**—it excels at solving specific, narrowly defined problems, not replicating human intelligence. Despite advances, AI still lacks the adaptability and learning capacity required for true cognitive autonomy. AI is a powerful tool, but it is not self-aware or sentient, and it’s unlikely to be in the foreseeable future.

In summary, while AI’s progress is remarkable, the hype surrounding its potential—especially regarding AGI—is best approached with skepticism, as AI remains fundamentally different from human-like intelligence.

Summer Can Turn to Winter
Inflated short-term expectations of AI can lead to significant setbacks if the technology inevitably falls short, resulting in reduced investment and slower progress. This cycle of optimism followed by disappointment has happened twice before.

In the 1960s and 70s, symbolic AI experienced high expectations, with predictions of imminent human-level AI, only to be followed by disillusionment and the first AI winter. A similar cycle occurred in the 1980s with expert systems, where initial successes led to large investments, but by the early 90s, these systems were costly and difficult to scale, leading to the second AI winter.

Currently, while AI is in a phase of intense optimism, it's unlikely to face a full retreat like in the 1990s. Instead, a "mild winter" could be on the horizon, as AI investments (over $100 billion annually) far exceed its current revenue generation (around $10 billion). Much of the hype about AI's future potential may not materialize soon, leading to adjustments in expectations, though the exact nature of these changes remains uncertain.

### Summer Can Turn to Winter

Inflated short-term expectations of AI can lead to significant setbacks if the technology inevitably falls short, resulting in reduced investment and slower progress. This cycle of optimism followed by disappointment has happened twice before.

In the 1960s and 70s, symbolic AI experienced high expectations, with predictions of imminent human-level AI, only to be followed by disillusionment and the first AI winter. A similar cycle occurred in the 1980s with expert systems, where initial successes led to large investments, but by the early 90s, these systems were costly and difficult to scale, leading to the second AI winter.

Currently, while AI is in a phase of intense optimism, it's unlikely to face a full retreat like in the 1990s. Instead, a "mild winter" could be on the horizon, as AI investments (over $100 billion annually) far exceed its current revenue generation (around $10 billion). Much of the hype about AI's future potential may not materialize soon, leading to adjustments in expectations, though the exact nature of these changes remains uncertain.

### The Promise of AI

While short-term expectations of AI may be overly optimistic, the long-term potential remains incredibly promising. We are still in the early stages of applying deep learning to a wide range of transformative problems, from medical diagnoses to digital assistants. Despite rapid progress in AI research, the deployment of these advancements is still limited.

As of 2024, many predictions about AI's future are already coming true: AI chatbots like ChatGPT and Gemini are daily assistants for millions, answering questions and helping with education. Autonomous vehicles, such as Waymo cars, are operational in cities like San Francisco and Phoenix. AI is also making substantial strides in science, with breakthroughs like AlphaFold, which aids biologists in predicting protein structures, and AI's potential role in mathematical research by 2026.

While setbacks may occur, such as an AI winter akin to the early internet industry's crash, AI's integration into daily life will only continue to grow. Like the internet, AI will eventually become central to almost every process in society, enhancing industries from healthcare to education and scientific research.

In conclusion, while short-term hype should be tempered, the long-term vision for AI's transformative potential is real and unfolding—though its full impact has yet to be fully realized.

## 1.2 Before Deep Learning: A Brief History of Machine Learning

Deep learning has gained significant attention, but it’s not the first successful machine learning approach. Most industry-used algorithms aren't deep learning-based. Deep learning isn't always the best tool, especially when data is limited or other algorithms are more suitable. It's important to understand classical machine learning methods to avoid over-relying on deep learning and to know when other techniques are more effective. This section offers a brief historical context, helping to place deep learning within the broader field of machine learning.

### Probabilistic Modeling

Probabilistic modeling applies statistical principles to data analysis and remains a foundational approach in machine learning. One well-known algorithm is **Naive Bayes**, a classifier based on Bayes' theorem that assumes input features are independent—an assumption that simplifies calculations. Though it predates modern computing, Naive Bayes continues to be widely used. Another classic model is **logistic regression** (logreg), a versatile classification algorithm often used as an introductory tool for data scientists. Despite its name, it’s used for classification, not regression, and remains a go-to method due to its simplicity and effectiveness.

### Early Neural Networks

Early neural networks, although now overshadowed by modern deep learning techniques, laid the foundation for current advancements. While neural networks were explored in the 1950s, they lacked an efficient training method. In the mid-1980s, the **Backpropagation** algorithm was rediscovered, enabling effective training of neural networks using gradient descent optimization.

A key milestone came in 1989 when **Yann LeCun** developed **LeNet**, a convolutional neural network (CNN) trained with backpropagation to classify handwritten digits. LeNet was later used by the U.S. Postal Service in the 1990s to automate ZIP code reading.

### Kernel Methods

In the 1990s, **kernel methods**, particularly **Support Vector Machines (SVMs)**, rose to prominence, briefly overshadowing neural networks. Developed by **Vladimir Vapnik** and **Corinna Cortes** in the early 1990s, SVMs excel at classifying data by finding decision boundaries (hyperplanes) that maximize the margin between classes. This is achieved through two steps: mapping data to a high-dimensional space and calculating a hyperplane that separates classes effectively.

The **kernel trick** allows SVMs to avoid directly computing high-dimensional representations, instead using kernel functions to calculate distances between points efficiently.

While SVMs were once state-of-the-art for simple classification tasks and had strong theoretical backing, they struggled with scaling to large datasets and performing well on perceptual problems (e.g., image classification). These issues arose because SVMs require manual **feature engineering**, making them less effective for raw data inputs like images.

### Decision Trees, Random Forests, and Gradient Boosting Machines

Decision trees are flowchart-like structures used for classification and prediction. They became popular in the 2000s for their simplicity and interpretability.

The Random Forest algorithm enhances decision trees by creating a large number of specialized trees and combining their outputs, making it a robust and versatile method. By 2010, random forests were widely used, particularly in machine learning competitions like Kaggle.

In 2014, Gradient Boosting Machines (GBMs), an ensemble method that improves weak models iteratively, surpassed random forests in performance. GBMs, when applied to decision trees, often outperform random forests, especially for non-perceptual tasks. They remain a top choice for structured data alongside deep learning, frequently appearing in Kaggle competitions.

### Back to Neural Networks

Around 2010, neural networks regained attention due to breakthroughs by researchers like Geoffrey Hinton, Yoshua Bengio, Yann LeCun, and others.

The first practical success of modern deep learning came in 2011 when Dan Ciresan’s GPU-trained neural networks won image-classification competitions. The pivotal moment occurred in 2012 when Hinton’s team achieved a top-five accuracy of **83.6%** in the **ImageNet** competition, significantly outperforming the previous year's winner (74.3%). By 2015, deep convolutional neural networks (CNNs) dominated the competition, reaching an accuracy of **96.4%**, solving the ImageNet classification task.

Since then, CNNs have become the standard for **computer vision** and **perceptual tasks**, gradually replacing classical methods like SVMs and decision trees. In fact, organizations like CERN transitioned from decision trees to deep neural networks (via Keras) for tasks like particle data analysis due to superior performance and ease of use with large datasets.

### What Makes Deep Learning Different

Deep learning has rapidly surpassed traditional machine learning due to two key advantages: superior performance and simplification of the problem-solving process.

1. **Automating Feature Engineering**: Unlike shallow learning methods (e.g., SVMs, decision trees), which required manual **feature engineering** — transforming input data into useful representations—deep learning automates this entire process. Deep learning models learn all features in one pass, eliminating the need for human intervention in creating representations.
2. **Layer-by-Layer Learning**: Deep learning models build complex representations incrementally, learning multiple layers of features jointly, rather than in succession. This joint learning allows for more sophisticated and abstract representations, with each layer adapting based on feedback from the entire network, making deep learning much more powerful than stacking shallow models.

These two factors—automated feature learning and joint, layer-by-layer training—have made deep learning significantly more successful and efficient compared to earlier machine learning techniques.

### The Modern Machine Learning Landscape

Machine learning today is largely dominated by **deep learning** and **gradient boosted trees**, as seen in platforms like Kaggle.

1. **Kaggle Competitions**: In 2019, a survey of top Kaggle teams revealed that winning solutions commonly used deep learning (via Keras) or gradient boosted trees (via LightGBM or XGBoost). These tools perform well in both structured data and perceptual tasks like image classification.
2. **Industry Trends**: Kaggle’s annual survey of data science professionals shows that from 2016 to 2020, deep learning and gradient boosting were the leading approaches. **Deep learning** is preferred for perceptual problems (e.g., image classification), while **gradient boosted trees** are best for structured data.
3. **Key Tools**: Popular machine learning libraries include:
	* **Gradient Boosted Trees**: Scikit-learn, XGBoost, LightGBM
	* **Deep Learning**: Keras (often with TensorFlow)

Python remains the dominant language in the field, and mastering these tools (Scikit-learn, XGBoost, Keras) is crucial for success in modern applied machine learning.

## 1.3 Why Deep Learning? Why Now?

Although deep learning algorithms like convolutional neural networks (CNNs) for computer vision and Long Short-Term Memory (LSTM) for time series were developed in the 1990s, deep learning only gained significant traction after 2012. Three key factors explain this:

1. **Hardware**: The rise of **high-performance graphics chips** (GPUs), originally designed for gaming, became essential for training large-scale deep learning models. GPUs enable faster and more efficient processing of data, a crucial factor for scaling deep learning algorithms.
2. **Datasets and Benchmarks**: The growth of **large-scale datasets** and established benchmarks, like ImageNet for computer vision, provided standardized ways to evaluate model performance. This helped accelerate progress and made comparisons easier.
3. **Algorithmic Advances**: Though algorithmic advancements were made, deep learning only thrived when there was enough data and hardware to implement and scale them effectively. Machine learning is more experimental than theoretical, requiring the right tools and infrastructure to see breakthroughs.
Thus, the combination of **data availability**, **advanced hardware**, and **improved algorithms** around 2012 fueled the explosion of deep learning's success.

### Hardware

Between 1990 and 2010, CPUs improved significantly, becoming 5,000 times faster, allowing small deep learning models to run on laptops. However, more complex models, such as those used in computer vision and speech recognition, require far greater computational power.

In the 2000s, companies like NVIDIA and AMD invested heavily in GPUs to enhance gaming graphics. These GPUs, designed to render complex 3D scenes in real time, proved highly useful for deep learning, particularly due to their parallel processing capabilities. In 2007, NVIDIA introduced **CUDA**, a programming interface that allowed GPUs to be used for general-purpose computing, including deep neural networks. Researchers like Dan Ciresan and Alex Krizhevsky were among the first to leverage CUDA for deep learning in 2011.

By 2019, GPUs like the **NVIDIA Titan RTX** delivered 16 teraFLOPS of computing power, about 500 times the computing power of the world’s fastest supercomputer in 1990. These advances allow deep learning models to be trained much faster, with tasks like training ImageNet models now taking only hours instead of weeks.

Beyond GPUs, the deep learning industry has shifted towards more specialized chips. **Google's Tensor Processing Unit** (TPU), introduced in 2016, is designed to accelerate deep learning further. The **third generation of TPUs** delivers 420 teraFLOPS of power, 10,000 times more than the 1990 supercomputer. TPU configurations, called "pods," can scale to 100 petaFLOPS, rivaling the power of current top supercomputers.

### Data

Data is crucial to the success of deep learning, often described as the "coal" that powers the AI revolution. Progress in storage hardware over the past 20 years, coupled with the rise of the internet, has enabled the collection and distribution of vast datasets essential for machine learning. Large companies now work with extensive image, video, and natural language datasets that would have been impossible to compile without the internet. For example, user-generated tags on platforms like **Flickr** and **YouTube videos** have provided valuable data for computer vision, while **Wikipedia** serves as a key resource for natural language processing.

The **ImageNet** dataset, consisting of 1.4 million images annotated with 1,000 categories, has been pivotal in deep learning's success. ImageNet's annual competition, the **ImageNet Large Scale Visual Recognition Challeng (ILSVRC)**, has driven significant advancements in the field. Competitions like these, also seen on **Kaggle**, serve as benchmarks, motivating researchers to push the boundaries of machine learning and demonstrating the superiority of deep learning over classical approaches.

###  Algorithms

Until the late 2000s, training deep neural networks was challenging, as neural networks were often shallow (with just one or two layers), limiting their ability to compete with refined shallow methods like SVMs and random forests. The issue was **gradient propagation** — the feedback signal needed for training would fade in deeper layers. This problem was addressed around **2009-2010** with several key algorithmic improvements:

* **Better activation functions**
* **Improved weight initialization**, like layer-wise pretraining (though it was later abandoned)
* **Optimized schemes** such as RMSProp and Adam

These innovations enabled the training of deeper models, with **10 or more layers**. Later advancements in **2014-2016**, including **batch normalization**, **residual connections**, and **depthwise separable convolutions**, further improved gradient propagation.

Today, deep learning models can be trained with arbitrary depth, unlocking the use of **extremely large models** with millions of parameters. This scalability has been critical in driving progress in **computer vision** (e.g., **ResNet**, **Inception**, **Xception**) and **natural language processing** (e.g., **BERT**, **GPT-3**, **XLNet**), showcasing the significant power of deep architectures in modern AI.

### A New Wave of Investment

As deep learning became the leading approach for computer vision and perceptual tasks (starting around 2012–2013), it sparked a wave of industry investment that revolutionized AI development.

* **2011**: Total global venture capital investment in AI was under $1 billion, mostly for shallow machine learning applications.
* **2015**: Investment grew to over $5 billion.
* **2017**: It surged to $16 billion, with hundreds of startups launching to capitalize on deep learning advancements.

Tech giants like **Google**, **Amazon**, and **Microsoft** also invested heavily in deep learning research and product integration, with Google CEO **Sundar Pichai** emphasizing machine learning as a core element of their strategy across products.

This influx of investment rapidly expanded the deep learning field, growing the number of professionals working in AI from a few hundred to tens of thousands, accelerating research progress at an unprecedented pace.

### The Democratization of Deep Learning

The democratization of deep learning tools has been key to attracting new talent to the field. In the past, deep learning required specialized knowledge of **C++** and **CUDA**, which limited accessibility. Today, **basic Python scripting** skills are enough to dive into advanced deep learning, thanks to:

* **Theano** and **TensorFlow**: Both symbolic tensor-manipulation frameworks with autodifferentiation capabilities that simplify model implementation.
* **Keras**: A user-friendly library released in 2015, making deep learning as accessible as assembling LEGO bricks.

Keras, in particular, became the go-to tool for many new startups, researchers, and graduate students, driving rapid growth in the deep learning community.

### Will It Last?

Deep learning is not a passing trend—it's here to stay due to several key properties that ensure its long-term relevance:

* **Simplicity**: Deep learning removes the need for manual feature engineering, offering **simple**, **end-to-end** models that can be trained with minimal complexity.
* **Scalability**: It's highly parallelizable on GPUs and TPUs, leveraging **Moore's law** and allowing models to scale on vast datasets.
* **Versatility & Reusability**: Models can be reused for various tasks (e.g., taking an image classification model and applying it to video). They can also be retrained on additional data without starting from scratch, facilitating **continuous learning**.

Deep learning has been transformative, particularly with the rise of **Transformer models** for natural language processing, and it continues to progress. While it’s currently in the second phase of its **sigmoid curve** (following a period of explosive growth), deep learning's potential is still unfolding, with more improvements and applications ahead.