# Cover Letter Draft for Neurocomputing

Dear Editor,

Please consider our manuscript, **"When Do Dendritic Artificial Neural Networks Help? A Controlled Benchmark Study with Parameter, Branching, and Timing Controls,"** for publication in *Neurocomputing*.

This work studies dendritic artificial neural networks through a controlled benchmark rather than through a state-of-the-art vision framing. The manuscript asks a focused question: when do dendritic architectures provide measurable benefit once parameter count, branching structure, and runtime are explicitly controlled? To answer that question, the paper compares three dendritic sampling strategies against a parameter-matched multilayer perceptron, a naive branching control that preserves routing without dendrite-level nonlinearity, and a higher-capacity dense reference.

The main contributions are:

1. A controlled benchmark centered on architectural fairness rather than leaderboard performance.
2. Validation-selected reporting reconstructed from per-epoch histories to avoid relying on historical best-test exports.
3. Evidence that DANN-LRF improves over a parameter-matched dense MLP on all full-data benchmarks studied, while the advantage over naive branching is smaller and depends on dataset structure.
4. CPU timing analysis showing that the extra cost of dendritic nonlinearity beyond branching is modest.

We believe the manuscript is a good fit for *Neurocomputing* because it combines biologically inspired neural architecture design, empirical machine learning evaluation, pattern-recognition benchmarks, and reproducible computational analysis. The paper does not claim state-of-the-art image classification performance; instead, it contributes a more careful and publication-relevant question about parameter-efficient neural computation and baseline discipline.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. The author declares no competing interests. Code, archived results, and manuscript-generation assets are being prepared for public repository release alongside submission at `https://github.com/canay/dendritic-ann-benchmarks`.

Thank you for your consideration.

Sincerely,

Ozkan Canay  
Department of Information Systems and Technologies  
Faculty of Computer and Information Sciences  
Sakarya University  
canay@sakarya.edu.tr  
ORCID: 0000-0001-7539-6001
