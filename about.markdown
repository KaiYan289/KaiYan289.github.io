---
layout: page
title: Tips of the Day
permalink: /about/ 
---

<br/>

*I express my heartfelt gratitude to the mentors of my research career, Dr. Jiecao Chen, Prof. Alexander Schwing, Prof. Yu-Xiong Wang, Dr. Jie Yan, Dr. Chuan Luo, Prof. Changliu Liu and Prof. Zongqing Lu (sorted by time), as well as all the wonderous people I have met at ByteDance, UIUC, 
MSRA, CMU and PKU. I learned all these tips from my experiences working with you.*


<br/>

<div class="quote"><p><i>"The palest ink is better than the strongest memory."</i></p></div> 

<br/>

1. Vanilla A2C/PPO without reward shaping/prolonged episode/ exploration skills are actually hard to deal with mountain car, as the reward is too sparse.

2. It is important to do state/reward normalization for PPO to maintain numerical stability.  

3. DO NOT put any function that changes global variables in Pycharm's watch! (e.g. a function in Pycharm's watch which adds a global counter by 1 may cause the wrong value of the counter).

4. Gurobi, Mosek and SCIP (not scipy!) are the best open-source optimization solver. CVXPY integrated many of them.

5. Don't use scipy in your research project as an optimization solver; use Gurobi instead. An academic license costs $0, yet Gurobi is ~250x faster than scipy (and also more numerically stable).

6. Normally, a good solver (e.g. Gurobi) will do some numerical tricks for actions that may cause singularity.

7. If you don't know what hyper-parameter to set, go find a previous work and inherit their params. This will help to convince the reviewers that your idea works.

8. Randomly initialized NN has a compressing effect (see Benjamin/Rechat's work), which means its output is probably a contract mapping (with possible shifts) with random inputs. This effect can be used in anomaly detection.

9. When dealing with a temporal sequence, use the first part (e.g. year 1-6 for a 10-year dataset) as the training set, then validation set, finally the test set.

10. Prediction models for time sequence (e.g. electricity/VM demand) usually underestimates, for there are systematic bias (e.g. peaks) in the dataset. On the other hand, underestimating the demands are usually more serious than overestimating in real life.

11. You can get Azure VM information from Kusto.

12. Exploration for RL matters, even for toy environment. The same environment with different default behavior for illegal actions (e.g. stay or randomly moving or giving large negative reward) causes huge performance gap for A2C. As for my own experience, the first two are better choices.

13. L1 loss fits better for data with **sparse** entries, and is more **robust against outliers**.

14. The goal of experimental parts in a paper is not stating "what we've done". It should be organized by "What we're going to validate" (e.g. Why do we design this experiment, and what is the conclusion).
 
15. The MIT book and the Boyd book are the two classical textbooks for convex optimization; strongly recommending the two books.

16. The difference of **\forall** and **for x \in X**: The former emphasizes "satisfaction of conditions", usually used in proofs of advanced mathematics; the latter is an enumeration. They are generally the same, but proper usage helps comprehension for readers.

17. A **sparse embedding** (e.g. holiday tag) with **small training set** is inherently infavorable over two-stage method and favors decision-focused method.

18.	Write papers! Only by writing papers can you be more rigorous in language for papers.

19.	*Constraint* is for decision variables' feasible domain. The relationship between problem parameters should not appear in the constraint part. 

20. tensor([0]) < 0.5 is **False**. Note the **round down of integer types of torch.tensor!**

21. To check the difference of two general distributions (e.g. When you are comparing the performance of two methods), mean and std are not enough. Try percentile, histograms and Maximum Mean Discrepancy!

22. Add axis label and title for debugging figures, as you may forget what you were plotting.

23. Do periodically save your **code** and model for an actively debugged program; preferably automatically doing so every time you run your code.

24. A L1/L2 regularization is by essence Lipschitz regularization for target function.

25. Some ways to note current update for your research field:  a) arxiv subscribing cs.AI cs.LG, plus manually searching the key word *proceedings of ICML、NeurIPS、ICLR、UAI、AISTATS, etc, and b) reddit.com/r/MachineLearning

26. Put a demo one-line run script for cmd/shell in your project readme. The most common one will do.

27. Do note your notations for theoretical parts, and try your best to make it coherent for each of the theorem / both main paper and appendix.

28. Recurrent DDPG is unreliable and hard to tune. MADDPG/Recurrent MADDPG is even more painful. So do recurrent TD3; try to avoid recurrent policy if you want stable performance.

29. Programming dataset, e.g. GAMS, has a very large number of dimensions for decisions (e.g. >100k).

30. A noise of ~0.05 over a value 1 causes a SNR less than 15db, and by this aspect is not a small noise.

31. If you can tell a good story / establish a good framework, then the experimental part will be much easier as it only serves as a validation. Otherwise, your research will be an empirical one, which requires high demand on performance.

32.	General Multi-Objective problem may seem luring, but it is not trivial: pareto optimal means balance over multiple goals, yet such goals usually depends on the settings of real scenario.

33. "Add noise then discretization(e.g. rounding)" is more close to reality than "discretization then add noise".

34. Sometimes, if the experiment code is not working, you can fix some elements to debug. 
E.g. for off-policy 2-step RL, you can fix the first step and try to train the 2nd step; if the current picture training set is not working, you can pick one picture as the training set to see if it can overfit; if not, the code may be buggy.
However, such practice (the one datapoint method) may face the problem of **not having enough support for optimization surface**, so it is not a panecea.

35. Intuitively, the following situation will put decision-focused method at advantage over 2-stage method: a) the optimization part, with surrogate, has a differentiable argmax and good generalization, and b) the prediction part has some outlier dimensions which has low weight on optimization quality.

36. If you find an unnecessary condition set in your experiment due to early decisions, If you have no time for re-runs, you can simply explain the condition in the appendix, and give a real-life example if necessary.

37. For a multi-dimensional decision vector in optimization, the influence of single/minority number of dimension may be overwhelmed.

38. 2-stage early stopping has an inherent logic of "doing prediction well first". Thus, it should be early stopping according to **prediction loss** instead of **optimization performance**.

39. Significance tests are usually conducted in traditional statistic works for hypotheses, especially where test set does not exist. 

40. Use **on-policy** methods for MARL, as stationarity is not preserved!  

41. When you are imitating someone else's code but failed, a useful debugging method is to take his code, and changing his code into yours function by function (instead of  changing yours onto his). You can try the differnet versions of code in parallel to quicker iterate.

42. Batchnorm is influenced by eval/train! By default, the running_stats is on. Then for training, the normalization is conducted with batch statistics; but for evaluation, the normalization is conducted with a fixed mean and variance estimate kept with a momentum of 0.1. This could have a VERY BIG influence if you ignore the difference.

43. You can try to feed feature^2 besides feature into MLP to get better expressivity, which works particularly well in fitting near-quadratic functions. 

44. torch implementations such as **logsumexp** are numerically stable, and should be used instead of self-implemented vanilla code.

45. Be patient when you are training a large network. For a classifier, the training loss may be not decreasing in a relatively long period at the beginning of the training (although the output is changing greatly), but the loss will decrease quicker in the later training process. 

46. One technique for serious outliers in a dataset is to clip the loss to a constant, e.g. minimize max(-log(y\|x), 0.1); this effectively "rejects" the gradient from the outliers and upper bounds the loss.

47. Note: Pytorch passes address, so if you want to only pass value to a function, make sure that you use clone() function! (e.g. for normalizing flows) 

48. Do not trust "manual design" too much against randomization in deep learning. (e.g. permutations of channels in normalizing flows)

49. Note that torch.KLDivLoss(q.log(), p) = KL(p\|\|q).

50. When you are tuning performance, try keep observing the curve for the first run if possible; this takes a little time, but it helps you to grab a sense of what is happening, and what epoch is the best. Also, try to run your code **through** before starting a long experiment (e.g. set epoch to 1 to see if the model can save correctly).

51. Use the following code to fix your pytorch random seeds, preferably at the beginning of main process:

{% highlight python %}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs
    torch.cuda.manual_seed(seed)     
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    # torch.use_deterministic_algorithms(True) use with caution; this line of code changes many behavior of program. 
    torch.backends.cudnn.benchmark = False # CUDNN will try different methods and use an optimal one if this is set to true. This could be harmful if your input size / architecture is changing.
{% endhighlight %}

And don't forget to use env.seed(seed) for your gym environment! 

Note: once the random seed is set anywhere in this process (regardless of which file it is in), the seed remain fixed (unless implicitly set by other libraries).

{:start="52"}
52. You should reduce the learning rate if you are using batchnorm. Batchnorm changes the landscape.

53. What is the difference between optimizing KL and reverse KL? **Mode-seeking (reverse) and mode-covering (forward)!** See https://www.tuananhle.co.uk/notes/reverse-forward-kl.html for a brief explanation.

54. You can use the following code to visualize your RL episode in your gym step: 
{% highlight python %}
    img = env.render(mode='rgb_array')
    IMG.append(img)
{% endhighlight %}
and at the end of the episode, you write
{% highlight python %}
 imageio.mimsave(name+'.mp4', IMG, fps=25)
{% endhighlight %}
{:start="55"}
55. If you need to change distribution in expectation in your derivation, try importance sampling. But as this introduces a possibly instable denominator, you may need surrogates to stabilize the whole thing.  

56. If you are encountering strange problems in python import (e.g. missing .so), there are a few possible things that you can do:

1) check if the import path is wrong. For example, is your python importing a package from /.local/lib/python3.x/site-packages instead of your conda directory? 

2) check both pip and conda envrionment, especially where there is a version mismatch between python xxx.\_\_version\_\_ and pip list / conda list. https://www.anaconda.com/blog/using-pip-in-a-conda-environment Actually, you may want to avoid using pip and conda together.
    
3) check your python, linux, and some system library version (e.g. MPI) under different settings. Sometimes the problem comes from version mismatch.

4) DO NOT try to copy-paste a file from another version into this version and simply change its name for a quick fix, unless you absolutely know what you are doing. While sometimes it can fix the problem, this creates an environment that is hard to reproduce and would be questioned by future readers of your papers.

{:start="57"}
57. There are many people write ML code, but many of them do not write the code well. Make sure you have seen others' code and wisely referred to them for your own code; you should not easily believe in one single reference, even if it has many stars.

58. Whenever you see a logsumexp() in your formulation, ask yourself if you have made it robust by subtracting the largest term. (torch.logsumexp already does this for you.)

59. When using remote Python debugger with pycharm professional, you can close "attach to subprocess" option if you witness strange bugs. (At settings -> build, execution, deployment -> Python debugger)

60. When developing a new architecture for deep learning, you cannot simply considering throwing a dart randomly and hoping it can work. You should deviate from original design gradually, that is, stand on the shoulders of the giants.

61. You should never use gradient clipping along with weight decay. The gradient clipping take effect **Before** weight decay, thus greatly amplifying the weight decay factor and cause the training to be weird.

62. You should not copy models by iterating through parameters(), as parameters() does not contain parameters from the batchnorm layers. Use copy.deepcopy() or load_state_dict() instead.

63. If you are confronting strange problems in installing a package with "pip install -e .", take a look at the setup.py, especially if you cannot find any version. Sometimes, authors will use "git+git@xxxx" to fetch dependency posts. you should change it to git+https://xxxx ... as you are not collaborator / author of the repo.

64. If you cannot run "git submodule update --init --recursive", you should check .gitmodule file to see if there is problem, especially that mentioned in 63. After that, "run git submodule sync" and the problem should be fixed.

65. Even the smallest library/package version change can cause a big difference in a particular environment. For example, mpi4py 3.1.1 and 3.1.3, though seemingly no big difference in the update log, can decide whether a program is runnable or not.

66. Different version of GPU (e.g. Nvidia A6000 and RTX 2080Ti) and different computation platform (e.g. GPU and CPU) could lead to non-egligible value difference when doing matrix multiplication! See https://forums.developer.nvidia.com/t/cpu-and-gpu-floating-point-calculations-results-are-different/18175 for details.

67. If we decrease the number of steps in the diffusion model, for each sampled diffusion timestep t, on average, the product of \alpha, which is \bar{\alpha} will increase as there are less terms less than 1. As we are fitting epsilon, this leads to lower signal-noise-ratio for epsilon and higher MSEloss. Therefore, fewer number of steps requires higher beta. (https://arxiv.org/pdf/2006.11239.pdf)

68. Remember to save your powerpoint every time you made a slide, and add a timestamp to your experiment results.

69. You can use the following code to output all attributes in args from argparse:
{% highlight python %}
for arg in vars(args): f.write(str(arg)+" "+str(getattr(args, arg))+"\n")
{% endhighlight %}
{:start="70"}
70. Use the same color for the same method throughout the paper of your work, and same notation for the same thing as well. 

71. Keep the citations nice, neat and simple for your papers. Just keep the platform of publication (e.g. ICML, Nature), paper name, year and author on it and don't put the pages / publishers etc.

72. Use grammarly to check the writing of your paper, but do not overly rely on it. It may not recognize terms in your field and make mistakes.

73. You can use a plain notation in Python for elementwise multiplication (Hadamard product), but need to state elementwise clearly when writing papers.

74. Parenthesis around sum symbols should be at least as large as the sum symbols.

75. Beware of any presence of your identification in the code of paper, including absolute path and platform username (e.g. for wandb)!

76. Figures and its captions should be self-contained, especially in appendix where the space is unlimited; put settings and brief conclusion there.

77. The most important hyperparam for PPO is #update epoch and update interval (# env steps).

78. Boil down your slides (in mind); nobody will parse a slide that is full of text.

79. BC is a good baseline if a complete trajectory presents, and if the initial position is of small variance. On discrete MDP, the best BC result is simply counting transitions and do random action if current state is never witnessed.

80. Continuing from 79, you should be pessimistic in offline RL / IL, so that you policy does not astray from what you have witnessed.

81. Wasserstein distance is a much "weaker" distance than f-divergences (e.g. KL divergence), which means in many scenario, the f-divergence method will give either a infinite value / being invalid, or being uncontinuous, or losing a gradient. Intuitively, this is because Wasserstein distance represents a much "weaker" norm in topology (see WGAN paper for details). Wasserstein-1 distance is also called earth mover's distance.

82. If you have a score model which estimates the gradient of log probability of some unknown distribution, you can sample from the distribution using **Langevin dynamics**. This is score matching method.

83. The notion of spaces:

A normed space is a special metric space, which means elements have the notion of "large / small" by norm. A complete normed space is called a **Banach space**; by "complete" it means the limit of a Cauchy sequence is still in the space (counterexample: rational number Q)

A Euclidean space is a finite-dimenisional linear space with an inner product. A **Hilbert space** is an expansion of Euclidean space; it means a complete inner product space, but can be infinite dimensional and not confined to real numbers.

{:start="84"}
84. By Mercer theorem, any semi-positive definite function can be a kernel function.

85. You should not put plotting plt and ax inside an object and make it a property of other object, especially one that is not a singleton (e.g. solver class for a ML solution). Remember that plt settings are global; object duplication will ruin your plotting.

86. Be wary of the subtle constraints on Lagrange multipliers when you try to derive the dual problem (without them the optimal value could be unbounded; e.g. when you derive dual for linear programming). You should be extra careful when you only apply Lagrange on part of the constraints; when the other part of constraints is a bounded closed set, the problem might be much harder to discover. 

87. Be very careful when you try to explain something with a toy example but change cases (e.g. when talking about something for continuous space, use discrete space as an example). 

88. In a rebuttal, write in a way that is considerate for the reviewers:

1) Answer the question clearly with a few words at the beginning of the problem;

2) Do not show them that you are lazy typing, but you are helping them (e.g. for brevity -> for readability);

3) If possible, do not force the reviewer to get back to the paper. List the points briefly besides reference to the paper. Similarly, avoid reference to the other reviewers;

4) Do not simply write "We will change this", but show them "how we will change this" (and in the case where you can update pdf, do it) and invite them for advice on further modification;

5) Reply after everything is ready, but immediately beyond that point.

{:start="89"}
89. Do not assume that the battle is over until the authors are not expected to say anything (e.g. reviewer-metareviewer discussion). For NeurIPS, finishing rebuttal period is only half-way there; there are still much work to do at author-reviewer discussion period.

90. For L-BFGS, increasing the history hyperparam will increase the stability of iteration. Also, you should use line search with 'strong wolfe' in pytorch.

L-BFGS needs optimizer.step(closure()) where closure() gives the loss function. It might be invoked multiple times in one timestep, sometimes with gradient and sometimes without. That's why you will sometimes get backward second time error if you do not put everything in the closure() function. Here are two examples: https://gist.github.com/tuelwer/0b52817e9b6251d940fd8e2921ec5e20#file-pytorch-lbfgs-example-py-L27; http://sagecal.sourceforge.net/pytorch/index.html.

{:start="91"}
91. Be very careful when you try to generate dataset with some tricks (e.g. manipulate the distribution so that the states are guaranteed to be covered) and handling the "default value" for corner case. They might lead to very counter-intuitive behavior if not considered properly.

92. Gurobi max (gp.max_) operator can only take constant and variable (that is, no expressions such as x+1) as of Sept. 2022.

93.  torch.nn.parameter.Parameter(a.double(), requires_grad=True) is correct; torch.nn.parameter.Parameter(a, requires_grad=True).double() is not.

94. Wasserstein-1 distance with hamming distance metric is total variation distance.

95. If you are doing stochastic gradient descent by sampling some pairs of variables (e.g. uniformly sampling (i,j) for x_i+x_j), you'd better sample each pair independently, instead of sampling two state uniformly and then select all pairs of chosen states. In the latter case, you cannot break the correlation between (i,j) and (i,*), (i,j) and (j, *), as they are always updated together.

96. While there is a rule of thumb that choices the learning rate, it really depends on your scale of the loss and batch size. Be open to rare learning rates when you finetune your algorithm. 

97. Remember to "git add" your new file when you are using git to do version control, especially writing script to auto-commit things. otherwise, you may find that your modifications are all untracked.

98. You can use gym.spaces.MultiBinary(n=10) for one-hot observation space.

99. torch.multinomial and torch.Categorical supports batch sampling, which means you only need to get a input of batchsize * n tensor and it will sample batchsize different groups of samples for you. You don't have to go over the whole array! And use F.one_hot(m, num_classes=n) if neccessary.

100. Some points on making slides:

1) Text size should be unified across all slides and inside figure. Don't be lazy and just using existing figures; draw a nice figure that expands as your presentation progresses. And change text instead of font size to fit in spaces.

2) The shorter the presentation is, the more rigorous logic your slides must have because you don't have too much time to do "overviews" to remind people what you are discussing.  

3) You can align the elements on your slides by choosing items and selecting "align". 

4) You can make changing text w.r.t. time by animation with time delay. No need to make a video using video edit tools.

5) Use animations to avoid making your slides to be overwhelming. Let the slide be filled gradually as your speak progresses.

6) Always introduce math symbols first before using them, even the most common-sense ones in the subfield (e.g. state s in reinforcement learning). You should use as less symbols as possible in a short presentation.

7) Colors and shapes matter; they can be a strong indicator in the figure. Ask yourself why use this color for this color / this shape?  

{:start="101"}
101. self-made dataloader based on torch.randperm could be much faster than torch dataloader, especially if the data is stored in dict for each dataset. torch dataloader need to concatenate them every time and that can be very slow.

102. If you are trying to overfit behavior cloning on a small dataset to debug, remember to add variance lower bound (e.g. clip / tanh) to avoid spikes.

103. If you are training an action distribution on a closed set (e.g. in behavior cloning in gym environment), and you are using Gaussian / GMM / normalizing flow. One thing you could try to optimize log probability a lot is to use tanh to converge your output into a bounded one. And probability tractable will still be tractable.

104. Wasserstein distance in the Rubinstein-Kantorovich form assumes the underlying metric to be Euclidean, unless the definition of 1-Lipschitz is modified.

105. The sample complexity of Wasserstein distance is bad, but for MMD it is good. Sinkhorn Divergence stands between them, and have a corresponding sample complexity. They are all called integral probability methods.

106. Do not undo commit in github desktop unless you are absolutely certain! Undoing commit makes you lose all the progresses during this commit. 

107. You need to use clf() instead of cla() to remove the old colorbar in your last figure in matplotlib. However, after that you need ax = fig.add_subplot() to re-insert subfigures in order to draw anything more on the canvas. 

108. If you feel lost about why your method is not working while the baseline is, a way out is to implement your method inside the codebase of the baseline. In that way, you can make your method to be as similar to the baseline as possible, and to rule out the factors that does not matter one by one.

109. Be bold and aggressive when you first try to tune your algorithm; often it takes longer than expected to train / bolder choice of hyperparameter than your expecation to make your algorithm work.

110. Do read the experiment details of your baselines, and make sure of how they set up their experiment, especially what do they do to their dataset (e.g. merging). You do not want to waste time on settings that is unnecessarily harder / easier than prior work.

111. When you don't know where is the problem of your algorithm, go and check if your dataset has problems.

112. If you are working optimizations of f-divergences on a probability simplex, consider Fenchel conjugate; consider Donsker-Varadhan representation and https://people.lids.mit.edu/yp/homepage/data/LN_fdiv.pdf Thm 7.14. 

113. Continuing from 112: when considering relaxing optimization (e.g. use Lagrange multiplier to relax some constraints), relax as less constraint as possible (as long as you can solve it). Relax to probability simplex is better than relax to positivity constraint.

114. remember to set CUDA_LAUNCH_BLOCKING=1 whenever you meet a device-side assert triggered error.

115. If you don't know what parameter to tune, try to do the following two things:
1) check very closely on your direct baseline to see how they solve the problem;
2) retry factors excluded before last bug fix. Sometimes bug fixes will make factors behave very differently and you may overlook some crucial factors.

116. For RL evaluation, you should try to use deterministic action (mean as output) as stochastic ones are often with fairly high variance and cannot do well, especially in those environments requiring accurate actions.

117. If you need to send your computer to repair, make sure you have copied everything you need out of it. Especially the private keys for the server.

118. If you need to copy datasets to different folders on your server, consider soft links; this saves your disk space and frees you from copying everytime you change your dataset.

119. If you were to build up a desktop, remember that do not throw the boxes until you have lighten up the machine. There might be some important information or some material (e.g. screws, cables) in the boxes.

120. When building up your desktop, remember to observe the minimal principle: use only as least as possible components to light up your mainboard first. Do not haste to install the extra memory / disk / graphics card. However, you should always make room for your GPU at the very beginning.

121. Make sure to check the debugging light and code on your mainboard to figure out the problem.

122. When swapping an element in an array and its index in python, be very careful: a[a[0]], a[0] = a[0], a[a[0]] might not behave the expected way. A better choice is to use a, b = copy.deepcopy(b), copy.deepcopy(a), or use the tmp variable.

123. (Pytorch official hint) If you need to move a model to GPU via .cuda() , please do so before constructing optimizers for it. Parameters of a model after .cuda() will be different objects with those before the call. In general, you should make sure that optimized parameters live in consistent locations when optimizers are constructed and used.

124. GLFW might be problematic on headless machines (i.e. servers); to fix this, try set MUJOCO_GL environment variable to egl or osmesa.

125. When you are using np.linalg.norm, make sure you know you are operating on a vector or matrix; they have very different behavior.

126. If you want to make modifications to a decision transformer, make sure your sequence is still one-way logical (always induce the terms on the right from the terms on the left).

127. The problem of beam search is that top-K choices may only include a very small portion of probability. You need to set K to be very large to include most possibility.

128. If torch.save cannot save because of errors like some local variables in class, try cloudpickle.dump(). example: cloudpickle.dump(agent, open("model/"+NAME+"/agent-iter"+str(i)+".pkl", mode="wb"))

129. Remember that decision transformer is a transformer, so as long as the timestep, state and attention mask are correctly matched and in the correct order, it does not matter where you put the paddings (front or latter). Remember, decision transformer can be used for unordered set once the positional encoding is removed, so it does not really matter if it is front-padded or latter-padded. 

130. GPT2 in huggingface is causal, and decision transformer is based on GPT2. It does not matter what padding action you are putting into the place where attention mask is 0.

131. Remember that AWR's implementation has a weight clipping of 20, and it is normal that initially there are either weight 20 or 0. Also, AWR is quite sensitive to buffer size; buffer size too small (50K recommended) will make the algorithm overfit to the dataset.

132.
```
import time
from tqdm import tqdm
lst, lst2, lst3, lst4 = [], [], [], []

for i in tqdm(range(100000)):
    lst.append(torch.zeros(1, 100))
    lst2.append(torch.zeros(100))

t0 = time.time()
for i in tqdm(range(100000)):
    lst3.append(lst[i][[0], :])
t1 = time.time()

for i in tqdm(range(100000)):
    lst4.append(lst2[i].reshape(1, 100))
t2 = time.time()

print("[0]:", t1 - t0, "reshape:", t2-t1)
``` 
The result is [0]: 1.9272778034210205 reshape: 0.3856058120727539. The latter is 5x faster than the former! (torch.stack is even faster!)

{:start="133"}

133. When you find that the training curve is strange, make sure to check whether you sampling process is fine; your program might only be trained on a small subset due to code bug.

134. remember torch.distributions.Normal takes **standard deviation** as input, but torch.distributions.multivariate_normal.MultivariateNormal takes **variance** (covariance) as input!

135. remember to check the original code by the author; there might be some special tricks in it or different hyperparams that are not specified in the paper.

136. Be very careful when you use xxx if yyy else zzz; adding () at the edge of the expressions is always a good practice.

137. A good way to write squashed Gaussian is through torch.distributions (from online DT):

```
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        
        # print("shape:", loc.shape, std.shape)
        
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]#  [] #
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)

```

{:start="138"}

138. A simple way to write a dataset for iteration:

```
class RepeatedDataset:
    def __init__(self, datas, batch_size, start_with_random=True):
        self.datas = []
        for data in datas: # list of arrays with the same first dimension.
            self.datas.append(data.clone())
        self.counter, self.idx, self.batch_size = 0, torch.randperm(self.datas[0].shape[0]), batch_size
        if start_with_random:
            for _ in range(len(self.datas)):
                print("shape:", self.datas[_].shape)
                self.datas[_] = self.datas[_][self.idx]
    
    def __len__(self):
        return self.datas[0].shape[0] // self.batch_size    
    
    def getitem(self):
        if self.counter + self.batch_size > len(self.idx):
            self.counter, self.idx = 0, torch.randperm(self.datas[0].shape[0])
            for _ in range(len(self.datas)):
                self.datas[_] = self.datas[_][self.idx]
        ret = []
        for _ in range(len(self.datas)):
            ret.append(self.datas[_][self.counter:self.counter+self.batch_size])
        self.counter += self.batch_size
        """
        print(self.counter, self.counter+self.batch_size)
        
        for _ in range(len(self.datas)):
            print(self.datas[_][self.counter:self.counter+self.batch_size])
        """
        if len(self.datas) == 1: return ret[0]
        else: return ret
```

{:start="139"}

139. You should not use multiprocessing in dataloader while loading tensors, or you might get a CUDA initialization error. Make sure that your torch.dataloader only loads numpy arrays instead of tensors. (besides, if the data is on GPU, why bother loading it via multiprocessing?)

140. use dataload = iter(dataloader); next(dataload) to get the next batch in the torch dataloader. (do not use next(iter()) as it is very slow!)

141. You must do **left-padding** for pretrained LLM models, because LLMs are decoder-only architectures and are not trained to continue from padding tokens! (https://huggingface.co/docs/transformers/main/en/llm_tutorial#wrong-padding-side)

142. If your self-implemented SAC algorithm is diverging, you should check whether the entropy sign is correct. If the entropy term is wrong, then the Q value will certainly diverge (which is different from other cases where the entropy is not involved in the TD function).

143. next(it), it=iter(dataloader) is very slow, probably because it does not use the parallelization of the torch dataloader; try iterate in a for loop instead.

144. If you find that the CPU usage of pytorch code is very high, try use torch.set_num_threads(1) (to reduce thread communication cost) or pin_memory=False (if you have ever explicitly set it to true). 

145. When making slides, the front "dot" recommendation: unicode 2022 (in custom), 100% height

146. Be very careful when you adapt random agents to deterministic algorithms (e.g. TD3 to ODT). You probably run the risk of not initiating exploration noise, which does not have to exist when it was a stochastic agent.

147. Be close to standard D4RL format; it is better that your program directly read from get_dataset() such that you have better reproducibility.

148. Transformer RL agents might have very different hyperparameters from MLP ones (e.g. critic learning rate).

149. If you are confronting weird critic divergence, check your data; if not a single state is "terminal" (i.e. all timeout), remember to set one to terminal.

150. If your RL agent is diverging due to strange reasons, try layernorm on the critic. However, adding layernorm to the critic is not always the best choice; sometimes (e.g. mujoco) it slows down the learning process, but sometimes (e.g. adroit) it is magical.

151. If you are wondering how people solve antmaze: they (CQL, IQL) sub reward by 1, making a sparse reward env becoming a dense one.

152. Make sure that you use \left and \right before the parentheses for complicated contents in the formula (e.g. \\exp\\left(\\frac\{a\}\{b\}\\right) ).

153. Remember that "by \[4\]" is not correct in writing papers; instead, you should write "by xxx et al. \[4\]".

154. Remember to use \eqref instead of \ref for equations.

155. Remember to use vector graph (i.e., pdf) for figures in the paper.

156. When you are updating posts in Jekyll, make sure that you add posts from a past point of time. The future posts will be skipped by Jekyll. To check this, use jekyll build --verbose.

157. Remember that openreview requires "\_" for latex formulas that are "_" in overleaf. https://docs.openreview.net/reference/openreview-tex/common-issues-with-latex-code-display

158. Check for the success file when you use hdfs to download items.

<!--
This is the base Jekyll theme. You can find out more info about customizing your Jekyll theme, as well as basic Jekyll usage documentation at [jekyllrb.com](https://jekyllrb.com/)

You can find the source code for Minima at GitHub:
[jekyll][jekyll-organization] /
[minima](https://github.com/jekyll/minima)

You can find the source code for Jekyll at GitHub:
[jekyll][jekyll-organization] /
[jekyll](https://github.com/jekyll/jekyll)


[jekyll-organization]: https://github.com/jekyll
-->
