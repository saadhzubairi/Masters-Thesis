# LBEADS-NET Thesis Defense -- Speaker Script

> **Total time**: ~22 minutes speaking + buffer for transitions
> **Format**: Read through this before the defense. Internalize the key points -- do NOT read from this during the presentation.
> **Tip**: Practice each act separately. Time yourself. Cut words if you're over, never cut pauses.

---

## ACT I: "The Problem" (5 minutes)

---

### Slide 1 -- Title Slide (10 seconds)

> Good morning/afternoon everyone. My name is Saad Zubairi, and today I'll be presenting my master's thesis, LBEADS-NET -- Algorithm Unrolling for Learned Baseline Estimation and Denoising in Chromatographic Signals. This work was conducted under the supervision of Professor Ivan Selesnick.

---

### Slide 2 -- What is a Chromatogram? (45 seconds)

> Let me start with the application domain. Chromatography is one of the most widely used analytical techniques in chemistry, pharmacology, and biochemistry. It works by separating chemical compounds in a mixture, and a detector records the response over time -- that's the chromatogram.

> What you see on screen is a typical chromatogram. The observed signal y -- the dark blue line -- is actually a mixture of three components. First, there are the **peaks**, shown in green. Each peak corresponds to a different chemical compound, and its area is proportional to the compound's concentration. Second, there's **baseline drift** -- the dashed blue curve -- which is a slowly varying distortion caused by things like column bleed, temperature gradients, and detector aging. And third, there's additive measurement **noise**.

> The fundamental model is simple: y equals x plus f plus w -- peaks plus baseline plus noise. The goal of baseline correction is to separate these components so we can accurately measure the peak areas.

---

### Slide 3 -- Why Baseline Correction Matters (45 seconds)

> Why does this matter? Because the way chromatographers determine concentration is through **peak area integration** -- the area between the corrected signal and the baseline.

> On the left, you see what happens with a correct baseline: the shaded green area accurately represents the analyte concentration. On the right, a wrong baseline -- in this case, one that's too high -- gives you a smaller area, which means you underestimate the concentration.

> This isn't just an academic concern. In pharmaceutical quality control, food safety testing, and clinical diagnostics, even small systematic errors in baseline estimation can compromise the reliability of the results. Regulatory standards like USP 621 specify acceptance criteria as tight as plus or minus 2 percent -- so baseline errors of even a few percent can cause a batch to fail quality control.

---

### Slide 4 -- The Classical Solution: BEADS (60 seconds)

> The state-of-the-art classical approach is BEADS -- Baseline Estimation And Denoising using Sparsity -- proposed by Ning, Selesnick, and Duval in 2014. Professor Selesnick, my advisor, is a co-author of this algorithm, so this thesis is a natural extension of that line of research.

> BEADS formulates baseline correction as a regularized optimization problem. Here's the cost function. The first term is data fidelity -- a high-pass filter H removes the low-frequency baseline, so this term measures how well the peak estimate explains the high-frequency content. The second term is an asymmetric sparsity penalty that encourages peaks to be sparse and non-negative. The third and fourth terms penalize the first and second derivatives of the peak signal, promoting smoothness.

> It's an elegant formulation. But there's a catch -- it requires careful specification of **six parameters**: the sparsity weight lambda-zero, the smoothness weights lambda-one and lambda-two, the asymmetry ratio r, the filter cutoff frequency f-c, and the filter order d. And as we'll see on the next slide, getting these right is the central challenge.

---

### Slide 5 -- The Tuning Problem (45 seconds)

> Here's the problem in one picture. I'm showing you the **same signal** processed with three different BEADS parameter configurations.

> On the left, under-regularized: the sparsity penalty is too weak, so the peaks are noisy and the baseline doesn't track the drift properly. In the middle, well-tuned: clean separation -- this is what you want, but it required careful manual adjustment. On the right, over-regularized: the sparsity penalty is too strong, the peaks are suppressed, and their energy gets absorbed into the baseline.

> The fundamental issue is that **no single parameter configuration generalizes** across different instruments, analytes, column conditions, or even different signals from the same run. In practice, analysts have to tune these parameters for each experimental setup -- a tedious, expert-dependent process that doesn't scale.

---

### Slide 6 -- The Deep Learning Alternative (30 seconds)

> Recent work has applied deep learning to this problem. Kensert and colleagues trained a 1D convolutional autoencoder on 190,000 synthetic chromatograms. Chen and colleagues used a ResNet-UNet hybrid with 2.35 million parameters. Aradhye and colleagues developed DIRAS+, a hybrid that uses a CNN to predict parameters for a classical solver.

> These methods achieve strong generalization without per-signal tuning. But they're **black boxes** -- millions of opaque parameters, no insight into the decomposition process. You can't inspect what the network learned, verify that it respects physical constraints, or diagnose why it fails on a particular signal.

---

### Slide 7 -- The Gap (45 seconds)

> So the landscape looks like this. On the vertical axis: interpretability. On the horizontal: generalizability.

> Classical methods like BEADS and arPLS sit in the top-left: highly interpretable -- every parameter has a physical meaning -- but they don't generalize without per-signal tuning. Deep learning methods sit in the bottom-right: they generalize well, but they're opaque.

> The top-right quadrant -- interpretable AND generalizable -- is empty. That's the gap. And that's where we're trying to place LBEADS-NET, using a technique called algorithm unrolling.

---

### Slide 8 -- Our Approach: Algorithm Unrolling (30 seconds)

> The idea is beautifully simple. Take an iterative optimization algorithm -- in our case, BEADS. Instead of running it for K iterations with fixed parameters, **unroll** it: map each iteration to one layer of a neural network, and make the algorithm parameters learnable.

> What this essentially unlocks for us is the ability to **learn a different set of parameters at each iteration**. Classical BEADS uses the same six parameters for every iteration -- the algorithm has no way to adapt its behavior as it refines the estimate. But in the unrolled version, each layer gets its own parameters, so the network can learn to do coarse separation in early layers and fine refinement in later layers. We get the interpretability of BEADS -- every parameter still has a physical meaning -- but with the adaptability of a learned model.

> On the left, the classical version: a loop with fixed theta, repeated K times. On the right, the unrolled version: K sequential layers, each with its own parameters theta-k, trained end-to-end via backpropagation.

> For LBEADS-NET, this gives us 20 layers times 5 parameters per layer equals **100 learnable scalars**. That's orders of magnitude fewer than any deep learning baseline, and every parameter has a direct physical interpretation.

---

## ACT II: "The Attempt" (10 minutes)

---

### Slide 9 -- From BEADS to BEADSLayer (60 seconds)

> Now let me show you how we actually transform BEADS into a neural network layer.

> On the left is what one iteration of classical BEADS looks like: compute reweighting matrices from the current estimate, assemble a system matrix M, solve a banded linear system -- this is the expensive part, it's a big matrix equation -- and then update the signal estimate.

> On the right is our ISTA reformulation. Instead of solving the full linear system exactly, we decompose the update into two simpler steps. First, a **gradient step** that moves the estimate toward the data along the smooth part of the objective. Second, a **proximal step** -- specifically, asymmetric soft-thresholding -- that handles the non-smooth sparsity penalty.

> This is the core reformulation of the thesis. We replace an exact but expensive linear system solve with an approximate but simple gradient-plus-proximal update. Each layer does less work than a classical iteration, but we can stack 20 of them and train them end-to-end.

---

### Slide 10 -- The ISTA Layer (45 seconds)

> Here are the two key equations. The gradient step combines three forces: the data fidelity gradient -- how well does the current estimate explain the high-frequency content of the observed signal -- minus the smoothness penalty gradients, all scaled by a learnable step size eta.

> The proximal step is asymmetric soft-thresholding. If the updated value is above the threshold lambda-zero times eta, we subtract the threshold -- shrinking positive values. If it's below the negative threshold times r, we add the threshold -- shrinking negative values more aggressively, since chromatographic peaks should be positive. If the value falls within the dead zone, it's set to zero.

> Two things to notice. First, the effective threshold lambda-zero times eta is something the network can modulate independently through either parameter. Second -- and this is critical for the rest of the talk -- this operator **shrinks all non-zero values**. That shrinkage is where baseline leakage originates. Keep this in mind.

---

### Slide 11 -- Why Not Conjugate Gradient? (45 seconds)

> A natural question: why not just unroll BEADS directly, using conjugate gradient to solve the linear system at each layer?

> We tried this -- it was actually the first version of the architecture. The problem is computational depth. With K equals 8 outer layers and 12 CG iterations each, the effective depth is 96 sequential differentiable operations. Backpropagation has to push gradients through all 96 steps, and as you can see on the right, the gradient magnitude decays exponentially. By the time gradients reach the early layers, they're essentially zero -- the classic vanishing gradient problem.

> The ISTA variant has effective depth of just 20. Each layer is a single gradient step plus a proximal operation. The gradient flow is dramatically better -- the green curve stays well above zero across all layers.

> The design principle that emerged: **simpler layers times more layers beats complex layers times fewer layers**, as long as each layer captures the essential structure of the original algorithm.

---

### Slide 12 -- Full Architecture (30 seconds)

> Here's the complete architecture. The observed signal y enters on the left. First, we compute a high-pass initialization -- x-zero equals y minus the low-pass filtered version of y -- which gives the network a rough starting estimate with the bulk of the baseline already removed.

> Then we chain 20 ISTA layers, each with its own five learnable parameters: lambda-zero, lambda-one, lambda-two, r, and eta. The output of the final layer is our peak estimate x-hat. The baseline estimate is computed by low-pass filtering the residual.

> The key numbers: signal length 4096, 20 layers, 5 parameters per layer, 100 total learnable scalars. That's the entire model.

---

### Slide 13 -- The Development Timeline (20 seconds)

> Now for the core of the talk -- the development journey. We ran six progressive experiments, each adding one key idea to the loss function. This timeline is your roadmap. We start with classical BEADS on the left, move through increasingly sophisticated loss configurations, and -- spoiler -- the story doesn't end where you'd expect. Notice the colors: the green node is where things peak, and the red node at the end is where they get worse.

---

### Slide 14 -- Three-Stage Curriculum Training (40 seconds)

> Before we dive into the experiments, let me explain how we actually train this network. With up to 12 loss terms competing for 100 parameters, you can't just turn everything on from the start -- the optimizer gets pulled in too many directions at once.

> So we use a three-stage curriculum. **Stage A** is pure warmup: only peak reconstruction MSE is active. The network learns basic peak-baseline separation without any competing objectives. This gives the parameters a stable starting point.

> **Stage B** is where the anti-leakage machinery comes online. We activate sparsity, total variation, non-negativity, masked baseline supervision, orthogonality -- ten terms total. The key insight is that these penalties need a reasonable decomposition to produce meaningful gradients. Applied to a random initialization, they'd just produce noise.

> **Stage C** activates the remaining terms -- envelope constraint, frequency separation -- and rebalances all the weights for final refinement. The progression is: learn to separate, then learn to separate *correctly*, then polish.

---

### Slide 15 -- Exp 1.0: Classical BEADS (40 seconds)

> Experiment 1.0 is our baseline: classical BEADS with published default parameters. These parameters were designed for unnormalized chromatograms with peak amplitudes in the hundreds. But our test signals are normalized to the range zero to one.

> The result is catastrophic: 99.99 percent area error. The classical algorithm treats essentially the entire signal as baseline. The peak correlation is a modest 0.746, and the BLI -- Baseline Leakage Index, which measures how much peak energy leaks into the baseline -- is 0.837. That means 84 percent of the peak energy is absorbed into the baseline estimate.

> Now, this is deliberately unfair to BEADS -- we're using the wrong parameters for the data scale. But that's exactly the point. This is what happens when you apply BEADS out of the box without per-signal tuning. And **that's the setting LBEADS-NET is designed for**.

---

### Slide 15 -- Exp 1.1: Naive Unrolling (50 seconds)

> Experiment 1.1 is the simplest possible learned model: just reconstruction loss, MSE between the predicted and true peaks. No sparsity priors, no baseline supervision, nothing else.

> And even this naive version shows dramatic improvement. Peak MSE drops by 2x, from 21.27 to 9.79. Area error drops from 99.99 percent to 49.47 percent -- still large, but a qualitatively different failure mode. Where classical BEADS erased nearly all peaks, the learned model preserves roughly half the peak area. BLI drops from 0.837 to 0.593.

> But look at the negative fraction -- 29.5 percent of the predicted peak values are negative, which is physically impossible. Without a non-negativity constraint, the network produces artifacts below zero.

> The takeaway: algorithm unrolling works out of the box. The architecture has the right inductive bias for this problem. But there's clearly room for improvement, and there's a hint of trouble -- that BLI of 0.593 is going to become very familiar.

---

### Slide 16 -- Exp 1.2: + Sparsity Priors (50 seconds)

> Experiment 1.2 adds three sparsity-related losses: an ell-one penalty on peaks, total variation to encourage sharp transitions, and a non-negativity penalty.

> The good news: peak correlation improves from 0.748 to 0.785 -- the peaks look more like real chromatographic peaks. The negative fraction drops from 29.5 percent to 21.6 percent.

> The bad news -- and this is the first critical finding of the thesis: the **BLI doesn't move**. It goes from 0.593 to 0.597. Area error is essentially unchanged at 49.51 percent.

> This tells us something important. Sparsity priors can improve the **shape** of the peaks -- smoother, sparser, more physically plausible. But they don't address the fundamental mechanism by which peak energy leaks into the baseline. The ell-one penalty is actually part of the *cause* of leakage, not the cure. We'll come back to why in Act III.

---

### Slide 17 -- Exp 1.3: + Baseline Supervision (50 seconds)

> Experiment 1.3 is the breakthrough. We add two terms: masked baseline reconstruction, which directly supervises the baseline estimate in non-peak regions, and a baseline smoothness penalty.

> The results: best peak MSE across all experiments at 9.30, best peak correlation at 0.793. This is the **single most impactful addition** in the entire development journey.

> Why does it work? Think about what the loss function sees without baseline supervision. It only penalizes peak reconstruction error. If the network produces a peak estimate that's slightly too small and a baseline that's slightly too high, the peak loss goes up a little -- but the baseline is unconstrained and can absorb the error freely. There's no gradient signal telling the optimizer "your baseline is wrong."

> The masked baseline loss closes this gap. It provides direct supervision in the regions where the baseline is observable -- where no peaks are present. This gives the optimizer the information it needs to keep the baseline honest. And it does all this with only 6 active loss terms.

---

### Slide 18 -- Exp 1.4: + Orthogonality (v5 -- Thesis Model) (40 seconds)

> Experiment 1.4 adds peak-baseline orthogonality and a baseline third-derivative penalty. The orthogonality loss says: wherever there's a peak, the baseline should be locally flat. If the baseline is tracking the peak shape, that's leakage, and we penalize it.

> This gives us the best baseline MSE at 7.40 and the best BLI at 0.592. Peak correlation dips slightly from 0.793 to 0.785 -- a mild trade-off between baseline quality and peak shape fidelity, which is characteristic of multi-objective optimization.

> This is the **thesis model configuration** -- 8 active loss terms, a good balance between reconstruction quality and leakage suppression.

---

### Slide 19 -- Exp 1.5: Full Anti-Leakage Battery (v7) (50 seconds)

> And then we went further. Experiment 1.5 activates **all twelve loss terms**: everything from before, plus an asymmetric baseline loss that penalizes overestimation 9 times more than underestimation, a high-frequency leakage penalty, an envelope constraint, and frequency-domain separation.

> And the result is... it gets **worse**. Peak correlation drops to 0.744 -- that's worse than the MSE-only baseline from Experiment 1.1. Peak MSE is the worst among all learned variants. Negative fraction actually increases to 30.5 percent, despite a 20x increase in the non-negativity weight. And BLI increases to 0.611 -- the worst leakage of any LBEADS-NET configuration.

> This is a genuinely surprising and important result. More loss terms, more carefully designed anti-leakage penalties -- and the model gets worse on every metric.

---

### Slide 20 -- The Over-Regularization Lesson (40 seconds)

> Here's the picture. Peak correlation on the y-axis, loss complexity on the x-axis. The curve rises as we add terms -- one, four, six -- peaks at 6 terms with Experiment 1.3, and then **declines** as we add more.

> What's happening? Twelve loss terms create twelve competing gradient directions. The asymmetric baseline loss pushes the baseline down. The envelope constraint also pushes it down. But the orthogonality loss pushes it up where peaks are absent. And the reconstruction loss pulls the peak estimate toward the ground truth regardless of what the baseline is doing.

> With only 100 parameters and 300 gradient steps, the optimizer cannot satisfy all twelve objectives simultaneously. The result is a compromise that satisfies none of them well. The lesson: the relationship between loss complexity and model quality is **non-monotonic**. This is what motivated the three-stage curriculum training strategy.

---

### Slide 21 -- The BLI Plateau (40 seconds)

> And now the central result of the thesis. Look at these bars. BLI across experiments 1.1 through 1.5. I want you to focus on experiments 1.1 through 1.4.

> 0.593. 0.597. 0.596. 0.592. The BLI varies by only **five thousandths** across four very different loss configurations -- from a single MSE term to eight terms including orthogonality and baseline supervision.

> That dashed red line at 0.59 is what I call the **leakage floor**. No matter what we put in the loss function -- sparsity priors, baseline supervision, orthogonality, asymmetric penalties -- the BLI won't go below approximately 0.59.

> This is the central negative result: **loss engineering cannot overcome the architectural inductive bias**. The ell-one shrinkage is embedded in the forward computation of every layer. The loss function can adjust the learned parameters, but it cannot change the fact that soft-thresholding shrinks every non-zero value. The question is: why?

---

## ACT III: "The Discovery" (5 minutes)

---

### Slide 22 -- Why Leakage is Fundamental (60 seconds)

> Here's the answer. Soft-thresholding -- the proximal operator at the core of every ISTA layer -- is defined as: sign of x times max of absolute x minus lambda, zero. It sets small values to zero and shrinks large values toward zero by exactly lambda.

> Look at this diagram. On the left, the true signal value x-star. On the right, the output of soft-thresholding: x-star minus lambda. The difference -- that's the shrinkage bias, and it's **exactly lambda for every non-zero entry**.

> Now, in sparse regions -- where most entries of x are zero -- this is fine. The non-zero peaks get slightly shrunk, the zeros stay zero, and the small amount of missing energy is negligible.

> But in **dense-peak regions** -- where peaks overlap and most of the signal is non-zero -- every single sample gets shrunk by lambda. The total missing energy is lambda times the number of non-zero samples, and this energy has to go somewhere. It gets absorbed into the baseline estimate through the low-pass filter. That's baseline leakage.

> This isn't a tuning problem or a loss engineering problem. It's a **structural property** of any architecture built on ell-one-based soft-thresholding. And it was independently confirmed by Gharbi and colleagues in their work on unrolled sparse-recovery networks.

---

### Slide 23 -- Three Root Causes (45 seconds)

> Let me break down the three mechanisms that compound to produce leakage.

> First: **sparsity assumption violation**. The ell-one penalty assumes that most entries of x are zero. In dense-peak regions -- common in metabolomics and complex mixture analysis -- peaks overlap extensively. The signal is simply not sparse, and the penalty over-shrinks it.

> Second: **fixed cutoff frequency**. The Butterworth filter cutoff f-c is fixed at 0.006 for all signals. Broad peaks have spectral energy below this cutoff, so the low-pass filter captures that energy and attributes it to the baseline. A signal with both narrow and broad peaks can't be served by a single cutoff.

> Third: **spatially uniform regularization**. The weights lambda-zero, lambda-one, lambda-two are the same everywhere in the signal. A sparse region with well-separated peaks needs strong regularization to enforce sparsity. A dense region needs weak regularization to preserve peak amplitudes. Uniform weights can't adapt.

> These three mechanisms are not independent -- they compound in dense-peak regions, which is precisely where accurate baseline correction is most needed.

---

### Slide 24 -- Emergent Parameter Specialization (50 seconds)

> There's a positive finding too. When we inspect the learned per-layer parameters, we see something remarkable: emergent specialization.

> Look at this table. Lambda-zero -- the sparsity weight -- increases monotonically from 0.481 at Layer 0 to 2.0 at Layer 5, where it hits the parameter clamp. Early layers apply gentle thresholding, preserving most of the signal. Later layers apply aggressive thresholding to refine the decomposition.

> The asymmetry ratio r follows a non-monotonic pattern: moderate at Layer 0, drops to the minimum at Layer 3, then jumps to 9.67 at Layer 5. The final layer applies very strong asymmetric penalization -- aggressively removing any remaining negative artifacts.

> The step size eta is approximately uniform across layers -- around 0.2 -- suggesting it's determined by the spectral properties of the shared filter matrices rather than the layer's role.

> What's remarkable is that this specialization was **not designed**. All layers start with identical parameters. The training process discovers a non-uniform iteration schedule -- coarse separation early, aggressive refinement late -- that classical BEADS, with its fixed parameters at every iteration, cannot express.

---

### Slide 25 -- What Actually Worked (50 seconds)

> Let me summarize the five key findings.

> First, check: algorithm unrolling eliminates per-signal tuning. All LBEADS-NET variants outperform classical BEADS with default parameters, with roughly 2x improvement in peak MSE.

> Second, check: masked baseline supervision is the single most impactful loss term. It provides gradient signal about baseline quality that pure peak reconstruction cannot.

> Third, warning: baseline leakage persists at BLI approximately 0.59 across all configurations. This is a fundamental limit of the ell-one-based architecture.

> Fourth, cross: more loss terms does not mean better performance. Over-regularization with 12 competing objectives degrades results.

> Fifth, check: learned parameters exhibit emergent layer specialization that mirrors the behavior of well-designed iterative algorithms.

> I want to emphasize: the negative results -- three and four -- are **as important as the positive ones**. They tell future researchers where not to invest effort and what structural changes are needed.

---

### Slide 26 -- Limitations (40 seconds)

> Let me be honest about what this work does not do perfectly.

> Our quantitative evaluation is on synthetic data. We did test on real gas chromatography signals and the model produces visually plausible baselines -- smooth, physically reasonable -- but without ground truth we can't compute metrics on those. We don't compare against arPLS, airPLS, or any deep learning baseline -- only against classical BEADS with default parameters. The dense matrix implementation uses O(N-squared) memory, which limits scalability. The cutoff frequency is fixed because of the SciPy-PyTorch autograd boundary. And area error on synthetic data is around 50 percent, which is not deployment-ready for quantitative analysis.

> That said, this is a **methodological exploration** -- and the architecture itself is very promising for real-world deployment, as I'll show on the next slide.

---

### Slide 27 -- Future Directions (50 seconds)

> Four concrete next steps.

> First, and most impactful for quality: replace the element-wise ell-one penalty with **group sparsity** -- an ell-two-one penalty. Group sparsity penalizes entire blocks of coefficients jointly: either the whole group is zero, or none of it is. Within an active group, amplitudes are NOT shrunk. This directly addresses the leakage mechanism.

> Second: implement banded matrix operations **natively in PyTorch**, reducing memory from O(N-squared) to O(N-times-b) and enabling signals with hundreds of thousands of data points on GPU.

> Third: real chromatographic validation. Spike-and-recovery experiments or consensus labeling by expert chromatographers. We've shown qualitatively that it works on real GC data -- now we need the quantitative proof.

> And fourth -- this is exciting -- **edge deployment on STM32 microcontrollers**. The entire model is just 100 scalar parameters. With banded matrix operations, inference on a 4096-sample signal requires roughly 200 kilobytes of RAM. That's well within the capability of an STM32H7 -- a Cortex-M7 running at 480 megahertz with over a megabyte of SRAM. This means real-time baseline correction could run directly on portable chromatography instruments, no cloud or PC required.

---

### Slide 28 -- Thank You (15 seconds)

> To conclude: algorithm unrolling provides a principled framework for bridging interpretability and generalization in chromatographic baseline correction. And perhaps more importantly, it reveals the fundamental limits of sparsity-based formulations when applied to dense-peak signals. The persistent leakage at BLI approximately 0.59 is not a failure of loss engineering -- it's an honest characterization of what is and isn't possible within an ell-one-regularized architecture.

> I'd like to thank Professor Selesnick for his guidance throughout this work, and I'm happy to take questions.

---

## Quick Reference Card

Keep these in your head for Q&A:

| If they ask about... | Redirect to... |
|---|---|
| Why area error is so high | ell-1 shrinkage bias (Slide 22) |
| Why no arPLS comparison | Limitation, acknowledge, scope (Slide 26) |
| Why not a black-box net | 100 params, interpretability, data efficiency (Slide 8) |
| Convergence guarantees | Untied params break MM convergence -- known trade-off (thesis Sec 2.4.3) |
| Is the BEADS comparison fair | Deliberately unfair to show the tuning problem (Slide 14) |
| What's the actual contribution | Unrolling + leakage characterization + negative results (Slide 25) |
| Why K=20 | Heuristic from classical BEADS iterations, ablation not yet run |
| Group sparsity details | Block soft-thresholding, no per-element shrinkage, future work (Slide 27) |

---

## Timing Checkpoints

| Checkpoint | You should be at... | Slide |
|---|---|---|
| 2 min | Finishing "Why Baseline Matters" | 3 |
| 5 min | Finishing "Algorithm Unrolling" intro | 8 |
| 8 min | Finishing architecture, starting timeline | 13 |
| 12 min | Finishing Exp 1.3 (baseline supervision) | 17 |
| 15 min | Finishing the BLI plateau | 21 |
| 18 min | Finishing "Three Root Causes" | 23 |
| 20 min | Starting limitations | 26 |
| 22 min | Thank you | 28 |
