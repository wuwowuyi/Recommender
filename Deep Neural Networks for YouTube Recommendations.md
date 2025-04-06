My notes and summary on paper [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf).

This is a concise paper packed with rich details! Also one of the best papers I've ever read!

## Introduction
Why recommending YouTube videos is hard
* Scale. over a billion users and millions of videos
* Freshness. very dynamic, lots of newly uploaded videos and user latest actions.
* Noise. users feedback are noisy, metadata poorly structured.

## System Overview
Two stage approach: retrieval (candidate generation) and ranking.

Retrieval (typically Two-Tower DNN)
* Inputs: **coarse features** of user history, context, and videos
* Output: 100-1000 candidate videos
* fast and scalable with massive data, using collaborative filtering
* Targeting high recall

Ranking
* Inputs: candidates generated from retrieval + other sources, **rich feature set** about videos and users
* Output: a highly personalized ranking list
* Targeting high precision

Rely on A/B testing via live experiments, measuring changes in CTR (Click-Through Rate), watch time, and many other metrics that measure user engagement.
Live A/B results are not always correlated with offline experiments. (??)

## Candidate Generation

### Recommendation Modeled as Classification problem

$\displaystyle P(w_t|U, C) = \frac{e^{v_iu}}{\sum_{j \in V}e^{v_ju}}$. 

* $w_t$ is video watch at time $t$
* video $i$ and video embedding $v \in R^N$
* user embedding $u \in R^N$ 

#### Training label
use implicit feedback of watches to train the model, where a user completing a video is a positive example.

This choice is based on the **orders of magnitude more implicit user history** available, allowing us to produce recommendations **deep in the tail** where **explicit feedback is extremely sparse**.

#### Negative Sampling + Importance sampling
Computing the full softmax over the entire video corpus $V$ is expensive. The paper:

* sample negative samples (several thousands) from a "background distribution" (e.g., item popularity)
* compute softmax over the one positive item (user watched video) and negative samples
* correct probability using importance sampling

🤔 In tensorflow, I think we can use `tf.nn.sampled_softmax_loss` to compute the sampled cross entropy loss. 

At serving time, the scoring problem reduces to a nearest neighbor search problem. The paper says A/B results were not particularly sensitive to the choice of nearest neighbor search algorithm. (🤔There are a number of choices to choose, like ScaNN, Faiss, etc.)

### Model architecture
Just two tower. Same idea as [word2vec](https://arxiv.org/abs/1301.3781). See Figure 3 in the paper.

### Features
Use various features:
* search history is treated in the same way as short text, tokenize + embedding + average like `GlobalAveragePooling1D` to generate a dense vector for each query
* Demographic features are important to new users
* User's geographic region and device
* booleans and continuous features are treated like numbers and normalized

#### Example Age (Not Video Age!)
The distribution of video popularity is **high non-stationary** but the trained multinomial distribution will reflect the average watch likelihood in the training window of several weeks. To correct this, **we feed the video age of the training examples as a feature during training. At serving time, this feature is set to zero (or slightly negative) to reflect that the model is making predictions at the very end of the training window**.  :boom:

🤔 I think the example_age is not about boosting new videos. It’s about modeling the non-stationary distribution of user behavior over time during training. Note here:

`example_age = training_time - watch_time`

`video_age = watch_time - video_upload_time`

The key point is that the distribution the model trained to learn is non-stationary, and therefore we feed in `example_age` to tell the model to down-weight older training examples.

At serving time, the model pretends all videos are “brand new” by setting `example_age = 0` or slightly negative — for **all** videos. We treat all candidate videos, regardless of upload time, as if they were just uploaded now, to simulate a “present-moment” popularity context.


