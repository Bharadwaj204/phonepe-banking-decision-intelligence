Domain: Banking

## How to run

Run these commands from the repository root on Python 3.10–3.12. The
project uses NumPy only and does not require internet access after the
package is installed.

``` bash
python -m venv .venv
```

Activate the virtual environment:

``` bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the pinned dependency and regenerate the deterministic dataset:

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
```

The generator should print:

``` text
800 10 170 e85e7605b784f4142844f79ad2dd0b8e
```

Run the four parts in order:

``` bash
python part1.py
python part2.py
python part3.py
python part4.py
```

The scripts write their results and CSV files into `outputs/`. Run them
from the repository root so the shared data and output paths resolve
correctly.

## Part 1 findings

### R1.1 --- Estimate and test the monthly-spend assumption

Across 800 synthetic records, mean monthly spend was 8.9382 thousand INR
(INR 8,938.20), with a 95% interval of 8.3026 to 9.5738 thousand INR.
The estimated mean is 0.9382 thousand INR (INR 938.20) above the
planning assumption of INR 8,000. The z-statistic was 2.8930, which
exceeds the two-tailed critical value of 1.96, so the null hypothesis
that the population mean is INR 8,000 is rejected at the 5% level. A
Type I error here would mean concluding that average borrower spend
differs from INR 8,000 when the true population mean is exactly INR
8,000. The difference is about 11.7% of the planning assumption; it may
matter when budgets are aggregated across many borrowers, although the
synthetic data should not be treated as a real spending forecast.

### R1.2 --- Required sample size

The estimated sample size required for a margin of error of 0.5 thousand
INR at 95% confidence is 1,293 records, compared with 800 available. The
current sample is 493 records short of that target, so the planning team
would need a larger sample to meet the stated precision target under
this calculation.

### R1.3 --- Gradient descent check

The absolute gap between the gradient-descent slope and the closed-form
slope was 0.0000. Gradient descent reached the same line because the
squared-error objective is convex and the standardised single predictor
with the specified learning rate allowed the batch updates to converge.

### R1.4 --- Residual spread

The raw model's upper-to-lower residual standard-deviation ratio was
3.7525, indicating a clear funnel: residual spread is much larger at
higher fitted values. After regressing log(value) on log(X4), the ratio
fell to 1.0396, so the log transformation largely removed the funnel
pattern in this analysis.

### R1.5 --- Multiple regression

Using the required mechanical absolute-coefficient cutoff of 0.05, the
predictors above the cutoff were X4 (log average monthly inflow;
coefficient 1.1576), X2 (failed transactions; 0.0940), X3
(credit-utilisation ratio; 0.1067) and X6 (irregular-salary flag;
0.1518). The adjusted R² of the multiple regression was 0.7403, compared
with 0.6561 for the log-log model. On this synthetic dataset, the added
predictors improve adjusted fit; the coefficient cutoff is not a ranking
of causal or practical importance because the predictors use different
units.

## Part 2 findings

### R2.1 --- Learning-rate comparison

The 200-update training losses, from lowest to highest, were η = 1.0
(0.3703), η = 0.1 (0.3776) and η = 0.01 (0.5091). The smallest learning
rate made slower progress over the fixed number of updates; η = 1.0
performed best in this experiment, although that does not establish that
a large learning rate is safe for every dataset or model.

### R2.2 --- Logistic-regression weights

The three largest absolute standardised weights were X3, credit
utilisation (0.9013); X1, account age (0.5554); and X2, failed
transactions (0.5505). X3 also had the largest odds ratio, 2.4629:
holding the other predictors fixed, a one-training-standard-deviation
increase in credit utilisation multiplies the modelled odds of the
outcome by about 2.46.

### R2.3 --- Threshold trade-off

The blind all-negative model achieved accuracy 0.7875, close to the
logistic model's 0.8042 accuracy at τ = 0.50. Accuracy alone is weak
here because only 51 of the 240 test records were positive, so an
all-negative model already appears accurate. Lowering the logistic
threshold from 0.50 to 0.30 increased recall from 0.4706 to 0.7451,
while precision decreased from 0.5455 to 0.4634.

### R2.4 --- Logistic regression versus KNN

At τ = 0.30, logistic regression achieved F1 = 0.5714 and recall =
0.7451. KNN, using the selected K = 5, achieved F1 = 0.3696 and recall =
0.3333. Logistic regression performed better on both measures in this
test split.

### R2.5 --- Cost-based alert threshold

I would pilot logistic regression using τ = 0.15, selected by minimising
the specified false-alarm and missed-default cost on the training rows.
Its test cost at that threshold was INR 6,050, compared with INR 11,800
at τ = 0.50 and INR 20,400 for flagging nobody. The threshold was
selected using training rows so the test set remains an evaluation set
rather than influencing the operating decision.

## Part 3 findings

### R3.1 --- K-means and the elbow

The successive WCSS drops were 659.8282 (K = 2 to 3), 108.0739 (K = 3 to
4), 171.0575 (K = 4 to 5) and 45.5466 (K = 5 to 6). The pinned elbow
rule selects K = 3 because the drop from K = 3 to 4 is less than half
the preceding drop. The drops do not shrink steadily; the prescribed
starting rows give a reproducible solution, not necessarily the global
optimum for each K.

### R3.2 --- Segment profiles and actions

  -----------------------------------------------------------------------
  Segment                 Profile from            Suggested action
                          `part3_segments.csv`
  ----------------------- ----------------------- -----------------------
  A --- Established,      322 records; mean       Keep servicing simple
  steady users            account age 47.5028     and monitor routinely;
                          months, utilisation     avoid unnecessary
                          0.2250, repayment score high-intensity
                          62.1584 and outcome     collections for this
                          rate 0.0186. Mean spend low-outcome-rate group.
                          is 3.5976 thousand INR.

  B --- High-activity     255 records; mean       Offer timely payment
  spenders                failed transactions     support and investigate
                          6.0706, inflow 28.7192  repeated transaction
                          thousand INR and spend  friction, while
                          17.1248 thousand INR.   monitoring repayment
                          Outcome rate is 0.2353. behaviour.

  C --- New,              223 records; mean       Use closer early-life
  high-utilisation        account age 14.3892     monitoring and clear
  borrowers               months, utilisation     due-date reminders;
                          0.7118 and repayment    review affordability
                          score 38.5215. Outcome  and repayment friction
                          rate is 0.4664.         before extending
                                                  further exposure.
  -----------------------------------------------------------------------

Segment C has the highest outcome rate, 0.4664 (46.64%). These are
descriptive profiles of synthetic clusters, not validated borrower
policies.

### R3.3 --- Hierarchical clustering

At the largest-gap cut, single linkage gives four clusters: `{E1}`,
`{E2, E3, E6}`, `{E4}` and `{E5}`. Complete linkage gives two clusters:
`{E1, E4, E5}` and `{E2, E3, E6}`. Single linkage uses the closest pair
between clusters, so a sequence of short links can create a chain;
complete linkage uses the farthest pair and therefore delays a merge
until all cross-cluster pairs are sufficiently close. This difference
explains why the two rules produce different cuts on the supplied
distance table.

### R3.4 --- Principal components

The standardised five-feature matrix has a zero mean vector (up to
floating-point rounding). PC1 is positive for account age (X1, loading
0.5317) and repayment score (X5, 0.4418), and negative for credit
utilisation (X3, −0.5187), failed transactions (X2, −0.4100) and inflow
(X4, −0.2916). Three components retain at least 80% of the variance
(86.16%). The mean PC1 scores---A: 1.7773, B: −0.7831 and C:
−1.6709---place the established, lower-utilisation segment A on the
positive side and the newer, higher-utilisation segment C on the
negative side of this component.

## Part 4 findings

### R4.1 --- Split quality

S4 (X1 ≤ 24) was the best split under both criteria: its information
gain was 0.1252 and its Gini gain was 0.0606, higher than the next-best
S3 values of 0.1046 and 0.0494. A tree grown until every leaf is pure
can memorise noise and small idiosyncratic groups in the training
sample, so its performance may not generalise. A maximum-depth limit is
one practical stopping rule.

### R4.2 --- Model comparison and voting

I would choose logistic regression from the five Task 4.2 candidates: it
had the highest F1 (0.5714) with recall of 0.7451. The stump had
slightly higher recall (0.7647) but lower F1 (0.5532). Combining the
models did not help F1 in this test set: hard voting scored 0.5344 and
soft voting 0.5507, both below logistic regression; KNN scored 0.3696.

### R4.3 --- AdaBoost

After round 1 normalisation, the two misclassified observations (points
3 and 7) each had weight 0.2500, for a combined weight of 0.5000. Round
2 gives h2 more influence because its weighted error was lower (0.1667
versus 0.2500 for h1), producing a larger alpha (0.8047 versus 0.5493).

### R4.4 --- Neural-network sizing

Model A is an OCR-style classifier for 28 × 28 grayscale inputs and 10
digit classes. The table gives the output shape and parameter count for
each stage of the CNN option; ReLU, pooling and flattening have no
trainable parameters.

  Model A CNN stage                         Output shape   Parameters
  --------------------------------------- -------------- ------------
  Input                                      28 × 28 × 1            0
  Conv1 (5 × 5, 6 filters, padding 2)        28 × 28 × 6          156
  ReLU                                       28 × 28 × 6            0
  Max-pool 1 (2 × 2, stride 2)               14 × 14 × 6            0
  Conv2 (5 × 5, 16 filters, no padding)     10 × 10 × 16        2,416
  ReLU                                      10 × 10 × 16            0
  Max-pool 2 (2 × 2, stride 2)                5 × 5 × 16            0
  Flatten                                            400            0
  Dense1 + ReLU (32 neurons)                          32       12,832
  Dense output (10 neurons)                           10          330
  **CNN total**                                      ---   **15,734**

The alternative MLP has 13,002 parameters. For Model B, the RNN weight
shapes are W_xh = 128 × 64, W_hh = 128 × 128 and W_hy = 16 × 128; the
bias shapes are b_h = 128 and b_y = 16, giving 26,768 parameters in
total. The RNN parameter count does not grow with sequence length
because the same recurrent weights are reused at each time step. ReLU is
a suitable hidden-layer activation for the CNN/MLP, and a 10-class
softmax output with categorical cross-entropy is a suitable choice for
digit classification. In a very deep plain network, repeated
multiplication by small derivatives can make gradients shrink before
they reach early layers; skip connections provide a more direct path for
gradient and signal flow.

## Model-choice memo

To: Head of Lending-Risk Team

I recommend logistic regression as the first structured-data model to
pilot. On the 240-row test set at the Part 4 comparison threshold of
0.30, it achieved F1 = 0.5714 and recall = 0.7451, ahead of hard voting
(F1 = 0.5344) and soft voting (F1 = 0.5507). The stump had slightly
higher recall (0.7647) but lower F1 (0.5532). For the cost-based
operating threshold, use τ = 0.15, selected on training data; the
logistic model's test cost at that threshold was INR 6,050, compared
with INR 11,800 at τ = 0.50 and INR 20,400 for flagging nobody. These
figures favour a controlled pilot, not automatic production deployment.
The evidence is limited by synthetic data, one stratified split and only
51 positive test cases. Neither the OCR digit CNN (Model A) nor the
remark-classification RNN (Model B) should be built before validating
the structured model; they address separate tasks and have not been
trained or evaluated here.

## Limitations

1.  **Synthetic data:** all 800 records are generated by the supplied
    script. They are not PhonePe records and cannot establish real-world
    default patterns, customer behaviour or operational performance.
2.  **One train/test split:** the classifier is evaluated on one
    stratified 70/30 split. Results may vary with another sample or time
    period, and there is no temporal or external validation.
3.  **Small positive test group:** only 51 of the 240 test records are
    positive, so a small number of changed predictions can noticeably
    affect recall, precision and F1.
4.  **Simplified modelling:** the models use the supplied features and
    prescribed methods; calibration, fairness analysis, drift
    monitoring, operational constraints and intervention outcomes have
    not been evaluated.
5.  **Architecture sizing only:** the MLP, CNN and RNN are
    parameter-count exercises. They have not been trained, benchmarked
    or tested on cheque images or payment remarks.
