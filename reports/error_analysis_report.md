# Robustness & Error Analysis Diagnostics Report

## 1. Classification Metrics Summary

- **Total Test Evaluation Samples**: 8835
- **Misclassified Instances**: 102 / 8835 (Accuracy: 98.8455%)
- **False Positives (Fake predicted as Real)**: 63
- **False Negatives (Real predicted as Fake)**: 39

### Detailed Classification Report
```
              precision    recall  f1-score   support

    Fake (0)       0.99      0.99      0.99      4552
    Real (1)       0.99      0.99      0.99      4283

    accuracy                           0.99      8835
   macro avg       0.99      0.99      0.99      8835
weighted avg       0.99      0.99      0.99      8835
```

## 2. Hardest Articles (Highest Confidence Errors)

### 1. Original Index: 24708 (Error Confidence: 99.94%)
- **Title**: BREAKING: MUSLIM OPENS FIRE ON JOURNALISTS [Video]
- **Actual Label**: Fake | **Model Prediction**: Real
- **Content Excerpt**:
  > *"Religion of peace h/t Weasel Zippers..."*

### 2. Original Index: 41982 (Error Confidence: 99.55%)
- **Title**: BREAKING: Republican Majority House Caves To Obama…Narrowly Passes TPA Bill
- **Actual Label**: Fake | **Model Prediction**: Real
- **Content Excerpt**:
  > *"Just when you thought the 2014 election results would provide America with some checks and balances U.S. lawmakers narrowly approved legislation key to securing a hallmark Pacific trade deal on Thursday, partly reversing a defeat less than a week before, in a boost to President Barack Obama s goal of strengthening U.S. economic ties with Asia.The H..."*

### 3. Original Index: 30044 (Error Confidence: 99.38%)
- **Title**: CHECKMATE! PUTIN Offers Up Proof Of Trump’s Innocence [Video]
- **Actual Label**: Fake | **Model Prediction**: Real
- **Content Excerpt**:
  > *"Putin just trolled the Democrats Haha!Vladimir Putin said on Wednesday that U.S. President Donald Trump had not divulged any secrets during a meeting in Washington with Russian officials and offered to prove it by supplying Congress with a transcript.But a leading U.S. Republican politician said he would have little faith in any notes Putin might s..."*

### 4. Original Index: 9023 (Error Confidence: 98.85%)
- **Title**: SENATE PASSES USA FREEDOM ACT
- **Actual Label**: Fake | **Model Prediction**: Real
- **Content Excerpt**:
  > *"McConnell was shut down completely in this debate as all amendments he put forth were rejected including the McConnell-Burr Amendment.Congress approved sweeping changes Tuesday to surveillance laws enacted after the Sept. 11 attacks, eliminating the National Security Agency s disputed bulk phone-records collection program and replacing it with a mo..."*

### 5. Original Index: 25652 (Error Confidence: 98.72%)
- **Title**: Disillusionment, U.S.A. Where voters are just hoping for change.
- **Actual Label**: Real | **Model Prediction**: Fake
- **Content Excerpt**:
  > *"ALGONAC, MICH.—Parker Fox drifted out of the Donald Trump rally in a sort of euphoric daze, along with the thousands emptying into the parking lot alongside him. “After leaving a Trump rally, you’re very pro-Trump,” he recalled a few weeks later, describing a noisy communion with people who understood that politics mattered, unlike some people he c..."*

## 3. Discriminative Vocabulary Coefficients

### Top 15 Words Predicting FAKE News:
- `century wire`: -12.4104
- `wire`: -11.8254
- `gop`: -11.4837
- `obama`: -11.2074
- `president trump`: -10.5740
- `doesn`: -9.8516
- `more`: -9.4026
- `america`: -9.0486
- `didn`: -8.9352
- `president obama`: -8.9331
- `sen`: -8.7030
- `century`: -8.4315
- `even`: -8.4292
- `isn`: -8.0016
- `flickr`: -7.5566

### Top 15 Words Predicting REAL News:
- `president donald`: 17.4973
- `nov`: 9.3769
- `did not`: 8.8088
- `republican`: 8.3305
- `president barack`: 8.1257
- `statement`: 7.0844
- `had`: 6.7388
- `bit`: 6.6240
- `rival`: 6.6172
- `democratic`: 6.4083
- `president`: 6.2902
- `london`: 6.1137
- `britain`: 6.0802
- `inc`: 6.0076
- `nov election`: 5.9505
