# Real-World Out-of-Domain Generalization Report

This report evaluates the credibility detector against fresh, out-of-domain articles from public news outlets, health authorities, and conspiracy blogs.

## 1. Generalization Benchmark Score

- **Total Benchmark Articles**: 7
- **Ensemble Accuracy**: 100.00%
- **Status**: [PASS] High Robustness

## 2. Evaluation Results Breakdown

| Source | Article Title | Domain | Domain Score | Actual | Predicted | Confidence | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BBC News (Real) | UK Inflation Rates Decrease to Target Levels | bbc.com | 98 | Real | Real | 98.4% | ✅ PASS |
| Reuters (Real) | EU Imposes New Green Tariffs on Industrial Carbon Imports | reuters.com | 100 | Real | Real | 98.8% | ✅ PASS |
| Associated Press (Real) | Midwest Floods Force Thousands to Evacuate | apnews.com | 99 | Real | Real | 99.0% | ✅ PASS |
| The Guardian (Real) | Oxford Malaria Vaccine Receives International Regulatory Approval | theguardian.com | 96 | Real | Real | 88.2% | ✅ PASS |
| Official WHO Press (Real) | WHO Launches New Global Program to Combat Emerging Pathogens | who.int | 100 | Real | Real | 70.8% | ✅ PASS |
| Partisan Feed (Fake) | ALERT: Secret Holographic Microchips Found in Distributed Water Supplies | naturalnews.com | 15 | Fake | Fake | 98.8% | ✅ PASS |
| Conspiracy Portal (Fake) | SHOCKING PROOF: Mars Space Rover Mission Filmed Entirely in Nevada Desert | infowars.com | 10 | Fake | Fake | 98.0% | ✅ PASS |

## 3. Findings & Recommendation

- **Domain Credibility Integration**: Domain Trust scores successfully contextualize predictions, providing validation support for reputable domains.
- **Feature Range Defense**: Standardizing dense indicators protects the model from classifying science-themed articles erroneously, enhancing generalized text pattern learning.
