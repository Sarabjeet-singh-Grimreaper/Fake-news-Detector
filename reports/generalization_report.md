# Real-World Out-of-Domain Generalization Report

This report benchmarks the Fake News Credibility Engine against raw out-of-domain articles from public news agencies, health authorities, and conspiracy feeds.

## 1. Executive Summary

- **Total Benchmark Articles**: 11
- **Baseline ML Accuracy**: 72.73% (8/11 passed)
- **Multi-Signal Real-Time Engine Accuracy**: 100.00% (11/11 passed)
- **Status**: [PASS] High Robustness

## 2. Multi-Signal Fusion Evaluation Breakdown

| Source File | Ground Truth | Multi-Signal Verdict | Score | Status | Primary Decision Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `real/AP.txt` | Real | Verified Real News | 88.0% | ✅ PASS | Verified in real-time across 3 recognized glo... |
| `real/BBC.txt` | Real | Verified Real News | 88.0% | ✅ PASS | Verified in real-time across 7 recognized glo... |
| `real/Government.txt` | Real | Likely Real | 60.8% | ✅ PASS | Linguistic markers, writing structure, and mo... |
| `real/Reuters.txt` | Real | Verified Real News | 88.0% | ✅ PASS | Verified in real-time across 4 recognized glo... |
| `real/WHO.txt` | Real | Verified Real News | 88.0% | ✅ PASS | Verified in real-time across 6 recognized glo... |
| `fake/AI_generated.txt` | Fake | Likely Fake | 26.1% | ✅ PASS | Stylistic markers strongly resemble fabricate... |
| `fake/Clickbait.txt` | Fake | Likely Fake | 27.9% | ✅ PASS | Stylistic markers strongly resemble fabricate... |
| `fake/Finance.txt` | Fake | Likely Fake | 47.7% | ✅ PASS | Stylistic irregularities, sensational markers... |
| `fake/Health.txt` | Fake | Likely Fake | 29.2% | ✅ PASS | Stylistic markers strongly resemble fabricate... |
| `fake/Political.txt` | Fake | Likely Fake | 26.2% | ✅ PASS | Stylistic markers strongly resemble fabricate... |
| `fake/Satire.txt` | Fake | Satire / Parody | 15.0% | ✅ PASS | Content exhibits recognized satirical phrasin... |

## 3. Baseline ML Standalone Breakdown

| Source File | Ground Truth | Model Prediction | Confidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| `real/AP.txt` | Real | Real | 68.0% | ✅ PASS |
| `real/BBC.txt` | Real | Real | 66.3% | ✅ PASS |
| `real/Government.txt` | Real | Real | 55.9% | ✅ PASS |
| `real/Reuters.txt` | Real | Fake | 53.8% | ❌ FAIL |
| `real/WHO.txt` | Real | Fake | 63.7% | ❌ FAIL |
| `fake/AI_generated.txt` | Fake | Fake | 73.2% | ✅ PASS |
| `fake/Clickbait.txt` | Fake | Fake | 90.1% | ✅ PASS |
| `fake/Finance.txt` | Fake | Fake | 55.3% | ✅ PASS |
| `fake/Health.txt` | Fake | Fake | 68.0% | ✅ PASS |
| `fake/Political.txt` | Fake | Fake | 73.0% | ✅ PASS |
| `fake/Satire.txt` | Fake | Real | 60.9% | ❌ FAIL |
