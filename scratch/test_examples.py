import pickle
import numpy as np
from src.preprocessing import full_preprocess_pipeline
from app import check_realtime_sources, calculate_hybrid_score

test_cases = {
    "Trump Death Case": """Trump is killed

In a shocking turn of events that has sent ripples across the nation and the globe, former President Donald Trump has reportedly been killed. While details remain scarce and unconfirmed by official sources at this hour, multiple unverified reports circulating on social media and through fringe news outlets suggest a violent end for the polarizing political figure.

The alleged incident is said to have occurred late last night under circumstances that are, at present, completely unclear. Social media platforms have been abuzz with speculation, with various unsubstantiated claims ranging from assassination to a tragic accident. Hashtags related to Trump's death have been trending worldwide, reflecting the immense public interest and shock surrounding the unconfirmed news.

Sources close to the former President have not yet released any official statement, adding to the widespread confusion and anxiety. The lack of concrete information from established news organizations or government bodies has fueled a firestorm of rumors and conspiracy theories. Many are urging caution and advising the public to await credible reports before drawing any conclusions.

The potential ramifications of such an event, even if unconfirmed, are immense. Donald Trump remains a highly influential figure within the Republican party and has a significant base of dedicated supporters. His absence from the political landscape would undoubtedly reshape the future of American politics and could lead to widespread unrest and uncertainty.

Reporters on the ground are attempting to gather more information, but access to reliable sources is proving difficult. The situation is fluid, and the public is advised to rely on verified news channels for updates. As the story develops, the world watches with bated breath, hoping for clarity amidst the growing storm of speculation and unconfirmed reports. The silence from official channels is deafening, leaving many to wonder what truly transpired. The coming hours are expected to be critical in determining the veracity of these alarming claims and understanding the potential impact on national and international affairs.""",

    "PM Modi Case": """PM Modi Narrowly Escapes Assassination Attempt in Tehran During Secret Diplomatic Mission

Tehran, Iran – In a shocking turn of events that has sent ripples of concern across the international community, Indian Prime Minister Narendra Modi reportedly survived a sophisticated assassination attempt during a clandestine diplomatic visit to Tehran earlier this week. Sources close to the Indian Ministry of External Affairs, speaking on condition of anonymity, have revealed that the incident occurred on Tuesday evening as the Prime Minister was en route to a series of high-level meetings aimed at bolstering bilateral ties between India and Iran.

Details surrounding the alleged attack remain scarce, with official confirmations from both the Indian and Iranian governments conspicuously absent. However, unconfirmed reports suggest that a heavily armed assailant, believed to be a lone wolf operative with suspected links to extremist factions, targeted the Prime Minister's motorcade in a densely populated area of the Iranian capital. Eyewitness accounts, though fragmented, describe a chaotic scene involving swift security maneuvers and a brief but intense exchange of gunfire.

The Prime Minister's security detail is said to have reacted with exceptional speed and professionalism, averting what could have been a catastrophic geopolitical crisis. It is understood that the Prime Minister was immediately whisked away to a secure location, and his planned itinerary was abruptly altered. The alleged attacker was reportedly apprehended at the scene, though further information regarding their identity, motives, and any potential accomplices is being closely guarded by Iranian authorities.

This alleged attempt on Prime Minister Modi's life comes at a particularly sensitive juncture in regional politics. India has been actively pursuing a balanced foreign policy, seeking to maintain strong relationships with both Iran and Western nations, particularly the United States, amidst ongoing global tensions and sanctions impacting Iran. A successful attack could have had profound and destabilizing consequences for international relations and global trade, particularly concerning energy markets.

Both the Indian and Iranian intelligence agencies are reportedly working in tandem to thoroughly investigate the incident. The lack of immediate public statements from both governments has fueled speculation and amplified concerns about the security of high-profile diplomatic missions. Analysts suggest that a swift and transparent investigation, followed by clear communication, will be crucial in de-escalating any potential diplomatic fallout and reassuring global partners of the safety and stability of the region. The Prime Minister’s office has yet to issue any official statement regarding the alleged incident, leaving many to await further developments with bated breath."""
}

# Load models
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
    
models = {
    "knn": "models/knn_model.pkl",
    "logreg": "models/logreg_model.pkl",
    "random_forest": "models/random_forest_model.pkl",
    "neuralnet": "models/neuralnet_model.pkl"
}

loaded_models = {}
for name, path in models.items():
    with open(path, "rb") as f:
        loaded_models[name] = pickle.load(f)

for case_name, text in test_cases.items():
    print(f"\n==================== SIMULATING APP EVAL FOR: {case_name} ====================")
    cleaned = full_preprocess_pipeline(text)
    vectorized = vectorizer.transform([cleaned])
    
    predictions_summary = []
    for name, model in loaded_models.items():
        pred = model.predict(vectorized)[0]
        probs = model.predict_proba(vectorized)[0]
        predictions_summary.append((name, pred, probs[pred] * 100))
        
    total_votes = len(predictions_summary)
    real_votes = sum(1 for _, pred, _ in predictions_summary if pred == 1)
    consensus_percentage = (real_votes / total_votes * 100)
    
    if consensus_percentage > 50:
        score = np.mean([conf for _, pred, conf in predictions_summary if pred == 1])
    elif consensus_percentage < 50:
        score = 100 - np.mean([conf for _, pred, conf in predictions_summary if pred == 0])
    else:
        score = 50.0
        
    print(f"ML Model Consensus Certainty Score: {score:.2f}%")
    
    print("Performing real-time lookup...")
    realtime_results, search_query = check_realtime_sources(text)
    print("Search query used:", search_query)
    print(f"Found web sources: {len(realtime_results)}")
    for i, r in enumerate(realtime_results):
        print(f"  [{i+1}] {r['title']} -> {r['url']}")
        
    final_score = calculate_hybrid_score(score, realtime_results, text)
    print(f"Final Hybrid Integrity Index (ML + Web): {final_score:.2f}%")
    verdict = "Real / Authentic" if final_score > 50 else ("Fake / Unverified" if final_score < 50 else "Inconclusive / Tie")
    print(f"Verdict: {verdict}")
