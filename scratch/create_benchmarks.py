import os

eval_data = {
    "evaluation/real/BBC.txt": "The UK economy grew by 0.2% in the latest quarter, official statistics show. The Office for National Statistics said growth was driven by the service sector, particularly retail and hospitality, offsetting a slight decline in manufacturing.",
    
    "evaluation/real/Reuters.txt": "Global oil prices stabilized on Friday as traders weighed supply disruptions in the Middle East against rising production in North America. Brent crude futures settled at $78.50 a barrel, while West Texas Intermediate rose slightly.",
    
    "evaluation/real/WHO.txt": "The World Health Organization has issued a new policy brief on mental health at work, highlighting recommendations to prevent psychosocial risks. Safe and healthy working environments are a fundamental right.",
    
    "evaluation/real/AP.txt": "Severe storms swept across the Midwest on Wednesday, leaving thousands without power and causing significant structural damage to homes and businesses. Local authorities have declared a state of emergency.",
    
    "evaluation/real/Government.txt": "The Department of Labor announced new initiatives to expand registered apprenticeship opportunities in high-growth industries, investing $50 million in grants to state agencies and workforce partnerships.",
    
    "evaluation/fake/AI_generated.txt": "In a historic and ground-breaking development, scientists have successfully engineered a quantum-biological hybrid leaf that absorbs light and outputs gold particles. This incredible green gold leaf could solve poverty instantly.",
    
    "evaluation/fake/Clickbait.txt": "SHOCKING TRUTH: You won't believe what these politicians did behind closed doors! This secret trick will make you rich in 24 hours, and doctors are absolutely shocked. Watch the video now before it gets banned!",
    
    "evaluation/fake/Satire.txt": "The nation's top conspiracy theorists officially conceded today that the Earth is indeed a sphere, citing that it's simply too difficult to keep drawing flat maps without falling off the edge of their desks during presentations.",
    
    "evaluation/fake/Political.txt": "BREAKING: Inside sources reveal secret deep-state shadow meetings designed to implement global controls on food supply systems. Elite forces are preparing covert operations to restrict citizens from growing home gardens.",
    
    "evaluation/fake/Health.txt": "A revolutionary miracle cure has been uncovered in the rainforest. Drinking warm water infused with rare blue orchids completely cures all metabolic diseases within three days. Big Pharma is desperately trying to hide this!",
    
    "evaluation/fake/Finance.txt": "URGENT WARNING: The global banking system is scheduled for a total manual reset tonight at midnight. All personal digital balances will be set to zero and replaced by standard state-issued credit coupons. Withdraw cash immediately!"
}

for path, content in eval_data.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created benchmark: {path}")
