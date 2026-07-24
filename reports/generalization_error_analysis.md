# Robustness Generalization Failure Analysis

This report catalogs and explains model failures on unseen news sources.

## Error Distribution

| Failure Category | Count |
| :--- | :--- |
| political misinformation | 198 |
| general news bias | 146 |
| health misinformation | 75 |
| satire / conspiracy | 16 |
| clickbait | 5 |


## Sample Failures & Rationale

### Title: BREAKING: OBAMA-APPOINTED JUDGE ORDERS Vote Recount To Begin At Noon On Monday
- **Actual Label**: Fake
- **Category**: political misinformation
- **Probabilities**: Real=0.92, Fake=0.08
- **Excerpt**: *"A federal judge has ordered Michigan election officials to begin a massive hand recount of 4.8 million ballots cast in the presidential election at noon Monday.U.S. District Judge Mark Goldsmith issued a ruling just after midnight Monday in favor of ..."*
- **Failure Rationale**: Highly structured political rhetoric mimics official transcripts, leading the classifier to focus on formal vocabulary rather than factual truth.

### Title: HYSTERICAL…THE DEMOCRAT CONVENTION Schedule Is Revealed
- **Actual Label**: Fake
- **Category**: political misinformation
- **Probabilities**: Real=0.13, Fake=0.87
- **Excerpt**: *"LOL! You ll want to share this with everyone Democrat Convention ScheduleMonday, 25 July 2016 11:30 AM Free lunch, medical marijuana, and bus ride to the Convention. Forms distributed for Food Stamp enrollment.2:30 PM Group Voter Registration for Und..."*
- **Failure Rationale**: Highly structured political rhetoric mimics official transcripts, leading the classifier to focus on formal vocabulary rather than factual truth.

### Title: BREAKING: TRUMP Announces “Phenomenal” Tax Cut Plan For Businesses In Next 2-3 Weeks…Stock Markets Respond [VIDEO]
- **Actual Label**: Fake
- **Category**: political misinformation
- **Probabilities**: Real=0.21, Fake=0.79
- **Excerpt**: *"President Donald Trump said his administration would be announcing a  big league  tax cut that would lower the burden on businesses within the next two or three weeks.The revelation saw the Dow Jones industrial average rise around 115 points to a rec..."*
- **Failure Rationale**: Highly structured political rhetoric mimics official transcripts, leading the classifier to focus on formal vocabulary rather than factual truth.

### Title: VICE PRESIDENT PENCE BREAKS TIE In Bill Allowing States To Deny Federal Funds For Killing Babies
- **Actual Label**: Fake
- **Category**: health misinformation
- **Probabilities**: Real=0.14, Fake=0.86
- **Excerpt**: *" A society can be judged by how it deals with its most vulnerable: the aged, the infirm, the disabled and the unborn.    Mike PenceVice President Pence cast a tie-breaking Senate vote Thursday to pass legislation that will allow states to withhold fe..."*
- **Failure Rationale**: Science-themed vocabulary and references to agencies (like WHO/FDA) mislead readability indices and TF-IDF parameters.

### Title: WATCH A SHOCKING VIEW Of A Woman’s Life Under Sharia Law…Women’s March Organizer Is Pro-Sharia Law! [Video]
- **Actual Label**: Fake
- **Category**: political misinformation
- **Probabilities**: Real=0.10, Fake=0.90
- **Excerpt**: *"PLEASE LISTEN TO LINDA SARSOUR SPEAK ABOUT HOW DISSAPOINTED SHE IS THAT 22 STATES HAVE VOTED AGAINST SHARIA LAW:  HAHA! Leader and organizer of Washington #WomensMarch is annoyed that 22 US states don t allow Sharia laws. Blames Islamaphobia! pic.twi..."*
- **Failure Rationale**: Highly structured political rhetoric mimics official transcripts, leading the classifier to focus on formal vocabulary rather than factual truth.

### Title: WHY OBAMA’S CORRUPT INNER CIRCLE Is Desperately Fighting To Protect His Failed Reputation
- **Actual Label**: Fake
- **Category**: political misinformation
- **Probabilities**: Real=0.11, Fake=0.89
- **Excerpt**: *" History shows that we can, if we must, tolerate nuclear weapons in North Korea. Those words were written by former National Security Adviser Susan Rice on Thursday in the New York Times, in arguing for appeasement towards Kim Jong-un.It was also the..."*
- **Failure Rationale**: Highly structured political rhetoric mimics official transcripts, leading the classifier to focus on formal vocabulary rather than factual truth.

### Title: NEWSFLASH FOR OUR IMPERIAL PRESIDENT: STATES CAN REFUSE IRAN DEAL [Video]
- **Actual Label**: Fake
- **Category**: political misinformation
- **Probabilities**: Real=0.76, Fake=0.24
- **Excerpt**: *"As Barack Hussein Obama tours around the country trying to convince the low information voter that the lopsided and dangerous deal he and John Kerry cut with Iran is somehow beneficial to the United States of America he may want to consider the state..."*
- **Failure Rationale**: Highly structured political rhetoric mimics official transcripts, leading the classifier to focus on formal vocabulary rather than factual truth.

### Title: FBI FILES REVEALED: VALERIE JARRETT’S FAMILY TIES TO COMMUNISM RUN DEEP
- **Actual Label**: Fake
- **Category**: political misinformation
- **Probabilities**: Real=0.58, Fake=0.42
- **Excerpt**: *"I m sure the apple doesn t fall far from the tree in this case. Many of us have known about the father-in-law but this digs deeper into Valerie Jarrett s own family. Federal Bureau of Investigation (FBI) files obtained by Judicial Watch reveal that t..."*
- **Failure Rationale**: Highly structured political rhetoric mimics official transcripts, leading the classifier to focus on formal vocabulary rather than factual truth.

### Title: Madrid representative in Catalonia apologizes for police violence during independence vote
- **Actual Label**: Real
- **Category**: health misinformation
- **Probabilities**: Real=0.06, Fake=0.94
- **Excerpt**: *"MADRID (Reuters) - The Spanish government s official representative in Catalonia  apologized on Friday for the violent response by Spanish police to protesters who were attempting to vote in a banned independence referendum in the region on Sunday.  ..."*
- **Failure Rationale**: Science-themed vocabulary and references to agencies (like WHO/FDA) mislead readability indices and TF-IDF parameters.

### Title: WOW! Why Florida Jews Deserted Hillary…Helped Trump To Win Florida
- **Actual Label**: Fake
- **Category**: political misinformation
- **Probabilities**: Real=0.80, Fake=0.20
- **Excerpt**: *"Hillary Clinton matched President Obama s historically low percentage of the Jewish vote, driving a third of Jews into the Republican camp, an unheard of percentage before Obama s pro-Iran policies. Obama also turned Jews off by his blatant personal ..."*
- **Failure Rationale**: Highly structured political rhetoric mimics official transcripts, leading the classifier to focus on formal vocabulary rather than factual truth.

